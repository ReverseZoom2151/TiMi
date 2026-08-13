"""Base agent class, and the shared parsing and validation boundary.

Everything a language model returns is untrusted text. It arrives wrapped in
markdown fences or prose, it can be cut short by a token limit, and the numbers
inside it can be negative, infinite, absurd, or not numbers at all. This module
holds the one set of helpers every agent uses to turn that text into values the
rest of the system is allowed to see:

* `extract_json_object` finds a genuinely balanced JSON object, tolerating
  fences and surrounding prose, and returns None rather than guessing.
* `response_was_truncated` spots a reply that stopped at the token limit, which
  is the usual reason a payload is malformed.
* `validate_numeric_parameters` bounds every numeric value against an explicit
  specification and reports what it rejected.

None of these helpers ever swallow `KeyboardInterrupt`, `SystemExit` or
`asyncio.CancelledError`: a run must always remain cancellable, even in the
middle of parsing a reply.
"""

import json
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from dataclasses import dataclass

from ..llm.client import LLMClient
from ..utils.config import Config
from ..utils.logging import get_logger


# Exceptions a decode attempt may legitimately raise. `json.JSONDecodeError`
# is a `ValueError`. Listing them explicitly keeps control-flow exceptions
# (cancellation, interrupt, exit) propagating.
DECODE_ERRORS: Tuple[type, ...] = (ValueError, TypeError)

# Stop reasons that mean the reply was cut short rather than finished. The
# providers do not agree on spelling, so both vocabularies are listed. The
# client records the value but does not act on it, so the check lives here.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "model_length"})


@dataclass
class AgentResult:
    """Result from an agent action."""
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class NumericBound:
    """Admissible range for a single numeric parameter.

    Attributes:
        minimum: Smallest accepted value, inclusive
        maximum: Largest accepted value, inclusive
        integer: Whether the value must be a whole number
    """

    minimum: float
    maximum: float
    integer: bool = False


# The band every numeric strategy parameter must sit inside before it is
# allowed anywhere near order sizing. These are deliberately generous outer
# limits: their job is to reject broken output, not to tune anything. The risk
# layer applies the tighter, account-specific limits afterwards, and both gates
# have to pass.
PARAMETER_BOUNDS: Dict[str, NumericBound] = {
    'capital_allocation': NumericBound(1.0, 1_000_000.0),
    'grid_levels': NumericBound(1, 50, integer=True),
    'grid_spacing': NumericBound(0.0001, 100.0),
    'max_position_pct': NumericBound(0.1, 100.0),
    'stop_loss_pct': NumericBound(0.0001, 0.5),
    'volatility_multiplier': NumericBound(0.0, 100.0),
}


def response_was_truncated(response: Any) -> bool:
    """Report whether a model reply stopped because it ran out of tokens.

    Args:
        response: An object carrying a `finish_reason` attribute, or anything
            else, in which case the answer is False

    Returns:
        True when the reply was cut short
    """

    reason = getattr(response, 'finish_reason', None)
    if not isinstance(reason, str):
        return False

    return reason.strip().lower() in TRUNCATION_REASONS


def _fenced_blocks(text: str) -> Iterator[str]:
    """Yield the contents of every markdown fenced block, outermost first.

    Args:
        text: Raw reply text

    Yields:
        The body of each fenced block, language tag removed
    """

    remainder = text
    while True:
        opening = remainder.find('```')
        if opening == -1:
            return

        after_fence = remainder.find('\n', opening)
        if after_fence == -1:
            return

        closing = remainder.find('```', after_fence)
        if closing == -1:
            # An unterminated fence is what a truncated reply looks like.
            yield remainder[after_fence + 1:]
            return

        yield remainder[after_fence + 1:closing]
        remainder = remainder[closing + 3:]


def _balanced_spans(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` span in the text, outermost first.

    Scanning with a depth counter that understands string literals is what
    separates this from `find('{')` with `rfind('}')`: a stray brace in the
    surrounding prose no longer stretches the span over the real payload, and
    a reply truncated part way through an object yields nothing rather than a
    span that happens to close on the wrong brace.

    Args:
        text: Text to scan

    Yields:
        Balanced object spans, including their braces
    """

    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = index
            depth += 1
        elif char == '}':
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start:index + 1]
                start = -1


def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """Extract one JSON object from a reply that may be wrapped in anything.

    Fenced blocks are searched first, because a reply that bothered to fence
    its payload has told us where the payload is. Otherwise the whole text is
    scanned for balanced spans. A reply with no decodable object, including one
    cut off mid-object, yields None so the caller can fall back deliberately.

    Args:
        text: Reply text

    Returns:
        The decoded object, or None when there is nothing trustworthy in it
    """

    if not isinstance(text, str) or '{' not in text:
        return None

    candidates: List[str] = []
    for block in _fenced_blocks(text):
        candidates.extend(_balanced_spans(block))
    candidates.extend(_balanced_spans(text))

    for span in candidates:
        try:
            decoded = json.loads(span)
        except DECODE_ERRORS:
            continue
        if isinstance(decoded, dict):
            return decoded

    return None


def validate_numeric_parameters(
    candidate: Any,
    bounds: Mapping[str, NumericBound],
    defaults: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Bound every known numeric parameter, falling back per key.

    A value is accepted only if it is a real number (booleans excluded, strings
    excluded, since a producer that sends "100" is a producer that might send
    "one hundred"), finite, and inside its declared band. Anything else is
    replaced by the documented default for that key. Keys with no declared
    bound are dropped entirely: an unrecognised knob must not travel onwards.

    Args:
        candidate: Parsed parameter mapping from a model reply, or anything
        bounds: Admissible range per parameter name
        defaults: Safe value per parameter name, used whenever one is rejected

    Returns:
        A tuple of the accepted parameters and the names that were rejected
    """

    accepted: Dict[str, Any] = dict(defaults)
    rejected: List[str] = []

    if not isinstance(candidate, Mapping):
        return accepted, sorted(bounds)

    for name, bound in bounds.items():
        if name not in candidate:
            continue

        value = candidate[name]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            rejected.append(name)
            continue

        numeric = float(value)

        if not math.isfinite(numeric):
            rejected.append(name)
            continue

        if numeric < bound.minimum or numeric > bound.maximum:
            rejected.append(name)
            continue

        if bound.integer:
            if numeric != int(numeric):
                rejected.append(name)
                continue
            accepted[name] = int(numeric)
        else:
            accepted[name] = numeric

    return accepted, rejected


class BaseAgent(ABC):
    """Base class for all TiMi agents.

    Each agent has access to specialized LLM capabilities:
    - φ (semantic analysis)
    - ψ (code programming)
    - γ (mathematical reasoning)
    """

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        config: Optional[Config] = None
    ):
        """Initialize agent.

        Args:
            name: Agent name
            llm_client: LLM client for generating responses
            config: Configuration object
        """
        self.name = name
        self.llm_client = llm_client
        self.config = config or Config()
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    async def execute(self, *args, **kwargs) -> AgentResult:
        """Execute the agent's primary function.

        Returns:
            AgentResult with execution outcome
        """
        pass

    def log_action(self, action: str, **kwargs):
        """Log an agent action.

        Args:
            action: Action description
            **kwargs: Additional context
        """
        self.logger.info(
            "agent_action",
            agent=self.name,
            action=action,
            **kwargs
        )

    def log_error(self, error: Exception, context: Optional[dict] = None):
        """Log an error.

        Args:
            error: Exception object
            context: Additional context
        """
        self.logger.error(
            "agent_error",
            agent=self.name,
            error_type=type(error).__name__,
            error_message=str(error),
            **(context or {})
        )

    def log_fallback(self, stage: str, reason: str, **kwargs) -> None:
        """Record that a documented fallback was used instead of a reply.

        A parse failure that leaves no trace is indistinguishable from a model
        that genuinely returned the default, so every fallback says so.

        Args:
            stage: Where the fallback was taken
            reason: Why the reply could not be used
            **kwargs: Additional context
        """

        self.logger.warning(
            "agent_fallback",
            agent=self.name,
            stage=stage,
            reason=reason,
            **kwargs
        )
