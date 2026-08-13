"""Tests for the agent layer, and for the boundary it defends.

The agents are the only part of this system that takes instructions from a
language model, and the only thing standing between a model reply and an order
quantity. So these tests are less about what the agents produce when everything
works, and more about what they refuse to produce when it does not: fenced
replies, prose-wrapped replies, replies cut off at the token limit, refusals,
and replies carrying numbers that are negative, zero, infinite, absurd, or not
numbers at all.

Two structural properties are also asserted here rather than left to review:
that no module under `timi/` can execute a generated string, and that a
cancelled or interrupted run stays cancelled instead of being caught by a
parser.
"""

from __future__ import annotations

import ast
import asyncio
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from timi.agents import (
    BotEvolutionAgent,
    FeedbackReflectionAgent,
    MacroAnalysisAgent,
    StrategyAdaptationAgent,
    extract_json_object,
    response_was_truncated,
)
from timi.agents.base import PARAMETER_BOUNDS, validate_numeric_parameters


TIMI_ROOT = Path(__file__).resolve().parents[1] / "timi"


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class StubResponse:
    """A stand-in for an LLM response, with a controllable finish reason."""

    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.model = "stub"
        self.tokens_used = 0
        self.finish_reason = finish_reason
        self.metadata: Dict[str, Any] = {}


class StubLLMClient:
    """An LLM client that returns canned text and never opens a socket.

    Records every prompt, so a test can assert that a call was made, or that
    one was deliberately not made.
    """

    def __init__(self, content: str = "", finish_reason: str = "stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls: List[str] = []

    async def _reply(self, prompt: str, system_prompt: str = None, **kwargs):
        self.calls.append(prompt)
        return StubResponse(self.content, self.finish_reason)

    generate_semantic = _reply
    generate_reasoning = _reply
    generate_code = _reply


class RaisingLLMClient:
    """An LLM client that raises a control-flow exception mid-call."""

    def __init__(self, error: BaseException):
        self.error = error

    async def _raise(self, prompt: str, system_prompt: str = None, **kwargs):
        raise self.error

    generate_semantic = _raise
    generate_reasoning = _raise
    generate_code = _raise


class ExplodingResponse:
    """A response whose content cannot be read without raising."""

    def __init__(self, error: BaseException):
        self._error = error

    @property
    def content(self):
        raise self._error

    @property
    def finish_reason(self):
        return "stop"


PAIR_PROFILE = {
    "pair": "BTC/USDT",
    "volatility": 0.04,
    "risk_category": "low",
    "liquidity_score": "high",
    "price": 30_000.0,
}


def _strategy_agent(content: str = "", finish_reason: str = "stop"):
    return StrategyAdaptationAgent(
        StubLLMClient(content, finish_reason),
        market_data=None
    )


def _reflection_agent(content: str = "", finish_reason: str = "stop"):
    return FeedbackReflectionAgent(StubLLMClient(content, finish_reason))


# --------------------------------------------------------------------------
# The no-execution property
# --------------------------------------------------------------------------


FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__"}


def _python_sources() -> List[Path]:
    return [
        path for path in TIMI_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_repository_contains_no_dynamic_execution():
    """No module under timi/ may execute, evaluate or compile a string.

    Generated bot code is stored as an inert string. This test is what keeps
    it that way: adding `exec` anywhere in the package fails the suite, and a
    reviewer never has to notice it by hand.
    """

    offenders = []

    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in FORBIDDEN_CALLS:
                    offenders.append(f"{path}:{node.lineno} calls {name}()")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "importlib":
                        offenders.append(f"{path}:{node.lineno} imports importlib")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "importlib":
                    offenders.append(f"{path}:{node.lineno} imports importlib")

    assert offenders == [], "dynamic execution found: " + "; ".join(offenders)


def test_repository_never_writes_a_python_module():
    """No module under timi/ may write generated text to a .py file.

    Writing a module is executing it one import later.
    """

    offenders = []

    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".py'" in line or '.py"' in line:
                if "open(" in line or "write" in line or "write_text" in line:
                    offenders.append(f"{path}:{number}: {stripped}")

    assert offenders == [], "python file writes found: " + "; ".join(offenders)


async def test_generated_bot_code_is_marked_inert():
    """The artefact says in its own payload that it is not to be run."""

    agent = BotEvolutionAgent(
        StubLLMClient("```python\nclass TradingBot:\n    pass\n```")
    )

    result = await agent.execute(
        {"name": "grid"}, {"capital_allocation": 100.0}, "BTC/USDT"
    )

    assert result.success is True
    assert result.data["executable"] is False
    assert isinstance(result.data["code"], str)


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def test_extracts_json_from_a_markdown_fence():
    text = 'Here you go:\n```json\n{"capital_allocation": 50}\n```\nHope that helps.'
    assert extract_json_object(text) == {"capital_allocation": 50}


def test_extracts_json_wrapped_in_prose_with_a_stray_brace():
    """A brace in the prose must not stretch the span over the payload."""

    text = (
        "I considered a set {of options} before deciding. "
        'The answer is {"capital_allocation": 50}. '
        "That concludes my {reasoning}."
    )
    assert extract_json_object(text) == {"capital_allocation": 50}


def test_truncated_json_yields_nothing():
    text = '```json\n{"capital_allocation": 50, "grid_lev'
    assert extract_json_object(text) is None


def test_malformed_and_refusal_replies_yield_nothing():
    assert extract_json_object("{not: valid, json}") is None
    assert extract_json_object("I cannot help with that request.") is None
    assert extract_json_object("") is None
    assert extract_json_object(None) is None


def test_truncation_is_detected_from_the_finish_reason():
    assert response_was_truncated(StubResponse("", "length")) is True
    assert response_was_truncated(StubResponse("", "max_tokens")) is True
    assert response_was_truncated(StubResponse("", "stop")) is False
    assert response_was_truncated(object()) is False


# --------------------------------------------------------------------------
# Numeric validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [-100.0, 0, float("nan"), float("inf"), float("-inf"), "100", "a lot",
     None, True, [100], {"value": 100}, 10 ** 9],
)
def test_bad_capital_allocation_is_rejected(bad_value):
    """A bad allocation never reaches the caller, whatever shape it arrives in."""

    defaults = {"capital_allocation": 75.0}

    accepted, rejected = validate_numeric_parameters(
        {"capital_allocation": bad_value},
        PARAMETER_BOUNDS,
        defaults
    )

    assert "capital_allocation" in rejected
    assert accepted["capital_allocation"] == 75.0


def test_good_capital_allocation_passes_through():
    accepted, rejected = validate_numeric_parameters(
        {"capital_allocation": 250.0},
        PARAMETER_BOUNDS,
        {"capital_allocation": 75.0}
    )

    assert rejected == []
    assert accepted["capital_allocation"] == 250.0


def test_unknown_parameters_are_dropped():
    """An unrecognised knob must not travel onwards."""

    accepted, _ = validate_numeric_parameters(
        {"capital_allocation": 120.0, "leverage": 100},
        PARAMETER_BOUNDS,
        {"capital_allocation": 75.0}
    )

    assert "leverage" not in accepted


def test_non_integer_value_for_an_integer_parameter_is_rejected():
    accepted, rejected = validate_numeric_parameters(
        {"grid_levels": 7.5},
        PARAMETER_BOUNDS,
        {"grid_levels": 5}
    )

    assert "grid_levels" in rejected
    assert accepted["grid_levels"] == 5


# --------------------------------------------------------------------------
# Strategy adaptation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I cannot provide trading parameters.",
        "{capital_allocation: fifty}",
        '```json\n{"capital_allocation": 50, "grid_lev',
        "Sorry, that request falls outside what I can do.",
    ],
)
def test_unusable_parameter_replies_fall_back_to_the_documented_defaults(reply):
    agent = _strategy_agent()
    defaults = agent._default_parameters(PAIR_PROFILE)

    parsed = agent._parse_parameters(StubResponse(reply), PAIR_PROFILE)

    assert parsed == defaults
    assert parsed["capital_allocation"] > 0


def test_fenced_parameter_reply_is_used_after_validation():
    agent = _strategy_agent()

    parsed = agent._parse_parameters(
        StubResponse('```json\n{"capital_allocation": 250, "grid_levels": 8}\n```'),
        PAIR_PROFILE
    )

    assert parsed["capital_allocation"] == 250.0
    assert parsed["grid_levels"] == 8


@pytest.mark.parametrize(
    "bad_value", [-50, 0, "lots", 10 ** 12, None]
)
def test_bad_allocation_in_a_reply_is_replaced_by_the_default(bad_value):
    agent = _strategy_agent()
    defaults = agent._default_parameters(PAIR_PROFILE)

    parsed = agent._parse_parameters(
        StubResponse('{"capital_allocation": %r}' % (bad_value,)
                     if not isinstance(bad_value, str)
                     else '{"capital_allocation": "%s"}' % bad_value),
        PAIR_PROFILE
    )

    assert parsed["capital_allocation"] == defaults["capital_allocation"]


def test_nan_allocation_in_a_reply_is_replaced_by_the_default():
    """NaN survives a JSON decode, so it has to be caught by value."""

    agent = _strategy_agent()
    defaults = agent._default_parameters(PAIR_PROFILE)

    parsed = agent._parse_parameters(
        StubResponse('{"capital_allocation": NaN}'), PAIR_PROFILE
    )

    assert parsed["capital_allocation"] == defaults["capital_allocation"]
    assert math.isfinite(parsed["capital_allocation"])


def test_truncated_parameter_reply_falls_back_and_stays_finite():
    agent = _strategy_agent()
    defaults = agent._default_parameters(PAIR_PROFILE)

    parsed = agent._parse_parameters(
        StubResponse('{"capital_allocation": 250, "grid', "length"),
        PAIR_PROFILE
    )

    assert parsed == defaults


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"selected_strategy": "trend"}', "trend"),
        ('```json\n{"selected_strategy": "mean-reversion"}\n```', "mean-reversion"),
        ('{"selected_strategy": "martingale-doubling"}', "grid"),
        ("I cannot help with that.", "grid"),
        ("", "grid"),
    ],
)
def test_strategy_selection_stays_inside_the_known_set(reply, expected):
    agent = _strategy_agent()
    assert agent._parse_strategy_selection(reply) == expected


# --------------------------------------------------------------------------
# Feedback reflection
# --------------------------------------------------------------------------


CURRENT_PARAMETERS = {
    "capital_allocation": 100.0,
    "grid_levels": 5,
    "max_position_pct": 10.0,
    "stop_loss_pct": 0.05,
}


def test_unusable_optimisation_reply_holds_the_current_parameters():
    agent = _reflection_agent()

    optimal = agent._parse_optimal_parameters(
        StubResponse("I am unable to assist with that."),
        CURRENT_PARAMETERS,
        {}
    )

    assert optimal == CURRENT_PARAMETERS


def test_absurd_proposed_allocation_is_rejected_and_stepped():
    """A proposal outside the band is dropped; one inside it is still stepped."""

    agent = _reflection_agent()

    optimal = agent._parse_optimal_parameters(
        StubResponse('{"capital_allocation": 1e12}'),
        CURRENT_PARAMETERS,
        {}
    )
    assert optimal["capital_allocation"] == 100.0

    optimal = agent._parse_optimal_parameters(
        StubResponse('{"capital_allocation": 900}'),
        CURRENT_PARAMETERS,
        {}
    )
    assert optimal["capital_allocation"] == pytest.approx(150.0)


def test_drawdown_shrinks_exposure_and_nothing_else():
    agent = _reflection_agent()

    feedback = {
        "risk": [{"type": "drawdown", "value": 0.10, "severity": "medium"}],
        "performance": [],
    }

    optimal = agent._parse_optimal_parameters(
        StubResponse("no json here"), CURRENT_PARAMETERS, feedback
    )

    # Half the 20% drawdown budget spent, so half the exposure.
    assert optimal["capital_allocation"] == pytest.approx(50.0)
    assert optimal["max_position_pct"] == pytest.approx(5.0)
    assert optimal["stop_loss_pct"] == pytest.approx(0.05)
    assert optimal["grid_levels"] == 5


def test_de_risking_never_goes_below_the_floor():
    agent = _reflection_agent()

    feedback = {
        "risk": [{"type": "drawdown", "value": 0.95, "severity": "critical"}],
        "performance": [{"type": "win_rate", "value": 0.01, "severity": "high"}],
    }

    optimal = agent._parse_optimal_parameters(
        StubResponse(""), CURRENT_PARAMETERS, feedback
    )

    assert optimal["capital_allocation"] == pytest.approx(25.0)


def test_broken_current_parameters_are_not_carried_forward():
    """Rubbish already in the parameter book does not survive a reflection."""

    agent = _reflection_agent()

    optimal = agent._parse_optimal_parameters(
        StubResponse(""),
        {"capital_allocation": float("inf"), "grid_levels": 5},
        {}
    )

    assert "capital_allocation" not in optimal
    assert optimal["grid_levels"] == 5


# --------------------------------------------------------------------------
# Macro analysis uses the reply, but only through the catalogue
# --------------------------------------------------------------------------


def test_macro_analysis_adds_only_known_strategy_names():
    agent = MacroAnalysisAgent(StubLLMClient(), market_data=None)

    strategies = agent._parse_strategies(
        "I recommend mean reversion, and also martingale-doubling.",
        {"avg_volatility": 0.0, "trend_distribution": {}}
    )

    names = [entry["name"] for entry in strategies]
    assert "mean-reversion" in names
    assert all(
        name in {"grid", "trend", "mean-reversion", "stat-arb"} for name in names
    )


def test_macro_analysis_always_returns_a_baseline_strategy():
    agent = MacroAnalysisAgent(StubLLMClient(), market_data=None)

    for reply in ("", "I cannot help with that.", "{broken"):
        strategies = agent._parse_strategies(reply, {})
        assert strategies
        assert strategies[0]["name"] == "grid"


# --------------------------------------------------------------------------
# Cancellation and interruption must not be swallowed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), asyncio.CancelledError(), SystemExit(1)]
)
def test_control_flow_exceptions_propagate_out_of_parsing(error):
    agent = _strategy_agent()

    with pytest.raises(type(error)):
        agent._parse_parameters(ExplodingResponse(error), PAIR_PROFILE)


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), asyncio.CancelledError()]
)
async def test_control_flow_exceptions_propagate_out_of_reflection(error):
    agent = FeedbackReflectionAgent(RaisingLLMClient(error))

    with pytest.raises(type(error)):
        await agent.execute(
            {"pair": "BTC/USDT", "parameters": CURRENT_PARAMETERS},
            {"max_drawdown": 0.5, "pnl": -500}
        )


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt(), asyncio.CancelledError()]
)
async def test_control_flow_exceptions_propagate_out_of_bot_generation(error):
    agent = BotEvolutionAgent(RaisingLLMClient(error))

    with pytest.raises(type(error)):
        await agent.execute({"name": "grid"}, {}, "BTC/USDT")


# --------------------------------------------------------------------------
# Stage II wiring
# --------------------------------------------------------------------------


def _system_with_stub_agents(bot_reply: str, reflection_reply: str):
    """Build a system with only the stage II agents wired, no exchange."""

    from timi.main import TiMiSystem
    from timi.utils.config import Config

    system = TiMiSystem(Config())
    system.bot_evolution_agent = BotEvolutionAgent(StubLLMClient(bot_reply))
    system.feedback_agent = FeedbackReflectionAgent(
        StubLLMClient(reflection_reply)
    )
    return system


async def test_optimization_stage_stores_an_inert_bot_and_skips_idle_reflection():
    """With no feedback there is nothing to reflect on, so nothing is asked."""

    system = _system_with_stub_agents(
        "```python\nclass TradingBot:\n    pass\n```", ""
    )

    configs = await system.run_optimization_stage({
        "BTC/USDT": {
            "pair": "BTC/USDT",
            "strategy": {"name": "grid"},
            "parameters": dict(CURRENT_PARAMETERS),
        }
    })

    entry = configs["BTC/USDT"]
    assert entry["bot"]["executable"] is False
    assert entry["parameters"] == CURRENT_PARAMETERS
    assert system.feedback_agent.llm_client.calls == []


async def test_optimization_stage_refuses_an_absurd_refined_allocation():
    """A refinement can narrow the parameters, never inflate them."""

    system = _system_with_stub_agents(
        "```python\nclass TradingBot:\n    pass\n```",
        '{"capital_allocation": 999999999, "max_position_pct": 90}'
    )

    configs = await system.run_optimization_stage(
        {
            "BTC/USDT": {
                "pair": "BTC/USDT",
                "strategy": {"name": "grid"},
                "parameters": dict(CURRENT_PARAMETERS),
            }
        },
        feedback_by_pair={"BTC/USDT": {"max_drawdown": 0.25, "pnl": -500}}
    )

    refined = configs["BTC/USDT"]["parameters"]
    assert refined["capital_allocation"] <= CURRENT_PARAMETERS["capital_allocation"]
    assert refined["max_position_pct"] <= CURRENT_PARAMETERS["max_position_pct"] * 1.5
    assert math.isfinite(refined["capital_allocation"])


# --------------------------------------------------------------------------
# Construction without an explicit configuration
# --------------------------------------------------------------------------


def test_agents_construct_without_an_explicit_config():
    """Every agent must be usable with the default configuration.

    Reading `config.get(...)` off the constructor parameter rather than
    `self.config` made three of these raise AttributeError.
    """

    llm = StubLLMClient()

    macro = MacroAnalysisAgent(llm, market_data=None)
    strategy = StrategyAdaptationAgent(llm, market_data=None)
    evolution = BotEvolutionAgent(llm)
    reflection = FeedbackReflectionAgent(llm)

    for agent in (macro, strategy, evolution, reflection):
        assert agent.config is not None

    assert isinstance(macro.time_windows, list)
    assert isinstance(evolution.enforce_laws, bool)
    assert reflection.drawdown_budget > 0
