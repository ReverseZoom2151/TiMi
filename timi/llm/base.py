"""Base LLM interface for TiMi system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class LLMCapability(Enum):
    """LLM capability types from the paper."""
    SEMANTIC_ANALYSIS = "semantic"  # φ - for macro/micro analysis
    CODE_PROGRAMMING = "code"       # ψ - for bot evolution
    MATHEMATICAL_REASONING = "reasoning"  # γ - for parameter optimization


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ):
        """Initialize LLM.

        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            **kwargs: Additional generation parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            LLM response
        """
        pass

    def format_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Format prompt into message format.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            List of formatted messages
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded error."""
    pass


class LLMAPIError(LLMError):
    """API communication error."""
    pass


class LLMValidationError(LLMError):
    """Response validation error."""
    pass
