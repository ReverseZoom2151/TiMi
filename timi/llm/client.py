"""LLM client with capability-based routing."""

from typing import Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from anthropic import Anthropic

from .base import BaseLLM, LLMResponse, LLMCapability, LLMRateLimitError, LLMAPIError
from ..utils.config import Config, LLMConfig
from ..utils.logging import get_logger


logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.client = openai.OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        messages = self.format_messages(prompt, system_prompt)
        return await self.generate_with_history(messages, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_with_history(
        self,
        messages: list[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response with history."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in self.kwargs.items() if k not in kwargs}
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )

        except openai.RateLimitError as e:
            logger.error("OpenAI rate limit exceeded", error=str(e))
            raise LLMRateLimitError(str(e))
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            raise LLMAPIError(str(e))


class AnthropicLLM(BaseLLM):
    """Anthropic (Claude) LLM implementation."""

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.client = Anthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )

        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            raise LLMAPIError(str(e))

    async def generate_with_history(
        self,
        messages: list[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response with history."""
        # Extract system message if present
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)

        # For simplicity, just use the last user message
        # In production, you'd want full conversation support
        last_message = user_messages[-1]["content"] if user_messages else ""
        return await self.generate(last_message, system_prompt, **kwargs)


class LLMClient:
    """Central LLM client with capability-based routing.

    Routes requests to appropriate LLM based on capability type:
    - Semantic analysis (φ): for macro/micro market analysis
    - Code programming (ψ): for bot evolution and code generation
    - Mathematical reasoning (γ): for parameter optimization
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize LLM client.

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.llms: Dict[LLMCapability, BaseLLM] = {}
        self._initialize_llms()

    def _initialize_llms(self):
        """Initialize LLM instances for each capability."""
        # Semantic analysis LLM
        semantic_config = self.config.llm_semantic
        self.llms[LLMCapability.SEMANTIC_ANALYSIS] = self._create_llm(
            semantic_config,
            "semantic"
        )

        # Code programming LLM
        code_config = self.config.llm_code
        self.llms[LLMCapability.CODE_PROGRAMMING] = self._create_llm(
            code_config,
            "code"
        )

        # Mathematical reasoning LLM
        reasoning_config = self.config.llm_reasoning
        self.llms[LLMCapability.MATHEMATICAL_REASONING] = self._create_llm(
            reasoning_config,
            "reasoning"
        )

    def _create_llm(self, config: LLMConfig, purpose: str) -> BaseLLM:
        """Create LLM instance based on provider.

        Args:
            config: LLM configuration
            purpose: Purpose identifier for logging

        Returns:
            LLM instance
        """
        provider = config.provider.lower()
        api_key = None

        if provider == "openai":
            api_key = self.config.get_api_key("openai")
            if not api_key:
                raise ValueError(f"OpenAI API key not found for {purpose}")

            return OpenAILLM(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )

        elif provider == "anthropic":
            api_key = self.config.get_api_key("anthropic")
            if not api_key:
                raise ValueError(f"Anthropic API key not found for {purpose}")

            return AnthropicLLM(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )

        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            api_key = self.config.get_api_key("deepseek")
            if not api_key:
                raise ValueError(f"DeepSeek API key not found for {purpose}")

            # Create OpenAI client with DeepSeek base URL
            llm = OpenAILLM(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            # Override base URL for DeepSeek
            llm.client.base_url = "https://api.deepseek.com/v1"
            return llm

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def generate(
        self,
        prompt: str,
        capability: LLMCapability,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using appropriate LLM for capability.

        Args:
            prompt: User prompt
            capability: Required LLM capability
            system_prompt: System prompt for context
            **kwargs: Additional generation parameters

        Returns:
            LLM response
        """
        if capability not in self.llms:
            raise ValueError(f"No LLM configured for capability: {capability}")

        llm = self.llms[capability]

        logger.info(
            "llm_request",
            capability=capability.value,
            model=llm.model,
            prompt_length=len(prompt)
        )

        response = await llm.generate(prompt, system_prompt, **kwargs)

        logger.info(
            "llm_response",
            capability=capability.value,
            model=response.model,
            tokens_used=response.tokens_used,
            response_length=len(response.content)
        )

        return response

    async def generate_semantic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using semantic analysis capability (φ).

        Args:
            prompt: Analysis prompt
            system_prompt: System context
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        return await self.generate(
            prompt,
            LLMCapability.SEMANTIC_ANALYSIS,
            system_prompt,
            **kwargs
        )

    async def generate_code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using code programming capability (ψ).

        Args:
            prompt: Code generation prompt
            system_prompt: System context
            **kwargs: Additional parameters

        Returns:
            LLM response with code
        """
        return await self.generate(
            prompt,
            LLMCapability.CODE_PROGRAMMING,
            system_prompt,
            **kwargs
        )

    async def generate_reasoning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using mathematical reasoning capability (γ).

        Args:
            prompt: Reasoning/optimization prompt
            system_prompt: System context
            **kwargs: Additional parameters

        Returns:
            LLM response with mathematical analysis
        """
        return await self.generate(
            prompt,
            LLMCapability.MATHEMATICAL_REASONING,
            system_prompt,
            **kwargs
        )
