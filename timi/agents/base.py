"""Base agent class for TiMi multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..llm.client import LLMClient
from ..utils.config import Config
from ..utils.logging import get_logger


@dataclass
class AgentResult:
    """Result from an agent action."""
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None


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
