"""Multi-agent system for TiMi - implementing the four specialized agents from the paper."""

from .base import (
    PARAMETER_BOUNDS,
    AgentResult,
    BaseAgent,
    NumericBound,
    extract_json_object,
    response_was_truncated,
    validate_numeric_parameters,
)
from .macro_analysis import MacroAnalysisAgent
from .strategy_adaptation import StrategyAdaptationAgent
from .bot_evolution import BotEvolutionAgent
from .feedback_reflection import FeedbackReflectionAgent

__all__ = [
    "PARAMETER_BOUNDS",
    "AgentResult",
    "BaseAgent",
    "NumericBound",
    "extract_json_object",
    "response_was_truncated",
    "validate_numeric_parameters",
    "MacroAnalysisAgent",
    "StrategyAdaptationAgent",
    "BotEvolutionAgent",
    "FeedbackReflectionAgent"
]
