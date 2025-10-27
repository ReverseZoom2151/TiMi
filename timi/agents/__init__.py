"""Multi-agent system for TiMi - implementing the four specialized agents from the paper."""

from .base import BaseAgent
from .macro_analysis import MacroAnalysisAgent
from .strategy_adaptation import StrategyAdaptationAgent
from .bot_evolution import BotEvolutionAgent
from .feedback_reflection import FeedbackReflectionAgent

__all__ = [
    "BaseAgent",
    "MacroAnalysisAgent",
    "StrategyAdaptationAgent",
    "BotEvolutionAgent",
    "FeedbackReflectionAgent"
]
