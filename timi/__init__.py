"""
TiMi - Trade in Minutes
A rationality-driven multi-agent system for quantitative financial trading.
"""

__version__ = "0.1.0"
__author__ = "TiMi Development Team"

from .utils.config import Config
from .utils.logging import setup_logging

__all__ = ["Config", "setup_logging"]
