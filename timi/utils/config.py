"""Configuration management for TiMi system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    primary: str
    testnet: bool = True


class StrategyConfig(BaseModel):
    """Trading strategy configuration."""
    execution_interval: int = 1
    lookback_period: int = 60
    min_volume: float = 1000000
    min_volatility: float = 0.5
    capital_per_pair: float = 100
    price_distribution: list[float] = Field(default_factory=list)
    quantity_distribution: list[float] = Field(default_factory=list)
    profit_loss_thresholds: list[float] = Field(default_factory=list)
    market_cap_coefficient: float = 1.0
    funding_rate_coefficient: float = 1.0
    entry_coefficient: float = 0.8


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_drawdown: float = 20.0
    max_position_pct: float = 10.0
    max_concurrent_positions: int = 5
    stop_loss_pct: float = 5.0
    max_price_deviation: float = 2.0
    position_size_divisor: int = 10


class Config:
    """Main configuration class for TiMi system."""

    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration."""
        if not self._config_data:
            self.load()

    def load(self, config_path: Optional[str] = None):
        """Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config file. Defaults to config.yaml in project root.
        """
        # Load environment variables
        load_dotenv()

        # Determine config file path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        # Load YAML configuration
        with open(config_path, 'r') as f:
            self._config_data = yaml.safe_load(f)

        # Override with environment variables where applicable
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Override safety settings from env
        if os.getenv('PAPER_TRADING_MODE'):
            self._config_data['mode'] = 'paper' if os.getenv('PAPER_TRADING_MODE') == 'true' else 'live'

        if os.getenv('EMERGENCY_STOP'):
            self._config_data['emergency_stop'] = os.getenv('EMERGENCY_STOP') == 'true'

        # Override exchange testnet
        if os.getenv('BINANCE_TESTNET'):
            self._config_data['exchange']['testnet'] = os.getenv('BINANCE_TESTNET') == 'true'

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'llm.semantic.model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def mode(self) -> str:
        """Get trading mode: backtest, paper, or live."""
        return self.get('mode', 'paper')

    @property
    def llm_semantic(self) -> LLMConfig:
        """Get semantic analysis LLM config."""
        return LLMConfig(**self.get('llm.semantic', {}))

    @property
    def llm_code(self) -> LLMConfig:
        """Get code programming LLM config."""
        return LLMConfig(**self.get('llm.code', {}))

    @property
    def llm_reasoning(self) -> LLMConfig:
        """Get mathematical reasoning LLM config."""
        return LLMConfig(**self.get('llm.reasoning', {}))

    @property
    def exchange(self) -> ExchangeConfig:
        """Get exchange configuration."""
        return ExchangeConfig(**self.get('exchange', {}))

    @property
    def strategy(self) -> StrategyConfig:
        """Get strategy configuration."""
        return StrategyConfig(**self.get('strategy', {}))

    @property
    def risk(self) -> RiskConfig:
        """Get risk management configuration."""
        return RiskConfig(**self.get('risk', {}))

    @property
    def trading_pairs_mainstream(self) -> list[str]:
        """Get mainstream trading pairs."""
        return self.get('trading_pairs.mainstream', [])

    @property
    def trading_pairs_altcoins(self) -> list[str]:
        """Get altcoin trading pairs."""
        return self.get('trading_pairs.altcoins', [])

    @property
    def trading_pairs_active(self) -> list[str]:
        """Get active trading pairs."""
        return self.get('trading_pairs.active', [])

    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode."""
        return self.mode == 'paper'

    def is_live_trading(self) -> bool:
        """Check if in live trading mode."""
        return self.mode == 'live'

    def is_backtesting(self) -> bool:
        """Check if in backtest mode."""
        return self.mode == 'backtest'

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables.

        Args:
            service: Service name (e.g., 'openai', 'binance')

        Returns:
            API key or None if not found
        """
        env_var = f"{service.upper()}_API_KEY"
        return os.getenv(env_var)

    def get_api_secret(self, service: str) -> Optional[str]:
        """Get API secret from environment variables.

        Args:
            service: Service name (e.g., 'binance')

        Returns:
            API secret or None if not found
        """
        env_var = f"{service.upper()}_API_SECRET"
        return os.getenv(env_var)


# Global config instance
config = Config()
