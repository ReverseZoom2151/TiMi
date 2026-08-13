"""Configuration management for TiMi system."""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Uses the standard library logger rather than timi.utils.logging, because that
# module imports this one and the dependency would be circular.
_logger = logging.getLogger(__name__)


class Mode(str, Enum):
    """The only trading modes this system recognises."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


#: The allow list. Anything outside it is a configuration error, never a
#: silently coerced value: a typo must not be allowed to resolve to live.
VALID_MODES: frozenset[str] = frozenset(m.value for m in Mode)

#: The mode assumed when none is configured. Deliberately the safe one.
DEFAULT_MODE: str = Mode.PAPER.value

_TRUE_LITERALS: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_FALSE_LITERALS: frozenset[str] = frozenset({"false", "0", "no", "off"})

#: Substrings that identify a Binance testnet host. The resolved endpoint of a
#: non-live run must contain one of these.
TESTNET_HOST_MARKERS: tuple[str, ...] = (
    "testnet.binance.vision",
    "testnet.binancefuture.com",
    "testnet",
    "sandbox",
)


class ConfigurationError(ValueError):
    """Raised when configuration is invalid or unsafe."""


class UnsafeModeError(ConfigurationError):
    """Raised when the requested mode and the reachable endpoint disagree."""


class LiveTradingNotUnlockedError(ConfigurationError):
    """Raised when live trading is requested without the explicit unlocks."""


def validate_mode(value: Any) -> str:
    """Validate a trading mode against the allow list.

    Args:
        value: The candidate mode. None or absent means the safe default.

    Returns:
        The validated mode string, one of VALID_MODES.

    Raises:
        ConfigurationError: If the value is not exactly an allowed mode. The
            comparison is exact: no case folding, no whitespace stripping, no
            aliasing. A value that merely looks like a mode ('Paper',
            'paper_trading', 'live ') is a mistake, and guessing what the
            operator meant is how real orders get placed by accident.
    """

    if value is None:
        return DEFAULT_MODE

    allowed = ", ".join(sorted(VALID_MODES))
    if not isinstance(value, str):
        raise ConfigurationError(
            f"Invalid trading mode {value!r} (type {type(value).__name__}). "
            f"Allowed modes are: {allowed}."
        )

    if value not in VALID_MODES:
        raise ConfigurationError(
            f"Invalid trading mode {value!r}. Allowed modes are: {allowed}. "
            "Modes are matched exactly: check the case and any stray whitespace."
        )

    return value


def parse_bool(value: Any, name: str = "value") -> bool:
    """Parse a boolean written the way a human writes one.

    Accepts true/1/yes/on and false/0/no/off in any case, with surrounding
    whitespace ignored. Booleans pass straight through.

    Args:
        value: The raw value, normally a string from the environment.
        name: The setting name, used in the error message.

    Returns:
        The parsed boolean.

    Raises:
        ConfigurationError: If the value is not recognised. It is never quietly
            treated as False: an unrecognised safety flag is an unknown state,
            not a disabled one.
    """

    if isinstance(value, bool):
        return value

    normalised = str(value).strip().lower()

    if normalised in _TRUE_LITERALS:
        return True
    if normalised in _FALSE_LITERALS:
        return False

    raise ConfigurationError(
        f"Cannot interpret {name}={value!r} as a boolean. "
        f"Use one of {sorted(_TRUE_LITERALS)} or {sorted(_FALSE_LITERALS)} "
        "(case insensitive)."
    )


def parse_bool_env(
    name: str,
    default: bool,
    safe_value: Optional[bool] = None,
) -> bool:
    """Read a boolean environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.
        safe_value: If given, an unparseable value resolves to this instead of
            raising, and a warning is logged saying so. Used for the flags
            where refusing to start is worse than quietly staying safe.

    Returns:
        The resolved boolean.

    Raises:
        ConfigurationError: If the value is unparseable and safe_value is None.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        return parse_bool(raw, name)
    except ConfigurationError:
        if safe_value is None:
            raise
        _logger.warning(
            "%s=%r is not a recognised boolean. Falling back to the safe "
            "value %r. Fix the value to silence this warning.",
            name,
            raw,
            safe_value,
        )
        return safe_value


def resolved_endpoint(exchange: Any) -> str:
    """Return the private API URL an exchange object would actually call.

    Accepts either a ccxt client (or the fake_ccxt double, which mirrors its
    shape) or a TiMi connector that wraps one on `.exchange`.

    Args:
        exchange: The constructed exchange object.

    Returns:
        The resolved private endpoint URL.

    Raises:
        UnsafeModeError: If no endpoint can be determined. An endpoint that
            cannot be read cannot be verified, and unverified is not safe.
    """

    client = getattr(exchange, "exchange", exchange)
    urls = getattr(client, "urls", None)

    if isinstance(urls, dict):
        api = urls.get("api")
        if isinstance(api, str):
            return api
        if isinstance(api, dict):
            endpoint = api.get("private") or api.get("public")
            if isinstance(endpoint, str):
                return endpoint

    raise UnsafeModeError(
        f"Cannot determine the API endpoint of {type(client).__name__}. "
        "Refusing to continue, because the mode cannot be verified against it."
    )


def is_testnet_endpoint(url: str) -> bool:
    """Whether a URL points at a testnet or sandbox host."""

    lowered = url.lower()
    return any(marker in lowered for marker in TESTNET_HOST_MARKERS)


def assert_mode_matches_endpoint(exchange: Any, mode: str) -> str:
    """Verify that the reachable endpoint agrees with the trading mode.

    Any mode other than live must resolve to a testnet host. This catches the
    class of defect where a testnet flag is set but does not reach every
    endpoint the client will use, which no amount of configuration review
    finds on its own.

    Args:
        exchange: The constructed exchange object, or its ccxt client.
        mode: The resolved trading mode.

    Returns:
        The resolved endpoint URL, so callers can display it.

    Raises:
        ConfigurationError: If the mode is not an allowed value.
        UnsafeModeError: If a non-live mode resolved to a production host.
    """

    mode = validate_mode(mode)
    endpoint = resolved_endpoint(exchange)

    if mode == Mode.LIVE.value:
        return endpoint

    if not is_testnet_endpoint(endpoint):
        raise UnsafeModeError(
            f"Mode is {mode!r} but the exchange resolved to {endpoint!r}, "
            "which is not a testnet host. Real orders could be placed. "
            "Refusing to start."
        )

    return endpoint


def assert_live_trading_unlocked(cli_flag_given: bool) -> None:
    """Refuse live trading unless both unlocks are present.

    Live trading needs a deliberate act in two separate places: the
    environment variable TIMI_ALLOW_LIVE=1 and the command line flag
    --i-understand-the-risk. One of them alone, or an interactive prompt
    alone, is too easy to reach by accident or by a copied command line.

    Args:
        cli_flag_given: Whether --i-understand-the-risk was passed.

    Raises:
        LiveTradingNotUnlockedError: If either unlock is missing.
    """

    env_value = (os.getenv("TIMI_ALLOW_LIVE") or "").strip()
    env_ok = env_value == "1"

    missing = []
    if not env_ok:
        missing.append("the environment variable TIMI_ALLOW_LIVE=1")
    if not cli_flag_given:
        missing.append("the command line flag --i-understand-the-risk")

    if missing:
        raise LiveTradingNotUnlockedError(
            "Live trading is locked. It requires both the environment "
            "variable TIMI_ALLOW_LIVE=1 and the command line flag "
            "--i-understand-the-risk. Missing: " + "; ".join(missing) + "."
        )


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

    def load(
        self,
        config_path: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Load configuration from YAML file and environment variables.

        Precedence, lowest to highest: YAML file, environment, CLI overrides.

        Args:
            config_path: Path to config file. Defaults to config.yaml in project root.
                A relative path is resolved against the current working
                directory first, then against the project root, so running
                from a subdirectory still finds the shipped config.
            cli_overrides: Top level keys that beat both file and environment.

        Raises:
            FileNotFoundError: If the config file cannot be found.
            ConfigurationError: If the resulting mode is not an allowed value.
        """
        # Load environment variables
        load_dotenv()

        # Determine config file path
        project_root = Path(__file__).parent.parent.parent
        if config_path is None:
            resolved_path = project_root / "config.yaml"
        else:
            resolved_path = Path(config_path)
            if not resolved_path.is_absolute() and not resolved_path.exists():
                fallback = project_root / resolved_path
                if fallback.exists():
                    resolved_path = fallback

        # Load YAML configuration. An empty file parses to None, which every
        # later lookup would trip over, so it becomes an empty mapping here.
        with open(resolved_path, 'r') as f:
            loaded = yaml.safe_load(f)

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigurationError(
                f"Config file {resolved_path} must contain a mapping at the "
                f"top level, got {type(loaded).__name__}."
            )

        self._config_data = loaded

        # Override with environment variables where applicable
        self._apply_env_overrides()

        # CLI wins over everything else
        for key, value in (cli_overrides or {}).items():
            if value is not None:
                self._config_data[key] = value

        # Fail closed: an unusable mode must stop the process here, not at the
        # point where an order would be placed.
        validate_mode(self._config_data.get('mode'))

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration.

        The safety flags are parsed case insensitively, and an unparseable
        value resolves towards safety (paper trading, testnet) rather than
        towards real money.
        """
        # Override safety settings from env
        if os.getenv('PAPER_TRADING_MODE') is not None:
            paper = parse_bool_env('PAPER_TRADING_MODE', True, safe_value=True)
            self._config_data['mode'] = Mode.PAPER.value if paper else Mode.LIVE.value

        if os.getenv('EMERGENCY_STOP') is not None:
            self._config_data['emergency_stop'] = parse_bool_env(
                'EMERGENCY_STOP', False, safe_value=True
            )

        # Override exchange testnet
        if os.getenv('BINANCE_TESTNET') is not None:
            exchange_section = self._config_data.get('exchange')
            if not isinstance(exchange_section, dict):
                exchange_section = {}
                self._config_data['exchange'] = exchange_section
            exchange_section['testnet'] = parse_bool_env(
                'BINANCE_TESTNET', True, safe_value=True
            )

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
        """Get trading mode: backtest, paper, or live.

        Returns:
            The validated mode. An absent mode is the safe default, paper.

        Raises:
            ConfigurationError: If the configured mode is not an allowed value.
        """
        return validate_mode(self.get('mode', DEFAULT_MODE))

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
        """Check whether order execution must be simulated.

        True for every mode except live. Backtesting simulates orders too, and
        callers use this single flag to decide whether to transmit, so the
        question it answers is 'is this not live?' rather than 'is the mode
        exactly paper?'. An invalid mode raises rather than answering False.
        """
        return self.mode != Mode.LIVE.value

    def is_live_trading(self) -> bool:
        """Check if in live trading mode."""
        return self.mode == Mode.LIVE.value

    def is_backtesting(self) -> bool:
        """Check if in backtest mode."""
        return self.mode == Mode.BACKTEST.value

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
