"""Logging configuration for TiMi system."""

import logging
import math
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import structlog
from .config import Config

# Denominators smaller than this are treated as zero. Spot holdings pulled
# straight from the exchange can have a cost basis of exactly 0.0 (no entry
# price was ever recorded), and floating point noise can leave a denominator
# that is technically nonzero but meaningless, so an exact `!= 0` comparison
# is not good enough.
_DUST = 1e-12


def setup_logging(config: Optional[Config] = None) -> None:
    """Set up structured logging for the system.

    Args:
        config: Configuration object. If None, will create new Config instance.
    """
    if config is None:
        config = Config()

    # Get logging configuration
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 'json')
    console_enabled = config.get('logging.console.enabled', True)
    file_enabled = config.get('logging.file.enabled', True)
    log_file = config.get('logging.file.path', 'logs/timi.log')

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(message)s',
        handlers=[]
    )

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == 'json':
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up handlers
    logger = logging.getLogger()

    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_enabled:
        max_size_mb = config.get('logging.file.max_size_mb', 100)
        backup_count = config.get('logging.file.backup_count', 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def _safe_pnl_pct(pnl: float, entry_price: float, size: float) -> Optional[float]:
    """Compute a PnL percentage, or ``None`` when it is not a defined quantity.

    The denominator is ``entry_price * size``, the cost basis of the
    position. That can legitimately be zero (or non-finite) in several
    situations that are ordinary rather than exceptional:

    - A spot holding discovered on the exchange with no recorded entry
      price (``entry_price == 0.0``).
    - A closed or not-yet-opened position (``size == 0``).
    - Upstream data that is momentarily ``inf``/``nan``.

    In every one of those cases there is no cost basis to measure a
    percentage against, so this returns ``None`` rather than ``0``: a
    fabricated 0% would read as "flat" (a real measurement), which is a
    different claim from "unknown" (no measurement is possible). Callers
    should omit the field, or pass the field through as ``None``, rather
    than ever substituting a fake zero.

    Args:
        pnl: Profit/loss in quote currency.
        entry_price: Position entry price. May legitimately be 0.0 for a
            spot holding with no cost basis.
        size: Position size.

    Returns:
        The PnL percentage, or ``None`` if it is undefined or unsafe to
        compute (zero/near-zero or non-finite denominator, or a
        non-finite input).
    """
    if not (math.isfinite(pnl) and math.isfinite(entry_price) and math.isfinite(size)):
        return None

    denominator = entry_price * size
    if not math.isfinite(denominator) or abs(denominator) < _DUST:
        return None

    return (pnl / denominator) * 100


class TradingLogger:
    """Specialized logger for trading operations.

    Logging must never be able to crash the code path it is observing.
    Every public method here degrades to a simpler, best-effort log line
    on unexpected failure (a malformed record, a bad kwarg, a renderer
    error) instead of letting the exception propagate into the caller.
    Programming errors are not silently discarded: they are still emitted,
    just as a plain log line rather than raised.
    """

    def __init__(self, name: str):
        """Initialize trading logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)

    def _safe_log(self, log_method, event: str, **fields) -> None:
        """Emit a structured log record without ever raising.

        Args:
            log_method: Bound structlog method to call (e.g. self.logger.info).
            event: Event name.
            **fields: Structured fields for the record.
        """
        try:
            log_method(event, **fields)
        except Exception as exc:  # noqa: BLE001 - logging must never crash the caller
            try:
                self.logger.warning(
                    "log_degraded",
                    original_event=event,
                    reason=str(exc),
                )
            except Exception:
                # Even the degraded log line failed (e.g. logger itself is
                # broken). There is nothing further that can be safely done
                # from a logging call; swallow rather than crash the caller.
                pass

    def log_trade(
        self,
        action: str,
        pair: str,
        price: float,
        quantity: float,
        order_id: Optional[str] = None,
        **kwargs
    ):
        """Log a trade execution.

        Args:
            action: Trade action (buy/sell)
            pair: Trading pair
            price: Execution price
            quantity: Trade quantity
            order_id: Order ID if available
            **kwargs: Additional context
        """
        self._safe_log(
            self.logger.info,
            "trade_executed",
            action=action,
            pair=pair,
            price=price,
            quantity=quantity,
            order_id=order_id,
            **kwargs
        )

    def log_position(
        self,
        pair: str,
        size: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        **kwargs
    ):
        """Log position update.

        Args:
            pair: Trading pair
            size: Position size
            entry_price: Entry price
            current_price: Current price
            pnl: Profit/loss
            **kwargs: Additional context

        Note:
            ``pnl_pct`` is ``None`` when the percentage is not a defined
            quantity (no cost basis, e.g. a spot holding with
            ``entry_price == 0.0``, or a zero/non-finite size or price).
            A missing cost basis is reported as unknown, never as a
            fabricated 0%, since 0% would misleadingly read as "flat".
        """
        self._safe_log(
            self.logger.info,
            "position_update",
            pair=pair,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            pnl_pct=_safe_pnl_pct(pnl, entry_price, size),
            **kwargs
        )

    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        **kwargs
    ):
        """Log a risk management event.

        Args:
            event_type: Type of risk event
            severity: Severity level (warning/critical)
            message: Event message
            **kwargs: Additional context
        """
        log_method = self.logger.warning if severity == 'warning' else self.logger.critical
        self._safe_log(
            log_method,
            "risk_event",
            event_type=event_type,
            severity=severity,
            message=message,
            **kwargs
        )

    def log_agent_action(
        self,
        agent: str,
        action: str,
        result: str,
        duration: Optional[float] = None,
        **kwargs
    ):
        """Log an agent action.

        Args:
            agent: Agent name
            action: Action performed
            result: Action result
            duration: Execution duration in seconds
            **kwargs: Additional context
        """
        self._safe_log(
            self.logger.info,
            "agent_action",
            agent=agent,
            action=action,
            result=result,
            duration=duration,
            **kwargs
        )

    def log_error(self, error: Exception, context: Optional[dict] = None):
        """Log an error with context.

        Args:
            error: Exception object
            context: Additional context information
        """
        try:
            error_type = type(error).__name__
            error_message = str(error)
        except Exception:
            # Even rendering the exception itself failed; fall back to
            # something that cannot raise.
            error_type = "UnknownError"
            error_message = "<unrenderable error>"

        self._safe_log(
            self.logger.error,
            "error_occurred",
            error_type=error_type,
            error_message=error_message,
            **(context or {})
        )
