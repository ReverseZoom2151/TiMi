"""Logging configuration for TiMi system."""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import structlog
from .config import Config


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


class TradingLogger:
    """Specialized logger for trading operations."""

    def __init__(self, name: str):
        """Initialize trading logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)

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
        self.logger.info(
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
        """
        self.logger.info(
            "position_update",
            pair=pair,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            pnl_pct=(pnl / (entry_price * size) * 100) if size > 0 else 0,
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
        log_method(
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
        self.logger.info(
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
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            **(context or {})
        )
