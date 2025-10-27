"""Exchange factory for creating exchange instances."""

from typing import Optional
from .base import BaseExchange
from .binance import BinanceExchange
from ..utils.config import Config
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ExchangeFactory:
    """Factory for creating exchange instances."""

    @staticmethod
    def create_exchange(
        exchange_name: str,
        config: Optional[Config] = None,
        **kwargs
    ) -> BaseExchange:
        """Create an exchange instance.

        Args:
            exchange_name: Name of exchange ('binance', 'cme', etc.)
            config: Configuration object
            **kwargs: Additional parameters to override config

        Returns:
            Exchange instance

        Raises:
            ValueError: If exchange not supported
        """
        if config is None:
            config = Config()

        exchange_name = exchange_name.lower()

        if exchange_name == 'binance':
            api_key = kwargs.get('api_key') or config.get_api_key('binance')
            api_secret = kwargs.get('api_secret') or config.get_api_secret('binance')
            testnet = kwargs.get('testnet', config.exchange.testnet)

            if not api_key or not api_secret:
                raise ValueError("Binance API key and secret required")

            logger.info(
                "Creating Binance exchange connector",
                testnet=testnet
            )

            return BinanceExchange(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                **{k: v for k, v in kwargs.items() if k not in ['api_key', 'api_secret', 'testnet']}
            )

        elif exchange_name == 'cme':
            # TODO: Implement CME connector
            raise NotImplementedError("CME connector not yet implemented")

        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

    @staticmethod
    def create_default_exchange(config: Optional[Config] = None) -> BaseExchange:
        """Create exchange instance based on config.

        Args:
            config: Configuration object

        Returns:
            Exchange instance
        """
        if config is None:
            config = Config()

        primary_exchange = config.exchange.primary
        return ExchangeFactory.create_exchange(primary_exchange, config)
