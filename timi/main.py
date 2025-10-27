"""Main entry point for TiMi trading system.

Orchestrates the three-stage process:
1. Policy Stage: Strategy development
2. Optimization Stage: Parameter refinement
3. Deployment Stage: Live trading execution
"""

import asyncio
import argparse
import sys
from typing import List, Optional

from .utils.config import Config
from .utils.logging import setup_logging, get_logger
from .llm.client import LLMClient
from .exchange.factory import ExchangeFactory
from .data import MarketDataManager
from .agents import (
    MacroAnalysisAgent,
    StrategyAdaptationAgent,
    BotEvolutionAgent,
    FeedbackReflectionAgent
)
from .core import BotEngine, BotConfig
from .core.position_manager import PositionManager
from .risk import RiskManager


logger = get_logger(__name__)


class TiMiSystem:
    """Main TiMi trading system orchestrator."""

    def __init__(self, config: Config):
        """Initialize TiMi system.

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger("timi_system")

        # Initialize components
        self.llm_client: Optional[LLMClient] = None
        self.exchange: Optional[any] = None
        self.market_data: Optional[MarketDataManager] = None

        # Agents
        self.macro_agent: Optional[MacroAnalysisAgent] = None
        self.strategy_agent: Optional[StrategyAdaptationAgent] = None
        self.bot_evolution_agent: Optional[BotEvolutionAgent] = None
        self.feedback_agent: Optional[FeedbackReflectionAgent] = None

        # Execution
        self.bot_engine: Optional[BotEngine] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None

    async def initialize(self):
        """Initialize all system components."""
        self.logger.info("Initializing TiMi system", mode=self.config.mode)

        # Initialize LLM client
        self.llm_client = LLMClient(self.config)
        self.logger.info("LLM client initialized")

        # Initialize exchange
        self.exchange = ExchangeFactory.create_default_exchange(self.config)
        self.logger.info(
            "Exchange initialized",
            exchange=self.config.exchange.primary,
            testnet=self.config.exchange.testnet
        )

        # Initialize market data
        self.market_data = MarketDataManager(self.exchange)
        self.logger.info("Market data manager initialized")

        # Initialize agents
        self.macro_agent = MacroAnalysisAgent(
            self.llm_client,
            self.market_data,
            self.config
        )

        self.strategy_agent = StrategyAdaptationAgent(
            self.llm_client,
            self.market_data,
            self.config
        )

        self.bot_evolution_agent = BotEvolutionAgent(
            self.llm_client,
            self.config
        )

        self.feedback_agent = FeedbackReflectionAgent(
            self.llm_client,
            self.config
        )

        self.logger.info("All agents initialized")

        # Initialize execution components
        self.bot_engine = BotEngine(
            self.exchange,
            self.market_data,
            self.config
        )

        self.position_manager = PositionManager()

        self.risk_manager = RiskManager(
            self.config,
            self.position_manager
        )

        # Initialize capital tracking
        initial_capital = self.config.get('risk.initial_capital', 10000)
        self.risk_manager.initialize_capital(initial_capital)

        self.logger.info("Execution components initialized")

    async def run_policy_stage(self, pairs: List[str]) -> dict:
        """Run Policy Stage: Generate strategies and bots.

        Args:
            pairs: List of trading pairs to analyze

        Returns:
            Dictionary with strategies and bot configurations
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE I: POLICY - Strategy Development")
        self.logger.info("=" * 60)

        # Step 1: Macro analysis
        self.logger.info("Running macro analysis on pairs", pairs=pairs)
        macro_result = await self.macro_agent.execute(pairs)

        if not macro_result.success:
            self.logger.error("Macro analysis failed", message=macro_result.message)
            return {}

        general_strategies = macro_result.data
        self.logger.info(
            "Macro analysis complete",
            strategies=len(general_strategies)
        )

        # Step 2: Strategy adaptation for each pair
        pair_configs = {}
        for pair in pairs:
            self.logger.info("Adapting strategy for pair", pair=pair)

            adaptation_result = await self.strategy_agent.execute(
                general_strategies,
                pair
            )

            if adaptation_result.success:
                pair_configs[pair] = adaptation_result.data
                self.logger.info(
                    "Strategy adapted",
                    pair=pair,
                    strategy=adaptation_result.data['strategy']['name']
                )
            else:
                self.logger.warning(
                    "Strategy adaptation failed",
                    pair=pair,
                    message=adaptation_result.message
                )

        self.logger.info("Policy stage complete", pairs_configured=len(pair_configs))
        return pair_configs

    async def deploy_bots(self, pair_configs: dict):
        """Deploy trading bots for configured pairs.

        Args:
            pair_configs: Dictionary of pair configurations
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE III: DEPLOYMENT - Live Trading")
        self.logger.info("=" * 60)

        for pair, config_data in pair_configs.items():
            parameters = config_data['parameters']

            # Create bot configuration
            bot_config = BotConfig(
                execution_interval=self.config.strategy.execution_interval,
                lookback_period=self.config.strategy.lookback_period,
                min_volume=self.config.strategy.min_volume,
                min_volatility=self.config.strategy.min_volatility,
                capital_per_pair=parameters.get('capital_allocation', 100),
                price_distribution=self.config.strategy.price_distribution,
                quantity_distribution=self.config.strategy.quantity_distribution,
                profit_loss_thresholds=self.config.strategy.profit_loss_thresholds
            )

            # Add bot to engine
            await self.bot_engine.add_bot(pair, bot_config)
            self.logger.info("Bot deployed", pair=pair)

        # Start all bots
        self.logger.info("Starting all trading bots")
        await self.bot_engine.start_all()

    async def run_paper_trading(self, pairs: List[str], duration_hours: int = 24):
        """Run paper trading mode.

        Args:
            pairs: Trading pairs
            duration_hours: How long to run (hours)
        """
        self.logger.info(
            "Starting paper trading mode",
            pairs=pairs,
            duration_hours=duration_hours
        )

        # Run policy stage
        pair_configs = await self.run_policy_stage(pairs)

        if not pair_configs:
            self.logger.error("No valid pair configurations generated")
            return

        # Deploy bots
        await self.deploy_bots(pair_configs)

        # Run for specified duration
        self.logger.info(f"Paper trading for {duration_hours} hours")

        # Monitor and log statistics periodically
        try:
            for hour in range(duration_hours):
                await asyncio.sleep(3600)  # Wait 1 hour

                # Log statistics
                bot_stats = self.bot_engine.get_all_stats()
                position_stats = self.position_manager.get_statistics()
                risk_report = self.risk_manager.get_risk_report()

                self.logger.info(
                    f"Hour {hour + 1}/{duration_hours} Statistics",
                    bot_stats=bot_stats,
                    positions=position_stats,
                    risk=risk_report
                )

                # Check for emergency stop
                if self.risk_manager.emergency_stop:
                    self.logger.error("Emergency stop triggered - halting trading")
                    break

        except KeyboardInterrupt:
            self.logger.info("Paper trading interrupted by user")

        finally:
            # Stop all bots
            await self.bot_engine.stop_all()

            # Final statistics
            self.logger.info("=" * 60)
            self.logger.info("PAPER TRADING COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info("Final bot statistics", stats=self.bot_engine.get_all_stats())
            self.logger.info("Final position statistics", stats=self.position_manager.get_statistics())
            self.logger.info("Final risk report", report=self.risk_manager.get_risk_report())

    async def shutdown(self):
        """Shutdown system gracefully."""
        self.logger.info("Shutting down TiMi system")

        if self.bot_engine:
            await self.bot_engine.stop_all()

        if self.exchange:
            await self.exchange.close()

        self.logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TiMi - Trade in Minutes")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["BTC/USDT"],
        help="Trading pairs"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=24,
        help="Duration in hours (for paper/live trading)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 60)
    logger.info("TiMi - Trade in Minutes")
    logger.info("Rationality-Driven Agentic System for Quantitative Trading")
    logger.info("=" * 60)

    # Load configuration
    config = Config()
    config.load(args.config)

    # Override mode if specified
    if args.mode:
        config._config_data['mode'] = args.mode

    # Initialize system
    system = TiMiSystem(config)

    try:
        await system.initialize()

        if args.mode == "paper":
            logger.info("Running in PAPER TRADING mode (safe)")
            await system.run_paper_trading(args.pairs, args.duration)

        elif args.mode == "live":
            logger.warning("Running in LIVE TRADING mode - REAL MONEY AT RISK!")
            response = input("Are you sure you want to proceed with live trading? (yes/no): ")
            if response.lower() == "yes":
                await system.run_paper_trading(args.pairs, args.duration)  # Same logic for now
            else:
                logger.info("Live trading cancelled")

        elif args.mode == "backtest":
            logger.info("Backtesting mode not yet implemented")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        raise
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
