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

from .utils.config import (
    Config,
    LiveTradingNotUnlockedError,
    Mode,
    UnsafeModeError,
    assert_live_trading_unlocked,
    assert_mode_matches_endpoint,
)
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

        # Initialize execution components. The position book is built first
        # because both the risk gate and the engine read from it: it is the
        # only place a cost basis exists, since a spot balance carries none.
        self.position_manager = PositionManager()

        self.risk_manager = RiskManager(
            self.config,
            self.position_manager
        )

        # Capital tracking must exist before the gate can size anything, so it
        # is set up before the engine that will call through it.
        initial_capital = self.config.get('risk.initial_capital', 10000)
        self.risk_manager.initialize_capital(initial_capital)

        # The engine shares both, so every order it places is checked and
        # every fill it takes lands in the same book the gate measures.
        self.bot_engine = BotEngine(
            self.exchange,
            self.market_data,
            self.config,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager
        )

        if self.risk_manager.emergency_stop:
            self.logger.error(
                "Emergency stop is set; the system will not place any orders"
            )

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

    async def run_optimization_stage(
        self,
        pair_configs: dict,
        feedback_by_pair: Optional[dict] = None
    ) -> dict:
        """Run Optimization Stage: bot artefacts and parameter refinement.

        Two things happen per pair, and neither of them can place an order.

        The bot evolution agent produces a trading bot as source text. It is
        checked for syntax and stored under the pair's 'bot' key with
        `executable` set to False. Nothing in this system executes it: it is an
        artefact for a person to read, and the engine keeps running the code in
        `timi.core` that was written and reviewed by hand.

        The feedback reflection agent then refines the pair's parameters, but
        only when there is feedback worth reflecting on. On a cold start there
        is none, so the call is skipped rather than paid for. Whatever comes
        back has already been bounded inside the agent, and the allocation is
        bounded again at the risk layer before deployment, so a refinement can
        only ever narrow what deployment accepts.

        Args:
            pair_configs: Pair configurations from the policy stage
            feedback_by_pair: Deployment or simulation feedback, keyed by pair

        Returns:
            The pair configurations, with refined parameters where available
        """

        self.logger.info("=" * 60)
        self.logger.info("STAGE II: OPTIMIZATION - Bots and Parameter Refinement")
        self.logger.info("=" * 60)

        feedback_by_pair = feedback_by_pair or {}
        evolution_enabled = self.config.get('agents.bot_evolution.enabled', True)
        reflection_enabled = self.config.get(
            'agents.feedback_reflection.enabled', True
        )

        for pair, config_data in pair_configs.items():
            strategy = config_data.get('strategy', {})
            parameters = config_data.get('parameters', {})

            if evolution_enabled:
                evolution_result = await self.bot_evolution_agent.execute(
                    strategy,
                    parameters,
                    pair
                )

                if evolution_result.success:
                    # Stored, never run.
                    config_data['bot'] = evolution_result.data
                    self.logger.info(
                        "Bot artefact generated (inert, not executed)",
                        pair=pair,
                        strategy=strategy.get('name')
                    )
                else:
                    self.logger.warning(
                        "Bot artefact generation failed",
                        pair=pair,
                        message=evolution_result.message
                    )

            feedback = feedback_by_pair.get(pair) or {}

            if not reflection_enabled or not feedback:
                self.logger.info(
                    "Skipping reflection, no feedback to reflect on",
                    pair=pair
                )
                continue

            reflection_result = await self.feedback_agent.execute(
                config_data.get('bot', {'pair': pair, 'parameters': parameters}),
                feedback
            )

            if not reflection_result.success:
                self.logger.warning(
                    "Reflection failed, keeping policy stage parameters",
                    pair=pair,
                    message=reflection_result.message
                )
                continue

            refined = reflection_result.data.get('optimal_parameters') or {}
            if refined:
                config_data['parameters'] = refined
                config_data['optimization_level'] = reflection_result.data.get(
                    'optimization_level'
                )
                self.logger.info(
                    "Parameters refined",
                    pair=pair,
                    level=config_data['optimization_level']
                )

        self.logger.info(
            "Optimization stage complete",
            pairs_processed=len(pair_configs)
        )
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

            # The allocation arrives as free-form JSON from the strategy
            # stage and multiplies every order quantity in the grid. It is
            # bounded at the risk layer before it can size anything; a value
            # that cannot be made safe costs this pair its bot, not the run.
            try:
                capital_per_pair = self.risk_manager.validate_capital_allocation(
                    parameters.get('capital_allocation',
                                   self.config.strategy.capital_per_pair)
                )
            except Exception as e:
                self.logger.error(
                    "Rejected capital allocation",
                    pair=pair,
                    reason=str(e)
                )
                continue

            # Build from configuration, so the scaling coefficients and the
            # position size divisor are the configured ones rather than the
            # dataclass defaults.
            bot_config = self.bot_engine.build_bot_config(
                capital_per_pair=capital_per_pair
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

        # Run optimization stage. Bot artefacts are generated and stored
        # inert, and parameters are refined only where feedback exists.
        pair_configs = await self.run_optimization_stage(pair_configs)

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
    parser.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        dest="i_understand_the_risk",
        help=(
            "Required for live trading, alongside the environment variable "
            "TIMI_ALLOW_LIVE=1. Live trading uses real money."
        )
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 60)
    logger.info("TiMi - Trade in Minutes")
    logger.info("Rationality-Driven Agentic System for Quantitative Trading")
    logger.info("=" * 60)

    # Load configuration. The CLI mode beats both the file and the
    # environment, and an unrecognised mode stops the run here.
    config = Config()
    config.load(args.config, cli_overrides={"mode": args.mode})

    mode = config.mode

    # Both unlocks must be present before anything is constructed, so a live
    # run cannot get as far as holding an authenticated client by mistake.
    if mode == Mode.LIVE.value:
        try:
            assert_live_trading_unlocked(args.i_understand_the_risk)
        except LiveTradingNotUnlockedError as e:
            logger.error("Live trading refused", reason=str(e))
            print(str(e))
            sys.exit(2)

    # Initialize system
    system = TiMiSystem(config)

    try:
        await system.initialize()

        # The mode and the endpoint the client actually resolved to must
        # agree. Nothing below this line may place an order until they do.
        endpoint = assert_mode_matches_endpoint(system.exchange, mode)
        logger.info("Endpoint verified against mode", mode=mode, endpoint=endpoint)

        if mode == Mode.PAPER.value:
            logger.info("Running in PAPER TRADING mode (safe)")
            await system.run_paper_trading(args.pairs, args.duration)

        elif mode == Mode.LIVE.value:
            logger.warning("Running in LIVE TRADING mode - REAL MONEY AT RISK!")
            # The prompt is the last barrier, not the only one, and it states
            # the facts so the operator confirms against them.
            print("About to trade with REAL MONEY.")
            print(f"  mode:     {mode}")
            print(f"  endpoint: {endpoint}")
            print(f"  pairs:    {', '.join(args.pairs)}")
            response = input("Type yes to proceed with live trading (yes/no): ")
            if response.strip().lower() == "yes":
                await system.run_paper_trading(args.pairs, args.duration)  # Same logic for now
            else:
                logger.info("Live trading cancelled")

        elif mode == Mode.BACKTEST.value:
            logger.info("Backtesting mode not yet implemented")
            sys.exit(1)

    except UnsafeModeError as e:
        logger.error("Startup safety check failed", error=str(e))
        print(str(e))
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        raise
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
