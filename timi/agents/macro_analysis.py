"""Macro Analysis Agent (Ama) - identifies market patterns and formulates general strategies.

Implements: Ama ◦ φ ◦ ψ : M × W → S
Uses semantic analysis (φ) capability to analyze market data and generate macro strategies.
"""

from typing import List, Dict, Any

from .base import BaseAgent, AgentResult
from ..llm.client import LLMClient
from ..data.market_data import MarketDataManager
from ..data.indicators import TechnicalIndicators
from ..utils.config import Config


class MacroAnalysisAgent(BaseAgent):
    """Macro Analysis Agent (Ama).

    Identifies macro-level market patterns and formulates general trading strategies
    based on technical indicators and statistical methods.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        market_data: MarketDataManager,
        config: Config = None
    ):
        """Initialize Macro Analysis Agent.

        Args:
            llm_client: LLM client for semantic analysis
            market_data: Market data manager
            config: Configuration
        """
        super().__init__("MacroAnalysisAgent", llm_client, config)
        self.market_data = market_data
        self.indicators_config = config.get('agents.macro_analysis.indicators', [])
        self.time_windows = config.get('agents.macro_analysis.time_windows', [1, 7, 30])

    async def execute(
        self,
        pairs: List[str],
        time_window: int = 7
    ) -> AgentResult:
        """Analyze market patterns and generate general strategies.

        Args:
            pairs: List of trading pairs to analyze
            time_window: Time window in days for analysis

        Returns:
            AgentResult with generated strategies
        """
        try:
            self.log_action("starting_macro_analysis", pairs=len(pairs))

            # Collect market data for all pairs
            market_data_collection = await self._collect_market_data(pairs)

            # Analyze patterns across markets
            pattern_analysis = await self._analyze_patterns(market_data_collection)

            # Generate general strategies using LLM
            strategies = await self._generate_strategies(pattern_analysis)

            self.log_action(
                "macro_analysis_complete",
                strategies_generated=len(strategies)
            )

            return AgentResult(
                success=True,
                data=strategies,
                message=f"Generated {len(strategies)} macro strategies",
                metadata={
                    "patterns_analyzed": pattern_analysis,
                    "pairs_count": len(pairs)
                }
            )

        except Exception as e:
            self.log_error(e, {"pairs": pairs})
            return AgentResult(
                success=False,
                data=None,
                message=f"Macro analysis failed: {str(e)}"
            )

    async def _collect_market_data(
        self,
        pairs: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect and process market data for analysis.

        Args:
            pairs: Trading pairs

        Returns:
            Dictionary of market data by pair
        """
        market_data = {}

        for pair in pairs:
            try:
                # Get historical data (last 100 candles, 1-minute)
                df = await self.market_data.get_historical_data(
                    pair,
                    timeframe='1m',
                    limit=100
                )

                # Calculate technical indicators
                df_with_indicators = TechnicalIndicators.calculate_all(df)

                # Get market summary
                summary = TechnicalIndicators.get_market_summary(df_with_indicators)

                # Calculate volatility
                volatility = await self.market_data.calculate_volatility(pair)

                market_data[pair] = {
                    'dataframe': df_with_indicators,
                    'summary': summary,
                    'volatility': volatility,
                    'current_price': df['close'].iloc[-1],
                    'volume_24h': df['volume'].sum()
                }

            except Exception as e:
                self.logger.warning(f"Failed to collect data for {pair}: {e}")
                continue

        return market_data

    async def _analyze_patterns(
        self,
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns across collected market data.

        Args:
            market_data: Collected market data

        Returns:
            Pattern analysis results
        """
        if not market_data:
            return {}

        # Aggregate patterns
        patterns = {
            'trend_distribution': {},
            'volatility_levels': {},
            'volume_patterns': {},
            'correlations': {}
        }

        # Analyze trends
        trend_counts = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
        volatility_values = []
        volume_trends = []

        for pair, data in market_data.items():
            summary = data['summary']

            # Trend distribution
            trend = summary.get('trend', 'unknown')
            if trend in trend_counts:
                trend_counts[trend] += 1

            # Volatility levels
            vol = data['volatility']
            volatility_values.append(vol)

            # Volume trends
            volume_trends.append(summary.get('volume_trend', 'unknown'))

        patterns['trend_distribution'] = trend_counts
        patterns['avg_volatility'] = sum(volatility_values) / len(volatility_values) if volatility_values else 0
        patterns['high_volatility_count'] = sum(1 for v in volatility_values if v > 0.05)  # >5%
        patterns['total_pairs_analyzed'] = len(market_data)

        return patterns

    async def _generate_strategies(
        self,
        pattern_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate general trading strategies using LLM.

        Args:
            pattern_analysis: Analysis of market patterns

        Returns:
            List of general strategies
        """
        # Build prompt for LLM
        prompt = self._build_strategy_prompt(pattern_analysis)

        # Use semantic analysis capability (φ)
        response = await self.llm_client.generate_semantic(
            prompt=prompt,
            system_prompt=self._get_system_prompt()
        )

        # Parse strategies from response
        strategies = self._parse_strategies(response.content, pattern_analysis)

        return strategies

    def _build_strategy_prompt(self, pattern_analysis: Dict[str, Any]) -> str:
        """Build prompt for strategy generation.

        Args:
            pattern_analysis: Pattern analysis results

        Returns:
            Formatted prompt
        """
        trend_dist = pattern_analysis.get('trend_distribution', {})
        avg_vol = pattern_analysis.get('avg_volatility', 0)
        high_vol_count = pattern_analysis.get('high_volatility_count', 0)

        prompt = f"""Based on the following market analysis, identify suitable general trading strategies:

Market Overview:
- Total pairs analyzed: {pattern_analysis.get('total_pairs_analyzed', 0)}
- Trend distribution: {trend_dist}
- Average volatility: {avg_vol:.4f} ({avg_vol*100:.2f}%)
- High volatility pairs: {high_vol_count}

Task: Identify 2-3 general trading strategies that would be effective in this market environment.

For each strategy, provide:
1. Strategy name (e.g., "grid", "trend", "stat-arb", "mean-reversion")
2. Market conditions where it applies
3. Key parameters to consider
4. Risk characteristics

Format your response as a structured analysis followed by specific strategy recommendations."""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for macro analysis.

        Returns:
            System prompt
        """
        return """You are a quantitative trading strategist with expertise in technical analysis and market microstructure.
Your role is to analyze market patterns and identify general trading strategies that can be applied systematically.

Focus on:
- Statistical patterns with demonstrable edge
- Strategies that can be programmatically implemented
- Risk-adjusted approaches suitable for automated trading
- Mechanical rules rather than discretionary judgment

Avoid:
- Emotional or sentiment-based reasoning
- Predictions about specific price movements
- Strategies requiring subjective interpretation"""

    def _parse_strategies(
        self,
        llm_response: str,
        pattern_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse strategies from LLM response.

        Args:
            llm_response: LLM response text
            pattern_analysis: Original pattern analysis

        Returns:
            List of parsed strategies
        """
        # For MVP, we'll extract strategies based on keywords
        # In production, you'd want more sophisticated parsing

        strategies = []

        # Default strategy based on market conditions
        avg_vol = pattern_analysis.get('avg_volatility', 0)
        trend_dist = pattern_analysis.get('trend_distribution', {})
        dominant_trend = max(trend_dist, key=trend_dist.get) if trend_dist else 'sideways'

        # Grid strategy for high volatility
        if avg_vol > 0.02:  # >2% volatility
            strategies.append({
                'name': 'grid',
                'description': 'Grid trading strategy for high volatility markets',
                'conditions': {'volatility': 'high', 'trend': 'any'},
                'applicable_to': 'high_volatility_pairs',
                'source': 'macro_analysis'
            })

        # Trend following for strong trends
        if trend_dist.get('uptrend', 0) > trend_dist.get('downtrend', 0) * 1.5:
            strategies.append({
                'name': 'trend',
                'description': 'Trend following strategy for bullish markets',
                'conditions': {'volatility': 'medium', 'trend': 'uptrend'},
                'applicable_to': 'trending_pairs',
                'source': 'macro_analysis'
            })

        # Mean reversion for sideways markets
        if trend_dist.get('sideways', 0) > sum(trend_dist.values()) * 0.5:
            strategies.append({
                'name': 'mean-reversion',
                'description': 'Mean reversion strategy for range-bound markets',
                'conditions': {'volatility': 'low_to_medium', 'trend': 'sideways'},
                'applicable_to': 'sideways_pairs',
                'source': 'macro_analysis'
            })

        # Always include at least one strategy
        if not strategies:
            strategies.append({
                'name': 'grid',
                'description': 'Default grid trading strategy',
                'conditions': {'volatility': 'any', 'trend': 'any'},
                'applicable_to': 'all_pairs',
                'source': 'macro_analysis'
            })

        return strategies
