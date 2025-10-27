"""Strategy Adaptation Agent (Asa) - customizes macro strategies for specific trading pairs.

Implements: Asa ◦ φ ◦ γ : S × P → Sp × Θp
Uses semantic analysis (φ) and mathematical reasoning (γ) to adapt strategies.
"""

import json
from typing import List, Dict, Any

from .base import BaseAgent, AgentResult
from ..llm.client import LLMClient
from ..data.market_data import MarketDataManager
from ..utils.config import Config


class StrategyAdaptationAgent(BaseAgent):
    """Strategy Adaptation Agent (Asa).

    Customizes general strategies for specific trading pairs, including:
    - Strategy selection based on pair characteristics
    - Parameter initialization tailored to volatility profiles
    - Adaptive risk management rules based on liquidity
    """

    def __init__(
        self,
        llm_client: LLMClient,
        market_data: MarketDataManager,
        config: Config = None
    ):
        """Initialize Strategy Adaptation Agent.

        Args:
            llm_client: LLM client
            market_data: Market data manager
            config: Configuration
        """
        super().__init__("StrategyAdaptationAgent", llm_client, config)
        self.market_data = market_data

    async def execute(
        self,
        general_strategies: List[Dict[str, Any]],
        pair: str
    ) -> AgentResult:
        """Adapt general strategies for a specific trading pair.

        Args:
            general_strategies: List of general strategies from macro analysis
            pair: Trading pair to customize for

        Returns:
            AgentResult with pair-specific strategy and parameters
        """
        try:
            self.log_action("starting_strategy_adaptation", pair=pair)

            # Analyze pair characteristics
            pair_profile = await self._analyze_pair(pair)

            # Select best strategy for this pair
            selected_strategy = await self._select_strategy(
                general_strategies,
                pair_profile
            )

            # Initialize parameters for the pair
            parameters = await self._initialize_parameters(
                selected_strategy,
                pair_profile
            )

            result_data = {
                'pair': pair,
                'strategy': selected_strategy,
                'parameters': parameters,
                'pair_profile': pair_profile
            }

            self.log_action(
                "strategy_adaptation_complete",
                pair=pair,
                strategy=selected_strategy.get('name')
            )

            return AgentResult(
                success=True,
                data=result_data,
                message=f"Adapted strategy for {pair}",
                metadata=result_data
            )

        except Exception as e:
            self.log_error(e, {"pair": pair})
            return AgentResult(
                success=False,
                data=None,
                message=f"Strategy adaptation failed: {str(e)}"
            )

    async def _analyze_pair(self, pair: str) -> Dict[str, Any]:
        """Analyze pair-specific characteristics.

        Args:
            pair: Trading pair

        Returns:
            Pair profile with characteristics
        """
        # Get market stats
        stats = await self.market_data.get_market_stats(pair)

        # Get historical data for additional analysis
        df = await self.market_data.get_historical_data(pair, '1h', 168)  # 1 week of hourly data

        # Calculate additional metrics
        price_std = df['close'].std() if not df.empty else 0
        volume_consistency = df['volume'].std() / df['volume'].mean() if not df.empty and df['volume'].mean() > 0 else 0

        return {
            'pair': pair,
            'volatility': stats.volatility,
            'volume_24h': stats.volume_24h,
            'price': stats.price,
            'price_std': price_std,
            'volume_consistency': volume_consistency,
            'liquidity_score': self._calculate_liquidity_score(stats.volume_24h),
            'risk_category': self._categorize_risk(stats.volatility)
        }

    def _calculate_liquidity_score(self, volume_24h: float) -> str:
        """Calculate liquidity score from volume.

        Args:
            volume_24h: 24-hour volume

        Returns:
            Liquidity category
        """
        if volume_24h > 100_000_000:
            return 'high'
        elif volume_24h > 10_000_000:
            return 'medium'
        else:
            return 'low'

    def _categorize_risk(self, volatility: float) -> str:
        """Categorize risk level from volatility.

        Args:
            volatility: Volatility value

        Returns:
            Risk category
        """
        if volatility > 0.10:  # >10%
            return 'high'
        elif volatility > 0.05:  # >5%
            return 'medium'
        else:
            return 'low'

    async def _select_strategy(
        self,
        general_strategies: List[Dict[str, Any]],
        pair_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select best strategy for the pair.

        Args:
            general_strategies: General strategies
            pair_profile: Pair characteristics

        Returns:
            Selected strategy
        """
        # Use semantic analysis to select strategy
        prompt = f"""Select the most appropriate trading strategy for this trading pair:

Pair Profile:
- Volatility: {pair_profile['volatility']:.4f} ({pair_profile['risk_category']} risk)
- Volume (24h): ${pair_profile['volume_24h']:,.0f}
- Liquidity: {pair_profile['liquidity_score']}
- Price: ${pair_profile['price']:,.2f}

Available Strategies:
{json.dumps(general_strategies, indent=2)}

Select the strategy that best matches this pair's characteristics and explain why.
Respond with JSON format: {{"selected_strategy": "strategy_name", "reason": "explanation"}}"""

        response = await self.llm_client.generate_semantic(
            prompt=prompt,
            system_prompt="You are a quantitative strategist. Select strategies based on statistical fit and risk-adjusted returns."
        )

        # Parse response to get selected strategy
        selected_name = self._parse_strategy_selection(response.content)

        # Find the strategy
        for strategy in general_strategies:
            if strategy['name'] == selected_name:
                return strategy

        # Default to first strategy if parsing failed
        return general_strategies[0] if general_strategies else {
            'name': 'grid',
            'description': 'Default grid strategy'
        }

    def _parse_strategy_selection(self, llm_response: str) -> str:
        """Parse strategy selection from LLM response.

        Args:
            llm_response: LLM response text

        Returns:
            Selected strategy name
        """
        # Try to parse JSON
        try:
            # Look for JSON in response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                data = json.loads(json_str)
                return data.get('selected_strategy', 'grid')
        except:
            pass

        # Fallback: look for strategy keywords
        response_lower = llm_response.lower()
        if 'trend' in response_lower:
            return 'trend'
        elif 'mean-reversion' in response_lower or 'mean reversion' in response_lower:
            return 'mean-reversion'
        elif 'stat-arb' in response_lower or 'statistical arbitrage' in response_lower:
            return 'stat-arb'
        else:
            return 'grid'

    async def _initialize_parameters(
        self,
        strategy: Dict[str, Any],
        pair_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize parameters for strategy and pair using mathematical reasoning.

        Args:
            strategy: Selected strategy
            pair_profile: Pair characteristics

        Returns:
            Initialized parameters (Θp from paper)
        """
        # Use mathematical reasoning capability (γ)
        prompt = f"""Initialize trading parameters for this strategy and trading pair:

Strategy: {strategy['name']}
Description: {strategy.get('description', '')}

Pair Characteristics:
- Volatility: {pair_profile['volatility']:.4f}
- Risk Level: {pair_profile['risk_category']}
- Liquidity: {pair_profile['liquidity_score']}
- Current Price: ${pair_profile['price']:,.2f}

Calculate optimal parameters for:
1. Capital allocation (base amount per trade)
2. Grid levels (number of price levels for orders)
3. Grid spacing (as multiple of volatility)
4. Position sizing (max position as % of capital)
5. Stop loss threshold (as % of entry price)

Provide mathematical justification for each parameter based on volatility and risk.
Format response as JSON with parameter names and values."""

        response = await self.llm_client.generate_reasoning(
            prompt=prompt,
            system_prompt="You are a quantitative analyst. Use mathematical reasoning to optimize parameters based on volatility, liquidity, and risk metrics."
        )

        # Parse parameters from response
        parameters = self._parse_parameters(response.content, pair_profile)

        return parameters

    def _parse_parameters(
        self,
        llm_response: str,
        pair_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse parameters from LLM response.

        Args:
            llm_response: LLM response
            pair_profile: Pair profile

        Returns:
            Parameter dictionary
        """
        # Try to extract JSON
        try:
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                params = json.loads(json_str)
                return params
        except:
            pass

        # Fallback: calculate parameters mathematically
        volatility = pair_profile['volatility']
        risk_level = pair_profile['risk_category']

        # Adjust parameters based on volatility and risk
        if risk_level == 'high':
            capital_multiplier = 0.5
            grid_levels = 10
            stop_loss = 0.03  # 3%
        elif risk_level == 'medium':
            capital_multiplier = 0.75
            grid_levels = 7
            stop_loss = 0.05  # 5%
        else:
            capital_multiplier = 1.0
            grid_levels = 5
            stop_loss = 0.08  # 8%

        return {
            'capital_allocation': 100 * capital_multiplier,  # Base USD amount
            'grid_levels': grid_levels,
            'grid_spacing': max(0.5, volatility * 2),  # 2x volatility
            'max_position_pct': 10 * capital_multiplier,
            'stop_loss_pct': stop_loss,
            'volatility_multiplier': volatility * 10  # For dynamic adjustments
        }
