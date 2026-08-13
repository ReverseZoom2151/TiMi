"""Bot Evolution Agent (Abe) - transforms strategies into executable trading bots.

Implements: Abe ◦ ψ : S × Θ × L → B
Uses code programming (ψ) capability to generate trading bot code.
Follows three programming laws:
1. Functional cohesion - one responsibility per component
2. Unidirectional dependency - dependencies flow from higher to lower layers
3. Parameter externalization - adjustable values centrally managed

The code this agent produces is inert. It is parsed with `ast.parse` to check
that it is syntactically Python and then kept as a string for an operator to
read. Nothing here, and nothing downstream, may execute it: no `exec`, no
`eval`, no `compile`, no import machinery, and no writing it to a module file.
Generated text that runs is generated text that trades, and the only thing
allowed to place an order in this system is code a person has reviewed.
"""

import ast
import re
from typing import Any, Dict

from .base import BaseAgent, AgentResult, response_was_truncated
from ..llm.client import LLMClient
from ..utils.config import Config


class BotEvolutionAgent(BaseAgent):
    """Bot Evolution Agent (Abe).

    Creates and evolves programmatic trading bots with layered design:
    - Strategy layer: Decision-making logic
    - Function layer: Technical computation
    - Parameter layer: Adjustable values
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Config = None
    ):
        """Initialize Bot Evolution Agent.

        Args:
            llm_client: LLM client for code generation
            config: Configuration
        """
        super().__init__("BotEvolutionAgent", llm_client, config)
        # Read through self.config, which the base class has already defaulted,
        # so the agent is constructable without one.
        self.enforce_laws = self.config.get(
            'agents.bot_evolution.enforce_laws', True
        )

    async def execute(
        self,
        strategy: Dict[str, Any],
        parameters: Dict[str, Any],
        pair: str
    ) -> AgentResult:
        """Generate executable trading bot code.

        Args:
            strategy: Trading strategy specification
            parameters: Strategy parameters
            pair: Trading pair

        Returns:
            AgentResult with generated bot code
        """
        try:
            self.log_action(
                "generating_bot",
                strategy=strategy.get('name'),
                pair=pair
            )

            # Generate bot code using LLM
            bot_code = await self._generate_bot_code(strategy, parameters, pair)

            # Validate code
            validation = {'valid': None, 'issues': []}
            if self.enforce_laws:
                validation = self._validate_code(bot_code)
                if not validation['valid']:
                    self.logger.warning(
                        "Bot code validation failed",
                        issues=validation['issues']
                    )

            # Create bot metadata. `executable` states the contract in the
            # payload itself: this string is a reviewable artefact, never
            # something to run.
            bot_metadata = {
                'pair': pair,
                'strategy': strategy['name'],
                'parameters': parameters,
                'code': bot_code,
                'executable': False,
                'validation': validation,
                'version': '1.0.0'
            }

            self.log_action(
                "bot_generated",
                pair=pair,
                code_length=len(bot_code)
            )

            return AgentResult(
                success=True,
                data=bot_metadata,
                message=f"Generated bot for {pair}",
                metadata=bot_metadata
            )

        except Exception as e:
            self.log_error(e, {"strategy": strategy, "pair": pair})
            return AgentResult(
                success=False,
                data=None,
                message=f"Bot generation failed: {str(e)}"
            )

    async def _generate_bot_code(
        self,
        strategy: Dict[str, Any],
        parameters: Dict[str, Any],
        pair: str
    ) -> str:
        """Generate trading bot code using LLM.

        Args:
            strategy: Strategy specification
            parameters: Parameters
            pair: Trading pair

        Returns:
            Python code for trading bot, as an inert string
        """
        # Build prompt for code generation
        prompt = self._build_code_generation_prompt(strategy, parameters, pair)

        # Use code programming capability (ψ)
        response = await self.llm_client.generate_code(
            prompt=prompt,
            system_prompt=self._get_code_system_prompt()
        )

        if response_was_truncated(response):
            self.log_fallback(
                'generate_bot_code',
                'reply truncated at the token limit; the artefact is partial',
                pair=pair
            )

        # Extract code from response
        bot_code = self._extract_code(getattr(response, 'content', ''))

        return bot_code

    def _build_code_generation_prompt(
        self,
        strategy: Dict[str, Any],
        parameters: Dict[str, Any],
        pair: str
    ) -> str:
        """Build prompt for code generation.

        Args:
            strategy: Strategy specification
            parameters: Parameters
            pair: Trading pair

        Returns:
            Formatted prompt
        """
        strategy_name = strategy.get('name', 'grid')
        description = strategy.get('description', '')

        prompt = f"""Generate a Python trading bot class for the following specification:

Strategy: {strategy_name}
Description: {description}
Trading Pair: {pair}

Parameters:
```python
{self._format_parameters(parameters)}
```

Requirements:
1. Create a TradingBot class with these methods:
   - __init__(self, exchange, parameters): Initialize with exchange connector and parameters
   - calculate_orders(self, current_price, volatility): Calculate order placements
   - should_execute(self, market_data): Determine if conditions met for trading
   - get_position_size(self, price, volatility): Calculate position sizing
   - check_exit_conditions(self, position, current_price): Check if position should be closed

2. Follow layered design:
   - Parameter layer: All adjustable values in self.params dict
   - Function layer: Reusable computation methods
   - Strategy layer: Decision-making logic

3. Implement grid trading logic from Algorithm 1 in the paper:
   - Calculate price levels: P_i = P_recent × (1 ± Φ)^MP[i]
   - Calculate quantities: Q_i = A × MQ[i] × c_m × c_f
   - Handle profit/loss thresholds with H parameters

4. Include error handling and logging

5. Use type hints for all methods

Generate complete, production-ready code with docstrings."""

        return prompt

    def _get_code_system_prompt(self) -> str:
        """Get system prompt for code generation.

        Returns:
            System prompt
        """
        return """You are an expert Python developer specializing in algorithmic trading systems.

Generate clean, efficient, production-ready code that:
- Follows PEP 8 style guidelines
- Includes comprehensive docstrings
- Uses type hints
- Has proper error handling
- Is modular and testable
- Follows the three programming laws:
  1. Functional cohesion (single responsibility)
  2. Unidirectional dependencies
  3. Parameter externalization

Output only the Python code, no explanations outside of code comments."""

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for prompt.

        Args:
            parameters: Parameter dictionary

        Returns:
            Formatted parameter string
        """
        lines = []
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key} = {value}")
            elif isinstance(value, str):
                lines.append(f"{key} = '{value}'")
            elif isinstance(value, (list, tuple)):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f"{key} = {repr(value)}")

        return "\n".join(lines)

    def _extract_code(self, llm_response: Any) -> str:
        """Extract Python code from LLM response.

        Args:
            llm_response: LLM response text

        Returns:
            Extracted Python code, as an inert string
        """
        if not isinstance(llm_response, str):
            return ''

        # Try to extract code block
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try without language specifier
        code_pattern = r'```\n(.*?)```'
        matches = re.findall(code_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, return entire response (might be just code)
        return llm_response.strip()

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code by reading it, never by running it.

        `ast.parse` builds a tree and evaluates nothing, which is why it is the
        only inspection used here.

        Args:
            code: Python code to validate

        Returns:
            Validation result
        """
        issues = []

        if not isinstance(code, str) or not code.strip():
            return {'valid': False, 'issues': ['Empty code artefact']}

        # Check if code is parseable
        try:
            ast.parse(code)
        except (SyntaxError, ValueError) as e:
            issues.append(f"Syntax error: {str(e)}")
            return {'valid': False, 'issues': issues}

        # Check for required class
        if 'class TradingBot' not in code:
            issues.append("Missing TradingBot class")

        # Check for required methods
        required_methods = [
            '__init__',
            'calculate_orders',
            'should_execute'
        ]

        for method in required_methods:
            if f'def {method}' not in code:
                issues.append(f"Missing required method: {method}")

        # Check for parameter externalization
        if 'self.params' not in code and 'self.parameters' not in code:
            issues.append("Parameters not externalized")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    async def refine_bot(
        self,
        bot_code: str,
        feedback: Dict[str, Any],
        optimization_level: str = 'parameter'
    ) -> AgentResult:
        """Refine bot based on feedback (hierarchical optimization).

        Args:
            bot_code: Current bot code
            feedback: Feedback from reflection agent
            optimization_level: 'parameter', 'function', or 'strategy'

        Returns:
            AgentResult with refined bot code
        """
        try:
            self.log_action(
                "refining_bot",
                level=optimization_level
            )

            prompt = self._build_refinement_prompt(
                bot_code,
                feedback,
                optimization_level
            )

            response = await self.llm_client.generate_code(
                prompt=prompt,
                system_prompt=self._get_code_system_prompt()
            )

            if response_was_truncated(response):
                self.log_fallback(
                    'refine_bot',
                    'reply truncated at the token limit; the artefact is partial',
                    level=optimization_level
                )

            refined_code = self._extract_code(getattr(response, 'content', ''))

            return AgentResult(
                success=True,
                data={
                    'code': refined_code,
                    'level': optimization_level,
                    'executable': False
                },
                message=f"Bot refined at {optimization_level} level"
            )

        except Exception as e:
            self.log_error(e, {"optimization_level": optimization_level})
            return AgentResult(
                success=False,
                data=None,
                message=f"Bot refinement failed: {str(e)}"
            )

    def _build_refinement_prompt(
        self,
        bot_code: str,
        feedback: Dict[str, Any],
        optimization_level: str
    ) -> str:
        """Build prompt for bot refinement.

        Args:
            bot_code: Current bot code
            feedback: Feedback data
            optimization_level: Optimization level

        Returns:
            Refinement prompt
        """
        if optimization_level == 'parameter':
            prompt = f"""Refine the following trading bot by updating ONLY the parameter values:

Current Code:
```python
{bot_code}
```

Feedback & New Parameters:
{feedback.get('parameters', {})}

Update the parameter values in the code without changing the logic or structure.
Return the complete updated code."""

        elif optimization_level == 'function':
            prompt = f"""Refine the following trading bot by improving or replacing functions:

Current Code:
```python
{bot_code}
```

Issues Identified:
{feedback.get('function_issues', [])}

Suggested Improvements:
{feedback.get('function_improvements', [])}

Update the functions while maintaining the overall strategy logic.
Return the complete updated code."""

        else:  # strategy level
            prompt = f"""Refine the following trading bot with strategic changes:

Current Code:
```python
{bot_code}
```

Performance Issues:
{feedback.get('performance_issues', [])}

Strategic Adjustments Needed:
{feedback.get('strategy_changes', [])}

Make strategic changes to the decision-making logic while maintaining code structure.
Return the complete updated code."""

        return prompt
