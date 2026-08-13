"""Feedback Reflection Agent (Afr) - optimizes parameters through mathematical reflection.

Implements: Afr ◦ γ : B × F × Θ → F* × Θ*
Uses mathematical reasoning (γ) to solve optimization problems:
Θ* = arg max Σ ωi·Ji(Θ,F) subject to C(Θ) = {Θ ∈ R^n | A(R)Θ ≤ b(R)}
"""

import json
import math
from typing import Any, Dict, List, Optional

from .base import (
    PARAMETER_BOUNDS,
    AgentResult,
    BaseAgent,
    extract_json_object,
    response_was_truncated,
    validate_numeric_parameters,
)
from ..llm.client import LLMClient
from ..utils.config import Config


# How far a proposal is allowed to move a parameter in one reflection. A
# proposal that wants to triple an allocation is not an optimisation, it is a
# different plan, and it has to arrive one bounded step at a time.
MAX_INCREASE_FACTOR = 1.5
MAX_DECREASE_FACTOR = 0.25

# Floor on the deterministic de-risking below, so a bad run shrinks the book
# rather than reducing it to a size that cannot trade at all.
MIN_RISK_SCALE = 0.25

# Win rate the de-risking treats as par. Below it, size comes down in
# proportion to the shortfall.
TARGET_WIN_RATE = 0.5


class FeedbackReflectionAgent(BaseAgent):
    """Feedback Reflection Agent (Afr).

    Reflects on deployment feedback and formulates precise optimization plans:
    1. Organizes risk scenarios from feedback
    2. Transforms scenarios into mathematical constraints
    3. Solves for optimal parameters
    4. Provides hierarchical refinement guidance
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Config = None
    ):
        """Initialize Feedback Reflection Agent.

        Args:
            llm_client: LLM client for mathematical reasoning
            config: Configuration
        """
        super().__init__("FeedbackReflectionAgent", llm_client, config)

        # The drawdown the account is willing to spend before it is considered
        # to have used up its budget. Configured as a percentage; the de-risking
        # below works in fractions.
        max_drawdown_pct = self.config.get('risk.max_drawdown', 20)
        try:
            self.drawdown_budget = max(float(max_drawdown_pct) / 100.0, 0.01)
        except (TypeError, ValueError):
            self.drawdown_budget = 0.20

    async def execute(
        self,
        bot_metadata: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> AgentResult:
        """Reflect on feedback and generate optimization plan.

        Args:
            bot_metadata: Trading bot metadata
            feedback: Feedback from deployment/simulation

        Returns:
            AgentResult with optimized parameters and refinement plan
        """
        try:
            self.log_action(
                "analyzing_feedback",
                pair=bot_metadata.get('pair')
            )

            # Extract and categorize feedback
            categorized_feedback = self._categorize_feedback(feedback)

            # Identify risk scenarios
            risk_scenarios = self._identify_risk_scenarios(categorized_feedback)

            # Transform to mathematical constraints
            constraints = await self._formulate_constraints(
                risk_scenarios,
                bot_metadata.get('parameters', {})
            )

            # Solve optimization problem
            optimal_parameters = await self._optimize_parameters(
                bot_metadata.get('parameters', {}),
                constraints,
                categorized_feedback
            )

            # Determine optimization level needed
            optimization_level = self._determine_optimization_level(
                categorized_feedback,
                optimal_parameters,
                bot_metadata.get('parameters', {})
            )

            # Generate hierarchical feedback
            hierarchical_feedback = {
                'optimization_level': optimization_level,
                'optimal_parameters': optimal_parameters,
                'constraints': constraints,
                'risk_scenarios': risk_scenarios,
                'recommendations': self._generate_recommendations(
                    optimization_level,
                    categorized_feedback
                )
            }

            self.log_action(
                "reflection_complete",
                level=optimization_level,
                parameters_changed=len(optimal_parameters)
            )

            return AgentResult(
                success=True,
                data=hierarchical_feedback,
                message=f"Optimization plan generated: {optimization_level} level",
                metadata=hierarchical_feedback
            )

        except Exception as e:
            self.log_error(e, {"feedback": feedback})
            return AgentResult(
                success=False,
                data=None,
                message=f"Reflection failed: {str(e)}"
            )

    def _categorize_feedback(self, feedback: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Categorize feedback into types.

        Args:
            feedback: Raw feedback data

        Returns:
            Categorized feedback
        """
        categorized = {
            'performance': [],
            'risk': [],
            'stability': [],
            'efficiency': []
        }

        # Performance metrics
        if 'pnl' in feedback:
            categorized['performance'].append({
                'type': 'pnl',
                'value': feedback['pnl'],
                'severity': 'high' if feedback['pnl'] < -100 else 'low'
            })

        if 'win_rate' in feedback:
            categorized['performance'].append({
                'type': 'win_rate',
                'value': feedback['win_rate'],
                'severity': 'high' if feedback['win_rate'] < 0.4 else 'low'
            })

        # Risk events
        if 'max_drawdown' in feedback:
            categorized['risk'].append({
                'type': 'drawdown',
                'value': feedback['max_drawdown'],
                'severity': 'critical' if feedback['max_drawdown'] > 0.2 else 'medium'
            })

        if 'risk_events' in feedback:
            for event in feedback['risk_events']:
                categorized['risk'].append(event)

        # Stability issues
        if 'errors' in feedback:
            categorized['stability'].extend(feedback['errors'])

        if 'execution_failures' in feedback:
            categorized['stability'].extend(feedback['execution_failures'])

        # Efficiency metrics
        if 'avg_latency' in feedback:
            categorized['efficiency'].append({
                'type': 'latency',
                'value': feedback['avg_latency']
            })

        return categorized

    def _identify_risk_scenarios(
        self,
        categorized_feedback: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Identify risk scenarios requiring constraints.

        Args:
            categorized_feedback: Categorized feedback

        Returns:
            List of risk scenarios
        """
        scenarios = []

        # Check for critical risk events
        for risk_item in categorized_feedback.get('risk', []):
            if risk_item.get('severity') in ['critical', 'high']:
                scenarios.append({
                    'type': 'risk_constraint',
                    'description': risk_item.get('type'),
                    'value': risk_item.get('value'),
                    'requires_constraint': True
                })

        # Check for poor performance
        performance_items = categorized_feedback.get('performance', [])
        poor_performance = any(
            item['severity'] == 'high'
            for item in performance_items
        )

        if poor_performance:
            scenarios.append({
                'type': 'performance_constraint',
                'description': 'poor_performance',
                'requires_optimization': True
            })

        return scenarios

    async def _formulate_constraints(
        self,
        risk_scenarios: List[Dict[str, Any]],
        current_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formulate mathematical constraints from risk scenarios.

        Args:
            risk_scenarios: Risk scenarios
            current_parameters: Current parameters

        Returns:
            Constraint specifications
        """
        if not risk_scenarios:
            return {}

        # Use mathematical reasoning to formulate constraints
        prompt = self._build_constraint_prompt(risk_scenarios, current_parameters)

        response = await self.llm_client.generate_reasoning(
            prompt=prompt,
            system_prompt=self._get_reasoning_system_prompt()
        )

        # Parse constraints from response
        constraints = self._parse_constraints(response)

        return constraints

    def _build_constraint_prompt(
        self,
        risk_scenarios: List[Dict[str, Any]],
        current_parameters: Dict[str, Any]
    ) -> str:
        """Build prompt for constraint formulation.

        Args:
            risk_scenarios: Risk scenarios
            current_parameters: Current parameters

        Returns:
            Formatted prompt
        """
        scenarios_text = "\n".join([
            f"- {s.get('type')}: {s.get('description')} (value: {s.get('value', 'N/A')})"
            for s in risk_scenarios
        ])

        prompt = f"""Formulate mathematical constraints to prevent these risk scenarios:

Risk Scenarios:
{scenarios_text}

Current Parameters:
```
{json.dumps(current_parameters, indent=2)}
```

Task: For each risk scenario, formulate a linear constraint of the form:
A·Θ ≤ b

Where:
- Θ is the parameter vector
- A is the constraint matrix coefficients
- b is the threshold vector

Example (from paper Appendix A.1):
For position size control under volatility:
Σ Q_i ≤ Q_max => Σ q_i ≤ Q_max / (A × c_m × c_f)

Provide:
1. Mathematical formulation for each scenario
2. Parameter bounds
3. Justification based on risk mitigation

Format as JSON with constraint specifications."""

        return prompt

    def _get_reasoning_system_prompt(self) -> str:
        """Get system prompt for mathematical reasoning.

        Returns:
            System prompt
        """
        return """You are a quantitative analyst specializing in portfolio optimization and risk management.

Use rigorous mathematical reasoning to:
- Formulate linear programming constraints from risk scenarios
- Derive parameter bounds from historical risk events
- Balance competing objectives using Pareto efficiency
- Provide mathematical justification for all recommendations

Output precise mathematical formulations with clear variable definitions."""

    def _parse_constraints(self, response: Any) -> Dict[str, Any]:
        """Parse constraints from LLM response.

        Constraints are descriptive only: they are recorded, and they are fed
        back into the next prompt, but nothing in this dictionary sizes an
        order. When no object can be decoded the reply is kept verbatim under
        'description' so an operator can read what was actually said, and the
        fallback is logged.

        Args:
            response: LLM response object, or the reply text

        Returns:
            Parsed constraints
        """

        content = response if isinstance(response, str) else getattr(
            response, 'content', ''
        )

        if response_was_truncated(response):
            self.log_fallback(
                'parse_constraints',
                'reply truncated at the token limit'
            )

        parsed = extract_json_object(content)
        if parsed is not None:
            return parsed

        self.log_fallback(
            'parse_constraints',
            'no decodable JSON object in the reply'
        )

        # Fallback: keep the text, claim no constraints
        return {
            'description': content if isinstance(content, str) else '',
            'constraints': []
        }

    async def _optimize_parameters(
        self,
        current_parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        categorized_feedback: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Solve optimization problem for optimal parameters.

        Implements: Θ* = arg max Σ ωi·Ji(Θ,F) s.t. constraints

        Args:
            current_parameters: Current parameters
            constraints: Constraint specifications
            categorized_feedback: Categorized feedback

        Returns:
            Optimal parameters
        """
        # Use mathematical reasoning for optimization
        prompt = self._build_optimization_prompt(
            current_parameters,
            constraints,
            categorized_feedback
        )

        response = await self.llm_client.generate_reasoning(
            prompt=prompt,
            system_prompt=self._get_reasoning_system_prompt()
        )

        # Parse optimal parameters
        optimal_params = self._parse_optimal_parameters(
            response,
            current_parameters,
            categorized_feedback
        )

        return optimal_params

    def _build_optimization_prompt(
        self,
        current_parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        categorized_feedback: Dict[str, List[Any]]
    ) -> str:
        """Build prompt for parameter optimization.

        Args:
            current_parameters: Current parameters
            constraints: Constraints
            categorized_feedback: Feedback

        Returns:
            Optimization prompt
        """
        # Calculate objective weights
        performance_items = categorized_feedback.get('performance', [])
        risk_items = categorized_feedback.get('risk', [])

        prompt = f"""Solve the parameter optimization problem:

Objective: Maximize risk-adjusted performance
Θ* = arg max Σ ωi·Ji(Θ,F)

Current Parameters:
```
{json.dumps(current_parameters, indent=2)}
```

Constraints:
```
{json.dumps(constraints, indent=2)}
```

Performance Feedback:
{json.dumps(performance_items, indent=2)}

Risk Feedback:
{json.dumps(risk_items, indent=2)}

Optimization Objectives (in order of priority):
1. Satisfy all risk constraints
2. Maximize win rate
3. Minimize drawdown
4. Maximize Sharpe ratio

Provide optimal parameter values with mathematical justification.
Format as JSON with parameter names and values."""

        return prompt

    def _parse_optimal_parameters(
        self,
        response: Any,
        current_parameters: Dict[str, Any],
        categorized_feedback: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """Turn a reply into parameters that are safe to hand onwards.

        Three gates, in order. A proposed value must decode, it must pass the
        type, finiteness and range checks in `PARAMETER_BOUNDS`, and it must
        sit within one bounded step of the value it is replacing. Whatever
        survives is then scaled by `_risk_scale_factor`, so a run that lost
        money cannot come back with a larger book than it started with. When
        nothing decodes, the de-risked current parameters are the answer.

        Args:
            response: LLM response object, or the reply text
            current_parameters: Current parameters
            categorized_feedback: Categorized feedback, used for de-risking

        Returns:
            Optimal parameters, every numeric value validated and bounded
        """

        feedback = categorized_feedback or {}
        scale = self._risk_scale_factor(feedback)

        baseline, baseline_rejected = validate_numeric_parameters(
            current_parameters,
            PARAMETER_BOUNDS,
            {}
        )

        if baseline_rejected:
            self.log_fallback(
                'optimize_parameters',
                'current parameters failed validation and were dropped',
                rejected=sorted(baseline_rejected)
            )

        content = response if isinstance(response, str) else getattr(
            response, 'content', None
        )

        if response_was_truncated(response):
            self.log_fallback(
                'optimize_parameters',
                'reply truncated at the token limit'
            )

        candidate = extract_json_object(content)

        if candidate is None:
            self.log_fallback(
                'optimize_parameters',
                'no decodable JSON object in the reply; holding current '
                'parameters and applying the risk scale',
                risk_scale=scale
            )
            return self._apply_scale(baseline, scale)

        proposed, rejected = validate_numeric_parameters(
            candidate,
            PARAMETER_BOUNDS,
            baseline
        )

        if rejected:
            self.log_fallback(
                'optimize_parameters',
                'proposed values failed validation and were held at current',
                rejected=sorted(rejected)
            )

        stepped = self._limit_step(baseline, proposed)

        return self._apply_scale(stepped, scale)

    def _risk_scale_factor(
        self,
        categorized_feedback: Dict[str, List[Any]]
    ) -> float:
        """Compute how far to shrink exposure given what the run actually did.

        Two signals contribute, and the smaller wins:

        * drawdown, as a fraction of the configured drawdown budget, so a run
          that has spent its whole budget trades at `MIN_RISK_SCALE`;
        * win rate, as a fraction of `TARGET_WIN_RATE`, so a strategy losing
          more often than it wins sizes down in proportion.

        With no usable signal the factor is 1.0, which leaves the parameters
        alone. That is deliberate: an arbitrary shrink on every reflection is
        not risk management, it is decay.

        Args:
            categorized_feedback: Categorized feedback

        Returns:
            A factor in [MIN_RISK_SCALE, 1.0]
        """

        scale = 1.0

        for item in categorized_feedback.get('risk', []):
            if not isinstance(item, dict) or item.get('type') != 'drawdown':
                continue
            drawdown = self._as_number(item.get('value'))
            if drawdown is None or drawdown <= 0:
                continue
            scale = min(scale, 1.0 - min(drawdown / self.drawdown_budget, 1.0))

        for item in categorized_feedback.get('performance', []):
            if not isinstance(item, dict) or item.get('type') != 'win_rate':
                continue
            win_rate = self._as_number(item.get('value'))
            if win_rate is None or win_rate < 0 or win_rate >= TARGET_WIN_RATE:
                continue
            scale = min(scale, win_rate / TARGET_WIN_RATE)

        return max(MIN_RISK_SCALE, min(1.0, scale))

    @staticmethod
    def _as_number(value: Any) -> Optional[float]:
        """Return a finite float, or None for anything that is not one.

        Args:
            value: Candidate value

        Returns:
            The value as a float, or None
        """

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _limit_step(
        baseline: Dict[str, Any],
        proposed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clamp each proposed value to one bounded step from the current one.

        Args:
            baseline: Validated current parameters
            proposed: Validated proposed parameters

        Returns:
            Proposed parameters, each within its step limit
        """

        stepped = dict(proposed)

        for name, current in baseline.items():
            if name not in stepped:
                continue
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                continue
            if current <= 0:
                continue

            upper = current * MAX_INCREASE_FACTOR
            lower = current * MAX_DECREASE_FACTOR
            value = min(max(float(stepped[name]), lower), upper)

            bound = PARAMETER_BOUNDS.get(name)
            stepped[name] = int(round(value)) if bound and bound.integer else value

        return stepped

    @staticmethod
    def _apply_scale(parameters: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """Scale the exposure parameters down by a de-risking factor.

        Only the two values that decide how much money is at stake are scaled.
        Grid geometry and stop distances are not exposure, and shrinking them
        alongside would tighten the strategy rather than de-risk it.

        Args:
            parameters: Validated parameters
            scale: Factor in [MIN_RISK_SCALE, 1.0]

        Returns:
            Parameters with exposure scaled
        """

        scaled = dict(parameters)

        if scale >= 1.0:
            return scaled

        for name in ('capital_allocation', 'max_position_pct'):
            value = scaled.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            bound = PARAMETER_BOUNDS[name]
            scaled[name] = max(float(value) * scale, bound.minimum)

        return scaled

    def _determine_optimization_level(
        self,
        categorized_feedback: Dict[str, List[Any]],
        optimal_parameters: Dict[str, Any],
        current_parameters: Dict[str, Any]
    ) -> str:
        """Determine which optimization level is needed.

        Args:
            categorized_feedback: Feedback
            optimal_parameters: Optimal parameters
            current_parameters: Current parameters

        Returns:
            Optimization level: 'parameter', 'function', or 'strategy'
        """
        # Check if parameter changes are sufficient
        param_changes = sum(
            1 for key in optimal_parameters
            if optimal_parameters.get(key) != current_parameters.get(key)
        )

        # Critical issues require higher-level optimization
        critical_risk = any(
            item.get('severity') == 'critical'
            for item in categorized_feedback.get('risk', [])
        )

        stability_issues = len(categorized_feedback.get('stability', [])) > 5

        if critical_risk or stability_issues:
            return 'strategy'
        elif param_changes > 3:
            return 'function'
        else:
            return 'parameter'

    def _generate_recommendations(
        self,
        optimization_level: str,
        categorized_feedback: Dict[str, List[Any]]
    ) -> List[str]:
        """Generate specific recommendations.

        Args:
            optimization_level: Optimization level
            categorized_feedback: Feedback

        Returns:
            List of recommendations
        """
        recommendations = []

        if optimization_level == 'parameter':
            recommendations.append("Adjust parameter values within current structure")
            recommendations.append("Monitor performance for 24 hours before further changes")

        elif optimization_level == 'function':
            recommendations.append("Review and update computational functions")
            recommendations.append("Consider alternative technical indicators")
            recommendations.append("Optimize order execution logic")

        else:  # strategy
            recommendations.append("Fundamental strategy revision needed")
            recommendations.append("Consider different strategy type")
            recommendations.append("Implement additional risk controls")

        # Add specific recommendations based on feedback
        if any(item.get('severity') == 'critical' for item in categorized_feedback.get('risk', [])):
            recommendations.append("CRITICAL: Implement emergency position limits")

        return recommendations
