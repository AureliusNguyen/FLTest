"""Bounded expectation rule - validates metric stays within specified bounds."""

from typing import List, Optional

from fl_testing.validator.rules.base import (
    ExpectationRule,
    RuleResult,
    RuleStatus,
    ExperimentResult,
)


class BoundedRule(ExpectationRule):
    """Validates that a metric stays within specified bounds.

    Example: Accuracy should always be between 0.0 and 1.0,
    or loss should be between 0.0 and some maximum value.
    """

    def __init__(
        self,
        metric: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """Initialize bounded rule.

        Args:
            metric: Name of the metric to validate (e.g., 'Final Round Accuracy').
            min_value: Minimum allowed value (optional).
            max_value: Maximum allowed value (optional).
        """
        super().__init__(metric)
        self.min_value = min_value
        self.max_value = max_value

        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

    def validate(self, experiments: List[ExperimentResult]) -> RuleResult:
        """Validate that metric stays within bounds for all experiments."""
        violations = []
        values = []

        successful = self._get_successful_experiments(experiments)

        if not successful:
            return RuleResult(
                rule_type="bounded",
                rule_config=self._get_config(),
                status=RuleStatus.SKIPPED,
                message="No successful experiments to validate",
                details={"values": [], "violations": []},
            )

        for exp in successful:
            metric_val = exp.results.get(self.metric)
            if metric_val is not None:
                values.append({"experiment": exp.id, "value": metric_val})

                if self.min_value is not None and metric_val < self.min_value:
                    violations.append(
                        {
                            "experiment": exp.id,
                            "value": metric_val,
                            "violation": "below_min",
                            "bound": self.min_value,
                        }
                    )
                if self.max_value is not None and metric_val > self.max_value:
                    violations.append(
                        {
                            "experiment": exp.id,
                            "value": metric_val,
                            "violation": "above_max",
                            "bound": self.max_value,
                        }
                    )

        passed = len(violations) == 0
        return RuleResult(
            rule_type="bounded",
            rule_config=self._get_config(),
            status=RuleStatus.PASSED if passed else RuleStatus.FAILED,
            message=self._build_message(passed, violations, values),
            details={"values": values, "violations": violations},
        )

    def _get_config(self) -> dict:
        """Get rule configuration as dict."""
        return {
            "metric": self.metric,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    def _build_message(self, passed: bool, violations: list, values: list) -> str:
        """Build human-readable result message."""
        if not values:
            return f"No values found for metric '{self.metric}'"

        actual_min = min(v["value"] for v in values)
        actual_max = max(v["value"] for v in values)

        bounds_str = self._format_bounds()

        if passed:
            return (
                f"{self.metric} within bounds {bounds_str}. "
                f"Actual range: [{actual_min:.4f}, {actual_max:.4f}]"
            )
        return (
            f"Bound violations: {len(violations)} experiment(s) outside {bounds_str}. "
            f"Actual range: [{actual_min:.4f}, {actual_max:.4f}]"
        )

    def _format_bounds(self) -> str:
        """Format bounds for display."""
        min_str = f"{self.min_value}" if self.min_value is not None else "-inf"
        max_str = f"{self.max_value}" if self.max_value is not None else "inf"
        return f"[{min_str}, {max_str}]"
