"""Monotonic expectation rule - validates metric changes monotonically with parameter."""

from typing import List, Literal

from fl_testing.validator.rules.base import (
    ExpectationRule,
    RuleResult,
    RuleStatus,
    ExperimentResult,
)


class MonotonicRule(ExpectationRule):
    """Validates that a metric changes monotonically with a parameter.

    Example: Accuracy should increase (or at least not decrease significantly)
    as the number of training rounds increases.
    """

    def __init__(
        self,
        metric: str,
        parameter: str,
        direction: Literal["increasing", "decreasing"],
        tolerance: float = 0.0,
    ):
        """Initialize monotonic rule.

        Args:
            metric: Name of the metric to validate (e.g., 'Final Round Accuracy').
            parameter: Name of the parameter being varied (e.g., 'num_rounds').
            direction: Expected direction - 'increasing' or 'decreasing'.
            tolerance: Allowed tolerance for violations (default 0.0).
        """
        super().__init__(metric)
        self.parameter = parameter
        self.direction = direction
        self.tolerance = tolerance

    def validate(self, experiments: List[ExperimentResult]) -> RuleResult:
        """Validate monotonicity of metric with respect to parameter."""
        values = self._extract_metric_values(experiments, self.parameter)

        if len(values) < 2:
            return RuleResult(
                rule_type="monotonic",
                rule_config=self._get_config(),
                status=RuleStatus.SKIPPED,
                message="Insufficient data points for monotonicity check (need >= 2)",
                details={"values": [{"param": p, "metric": m} for p, m in values]},
            )

        violations = []
        for i in range(1, len(values)):
            prev_param, prev_metric = values[i - 1]
            curr_param, curr_metric = values[i]

            if self.direction == "increasing":
                # curr should be >= prev (with tolerance)
                if curr_metric < prev_metric - self.tolerance:
                    violations.append(
                        {
                            "index": i,
                            f"{self.parameter}_prev": prev_param,
                            f"{self.parameter}_curr": curr_param,
                            "metric_prev": prev_metric,
                            "metric_curr": curr_metric,
                            "expected": "increasing",
                            "actual_change": curr_metric - prev_metric,
                        }
                    )
            else:  # decreasing
                if curr_metric > prev_metric + self.tolerance:
                    violations.append(
                        {
                            "index": i,
                            f"{self.parameter}_prev": prev_param,
                            f"{self.parameter}_curr": curr_param,
                            "metric_prev": prev_metric,
                            "metric_curr": curr_metric,
                            "expected": "decreasing",
                            "actual_change": curr_metric - prev_metric,
                        }
                    )

        passed = len(violations) == 0
        return RuleResult(
            rule_type="monotonic",
            rule_config=self._get_config(),
            status=RuleStatus.PASSED if passed else RuleStatus.FAILED,
            message=self._build_message(passed, violations),
            details={
                "values": [
                    {self.parameter: p, self.metric: m} for p, m in values
                ],
                "violations": violations,
                "tolerance": self.tolerance,
            },
        )

    def _get_config(self) -> dict:
        """Get rule configuration as dict."""
        return {
            "parameter": self.parameter,
            "metric": self.metric,
            "direction": self.direction,
            "tolerance": self.tolerance,
        }

    def _build_message(self, passed: bool, violations: list) -> str:
        """Build human-readable result message."""
        if passed:
            return (
                f"{self.metric} changes {self.direction}ly with {self.parameter} "
                f"(tolerance: {self.tolerance})"
            )
        return (
            f"Monotonicity violated: {len(violations)} violation(s) found. "
            f"Expected {self.metric} to be {self.direction} with {self.parameter}."
        )
