"""Relative threshold rule - validates adjacent metric changes don't exceed threshold."""

from typing import List

from fl_testing.validator.rules.base import (
    ExpectationRule,
    RuleResult,
    RuleStatus,
    ExperimentResult,
)


class RelativeThresholdRule(ExpectationRule):
    """Validates that metric changes between adjacent parameter values don't exceed threshold.

    Example: When increasing num_rounds from 5 to 10, the loss shouldn't change
    by more than 50% - a sudden dramatic change might indicate a bug.
    """

    def __init__(
        self,
        metric: str,
        parameter: str,
        max_change_percent: float,
    ):
        """Initialize relative threshold rule.

        Args:
            metric: Name of the metric to validate (e.g., 'Final Round Loss').
            parameter: Name of the parameter being varied (e.g., 'num_rounds').
            max_change_percent: Maximum allowed percent change between adjacent values.
        """
        super().__init__(metric)
        self.parameter = parameter
        self.max_change_percent = max_change_percent

        if max_change_percent <= 0:
            raise ValueError("max_change_percent must be positive")

    def validate(self, experiments: List[ExperimentResult]) -> RuleResult:
        """Validate that adjacent metric changes don't exceed threshold."""
        values = self._extract_metric_values(experiments, self.parameter)

        if len(values) < 2:
            return RuleResult(
                rule_type="relative_threshold",
                rule_config=self._get_config(),
                status=RuleStatus.SKIPPED,
                message="Insufficient data points (need >= 2)",
                details={"values": [{"param": p, "metric": m} for p, m in values]},
            )

        violations = []
        changes = []

        for i in range(1, len(values)):
            prev_param, prev_metric = values[i - 1]
            curr_param, curr_metric = values[i]

            # Calculate percent change (handle zero case)
            if prev_metric != 0:
                pct_change = abs((curr_metric - prev_metric) / prev_metric) * 100
            else:
                pct_change = abs(curr_metric) * 100 if curr_metric != 0 else 0

            change_record = {
                f"{self.parameter}_from": prev_param,
                f"{self.parameter}_to": curr_param,
                "metric_from": prev_metric,
                "metric_to": curr_metric,
                "percent_change": round(pct_change, 2),
            }
            changes.append(change_record)

            if pct_change > self.max_change_percent:
                violations.append(change_record)

        passed = len(violations) == 0
        return RuleResult(
            rule_type="relative_threshold",
            rule_config=self._get_config(),
            status=RuleStatus.PASSED if passed else RuleStatus.FAILED,
            message=self._build_message(passed, violations, changes),
            details={
                "changes": changes,
                "violations": violations,
                "max_change_percent": self.max_change_percent,
            },
        )

    def _get_config(self) -> dict:
        """Get rule configuration as dict."""
        return {
            "parameter": self.parameter,
            "metric": self.metric,
            "max_change_percent": self.max_change_percent,
        }

    def _build_message(self, passed: bool, violations: list, changes: list) -> str:
        """Build human-readable result message."""
        if not changes:
            return "No adjacent changes to validate"

        max_observed = max(c["percent_change"] for c in changes)

        if passed:
            return (
                f"All adjacent {self.metric} changes within {self.max_change_percent}%. "
                f"Max observed: {max_observed:.1f}%"
            )
        return (
            f"Threshold violations: {len(violations)} change(s) exceed "
            f"{self.max_change_percent}%. Max observed: {max_observed:.1f}%"
        )
