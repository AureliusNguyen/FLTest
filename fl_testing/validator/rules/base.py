"""Base classes for expectation rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class RuleStatus(Enum):
    """Status of a rule validation."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class RuleResult:
    """Result of a rule validation."""

    rule_type: str
    rule_config: Dict[str, Any]
    status: RuleStatus
    message: str
    details: Dict[str, Any]


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    id: str
    parameters: Dict[str, Any]
    status: str  # success, failed, timeout
    duration_seconds: float
    results: Optional[Dict[str, float]]  # The FL results dict
    error: Optional[str]
    cached: bool


class ExpectationRule(ABC):
    """Abstract base class for validation rules."""

    def __init__(self, metric: str, **kwargs):
        """Initialize rule.

        Args:
            metric: Name of the metric to validate.
            **kwargs: Additional rule-specific parameters.
        """
        self.metric = metric
        self.config = kwargs

    @abstractmethod
    def validate(self, experiments: List[ExperimentResult]) -> RuleResult:
        """Validate the rule against experiment results.

        Args:
            experiments: List of experiment results.

        Returns:
            RuleResult indicating pass/fail and details.
        """
        pass

    def _extract_metric_values(
        self, experiments: List[ExperimentResult], parameter: str
    ) -> List[tuple]:
        """Extract (parameter_value, metric_value) pairs sorted by parameter.

        Args:
            experiments: List of experiment results.
            parameter: Name of the parameter to extract.

        Returns:
            List of (param_value, metric_value) tuples sorted by param_value.
        """
        values = []
        for exp in experiments:
            if exp.status == "success" and exp.results:
                param_val = exp.parameters.get(parameter)
                metric_val = exp.results.get(self.metric)
                if param_val is not None and metric_val is not None:
                    values.append((param_val, metric_val))
        return sorted(values, key=lambda x: x[0])

    def _get_successful_experiments(
        self, experiments: List[ExperimentResult]
    ) -> List[ExperimentResult]:
        """Filter to only successful experiments with results."""
        return [
            exp
            for exp in experiments
            if exp.status == "success" and exp.results is not None
        ]
