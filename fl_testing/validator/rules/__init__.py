"""Expectation rules for validating FL experiment results."""

from typing import Dict, Any

from fl_testing.validator.rules.base import ExpectationRule, RuleResult, RuleStatus
from fl_testing.validator.rules.monotonic import MonotonicRule
from fl_testing.validator.rules.bounded import BoundedRule
from fl_testing.validator.rules.relative_threshold import RelativeThresholdRule


RULE_REGISTRY = {
    "monotonic": MonotonicRule,
    "bounded": BoundedRule,
    "relative_threshold": RelativeThresholdRule,
}


def create_rule(config: Dict[str, Any]) -> ExpectationRule:
    """Factory function to create rule instances from config dict."""
    rule_type = config.get("type")
    if rule_type not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule type: {rule_type}. Available: {list(RULE_REGISTRY.keys())}")

    rule_class = RULE_REGISTRY[rule_type]
    params = {k: v for k, v in config.items() if k != "type"}

    return rule_class(**params)


__all__ = [
    "ExpectationRule",
    "RuleResult",
    "RuleStatus",
    "MonotonicRule",
    "BoundedRule",
    "RelativeThresholdRule",
    "create_rule",
    "RULE_REGISTRY",
]
