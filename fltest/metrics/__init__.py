"""Metric listeners (the proposal's ``FrameworkMetricListener``), as composable hooks.

Core metrics (``accuracy``, ``loss``) are produced directly by every backend. Additional
listeners registered here add evaluation dimensions — e.g. per-client (personalized)
accuracy, which the proposal flags as commonly missing (Pitfall-3). Declarations are
lazy: :mod:`fltest.metrics.listeners` imports torch, so it is loaded on first use.
"""

from fltest.core.registry import METRICS
from fltest.metrics.base import MetricListenerBaseClass

#: registry name -> module that defines and registers it
BUILTIN_METRICS = {
    "accuracy": "fltest.metrics.listeners",
    "loss": "fltest.metrics.listeners",
    "per_client": "fltest.metrics.listeners",
}

for _name, _module in BUILTIN_METRICS.items():
    METRICS.register_lazy(_name, _module)

__all__ = ["MetricListenerBaseClass", "BUILTIN_METRICS"]
