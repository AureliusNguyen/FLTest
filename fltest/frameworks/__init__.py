"""FL framework adapters.

Each backend implements :class:`fltest.frameworks.base.FrameworkAdapter` and registers
itself by name. Backends are declared lazily here: the adapter module is imported the
first time its name is requested, so listing the catalog costs nothing. Optional heavy
backends (NVFlare) are declared only when their dependency is installed, which is
checked with :func:`importlib.util.find_spec` and does not import it.
"""

from importlib.util import find_spec

from fltest.core.registry import FRAMEWORKS
from fltest.frameworks.base import FrameworkAdapter, RunResult, get_adapter

#: registry name -> adapter module. Several names may share one module (aliases).
BUILTIN_FRAMEWORKS = {
    "reference": "fltest.frameworks.reference",
    "flower": "fltest.frameworks.flower",
    "flwr": "fltest.frameworks.flower",
}

#: optional backends, declared only if the third-party dependency is importable
OPTIONAL_FRAMEWORKS = {
    "nvflare": ("nvflare", "fltest.frameworks.nvflare"),
    "flare": ("nvflare", "fltest.frameworks.nvflare"),
}


def _declare() -> None:
    for name, module in BUILTIN_FRAMEWORKS.items():
        FRAMEWORKS.register_lazy(name, module)
    for name, (dependency, module) in OPTIONAL_FRAMEWORKS.items():
        try:
            available = find_spec(dependency) is not None
        except (ImportError, ValueError):
            available = False
        if available:
            FRAMEWORKS.register_lazy(name, module)


_declare()

__all__ = ["FrameworkAdapter", "RunResult", "get_adapter"]
