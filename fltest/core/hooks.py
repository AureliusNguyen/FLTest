"""
Global hook registry (schemathesis-style). Decorate functions to register them;
they are applied to the HookRunner when the simulation runs.

Set FLTEST_HOOKS to a comma-separated list of module names or paths (with or
without .py). They are loaded in order and hooks attach in that order.
Example: export FLTEST_HOOKS=fltest_hooks_a,fltest_hooks_b
Or: export FLTEST_HOOKS=examples/hooks/fltest_hooks_a,examples/hooks/fltest_hooks_b
Or import hook modules explicitly before run_fl_simulation.
"""

import importlib.util
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Any

_ENV_HOOKS = "FLTEST_HOOKS"

from fltest.core.hook_runner import HookRunner


_REGISTRY: dict[str, list[Callable[[Any], None]]] = defaultdict(list)


def _register(hook_name: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        _REGISTRY[hook_name].append(fn)
        return fn
    return decorator


# Lifecycle hook decorators: use @fltest.core.hooks.<name> or @fltest.hooks.<name> to register
before_simulation = _register("before_simulation")
on_data_partition = _register("on_data_partition")
on_data_distribute = _register("on_data_distribute")
before_round = _register("before_round")
before_client_train = _register("before_client_train")
after_client_train = _register("after_client_train")
before_aggregate = _register("before_aggregate")
on_aggregate = _register("on_aggregate")
after_aggregate = _register("after_aggregate")
after_round = _register("after_round")
after_simulation = _register("after_simulation")


def apply_to(runner: HookRunner) -> None:
    """Register all globally registered handlers onto the given HookRunner."""
    for hook_name, handlers in _REGISTRY.items():
        for h in handlers:
            runner.register(hook_name, h)


# Hooks that run inside Ray workers (client app). Handlers must be from importable
# modules (e.g. fltest.*); path-loaded FLTEST_HOOKS modules are driver-only.
_CLIENT_SIDE_HOOK_NAMES = ("before_client_train", "after_client_train")


def runner_for_workers(
    runner: HookRunner,
    hook_names: tuple[str, ...] = _CLIENT_SIDE_HOOK_NAMES,
    safe_module_prefix: str = "fltest.",
) -> HookRunner:
    """
    Return a new HookRunner with only handlers that are safe to pickle and run
    inside Ray workers (i.e. defined in an importable module like fltest.*).
    Path-loaded hook modules (e.g. from FLTEST_HOOKS) exist only in the driver;
    their handlers must not be passed to workers or deserialization fails.
    """
    out = HookRunner()
    for name in hook_names:
        for h in runner._registry.get(name, []):
            mod = getattr(h, "__module__", "")
            if mod.startswith(safe_module_prefix):
                out.register(name, h)
    return out


def _load_hooks_module(name: str) -> None:
    """Load a single hooks module by name or path (with or without .py)."""
    raw = name.strip()
    if not raw:
        return
    stem = raw.removesuffix(".py")

    # Path-like: load from file relative to cwd
    if "/" in stem or "\\" in stem:
        path = Path.cwd() / stem
        if not path.suffix:
            path = path.with_suffix(".py")
        if not path.is_file():
            return
        mod_name = re.sub(r"[\\/.]", "_", stem).strip("_") or "hooks_module"
        if mod_name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        return

    if stem in sys.modules:
        return
    try:
        importlib.import_module(stem)
        return
    except ImportError:
        pass
    if "." in stem:
        return
    cwd = Path.cwd()
    for candidate in (cwd, cwd.parent):
        for path in (candidate / f"{stem}.py", candidate / stem):
            if path.is_file():
                spec = importlib.util.spec_from_file_location(stem, path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[stem] = mod
                    spec.loader.exec_module(mod)
                return


def import_convention_hooks() -> None:
    """
    Import hook modules from FLTEST_HOOKS (comma-separated list of names or
    paths, with or without .py). Loaded in order; hooks attach in that order.
    If FLTEST_HOOKS is unset, nothing is loaded.
    """
    raw = os.environ.get(_ENV_HOOKS)
    if not raw:
        return
    for name in (s.strip() for s in raw.split(",") if s.strip()):
        _load_hooks_module(name)
