"""Hook runner: registry of handlers per hook name, run(name, ctx) invokes them in order."""

from typing import Any, Callable, Dict, List


class HookRunner:
    """Runs registered handlers for each lifecycle hook. Handlers can mutate ctx."""

    def __init__(self) -> None:
        self._registry: Dict[str, List[Callable[[Any], None]]] = {}

    def register(self, hook_name: str, handler: Callable[[Any], None]) -> None:
        """Append a handler for the given hook. Order is preserved."""
        if hook_name not in self._registry:
            self._registry[hook_name] = []
        self._registry[hook_name].append(handler)

    def run(self, hook_name: str, ctx: Any) -> None:
        """Run all handlers registered for hook_name in order. Handlers receive ctx."""
        for handler in self._registry.get(hook_name, []):
            handler(ctx)
