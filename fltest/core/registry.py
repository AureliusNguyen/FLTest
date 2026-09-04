"""Name-keyed plugin registries for the four extension points of FLTest.

Frameworks, attacks, defenses, and metric listeners all register here by name so the
orchestrator can wire a run purely from a YAML config (``framework: flwr``,
``attacks: [label_flip]`` …) without importing concrete classes. This is what makes
the testbed extensible: dropping a new ``@register_attack("foo")`` makes ``foo`` usable
from config immediately.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object registry with a decorator-based ``register``.

    An entry may also be declared *lazily*: the name is registered alongside the module
    that implements it, and that module is imported only when the entry is first
    requested. :meth:`names` and ``in`` see lazy entries without importing anything,
    which is what lets ``fltest list`` print the catalog without loading torch.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}
        self._lazy: Dict[str, str] = {}  # name -> module whose import registers it

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            key = name.lower()
            if key in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[key] = obj
            self._lazy.pop(key, None)  # the concrete entry supersedes its declaration
            return obj

        return deco

    def register_lazy(self, name: str, module: str) -> None:
        """Declare that ``module`` provides ``name``, without importing it."""
        key = name.lower()
        if key not in self._items:
            self._lazy[key] = module

    def get(self, name: str) -> T:
        key = name.lower()
        if key not in self._items:
            module = self._lazy.get(key)
            if module is None:
                raise KeyError(
                    f"Unknown {self.kind} '{name}'. Available: {self.names()}"
                )
            import_module(module)  # its @register_* decorator fills _items
            if key not in self._items:
                raise KeyError(
                    f"{self.kind} '{name}' is declared as provided by '{module}', but "
                    f"importing that module did not register it."
                )
        return self._items[key]

    def names(self):
        return sorted(set(self._items) | set(self._lazy))

    def __contains__(self, name: str) -> bool:
        key = name.lower()
        return key in self._items or key in self._lazy


# Global registries. Modules populate these on import.
FRAMEWORKS: "Registry[Type]" = Registry("framework")
ATTACKS: "Registry[Type]" = Registry("attack")
DEFENSES: "Registry[Type]" = Registry("defense")
METRICS: "Registry[Type]" = Registry("metric")


def register_framework(name: str):
    return FRAMEWORKS.register(name)


def register_attack(name: str):
    return ATTACKS.register(name)


def register_defense(name: str):
    return DEFENSES.register(name)


def register_metric(name: str):
    return METRICS.register(name)
