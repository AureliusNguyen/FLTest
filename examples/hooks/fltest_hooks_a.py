"""Hook handlers (A). Load via FLTEST_HOOKS=examples/hooks/fltest_hooks_a or from fltest_hooks."""

from pathlib import Path
from datetime import datetime

from fltest.core import hooks


def _write_signal(ctx, suffix: str):
    """Write tmp/client_<cid>_round_<r>_<suffix>.txt so client-side hook runs are visible (workers often suppress stdout)."""
    cid = getattr(ctx, "client_id", None)
    r = getattr(ctx, "round", None)
    base = getattr(ctx.cfg, "hook_signal_dir", None) if getattr(ctx, "cfg", None) else None
    dir_path = Path(base if base else "tmp")
    dir_path.mkdir(parents=True, exist_ok=True)
    f = dir_path / f"client_{cid}_round_{r}_{suffix}.txt"
    f.write_text(f"[Hook A] after_client_train client_id={cid} round={r} at {datetime.now().isoformat()}\n")


@hooks.before_simulation
def log_before_simulation_a(ctx):
    print("[Hook A] before_simulation")


@hooks.on_data_distribute
def log_on_data_distribute_a(ctx):
    print(f"[Hook A] on_data_distribute {d}")


@hooks.before_round
def log_before_round_a(ctx):
    r = getattr(ctx, "round", None)
    print(f"[Hook A] before_round round={r}")


@hooks.before_client_train
def log_before_client_train_a(ctx):
    cid = getattr(ctx, "client_id", None)
    r = getattr(ctx, "round", None)
    print(f"[Hook A] before_client_train round={r} client_id={cid}")
    _write_signal(ctx, "a")


@hooks.after_client_train
def log_after_client_train_a(ctx):
    cid = getattr(ctx, "client_id", None)
    r = getattr(ctx, "round", None)
    print(f"[Hook A] after_client_train round={r} client_id={cid}")
    _write_signal(ctx, "a")


@hooks.after_round
def log_after_round_a(ctx):
    r = getattr(ctx, "round", None)
    print(f"[Hook A] after_round round={r}")


@hooks.after_simulation
def log_after_simulation_a(ctx):
    print("[Hook A] after_simulation")
