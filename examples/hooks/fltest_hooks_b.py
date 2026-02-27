"""Hook handlers (B). Load via FLTEST_HOOKS=examples/hooks/fltest_hooks_b or from fltest_hooks."""

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
    f.write_text(f"[Hook B] after_client_train client_id={cid} round={r} at {datetime.now().isoformat()}\n")


@hooks.before_simulation
def log_before_simulation_b(ctx):
    print("[Hook B] before_simulation")


@hooks.on_data_distribute
def log_on_data_distribute_b(ctx):
    print("[Hook B] on_data_distribute")


@hooks.before_round
def log_before_round_b(ctx):
    r = getattr(ctx, "round", None)
    print(f"[Hook B] before_round round={r}")


@hooks.after_client_train
def log_after_client_train_b(ctx):
    cid = getattr(ctx, "client_id", None)
    r = getattr(ctx, "round", None)
    print(f"[Hook B] after_client_train round={r} client_id={cid}")
    _write_signal(ctx, "b")


@hooks.after_round
def log_after_round_b(ctx):
    r = getattr(ctx, "round", None)
    print(f"[Hook B] after_round round={r}")


@hooks.after_simulation
def log_after_simulation_b(ctx):
    print("[Hook B] after_simulation")
