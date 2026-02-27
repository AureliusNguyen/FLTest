"""Entry point for the hook-based FL simulation. Run with: poetry run python fltest/main.py"""

import os
from datetime import datetime
from pathlib import Path

import hydra
from diskcache import Index

from fl_testing.frameworks.pytorch_fl_dataset import (
    get_cached_federated_dataset,
    visualize_data_split,
)
from fl_testing.frameworks.utils import seed_every_thing

from fltest.core import HookRunner, hooks
from fltest.adapters.flower import run_flower_simulation

os.environ["PYTHONHASHSEED"] = "786"


def run_fl_simulation(cfg, hook_runner: HookRunner):
    seed_every_thing(cfg.seed)
    key = f"{cfg.exp_name}-{cfg.framework}"
    current_result = run_flower_simulation(cfg, hook_runner)
    cache = Index(cfg.fw_cache_path)
    prev_result = cache.get(key)
    cache[key] = current_result
    return prev_result, current_result


# Config path: from fltest/ we need ../fl_testing/config so Hydra finds the config at repo root
_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "fl_testing" / "config")


@hydra.main(
    config_path=_CONFIG_PATH,
    config_name="config",
    version_base=None,
)
def main(cfg):
    seed_every_thing(cfg.seed)
    raw_dataset = get_cached_federated_dataset(
        cfg.dataset,
        cfg.DATASET_DIVISION_CLIENTS,
        cfg.dataset_cache_path,
        cfg.data_distribution,
    )
    hooks.import_convention_hooks()
    hook_runner = HookRunner()
    hooks.apply_to(hook_runner)

    prev_result, current_result = run_fl_simulation(cfg, hook_runner)


if __name__ == "__main__":
    main()
