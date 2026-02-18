import os
from datetime import datetime
from pathlib import Path

import hydra
from diskcache import Index

from fl_testing.frameworks.apple_pfl.server import run_pfl_simulation
from fl_testing.frameworks.flower.simulation import run_flower_simulation
from fl_testing.frameworks.nvidia_flare.server import run_flare_simulation
from fl_testing.frameworks.pytorch_fl_dataset import (
    get_cached_federated_dataset,
    visualize_data_split,
)
from fl_testing.frameworks.utils import seed_every_thing

os.environ['PYTHONHASHSEED'] = '786'

FRAMEWORK2SIMULATION = {
    "flower": run_flower_simulation,
    "flare": run_flare_simulation,
    "pfl": run_pfl_simulation
}


def run_fl_simulation(cfg):
    seed_every_thing(cfg.seed)
    key = f"{cfg.exp_name}-{cfg.framework}"
    current_result = FRAMEWORK2SIMULATION[cfg.framework](cfg)
    cache = Index(cfg.fw_cache_path)
    prev_result = cache.get(key)
    cache[key] = current_result
    return prev_result, current_result


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    seed_every_thing(cfg.seed)
    raw_dataset = get_cached_federated_dataset(
        cfg.dataset,
        cfg.DATASET_DIVISION_CLIENTS,
        cfg.dataset_cache_path,
        cfg.data_distribution,
    )
    tmp_dir = Path(__file__).resolve().parent.parent.parent / "tmp"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = tmp_dir / f"data_split_{cfg.dataset}_{cfg.data_distribution}_{timestamp}.png"
    visualize_data_split(
        raw_dataset,
        dataset_name=cfg.dataset,
        data_distribution=cfg.data_distribution,
        num_clients_show=cfg.num_clients,
        save_path=save_path,
    )
    input("Press Enter to start simulation...")
    prev_result, current_result = run_fl_simulation(cfg)
    if prev_result is not None:
        for k, v in current_result.items():
            print(f'{k} -> prev {prev_result[k]}, current {v}')
        print("Tests:")
        for k, v in current_result.items():
            diff = abs(v-prev_result[k])
            print(f'{k} -> diff {diff}')
            assert diff < 1e-4, f"Prev : {prev_result[k]}, Current: {v}"
            print(f'{k} Passed')


if __name__ == "__main__":
    main()
