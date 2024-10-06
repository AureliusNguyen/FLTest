from diskcache import Index
import hydra
import os


from fl_testing.frameworks.utils import seed_every_thing
from fl_testing.frameworks.nvidia_flare.server import run_flare_simulation
from fl_testing.frameworks.flower.server import run_flower_simulation
from fl_testing.frameworks.apple_pfl.server import run_pfl_simulation

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
    prev_result, current_result = run_fl_simulation(cfg)
    if prev_result is not None:
        for k, v in current_result.items():
            print(f'{k} -> prev {prev_result[k]}, current {v}')

        for k, v in current_result.items():
            assert v == prev_result[k], f"For {k}, Prev : {prev_result[k]}, Current: {v}"
            print(f'{k} Passed')


if __name__ == "__main__":
    main()
