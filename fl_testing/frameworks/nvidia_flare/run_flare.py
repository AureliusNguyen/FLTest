from diskcache import Index
import hydra
from fl_testing.frameworks.nvidia_flare.server import run_simulation
from fl_testing.frameworks.utils  import seed_every_thing
import os
os.environ['PYTHONHASHSEED'] = '786'

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    seed_every_thing(cfg.seed)
    key = f"{cfg.exp_name}-{cfg.framework}"

    current_result = run_simulation(cfg)
    cache = Index(cfg.framework_cache_path)
    prev_result = cache.get(key)

    cache[key] = current_result

    if prev_result is not None:
        for k,v in current_result.items():
            print(f'{k} -> prev {prev_result[k]}, current {v}')

        for k,v in current_result.items():
            assert v == prev_result[k], f"For {k}, Prev : {prev_result[k]}, Current: {v}"
            print(f'{k} Passed')
    
    

    
if __name__ == "__main__":
    main()

