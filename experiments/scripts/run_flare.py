from diskcache import Index
import hydra
from fl_testing.frameworks.nvidia_flare.server import run_simulation
from fl_testing.frameworks.utils  import seed_every_thing





@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    seed_every_thing(cfg.seed)
    key = f"{cfg.exp_name}-{cfg.framework}"

    current_result =  run_simulation(cfg)
    cache = Index(cfg.framework_cache_path)
    prev_result = cache.get(key)

    cache[key] = current_result 


    if prev_result is not None:
        for k in current_result.keys():
            print(f'{k} -> prev {prev_result[k]}, current {current_result[k]}')
        
        for k in current_result.keys():
            assert current_result[k] == prev_result[k], f"For {k}, Prev : {prev_result[k]}, Current: {current_result[k]}"
            print(f'{k} Passed' )

    
if __name__ == "__main__":
    main()
