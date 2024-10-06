import pytest
from diskcache import Index
from fl_testing.frameworks.utils import seed_every_thing
from fl_testing.frameworks.nvidia_flare.server import run_flare_simulation
from fl_testing.frameworks.flower.server import run_flower_simulation
from fl_testing.frameworks.apple_pfl.server import run_pfl_simulation
import os
import hydra

# Dictionary to map frameworks to their respective simulation functions
FRAMEWORK2SIMULATION = {
    "flower": run_flower_simulation,
    "flare": run_flare_simulation,
    "pfl": run_pfl_simulation
}

@pytest.fixture(params=["flower", "flare", "pfl"])
def get_cfg(request):
    # Assuming you have a configuration file that can be loaded directly
    # Adjust the path accordingly if different
    with hydra.initialize(config_path="../config"):
        config = hydra.compose(config_name="config")
        config.framework = request.param
    return config

def test_run_fl_simulation(get_cfg):
    cfg = get_cfg
    os.environ['PYTHONHASHSEED'] = '786'
    # Run the run_fl_simulation function with the actual config
    seed_every_thing(cfg.seed)
    current_result = FRAMEWORK2SIMULATION[cfg.framework](cfg)

    # Verify that results are cached
    cache = Index(cfg.framework_cache_path)
    key = f"{cfg.exp_name}-{cfg.framework}"
    prev_result = cache.get(key)
    cache[key] = current_result
    
    assert key in cache, f"Expected key '{key}' not found in cache."
    
    # Verify the correctness of the result by comparing current and previous
    assert current_result is not None, "Current result should not be None"
    
    if prev_result is not None:
        for k, v in current_result.items():
            assert v == prev_result[k], f"For {k}, Prev: {prev_result[k]}, Current: {v}"
            print(f'{k} Passed')

