import os
import pytest
import hydra
from diskcache import Index
from fl_testing.frameworks.utils import seed_every_thing
from fl_testing.frameworks.nvidia_flare.server import run_flare_simulation
from fl_testing.frameworks.flower.server import run_flower_simulation
from fl_testing.frameworks.apple_pfl.server import run_pfl_simulation

# Mapping frameworks to their simulation functions
FRAMEWORK_SIMULATIONS = {
    "flower": run_flower_simulation,
    "flare": run_flare_simulation,
    "pfl": run_pfl_simulation
}


@pytest.fixture(scope="session")
def default_config():
    """Load the default Hydra configuration."""
    with hydra.initialize(config_path="../config"):
        config = hydra.compose(config_name="config")
    return config


@pytest.fixture(scope="class")
def get_frameworks_results(default_config):
    num_rounds = 10
    results = {}
    for framework in FRAMEWORK_SIMULATIONS.keys():
        cfg = default_config.copy()
        cfg.framework = framework
        cfg.num_rounds = num_rounds
        results[framework] = run_simulation(cfg)

    return results


@pytest.fixture(scope="function", autouse=True)
def setup_environment():
    os.environ['PYTHONHASHSEED'] = '786'
    yield
    # Add any necessary teardown steps here


def check_results_equal(current_result, prev_result):
    assert current_result is not None, "Current result should not be None."
    assert prev_result is not None, "Previous result should not be None."

    for key, current_value in current_result.items():
        assert key in prev_result, f"Key '{key}' missing in previous results."

        diff = abs(prev_result[key] - current_value)

        # assert  , (
        #     f"Mismatch for '{key}': Previous={prev_result[key]}, Current={current_value}"
        # )
        assert diff < 1e-6, ( # 1e-6 is the tolerance
            f"Mismatch for '{key}': Previous={prev_result[key]}, Current={current_value}"
        )


def run_simulation(cfg):
    seed_every_thing(cfg.seed)
    simulation_func = FRAMEWORK_SIMULATIONS[cfg.framework]
    current_result = simulation_func(cfg)

    cache = Index(cfg.fw_cache_path)
    cache_key = f"{cfg.exp_name}-{cfg.framework}"
    prev_result = cache.get(cache_key)
    cache[cache_key] = current_result

    return {
        'current_result': current_result,
        'prev_result': prev_result,
        'cache_key': cache_key
    }


class TestFrameworkSimulations:
    def test_flower_base(self, get_frameworks_results):
        results = get_frameworks_results
        current_result = results["flower"]["current_result"]
        prev_result = results["flower"]["prev_result"]
        check_results_equal(current_result, prev_result)

    def test_flare_base(self, get_frameworks_results):
        results = get_frameworks_results
        current_result = results["flare"]["current_result"]
        prev_result = results["flare"]["prev_result"]
        check_results_equal(current_result, prev_result)

    def test_pfl_base(self, get_frameworks_results):
        results = get_frameworks_results
        current_result = results["pfl"]["current_result"]
        prev_result = results["pfl"]["prev_result"]
        check_results_equal(current_result, prev_result)
