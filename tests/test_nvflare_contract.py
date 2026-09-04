"""The contract the NVFlare backend depends on, checked without running NVFlare.

NVFlare rebuilds a model in the server process from its class path and recovers the
constructor arguments by reading attributes of the same name off the instance. A model that
does not expose them silently falls back to its defaults, which sends every client a
1-channel, 10-class model regardless of the dataset.
"""

import pytest

from fltest.data.models import MODEL_REGISTRY, get_model


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_built_in_models_expose_their_constructor_arguments(name):
    model = get_model(name, "data/models_cache", channels=3, num_classes=100)
    assert model.channels == 3, f"{name} must expose `channels` for NVFlare to recover it"
    assert model.num_classes == 100, f"{name} must expose `num_classes` for NVFlare"


def test_nvflare_refuses_a_model_it_cannot_serialise():
    """torchvision architectures take a class argument, which NVFlare cannot JSON-encode."""
    pytest.importorskip("nvflare", reason="NVFlare is an optional extra")
    from fltest.core.config import RunSpec
    from fltest.frameworks.nvflare.adapter import NVFlareAdapter

    spec = RunSpec(run_id="r", run_name="r", framework="nvflare", model_name="ResNet18")
    with pytest.raises(ValueError, match="cannot run 'ResNet18'"):
        NVFlareAdapter().run_simulation(spec, {}, None)
