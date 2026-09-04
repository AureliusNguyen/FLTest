"""Dataset specs and the model zoo, covering the paths that need no network."""

import pytest
import torch

from fltest.data.datasets import (
    DATASET_CONFIG,
    dataset_meta,
    get_federated_dataset,
    list_partitioners,
    resolve_dataset,
)
from fltest.data.models import TORCHVISION_MODELS, get_model, list_models


def test_dataset_specs_describe_their_label_columns():
    """A dataset whose labels are not called 'label' must say so."""
    assert resolve_dataset("cifar100").label_column == "fine_label"
    assert resolve_dataset("femnist").label_column == "character"
    assert resolve_dataset("mnist").label_column == "label"


def test_dataset_meta_covers_the_new_datasets():
    assert dataset_meta("cifar100") == (3, 100)
    assert dataset_meta("femnist") == (1, 62)


def test_femnist_is_the_only_naturally_partitioned_dataset():
    """FEMNIST carries a writer id, which is what `data_distribution: natural` needs."""
    natural = {n for n, s in DATASET_CONFIG.items() if s.natural_partition_by}
    assert natural == {"femnist"}
    assert resolve_dataset("femnist").natural_partition_by == "writer_id"
    assert "natural" in list_partitioners()


def test_femnist_holds_out_its_own_test_set():
    """FEMNIST ships only a train split, so a held-out set has to be carved from it."""
    assert resolve_dataset("femnist").test_split == ""
    assert resolve_dataset("cifar100").test_split == "test"


def test_natural_partitioning_is_refused_where_it_does_not_apply():
    with pytest.raises(ValueError, match="no natural client column"):
        get_federated_dataset("mnist", 2, "natural")


@pytest.mark.parametrize("name", ["LeNet", "MLP", "ResNet18", "MobileNetV3"])
def test_models_accept_the_dataset_shape(name):
    """Every model adapts to the channel count and class count it is given."""
    model = get_model(name, "data/models_cache", channels=1, num_classes=62)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 62)


def test_torchvision_architectures_are_listed():
    assert set(TORCHVISION_MODELS) <= set(list_models())
    assert {"LeNet", "ConvNet", "MLP"} <= set(list_models())


def test_initial_weight_cache_separates_class_counts():
    """The cache key carries num_classes, or a 10-class head would be loaded into a 100."""
    ten = get_model("LeNet", "data/models_cache", channels=3, num_classes=10)
    hundred = get_model("LeNet", "data/models_cache", channels=3, num_classes=100)
    assert ten.fc3.out_features == 10
    assert hundred.fc3.out_features == 100


def test_unknown_model_name_explains_the_options():
    with pytest.raises(ValueError, match="hf:"):
        get_model("NoSuchNet", "data/models_cache", channels=1, num_classes=10)
