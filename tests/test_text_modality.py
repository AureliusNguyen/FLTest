"""Text datasets and models, and the guards that keep image-only plugins honest."""

import pytest
import torch

from fltest.attacks.data_poison_backdoor import _BackdoorLoader
from fltest.core.config import TestConfig
from fltest.core.orchestrator import expand_run_specs
from fltest.data.datasets import resolve_dataset
from fltest.data.models import forward_batch, get_model


def _text_spec(**overrides):
    cfg = TestConfig(
        name="t", dataset="ag_news", model_name="hf:google/bert_uncased_L-2_H-128_A-2",
        runs=[{"framework": "reference"}], **overrides,
    )
    return expand_run_specs(cfg)[0]


def test_text_dataset_reports_no_channels():
    spec = resolve_dataset("ag_news")
    assert spec.modality == "text"
    assert (spec.channels, spec.num_classes) == (0, 4)


def test_tokenizer_defaults_to_the_hugging_face_model():
    assert _text_spec().tokenizer_id() == "google/bert_uncased_L-2_H-128_A-2"


def test_explicit_tokenizer_wins():
    """Needed when a model repo ships no fast tokenizer but shares another's vocabulary."""
    assert _text_spec(tokenizer="google-bert/bert-base-uncased").tokenizer_id() == (
        "google-bert/bert-base-uncased"
    )


def test_image_model_on_text_data_is_refused_with_a_useful_message():
    with pytest.raises(ValueError, match="image model, but this dataset is text"):
        get_model("LeNet", "data/models_cache", channels=0, num_classes=4)


def test_forward_batch_routes_on_the_batch_contents():
    """One place knows the difference between an image batch and a text batch."""
    seen = {}

    class Recorder(torch.nn.Module):
        def forward(self, *args):
            seen["arity"] = len(args)
            return torch.zeros(1, 2)

    forward_batch(Recorder(), {"img": torch.zeros(1, 1, 32, 32)}, "cpu")
    assert seen["arity"] == 1
    forward_batch(
        Recorder(),
        {"input_ids": torch.zeros(1, 4, dtype=torch.long),
         "attention_mask": torch.ones(1, 4, dtype=torch.long)},
        "cpu",
    )
    assert seen["arity"] == 2


def test_backdoor_refuses_text_rather_than_raising_a_key_error():
    """Stamping a trigger patch needs pixels, so say that instead of failing obscurely."""
    text_batch = [{"input_ids": torch.zeros(2, 4), "label": torch.tensor([0, 1])}]
    poisoned = _BackdoorLoader(text_batch, frac=0.5, size=4, value=1.0, target=0)
    with pytest.raises(ValueError, match="needs an image dataset"):
        list(poisoned)
