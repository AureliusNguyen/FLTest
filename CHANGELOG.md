# Changelog

Every release of FLTest is recorded here. Versions follow [semantic
versioning](https://semver.org). The patch number changes for a fix and the minor number
for new capability that leaves existing configs working. The major number changes when the
configuration schema or the plugin API breaks.

## 0.3.0

**Text.** Added `ag_news` and the plumbing federated text needs. A dataset now declares its
modality, text splits are tokenized rather than transformed, and a model named `hf:<id>` on
a text dataset is built as a sequence classifier. One function, `forward_batch`, is the only
place that knows an image batch carries `img` while a text batch carries `input_ids` and
`attention_mask`, so the training and evaluation loops serve both.

A `tokenizer` knob was added. It defaults to the Hugging Face model's own tokenizer and is
set explicitly when a repository ships no fast tokenizer but shares another model's
vocabulary. Token ids are part of the dataset cache key.

`examples/configs/text_domains.yaml` gives each client a single news topic, which is the
extreme non-IID setting for language data, and runs an IID baseline beside it. The IID run
reaches 0.4629 accuracy with a worst client at 0.4617, while one topic per client falls to
0.2402 against a 1/4 chance baseline with a worst client at 0.0000.

**Guards.** `backdoor` stamps a trigger onto pixels and `dlg` reconstructs pixels from
gradients, so neither applies to text. Both now refuse a text run with a message naming the
attacks that do apply, rather than failing on a missing column.

## 0.2.0

**Datasets.** Added `cifar100` and `femnist`. FEMNIST is the answer to Pitfall-2, because it
labels every character by the writer who produced it. The new `natural` partitioner gives
each client one real writer instead of a synthetic shard. A dataset name FLTest does
not recognise is now treated as a Hugging Face id and described from its metadata, so any
Hub image-classification dataset works without a code change. A dataset that ships no test
split, as FEMNIST does, gets 10,000 examples held out under a fixed seed before
partitioning. Slicing the test set out of the client shards instead would evaluate the
global model on data its own clients trained on.

**Models.** Added the torchvision architectures `ResNet18`, `ResNet34`, `ResNet50`,
`VGG11`, `MobileNetV3`, and `EfficientNetB0`, each adapted to the dataset's channel count,
with the ResNet stem replaced by the 3x3 CIFAR variant. A name written as `hf:<id>` is
fetched from the Hugging Face Hub through timm or transformers, which installs with
`pip install -e ".[hf]"`.

**Pitfall checker.** CIFAR-100 joins the class-balanced set and FEMNIST is deliberately
outside it, so `dataset: femnist` now clears `P2_dataset` rather than downgrading it. The
recommender suggests FEMNIST, which previously proposed a counter-experiment that could not
clear the pitfall it was answering.

**Fixed.** The deterministic initial-weight cache was keyed on model name and channel count
alone. Adding CIFAR-100 exposed it: a 10-class head would have been loaded into a 100-class
model. The key now carries the class count.

## 0.1.0

First versioned release. It covers Tasks 1 to 3 of the project plan, which are automated
test orchestration, FL input configuration, and evaluation metrics with reporting.

**Orchestration.** A single YAML config expands into a grid of runs through the config
fuzzer, and every run executes behind one `run_simulation()` adapter.

**Backends.** A dependency-light reference oracle, Flower, and NVFlare as an optional
extra. Backends are declared lazily, so `fltest list` and `fltest pitfalls` return without
importing torch.

**Attacks.** `label_flip`, `sign_flip`, `gaussian`, `backdoor` with attack success rate,
and `dlg` gradient inversion with reconstruction MSE, PSNR, and label recovery.

**Defenses.** `gradient_noise`, `norm_clip`, and robust aggregation by `krum`,
`trimmed_mean`, and `median`.

**Testing.** Cross-framework differential parity, a determinism mode, four metamorphic
relations, and a pitfall checker that emits counter-experiments.

**Reporting.** An aligned run-matrix table that states shared parameters once and gives a
column to every parameter that differs, alongside a JSON report for each command.
