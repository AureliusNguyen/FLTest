# Changelog

Every release of FLTest is recorded here. Versions follow [semantic
versioning](https://semver.org). The patch number changes for a fix and the minor number
for new capability that leaves existing configs working. The major number changes when the
configuration schema or the plugin API breaks.

## 0.4.4

**Fixed.** The built-in datasets used bare Hugging Face ids, so `mnist`, `fashion_mnist`,
and `cifar10` were passed to the Hub as-is. huggingface-hub 1.16 removed that form, and
a fresh environment installs a later version, so any of those datasets failed with
`HfUriError` on a machine without a warm cache. The ids are now namespaced as
`ylecun/mnist`, `zalando-datasets/fashion_mnist`, and `uoft-cs/cifar10`, which works on
old and new versions alike. The short names in a config are unchanged.

The failure only appeared on a cold cache, because a machine that had already downloaded
the dataset answered from disk and printed a note about the Hub lookup failing. A test now
asserts every built-in id is namespaced, since a run on a warm machine cannot catch this.

A Hub id given directly in a config is checked too. A bare name that is not a built-in now
explains that the id needs a namespace, instead of surfacing the Hub's own error.

## 0.4.3

**Worked example.** The exhaustive attack and defense matrix moves from MNIST with an MLP
to CIFAR-10 with LeNet, and gains two privacy scenarios. Ten balanced classes make a
collapsed model obvious, since chance sits at 0.10.

The new numbers carry findings the MNIST version could not show. `norm_clip` at 0.5 reports
the worst attack success rate in the matrix, 0.9924, while its accuracy falls to 0.1094.
Clipping that hard stopped the model learning and it collapsed to predicting the attacker's
target label, which is a constant predictor scoring near 1.0 on a triggered test set. That
makes the case that attack success rate means nothing read alone.

Membership inference returns 0.4831, which is chance, because this model underfits and has
memorized nothing to expose. The dedicated example reaches 0.67. A privacy result of no
leakage describes the training regime rather than the defense.

`differential_cifar10_3way.yaml` now trains to a comparable accuracy, so its parity check
compares three backends that have actually learned rather than three sitting at chance.

## 0.4.2

`docs/assets` holds only assets now. The brand kit's README, brand guide, mkdocs snippet,
and preview page were build-time material rather than anything the site or the package
uses, and none of them was referenced.

## 0.4.1

**Reporting.** The aggregation rule is now an explicit field. `RunSpec.aggregation()` names
`fedavg` or the robust rule that replaced it, it appears in each run's recorded parameters,
and the run matrix gives it a column or lists it among the shared settings. It was
previously only inferable from the defense list.

Every shortened column is explained in a legend printed under the table, so `asr`,
`pc-min`, and `mia-auc` no longer send a reader to the source. A per-round trace of the
headline metric sits alongside it, since the table alone showed where a run ended but not
how it got there.

## 0.4.0

**Membership inference.** Added the `membership_inference` attack, which asks whether a
record was in a client's training data. It is the canonical privacy attack the proposal
cites and the one Pitfall-1 says evaluations skip. An honest-but-curious server scores the
global model each round by per-sample loss, following Yeom et al., since a model assigns
lower loss to what it trained on. Members are the target client's data and non-members are
the held-out test set.

It records `membership_inference_auc`, where 0.5 is no leakage, and `membership_loss_gap`.
No shadow model is needed, and because it reads only losses it works on text as well as
images, which gradient inversion does not.

`examples/configs/membership_inference.yaml` runs one overfitted setup twice. Undefended it
reaches AUC 0.67 with a loss gap of 1.39. Clipping with Gaussian noise takes the AUC to
0.50, which is chance, and the gap to 0.01, for about six points of accuracy. That is the
privacy and utility trade-off of Pitfall-4, measured.

The pitfall checker counts it as a privacy attack, so `P5_subtle_leakage` now recommends it
ahead of gradient inversion, which is cheaper to run and applies to more datasets.

## 0.3.4

**Fixed.** The NVFlare backend only ever worked on 1-channel, 10-class data. It rebuilds the
model in its server process from the class path and recovers constructor arguments by
reading attributes of the same name off the instance. The built-in models did not expose
`channels` or `num_classes`, so NVFlare fell back to their defaults and sent every client a
model shaped for MNIST. CIFAR-10 failed on that path before this release, and so did
CIFAR-100 and FEMNIST. The models now expose those arguments, and
`examples/configs/differential_cifar10_3way.yaml` is a three-way parity check on 3-channel
data, which the MNIST example could not catch.

**Fixed.** NVFlare round snapshots are keyed by round number in a cache that was never
cleared between runs. A run that produced no snapshots of its own replayed the previous
run's and reported them as its results. The cache is now cleared alongside the workspace.

**Changed.** The NVFlare backend now refuses a torchvision or Hugging Face model with a
message naming the built-in models and the backends that do run it. It previously failed
inside NVFlare with a JSON encoding error.

## 0.3.3

Documentation uses the light scheme only, so the dark toggle and its maroon page background
are gone. Removed the brand section from the landing page.

## 0.3.2

**CI.** Added `.github/workflows/ci.yml`, which runs on every pull request and on pushes to
the default branch. One job installs from a clean checkout and then runs `fltest list`,
checking the catalog rather than the exit code alone. That is the job that would have
caught the packaging bug where `fltest/data` existed locally but was never committed. A
second job runs the test suite, and a third builds the documentation with `--strict`, so a
broken link or a missing asset fails the build.

**Docs.** The worked example claimed its report file was present. Reports are generated
rather than checked in, so it now says what running the config writes.

## 0.3.1

**Branding.** The documentation site now carries the FLTest identity. `brand.css` is loaded
as `extra_css`, and it binds the maroon, orange, and stone tokens to Material's variables
for both the light and dark schemes. The palette is declared as `custom` so those tokens
govern the colours, since naming a built-in Material palette would fight them.

The header uses the white and orange mark, which is the variant drawn to read on maroon,
and the favicon comes from the same set. The home page and the README show the full lockup
and swap it by colour scheme, through `#only-light` and `#only-dark` on the site and a
`<picture>` element on GitHub. The brand kit itself is linked from the home page.

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
