# Worked example: exhaustively testing attacks & defenses

This page evaluates one federated setup against a **matrix of attacks and defenses from a
single config**, then cross-checks it with differential, metamorphic, and pitfall testing.
Every number below is real output from `examples/configs/exhaustive_eval.yaml`, which runs
CIFAR-10 with LeNet on 6 clients for 5 rounds on the reference backend, on CPU.

## 1. One config, a whole scenario matrix

Each entry in `runs:` is an independent scenario with its **own** `attacks`/`defenses`
(per-run overrides). The shared settings are held fixed so scenarios stay comparable.

```yaml
name: exhaustive_eval
dataset: cifar10
model_name: LeNet
num_clients: 6
num_rounds: 5
client_epochs: 2
client_lr: 0.001
optimizer: Adam
metrics: [accuracy, loss, per_client]

runs:
  - {framework: reference, name: baseline, attacks: []}
  - {framework: reference, name: label_flip,
     attacks: [{name: label_flip, params: {shift: 1}, target_clients: [0, 1]}]}
  - {framework: reference, name: sign_flip,
     attacks: [{name: sign_flip, params: {scale: 3.0}, target_clients: [0, 1]}]}
  - {framework: reference, name: gaussian,
     attacks: [{name: gaussian, params: {sigma: 0.5}, target_clients: [0, 1]}]}
  - {framework: reference, name: backdoor,
     attacks: [{name: backdoor, params: {target_label: 0, infection_rate: 0.8, patch_size: 5}, target_clients: [0, 1]}]}
  - {framework: reference, name: backdoor+median,    ...same backdoor..., defenses: [{name: median}]}
  - {framework: reference, name: backdoor+trimmed,   ...same backdoor..., defenses: [{name: trimmed_mean, params: {trim: 2}}]}
  - {framework: reference, name: backdoor+krum,      ...same backdoor..., defenses: [{name: krum, params: {num_byzantine: 2}}]}
  - {framework: reference, name: backdoor+normclip,  ...same backdoor..., defenses: [{name: norm_clip, params: {clip_norm: 0.5}}]}
  - {framework: reference, name: membership_inference,
     attacks: [{name: membership_inference, params: {target_client: 0}}]}
  - {framework: reference, name: mi+gradient_noise,  ...same attack..., defenses: [{name: gradient_noise, params: {sigma: 0.05}}]}
```

Run it:

```bash
fltest run examples/configs/exhaustive_eval.yaml
```

## 2. Results

| Scenario | accuracy | attack success | per-client (mean / min) | membership AUC |
|----------|:--------:|:--------------:|:-----------------------:|:--------------:|
| baseline | 0.3750 | – | 0.3875 / 0.3537 | – |
| label_flip | 0.3281 | – | 0.2978 / 0.2722 | – |
| sign_flip | **0.0986** | – | 0.0937 / 0.0803 | – |
| gaussian | **0.0830** | – | 0.0891 / 0.0755 | – |
| backdoor | 0.2891 | **0.4421** | 0.2554 / 0.2290 | – |
| backdoor + median | 0.3320 | **0.2369** | 0.3002 / 0.2794 | – |
| backdoor + trimmed_mean | 0.3320 | **0.2369** | 0.3002 / 0.2794 | – |
| backdoor + krum | 0.2520 | **0.0404** | 0.2194 / 0.1954 | – |
| backdoor + norm_clip | 0.1094 | **0.9924** | 0.1023 / 0.0887 | – |
| membership_inference | 0.3828 | – | 0.3891 / 0.3525 | 0.4831 |
| membership_inference + gradient_noise | 0.2969 | – | 0.2960 / 0.2806 | 0.4978 |

The whole matrix takes about nine minutes on a laptop CPU.

## 3. What this tells you

**Accuracy alone hides an attack.** The `backdoor` run holds 0.2891 accuracy against a
baseline of 0.3750, which reads as ordinary training noise. Its attack success rate is
0.4421. Without that metric the model looks merely mediocre rather than compromised.

**A naive attack is not a harmless one.** Two adversaries out of six collapse the model
with `sign_flip`, to 0.0986, and with `gaussian`, to 0.0830. CIFAR-10 has ten balanced
classes, so both sit at chance. Undefended FedAvg is fragile against attacks the survey
calls weak.

**Robust aggregation helps on both axes.** `median` and `trimmed_mean` cut the backdoor
from 0.4421 to 0.2369 and *raise* accuracy from 0.2891 to 0.3320, because discarding the
poisoned updates also discards their damage. `krum` is the strongest against the trigger,
at 0.0404, and costs accuracy, at 0.2520, since it keeps a single client's update per
round.

**The wrong defense is worse than no defense.** `norm_clip` at 0.5 reports an attack
success rate of 0.9924, the worst in the matrix. Read it beside the accuracy of 0.1094.
Clipping that hard stopped the model learning, it collapsed to predicting the attacker's
target label for nearly everything, and a constant predictor scores near 1.0 on a triggered
test set. The lesson is that attack success rate is only meaningful next to accuracy, and
FLTest surfaces the pair rather than letting either stand alone.

**Leakage needs something to leak.** Membership inference reaches AUC 0.4831 here, which is
chance, and the loss gap is -0.0108. This model underfits at 0.3828 accuracy, so it has
memorized nothing to expose. The same attack reaches 0.67 in
`examples/configs/membership_inference.yaml`, where local epochs are high and shards are
small. A privacy result of no leakage is a statement about the training regime, not a
property of the defense.

**Personalized evaluation exposes who pays.** `per_client_acc_min` tracks the worst-served
client. It falls to 0.0887 under `norm_clip` while the mean reads 0.1023, and a single
global number would hide that gap.

Running the config writes every value above to `reports/exhaustive_eval_run.json`, with the
per-round `history` alongside the `final` metrics. Reports are generated rather than checked
in, so `reports/` is empty in a fresh clone until you run something.

## 4. Cross-framework differential check

Do the backends agree on a clean run? `examples/configs/differential_cifar10_3way.yaml`
runs the same CIFAR-10 setup on all three:

```bash
fltest diff examples/configs/differential_cifar10_3way.yaml
```

```
reference  acc=0.3867
flower     acc=0.3809
nvflare    acc=0.3789
[✓ PASS] differential: cifar10/iid LeNet c3 r5 :: ['reference', 'flwr', 'nvflare']
        max|Δ|=0.0078 (tol=0.08)
```

All three land within 0.008, and a divergence past tolerance would point to a framework
bug. This config uses 3-channel data deliberately. The MNIST example cannot catch a backend
that loses the real channel and class counts, because 1 channel and 10 classes are the
defaults of every built-in model.

## 5. Metamorphic check

`examples/configs/metamorphic.yaml` sweeps one parameter at a time and asserts a relation
that ought to hold:

```bash
fltest metamorphic examples/configs/metamorphic.yaml
```

```
[✓ PASS] clients_scale (num_clients)    non-decreasing over [4.0, 8.0]
[✓ PASS] rounds_monotonic (num_rounds)  non-decreasing over [1.0, 3.0]
```

Doubling the clients on IID data did not drop accuracy, and more rounds did not decrease
it.

## 6. Pitfall check on this very config

```bash
fltest pitfalls examples/configs/exhaustive_eval.yaml
```

```
[LOW   ] Class-balanced datasets only  (P2_dataset)
[HIGH  ] IID-only data distribution  (P3_iid_only)
```

The checker reads the per-run attacks, so it raises neither the threat-model pitfall nor
the privacy-leakage one. This matrix carries strong attacks and a privacy attack already.
It still flags two things. CIFAR-10 is class balanced, which `femnist` fixes since it is
partitioned by writer, and every run here is IID. Merge the printed counter-experiments and
the [fuzzer](fuzzing.md) expands the matrix accordingly.

## 7. Reproduce everything

```bash
conda activate fltest
pytest tests/ -q                                          # unit + smoke suite (green)
fltest run         examples/configs/exhaustive_eval.yaml
fltest diff        examples/configs/differential_cifar10_3way.yaml
fltest metamorphic examples/configs/metamorphic.yaml
fltest pitfalls    examples/configs/exhaustive_eval.yaml
```

To add a scenario, append a run with your own attack or defense. To add a *new* attack or
defense, see **[Port your attacks & defenses](extending.md)**.
