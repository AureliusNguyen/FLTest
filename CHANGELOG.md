# Changelog

Every release of FLTest is recorded here. Versions follow [semantic
versioning](https://semver.org). The patch number changes for a fix and the minor number
for new capability that leaves existing configs working. The major number changes when the
configuration schema or the plugin API breaks.

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
