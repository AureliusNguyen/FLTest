# CLI reference

```bash
fltest <command> [args]
```

| Command | Purpose |
|---------|---------|
| `fltest list` | List registered frameworks / attacks / defenses / metrics |
| `fltest run <conf>` | Run the orchestrated experiment matrix; print a table; write a JSON report |
| `fltest diff <conf>` | Differential test — cross-framework parity (default) or determinism |
| `fltest metamorphic <conf>` | Check metamorphic relations under `testing.metamorphic` |
| `fltest pitfalls <conf>` | Pitfall check + counter-experiment recommendations |

## Options

`run`, `diff`, `metamorphic` accept:

| Option | Default | Meaning |
|--------|---------|---------|
| `-o, --output DIR` | `reports` | directory for the JSON report |
| `-v, --verbose` | off | show framework/Ray/HF logs (otherwise INFO is suppressed) |

## Exit codes

- `fltest diff` / `fltest metamorphic` exit **0** if all checks pass, **1** if any fail —
  suitable for CI gating.
- `fltest run` / `fltest pitfalls` exit 0 on completion.

## Console output

`fltest run` prints an aligned table. Settings shared by every run appear once above it,
and only settings that differ between runs become columns, so a fuzzed grid shows what
varies rather than repeating itself. The aggregation rule is one of those settings, naming
`fedavg` or the robust rule that replaced it.

Below the table come a per-round trace of the headline metric and a legend that expands
every shortened column, so `asr` and `mia-auc` explain themselves without a trip to the
source.

## Reports

Each command writes JSON to the output directory:

- `<name>_run.json` — every run's `final`, `history`, `extras`, status, duration.
- `<name>_differential.json` — mode, metric, tolerance, and PASS/FAIL outcomes.
- `<name>_metamorphic.json` — relation outcomes with the swept values and metric trace.

## Environment variables

| Var | Meaning |
|-----|---------|
| `FLTEST_HOOKS` | comma-separated hook files/modules to load (attacks/defenses/validators) |
| `HF_HUB_DISABLE_PROGRESS_BARS` | set to `1` to silence dataset download bars (the CLI sets this by default) |

## Programmatic API

```python
from fltest.core.config import load_config
from fltest.core.orchestrator import Orchestrator
from fltest.testing import DifferentialTester, MetamorphicTester

cfg = load_config("my_conf.yaml")
matrix = Orchestrator().run(cfg)                       # -> RunMatrix (results)
diff = DifferentialTester().cross_framework(matrix)    # -> DifferentialReport
```
