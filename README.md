# FL Testing Framework

### Requirements
Please ensure `poetry` is installed on your system.

### Setup
```
poetry install
```

### Example Run
```
export FLTEST_HOOKS=examples/hooks/atk_label_flip
poetry run python fltest/main.py num_rounds=20
```

## Configuration

Any parameter you see in `fl_testing/config/config.yaml` can be overridden via CLI. Examples: 

| Parameter | Default | Options |
|-----------|---------|---------|
| `framework` | flower | `flower`, `flare`, `pfl` |
| `dataset` | mnist | `mnist`, `cifar10` |
| `data_distribution` | iid | `iid`, `dirichlet`, `pathological` |
| `num_clients` | 10 | |
| `num_rounds` | 10 | |
| `client_epochs` | 1 | |
| `client_lr` | 0.001 | |
| `device` | cpu | `cpu`, `cuda` |

---

