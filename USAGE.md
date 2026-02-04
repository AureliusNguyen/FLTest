# FL Framework Testing - Usage Guide

## Quick Start

```bash
poetry run python fl_testing/scripts/main.py                    # Default: Flower, MNIST, IID
poetry run python fl_testing/scripts/main.py framework=pfl      # Switch framework
poetry run python fl_testing/scripts/main.py data_distribution=dirichlet  # Non-IID
```

## Configuration

Override any parameter via CLI. Config file: `fl_testing/config/config.yaml`

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

## Data Distribution: IID vs Non-IID

| Strategy | Description |
|----------|-------------|
| `iid` | Uniform random split - each client sees all classes |
| `dirichlet` | Label skew (alpha=0.5). Lower alpha = more heterogeneous |
| `pathological` | Each client gets only 2 classes (extreme) |

**Dirichlet alpha** (in `pytorch_fl_dataset.py`): `0.1` = extreme non-IID, `100` = nearly IID

---

## Adding New Models

In `fl_testing/frameworks/models.py`:

```python
# 1. Define model (must accept channels, num_classes)
class MyModel(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        ...

# 2. Register it
model_name2class = {
    'LeNet': LeNet,
    'MyModel': MyModel,  # Add here
}
```

```bash
# 3. Use it
poetry run python fl_testing/scripts/main.py model_name=MyModel
```

---

## Adding New Datasets

1. Add to `constants.yaml`:
   ```yaml
   dataset_channels:
       fashion_mnist: 1
   ```

2. Add to `DATASET_CONFIG` in `pytorch_fl_dataset.py`:
   ```python
   DATASET_CONFIG = {
       'fashion_mnist': ('grayscale', 'image'),
   }
   ```

---

## Adding New Partitioners

Add to `PARTITIONERS` in `pytorch_fl_dataset.py`:

```python
PARTITIONERS = {
    'my_partitioner': lambda n: MyPartitioner(num_partitions=n),
}
```

---

## Framework Notes

| Framework | GPU | Notes |
|-----------|-----|-------|
| Flower | Yes | Fastest |
| FLARE | Yes | Slower startup |
| PFL | CPU only | Central optimizer must use `lr=1.0` for FedAvg |
