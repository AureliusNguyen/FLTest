# Bug Report: PyTorchModel Defaults to CUDA instead of CPU

**Description:**
The `PyTorchModel` class in `pfl.model.pytorch` automatically uploads the model to CUDA if CUDA is available on the system, regardless of whether the user intends to use CPU for FL training. This behavior is unexpected and deviates from standard PyTorch practices, where models remain on CPU unless explicitly moved to CUDA.

**Steps to Reproduce:**
1. Ensure CUDA is available on your system.
2. Initialize a `PyTorchModel` instance :
    ```python
    from torchvision.models import resnet18
    from pfl.model.pytorch import PyTorchModel
    pytorch_model = resnet18(pretrained=False)
    pytorch_model.loss = None
    pytorch_model.metrics = None    
    print(f"Original model device: {next(pytorch_model.parameters()).device}")
    # Initialize the PFL PyTorch model 
    pfl_pt_model = PyTorchModel(pytorch_model, local_optimizer_create=None, central_optimizer=None)
    print(f"PFL model device: {next(pfl_pt_model.pytorch_model.parameters()).device}")
    ```
    Output of the above code:
    ```output
    Original model device: cpu
    PFL model device: cuda:0    
    ```

3. Observe that `pytorch_model` is moved to CUDA despite intending to use CPU.

**Expected Behavior:**
Similar to centralized training in PyTorch, the model should remain on CPU by default and only move to CUDA when the developer explicitly specifies.

**Proposed Solution:**
Introduce a `device` parameter to the `PyTorchModel` class, allowing users to specify the desired device (`'cpu'` or `'cuda'`). Additionally, provide a `.to(device)` method to facilitate moving the model as needed.

**Affected Code:**
In `pfl/model/pytorch.py`:
```python
self._model = model.to(pytorch_ops.get_default_device())
```

In `pfl/internal/ops/pytorch_ops.py`:

```python    
def get_default_device():
    manual_device = os.environ.get('PFL_PYTORCH_DEVICE', None)
    if manual_device:
        default_device = torch.device(manual_device)
    elif is_pytest_running():
        default_device = torch.device('cpu')
    elif torch.cuda.is_available():
        default_device = torch.device('cuda')
    elif (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        default_device = torch.device('mps')
    else:
        default_device = torch.device('cpu')
    return default_device
```

**Suggested Fix:** Modify PyTorchModel to accept a device parameter and adjust the get_default_device function to respect user preferences more accurately.

**Additional Information:**
- pfl version: 0.2.0 
- PyTorch version: 2.0.1


**I am willing to work on a *Pull Request* to implement this fix if the maintainers agree.**