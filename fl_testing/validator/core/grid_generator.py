"""Parameter grid generation for FL experiments."""

from itertools import product
from typing import Dict, List, Any
from dataclasses import dataclass
import hashlib
import json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    id: str
    parameters: Dict[str, Any]
    cache_key: str


class GridGenerator:
    """Generates parameter combinations for experiments."""

    def __init__(self, base_config: Dict[str, Any], parameter_grid: Dict[str, Any]):
        """Initialize grid generator.

        Args:
            base_config: Base configuration dict with default values.
            parameter_grid: Parameter grid specification with mode and parameters.
        """
        self.base_config = base_config
        self.mode = parameter_grid.get("mode", "single_sweep")
        self.parameters = parameter_grid.get("parameters", {})
        self.sweep_parameter = parameter_grid.get("sweep_parameter")

    def generate(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations.

        Returns:
            List of ExperimentConfig objects.
        """
        if self.mode == "single_sweep":
            return self._generate_single_sweep()
        elif self.mode == "combinatorial":
            return self._generate_combinatorial()
        else:
            raise ValueError(f"Unknown grid mode: {self.mode}")

    def _generate_single_sweep(self) -> List[ExperimentConfig]:
        """Generate configs for single-variable sweep (ablation study)."""
        if not self.sweep_parameter:
            raise ValueError("sweep_parameter required for single_sweep mode")

        if self.sweep_parameter not in self.parameters:
            raise ValueError(
                f"Sweep parameter '{self.sweep_parameter}' not in parameters"
            )

        configs = []
        param_values = self._get_param_values(self.sweep_parameter)

        for i, value in enumerate(param_values):
            params = self.base_config.copy()
            params[self.sweep_parameter] = value

            config = ExperimentConfig(
                id=f"exp_{i + 1:03d}",
                parameters=params,
                cache_key=self._compute_cache_key(params),
            )
            configs.append(config)

        return configs

    def _generate_combinatorial(self) -> List[ExperimentConfig]:
        """Generate all combinations (full grid search)."""
        param_names = list(self.parameters.keys())
        param_values_list = [self._get_param_values(p) for p in param_names]

        configs = []
        for i, combination in enumerate(product(*param_values_list)):
            params = self.base_config.copy()
            for name, value in zip(param_names, combination):
                params[name] = value

            config = ExperimentConfig(
                id=f"exp_{i + 1:03d}",
                parameters=params,
                cache_key=self._compute_cache_key(params),
            )
            configs.append(config)

        return configs

    def _get_param_values(self, param_name: str) -> List[Any]:
        """Get values for a parameter from specification."""
        spec = self.parameters[param_name]

        # Handle ParameterSpec object (from Pydantic)
        if hasattr(spec, "get_values"):
            return spec.get_values()

        # Handle dict representation
        if isinstance(spec, dict):
            if "values" in spec:
                return spec["values"]
            elif "range" in spec:
                r = spec["range"]
                start = r["start"]
                end = r["end"]
                step = r.get("step", 1)
                values = []
                current = start
                while current <= end:
                    values.append(current)
                    current += step
                return values

        raise ValueError(f"Invalid parameter spec for {param_name}: {spec}")

    def _compute_cache_key(self, params: Dict[str, Any]) -> str:
        """Compute deterministic cache key for parameters."""
        # Sort keys for deterministic ordering
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(sorted_params.encode()).hexdigest()[:16]

    def get_total_experiments(self) -> int:
        """Get total number of experiments that will be generated."""
        if self.mode == "single_sweep":
            return len(self._get_param_values(self.sweep_parameter))
        else:
            total = 1
            for param_name in self.parameters:
                total *= len(self._get_param_values(param_name))
            return total

    def preview(self) -> List[Dict[str, Any]]:
        """Preview experiments without full generation.

        Returns:
            List of parameter dicts for each experiment.
        """
        configs = self.generate()
        return [
            {"id": c.id, "varying_params": self._get_varying_params(c.parameters)}
            for c in configs
        ]

    def _get_varying_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the varying parameters for preview."""
        if self.mode == "single_sweep":
            return {self.sweep_parameter: params.get(self.sweep_parameter)}
        else:
            return {k: params.get(k) for k in self.parameters.keys()}
