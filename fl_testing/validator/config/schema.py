"""Pydantic models for validation configuration schema."""

from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator


class ParameterRange(BaseModel):
    """Range specification for generating parameter values."""

    start: float
    end: float
    step: float = 1.0

    def get_values(self) -> List[float]:
        """Generate list of values from range."""
        values = []
        current = self.start
        while current <= self.end:
            values.append(current)
            current += self.step
        return values


class ParameterSpec(BaseModel):
    """Specification for a parameter's values - either explicit list or range."""

    values: Optional[List[Any]] = None
    range: Optional[ParameterRange] = None

    @model_validator(mode="after")
    def validate_values_or_range(self):
        if self.values is None and self.range is None:
            raise ValueError("Either 'values' or 'range' must be specified")
        if self.values is not None and self.range is not None:
            raise ValueError("Cannot specify both 'values' and 'range'")
        return self

    def get_values(self) -> List[Any]:
        """Get list of values for this parameter."""
        if self.values is not None:
            return self.values
        if self.range is not None:
            return self.range.get_values()
        raise ValueError("No values or range specified")


class ParameterGrid(BaseModel):
    """Parameter grid configuration."""

    mode: Literal["single_sweep", "combinatorial"]
    parameters: Dict[str, ParameterSpec]
    sweep_parameter: Optional[str] = None

    @model_validator(mode="after")
    def validate_sweep_parameter(self):
        if self.mode == "single_sweep":
            if not self.sweep_parameter:
                raise ValueError("'sweep_parameter' is required for 'single_sweep' mode")
            if self.sweep_parameter not in self.parameters:
                raise ValueError(
                    f"sweep_parameter '{self.sweep_parameter}' not found in parameters"
                )
        return self


class MonotonicExpectation(BaseModel):
    """Expectation that metric changes monotonically with parameter."""

    type: Literal["monotonic"]
    parameter: str
    metric: str
    direction: Literal["increasing", "decreasing"]
    tolerance: float = 0.0


class BoundedExpectation(BaseModel):
    """Expectation that metric stays within bounds."""

    type: Literal["bounded"]
    metric: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be specified")
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("'min_value' cannot be greater than 'max_value'")
        return self


class RelativeThresholdExpectation(BaseModel):
    """Expectation that adjacent metric changes don't exceed threshold."""

    type: Literal["relative_threshold"]
    parameter: str
    metric: str
    max_change_percent: float

    @model_validator(mode="after")
    def validate_threshold(self):
        if self.max_change_percent <= 0:
            raise ValueError("'max_change_percent' must be positive")
        return self


# Union of all expectation types
Expectation = Union[MonotonicExpectation, BoundedExpectation, RelativeThresholdExpectation]


class ExecutionConfig(BaseModel):
    """Execution configuration for running experiments."""

    mode: Literal["sequential", "parallel"] = "sequential"
    max_workers: int = Field(default=4, ge=1)
    timeout_seconds: int = Field(default=600, ge=1)
    continue_on_failure: bool = True
    dry_run: bool = False
    use_cache: bool = True
    cache_path: str = "data/validator_cache"


class OutputConfig(BaseModel):
    """Output configuration for validation results."""

    format: Literal["json"] = "json"
    path: str = "validation_results/"
    filename_template: str = "{name}_{framework}_{timestamp}.json"
    include_raw_results: bool = True
    verbose: bool = True


class FrameworkConstraints(BaseModel):
    """Framework-specific configuration overrides."""

    pfl: Optional[Dict[str, Any]] = None
    flower: Optional[Dict[str, Any]] = None
    flare: Optional[Dict[str, Any]] = None


class BaseConfig(BaseModel):
    """Base configuration for FL experiments."""

    seed: int = 786
    device: str = "cpu"
    num_clients: int = 10
    num_rounds: int = 10
    client_lr: float = 0.001
    client_epochs: int = 1
    client_batch_size: int = 32
    server_batch_size: int = 512
    dataset: str = "mnist"
    model_name: str = "LeNet"
    deterministic: bool = True
    exp_name: str = "validation_experiment"
    DATASET_DIVISION_CLIENTS: int = 1000
    max_test_data_size: int = 2048
    total_cpus: int = 10
    total_gpus: int = 0
    model_cache_path: str = "data/models_cache"
    dataset_cache_path: str = "data/dataset_cache"
    framework_constraints: Optional[FrameworkConstraints] = None

    model_config = {"extra": "allow"}


class ValidationConfig(BaseModel):
    """Root validation configuration."""

    name: str
    description: Optional[str] = None
    version: str = "1.0"
    framework: Literal["flower", "flare", "pfl"]
    base_config: BaseConfig = Field(default_factory=BaseConfig)
    parameter_grid: ParameterGrid
    expectations: List[Expectation]
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def get_framework_base_config(self) -> Dict[str, Any]:
        """Get base config with framework-specific overrides applied."""
        config = self.base_config.model_dump()
        config["framework"] = self.framework

        # Apply framework constraints
        if self.base_config.framework_constraints:
            constraints = getattr(self.base_config.framework_constraints, self.framework, None)
            if constraints:
                config.update(constraints)

        # PFL always requires CPU
        if self.framework == "pfl":
            config["device"] = "cpu"

        # Set framework cache path
        config["fw_cache_path"] = f"data/caches/{self.framework}"

        return config
