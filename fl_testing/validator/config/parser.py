"""YAML configuration parser for validation configs."""

import yaml
from pathlib import Path
from typing import Union

from fl_testing.validator.config.schema import ValidationConfig


def load_validation_config(config_path: Union[str, Path]) -> ValidationConfig:
    """Load and validate a validation configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ValidationConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Handle nested 'validation' key if present
    if "validation" in raw_config:
        raw_config = raw_config["validation"]

    # Validate and parse with Pydantic
    config = ValidationConfig(**raw_config)

    return config
