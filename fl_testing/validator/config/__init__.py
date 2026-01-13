"""Configuration parsing and validation for FL Parameter Grid Validator."""

from fl_testing.validator.config.parser import load_validation_config
from fl_testing.validator.config.schema import ValidationConfig

__all__ = ["load_validation_config", "ValidationConfig"]
