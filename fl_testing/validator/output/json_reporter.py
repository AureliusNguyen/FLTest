"""JSON report generator for validation results."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONReporter:
    """Generates JSON reports for validation results."""

    def __init__(self, output_dir: str = "validation_results"):
        """Initialize JSON reporter.

        Args:
            output_dir: Directory to save reports to.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results, config, filename: Optional[str] = None) -> Path:
        """Generate JSON report and return path.

        Args:
            results: GridResults object with experiment and rule results.
            config: ValidationConfig that was used.
            filename: Optional filename override.

        Returns:
            Path to the generated report file.
        """
        report = {
            "meta": self._build_meta(config, results),
            "summary": self._build_summary(results),
            "experiments": self._build_experiments(results.experiments),
            "rule_results": self._build_rule_results(results.rule_results),
            "parameter_grid": self._build_grid_info(config),
        }

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.name}_{config.framework}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _build_meta(self, config, results) -> dict:
        """Build metadata section."""
        return {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "validation_name": config.name,
            "description": config.description,
            "framework": config.framework,
            "duration_seconds": round(results.total_duration, 2),
            "git_commit": self._get_git_commit(),
            "python_version": self._get_python_version(),
        }

    def _build_summary(self, results) -> dict:
        """Build summary section."""
        experiments = results.experiments
        rules = results.rule_results

        successful = sum(1 for e in experiments if e.status == "success")
        failed = sum(1 for e in experiments if e.status == "failed")
        timeout = sum(1 for e in experiments if e.status == "timeout")
        cached = sum(1 for e in experiments if e.cached)

        passed_rules = sum(1 for r in rules if r.status.value == "PASSED")
        failed_rules = sum(1 for r in rules if r.status.value == "FAILED")
        skipped_rules = sum(1 for r in rules if r.status.value == "SKIPPED")

        if failed_rules == 0:
            overall = "PASSED"
        elif passed_rules == 0:
            overall = "FAILED"
        else:
            overall = "PARTIAL"

        return {
            "total_experiments": len(experiments),
            "successful_experiments": successful,
            "failed_experiments": failed,
            "timeout_experiments": timeout,
            "cached_experiments": cached,
            "total_rules": len(rules),
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "skipped_rules": skipped_rules,
            "overall_status": overall,
        }

    def _build_experiments(self, experiments) -> list:
        """Build experiments section."""
        return [
            {
                "id": e.id,
                "parameters": self._serialize_params(e.parameters),
                "status": e.status,
                "duration_seconds": round(e.duration_seconds, 2),
                "results": e.results,
                "error": e.error,
                "cached": e.cached,
            }
            for e in experiments
        ]

    def _serialize_params(self, params: dict) -> dict:
        """Serialize parameters, handling non-JSON-serializable types."""
        serialized = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                serialized[k] = v
            elif isinstance(v, list):
                serialized[k] = v
            else:
                serialized[k] = str(v)
        return serialized

    def _build_rule_results(self, rules) -> list:
        """Build rule results section."""
        return [
            {
                "rule_type": r.rule_type,
                "rule_config": r.rule_config,
                "status": r.status.value,
                "message": r.message,
                "details": r.details,
            }
            for r in rules
        ]

    def _build_grid_info(self, config) -> dict:
        """Build parameter grid info section."""
        from fl_testing.validator.core.grid_generator import GridGenerator

        base_config = config.get_framework_base_config()
        param_grid_dict = {
            "mode": config.parameter_grid.mode,
            "parameters": {
                k: {"values": v.get_values()}
                for k, v in config.parameter_grid.parameters.items()
            },
            "sweep_parameter": config.parameter_grid.sweep_parameter,
        }

        generator = GridGenerator(base_config, param_grid_dict)

        return {
            "mode": config.parameter_grid.mode,
            "sweep_parameter": config.parameter_grid.sweep_parameter,
            "parameters": {
                k: v.get_values() for k, v in config.parameter_grid.parameters.items()
            },
            "total_combinations": generator.get_total_experiments(),
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_python_version(self) -> str:
        """Get current Python version."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
