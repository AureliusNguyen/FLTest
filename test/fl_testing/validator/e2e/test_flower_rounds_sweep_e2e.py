import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from fl_testing.validator.cli import cli


CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "resources" / "rounds_sweep.yaml"
)


def _run_fl_validate_rounds_sweep():
    """Run `fl-validate run rounds_sweep.yaml` once and return parsed JSON."""
    runner = CliRunner()

    assert CONFIG_PATH.is_file(), f"Config file not found: {CONFIG_PATH}"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(
            cli,
            [
                "run",
                str(CONFIG_PATH),
                "--no-cache",
                "--output",
                tmpdir,
            ],
        )

        assert (
            result.exit_code == 0
        ), f"CLI exited with {result.exit_code}: {result.output}"

        output_files = list(Path(tmpdir).glob("*.json"))
        assert (
            len(output_files) == 1
        ), f"Expected one JSON report, found {len(output_files)}"

        with output_files[0].open() as f:
            return json.load(f)


def _format_sigfig(value: float, sigfigs: int = 4) -> str:
    """Format a float to a string with the given number of significant figures."""
    return f"{value:.{sigfigs}g}"


def test_flower_rounds_sweep_matches_golden_results():
    """For the fixed config, results must match the golden JSON (4 significant figures)."""
    result = _run_fl_validate_rounds_sweep()

    # Sanity check: validation should pass.
    assert result["summary"]["overall_status"] == "PASSED"

    experiments_by_id = {e["id"]: e for e in result["experiments"]}

    expected_results = {
        "exp_001": {
            "Final Round Loss": 0.004492433043196797,
            "Final Round Accuracy": 0.1015625,
            "PyTorch Local GM Sum": -2.5355266593396664,
            "GM Framework Sum": -2.5355268716812134,
        },
        "exp_002": {
            "Final Round Loss": 0.00447628740221262,
            "Final Round Accuracy": 0.1015625,
            "PyTorch Local GM Sum": 12.673761874437332,
            "GM Framework Sum": 12.673762172460556,
        },
        "exp_003": {
            "Final Round Loss": 0.004412354901432991,
            "Final Round Accuracy": 0.3828125,
            "PyTorch Local GM Sum": 56.691800355911255,
            "GM Framework Sum": 56.69179804623127,
        },
    }

    for exp_id, expected_metrics in expected_results.items():
        assert exp_id in experiments_by_id, f"Missing experiment {exp_id} in results"
        actual_results = experiments_by_id[exp_id]["results"]

        for metric_name, expected_value in expected_metrics.items():
            assert (
                metric_name in actual_results
            ), f"Missing metric '{metric_name}' in experiment {exp_id}"

            actual_value = actual_results[metric_name]
            assert isinstance(
                actual_value, (int, float)
            ), f"Metric '{metric_name}' in {exp_id} is not numeric"

            expected_str = _format_sigfig(expected_value, 4)
            actual_str = _format_sigfig(actual_value, 4)

            assert (
                actual_str == expected_str
            ), f"{exp_id} / {metric_name}: expected {expected_str}, got {actual_str}"

