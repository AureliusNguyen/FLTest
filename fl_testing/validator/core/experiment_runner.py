"""Experiment execution engine with timeout, caching, and parallel support."""

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp

from diskcache import Index
from omegaconf import OmegaConf, DictConfig

from fl_testing.validator.core.grid_generator import GridGenerator, ExperimentConfig
from fl_testing.validator.rules.base import ExperimentResult, RuleResult
from fl_testing.validator.rules import create_rule


@dataclass
class GridResults:
    """Results from running the entire grid."""

    experiments: List[ExperimentResult]
    rule_results: List[RuleResult]
    total_duration: float
    all_passed: bool


class ConsoleReporter:
    """Simple console output handler."""

    def __init__(self, verbose: bool = True, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet

    def info(self, message: str):
        if not self.quiet:
            print(f"[INFO] {message}")

    def success(self, message: str):
        if not self.quiet:
            print(f"[OK] {message}")

    def error(self, message: str):
        print(f"[ERROR] {message}")

    def warning(self, message: str):
        if not self.quiet:
            print(f"[WARN] {message}")


def _run_experiment_worker(params: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for running a single experiment in a separate process.

    This function is designed to be run in a ProcessPoolExecutor.
    """
    import os

    os.environ["PYTHONHASHSEED"] = "786"

    try:
        # Build Hydra-compatible config
        base_cfg = OmegaConf.load("fl_testing/config/config.yaml")
        constants = OmegaConf.load("fl_testing/config/constants.yaml")
        cfg = OmegaConf.merge(constants, base_cfg)

        # Apply experiment parameters
        for key, value in params.items():
            if key in cfg or key in ["framework", "fw_cache_path", "exp_name"]:
                OmegaConf.update(cfg, key, value)

        # Resolve interpolations
        OmegaConf.resolve(cfg)

        # Import and run
        from fl_testing.scripts.main import run_fl_simulation

        _, current_result = run_fl_simulation(cfg)

        return {"success": True, "results": current_result}

    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        }


class ExperimentRunner:
    """Orchestrates experiment execution with timeout and caching."""

    def __init__(self, config, console: Optional[ConsoleReporter] = None):
        """Initialize experiment runner.

        Args:
            config: ValidationConfig object.
            console: Optional console reporter for output.
        """
        self.config = config
        self.console = console or ConsoleReporter()
        self.cache = None

        if config.execution.use_cache:
            cache_path = config.execution.cache_path
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            self.cache = Index(cache_path)

    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate experiment configurations from grid."""
        base_config = self.config.get_framework_base_config()

        # Convert parameter_grid to dict if needed
        param_grid = self.config.parameter_grid
        param_grid_dict = {
            "mode": param_grid.mode,
            "parameters": {
                k: {"values": v.get_values()} for k, v in param_grid.parameters.items()
            },
            "sweep_parameter": param_grid.sweep_parameter,
        }

        generator = GridGenerator(base_config=base_config, parameter_grid=param_grid_dict)
        return generator.generate()

    def run(self) -> GridResults:
        """Execute all experiments and validate rules.

        Returns:
            GridResults with experiment results and rule validations.
        """
        start_time = time.time()

        experiments = self.generate_experiments()
        self.console.info(f"Generated {len(experiments)} experiments for {self.config.framework}")

        # Execute experiments
        if self.config.execution.mode == "parallel":
            results = self._run_parallel(experiments)
        else:
            results = self._run_sequential(experiments)

        # Count successes
        success_count = sum(1 for r in results if r.status == "success")
        self.console.info(f"Completed {success_count}/{len(results)} experiments successfully")

        # Validate rules
        rule_results = self._validate_rules(results)

        total_duration = time.time() - start_time

        # Determine overall pass/fail
        all_passed = all(
            r.status.value == "PASSED" or r.status.value == "SKIPPED" for r in rule_results
        )

        return GridResults(
            experiments=results,
            rule_results=rule_results,
            total_duration=total_duration,
            all_passed=all_passed,
        )

    def _run_sequential(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        for i, exp in enumerate(experiments):
            self.console.info(
                f"Running experiment {exp.id} ({i + 1}/{len(experiments)})..."
            )
            result = self._run_single_experiment(exp)
            results.append(result)

            status_str = "cached" if result.cached else result.status
            self.console.info(
                f"  -> {status_str} in {result.duration_seconds:.1f}s"
            )

        return results

    def _run_parallel(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments in parallel using ProcessPoolExecutor."""
        max_workers = min(self.config.execution.max_workers, len(experiments))
        self.console.info(f"Running {len(experiments)} experiments with {max_workers} workers")

        results = []

        # First, check cache for all experiments
        cached_results = {}
        uncached_experiments = []

        for exp in experiments:
            if self.cache and exp.cache_key in self.cache:
                cached_results[exp.id] = ExperimentResult(
                    id=exp.id,
                    parameters=exp.parameters,
                    status="success",
                    duration_seconds=0,
                    results=self.cache[exp.cache_key],
                    error=None,
                    cached=True,
                )
            else:
                uncached_experiments.append(exp)

        if cached_results:
            self.console.info(f"Found {len(cached_results)} cached results")

        if uncached_experiments:
            # Run uncached experiments in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_exp = {
                    executor.submit(_run_experiment_worker, exp.parameters): exp
                    for exp in uncached_experiments
                }

                for future in future_to_exp:
                    exp = future_to_exp[future]
                    start_time = time.time()

                    try:
                        worker_result = future.result(
                            timeout=self.config.execution.timeout_seconds
                        )
                        duration = time.time() - start_time

                        if worker_result["success"]:
                            result = ExperimentResult(
                                id=exp.id,
                                parameters=exp.parameters,
                                status="success",
                                duration_seconds=duration,
                                results=worker_result["results"],
                                error=None,
                                cached=False,
                            )
                            # Cache the result
                            if self.cache:
                                self.cache[exp.cache_key] = worker_result["results"]
                        else:
                            result = ExperimentResult(
                                id=exp.id,
                                parameters=exp.parameters,
                                status="failed",
                                duration_seconds=duration,
                                results=None,
                                error=worker_result["error"],
                                cached=False,
                            )

                    except FuturesTimeout:
                        result = ExperimentResult(
                            id=exp.id,
                            parameters=exp.parameters,
                            status="timeout",
                            duration_seconds=self.config.execution.timeout_seconds,
                            results=None,
                            error=f"Timeout after {self.config.execution.timeout_seconds}s",
                            cached=False,
                        )

                    except Exception as e:
                        result = ExperimentResult(
                            id=exp.id,
                            parameters=exp.parameters,
                            status="failed",
                            duration_seconds=time.time() - start_time,
                            results=None,
                            error=str(e),
                            cached=False,
                        )

                    cached_results[exp.id] = result

        # Preserve original order
        for exp in experiments:
            results.append(cached_results[exp.id])

        return results

    def _run_single_experiment(self, exp: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with timeout handling."""
        # Check cache first
        if self.cache and exp.cache_key in self.cache:
            return ExperimentResult(
                id=exp.id,
                parameters=exp.parameters,
                status="success",
                duration_seconds=0,
                results=self.cache[exp.cache_key],
                error=None,
                cached=True,
            )

        start_time = time.time()

        try:
            # Run in current process for sequential mode
            worker_result = _run_experiment_worker(exp.parameters)
            duration = time.time() - start_time

            if worker_result["success"]:
                # Cache successful result
                if self.cache:
                    self.cache[exp.cache_key] = worker_result["results"]

                return ExperimentResult(
                    id=exp.id,
                    parameters=exp.parameters,
                    status="success",
                    duration_seconds=duration,
                    results=worker_result["results"],
                    error=None,
                    cached=False,
                )
            else:
                return ExperimentResult(
                    id=exp.id,
                    parameters=exp.parameters,
                    status="failed",
                    duration_seconds=duration,
                    results=None,
                    error=worker_result["error"],
                    cached=False,
                )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            return ExperimentResult(
                id=exp.id,
                parameters=exp.parameters,
                status="failed",
                duration_seconds=duration,
                results=None,
                error=error_msg,
                cached=False,
            )

    def _validate_rules(self, results: List[ExperimentResult]) -> List[RuleResult]:
        """Validate all expectation rules against results."""
        rule_results = []

        for rule_config in self.config.expectations:
            # Convert Pydantic model to dict
            if hasattr(rule_config, "model_dump"):
                rule_dict = rule_config.model_dump()
            else:
                rule_dict = dict(rule_config)

            rule = create_rule(rule_dict)
            result = rule.validate(results)
            rule_results.append(result)

            # Log result
            if result.status.value == "PASSED":
                self.console.success(f"Rule [{rule_dict['type']}]: {result.message}")
            elif result.status.value == "FAILED":
                self.console.error(f"Rule [{rule_dict['type']}]: {result.message}")
            else:
                self.console.warning(f"Rule [{rule_dict['type']}]: {result.message}")

        return rule_results

    def preview(self) -> List[Dict[str, Any]]:
        """Preview experiments without running them.

        Returns:
            List of experiment previews with varying parameters.
        """
        experiments = self.generate_experiments()

        # Get the varying parameter names
        if self.config.parameter_grid.mode == "single_sweep":
            varying = [self.config.parameter_grid.sweep_parameter]
        else:
            varying = list(self.config.parameter_grid.parameters.keys())

        previews = []
        for exp in experiments:
            preview = {
                "id": exp.id,
                "varying_params": {k: exp.parameters.get(k) for k in varying},
                "cache_key": exp.cache_key,
                "cached": self.cache and exp.cache_key in self.cache if self.cache else False,
            }
            previews.append(preview)

        return previews
