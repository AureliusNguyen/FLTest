"""CLI entry point for FL Parameter Grid Validator."""

import sys
from pathlib import Path
from typing import Optional

import click

from fl_testing.validator.config.parser import load_validation_config
from fl_testing.validator.core.experiment_runner import ExperimentRunner, ConsoleReporter
from fl_testing.validator.output.json_reporter import JSONReporter


@click.group()
@click.version_option(version="1.0.0", prog_name="fl-validate")
def cli():
    """FL Parameter Grid Validator - Validate FL framework behavior with parameter sweeps."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--framework",
    "-f",
    type=click.Choice(["flower", "flare", "pfl"]),
    help="Override framework from config",
)
@click.option("--parallel", "-p", is_flag=True, help="Enable parallel execution")
@click.option(
    "--workers", "-w", default=4, type=int, help="Number of parallel workers"
)
@click.option(
    "--timeout",
    "-t",
    default=600,
    type=int,
    help="Timeout per experiment in seconds",
)
@click.option("--dry-run", "-d", is_flag=True, help="Preview experiments without running")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--no-cache", is_flag=True, help="Disable result caching")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def run(
    config_path: str,
    framework: Optional[str],
    parallel: bool,
    workers: int,
    timeout: int,
    dry_run: bool,
    output: Optional[str],
    no_cache: bool,
    verbose: bool,
    quiet: bool,
):
    """Run validation tests from CONFIG_PATH.

    Example:
        fl-validate run config/validation/rounds_test.yaml
        fl-validate run config/validation/test.yaml --parallel --workers 4
        fl-validate run config/validation/test.yaml --dry-run
    """
    console = ConsoleReporter(verbose=verbose, quiet=quiet)

    try:
        # Load and parse configuration
        console.info(f"Loading configuration from {config_path}")
        config = load_validation_config(config_path)

        # Apply CLI overrides
        if framework:
            config.framework = framework
            console.info(f"Override framework: {framework}")

        if parallel:
            config.execution.mode = "parallel"
            config.execution.max_workers = workers

        config.execution.timeout_seconds = timeout
        config.execution.use_cache = not no_cache

        # Create runner
        runner = ExperimentRunner(config, console=console)

        if dry_run:
            # Preview mode
            console.info("Dry run mode - previewing experiments")
            previews = runner.preview()

            click.echo(f"\n{'='*60}")
            click.echo(f"Validation: {config.name}")
            click.echo(f"Framework: {config.framework}")
            click.echo(f"Mode: {config.parameter_grid.mode}")
            click.echo(f"Total experiments: {len(previews)}")
            click.echo(f"{'='*60}\n")

            click.echo("Experiments to run:")
            for p in previews:
                cache_str = " (cached)" if p["cached"] else ""
                click.echo(f"  {p['id']}: {p['varying_params']}{cache_str}")

            click.echo(f"\nExpectations to validate:")
            for exp in config.expectations:
                if hasattr(exp, "model_dump"):
                    exp_dict = exp.model_dump()
                else:
                    exp_dict = dict(exp)
                click.echo(f"  - {exp_dict['type']}: {exp_dict.get('metric', 'N/A')}")

            return

        # Execute experiments
        console.info(f"Starting validation: {config.name}")
        console.info(f"Framework: {config.framework}")
        console.info(f"Execution mode: {config.execution.mode}")

        results = runner.run()

        # Generate report
        output_dir = output or config.output.path
        reporter = JSONReporter(output_dir=output_dir)
        report_path = reporter.generate(results, config)

        # Print summary
        click.echo(f"\n{'='*60}")
        click.echo("VALIDATION SUMMARY")
        click.echo(f"{'='*60}")

        summary = {
            "Total experiments": len(results.experiments),
            "Successful": sum(1 for e in results.experiments if e.status == "success"),
            "Failed": sum(1 for e in results.experiments if e.status == "failed"),
            "Cached": sum(1 for e in results.experiments if e.cached),
            "Rules passed": sum(
                1 for r in results.rule_results if r.status.value == "PASSED"
            ),
            "Rules failed": sum(
                1 for r in results.rule_results if r.status.value == "FAILED"
            ),
            "Duration": f"{results.total_duration:.1f}s",
        }

        for key, value in summary.items():
            click.echo(f"  {key}: {value}")

        click.echo(f"\nReport saved to: {report_path}")

        status = "PASSED" if results.all_passed else "FAILED"
        click.echo(f"\nOverall status: {status}")
        click.echo(f"{'='*60}\n")

        # Exit with appropriate code
        sys.exit(0 if results.all_passed else 1)

    except FileNotFoundError as e:
        console.error(str(e))
        sys.exit(2)
    except ValueError as e:
        console.error(f"Configuration error: {e}")
        sys.exit(2)
    except Exception as e:
        console.error(f"Validation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def preview(config_path: str):
    """Preview experiments without executing (alias for run --dry-run).

    Example:
        fl-validate preview config/validation/rounds_test.yaml
    """
    ctx = click.get_current_context()
    ctx.invoke(run, config_path=config_path, dry_run=True)


@cli.command("list")
@click.option(
    "--path",
    "-p",
    default="fl_testing/config/validation",
    help="Path to search for configs",
)
def list_configs(path: str):
    """List available validation configurations.

    Example:
        fl-validate list
        fl-validate list --path config/validation
    """
    config_dir = Path(path)
    if not config_dir.exists():
        click.echo(f"Config directory not found: {path}")
        click.echo("Try creating validation configs in fl_testing/config/validation/")
        return

    configs = list(config_dir.glob("**/*.yaml"))
    configs.extend(config_dir.glob("**/*.yml"))

    if not configs:
        click.echo(f"No validation configs found in {path}")
        return

    click.echo("Available validation configurations:")
    for config in sorted(configs):
        try:
            vc = load_validation_config(config)
            click.echo(f"  {config.relative_to(config_dir)}")
            click.echo(f"    Name: {vc.name}")
            click.echo(f"    Framework: {vc.framework}")
            click.echo(f"    Mode: {vc.parameter_grid.mode}")
            click.echo()
        except Exception as e:
            click.echo(f"  {config.relative_to(config_dir)} (error: {e})")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
