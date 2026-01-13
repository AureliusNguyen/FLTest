# FL Parameter Grid Validator - Complete Guide

## Table of Contents
1. [What Is This?](#what-is-this)
2. [The Core Concept](#the-core-concept)
3. [Architecture Overview](#architecture-overview)
4. [Component Breakdown](#component-breakdown)
5. [YAML Configuration Format](#yaml-configuration-format)
6. [The Three Rule Types](#the-three-rule-types)
7. [How to Run It](#how-to-run-it)
8. [Understanding the Output](#understanding-the-output)
9. [File Locations](#file-locations)

---

## What Is This?

The FL Parameter Grid Validator is a **sanity-checking tool** for federated learning frameworks. Instead of comparing different frameworks against each other (which your existing benchmarks do), this tool validates that a **single framework behaves as expected** when you change its parameters.

Think of it like this:
- **Before**: "Flower takes 221s, FLARE takes 835s" (comparison)
- **Now**: "When I increase training rounds from 5 to 10, accuracy should go up. Did it?" (validation)

This is similar to scikit-learn's `GridSearchCV`, but instead of finding optimal hyperparameters, we're **checking that the framework behaves correctly**.

---

## The Core Concept

### The Problem We're Solving

When working with FL frameworks, you might encounter bugs where:
- More training rounds **don't** improve accuracy (something's broken)
- Changing learning rate has **no effect** (optimizer not working)
- Results are **wildly inconsistent** (non-determinism bug)

These bugs are hard to catch manually. The validator automates this.

### The Solution

1. **Define a parameter to vary** (e.g., `num_rounds: [2, 5, 10, 20]`)
2. **Define expected behavior** (e.g., "accuracy should increase")
3. **Run experiments** for each parameter value
4. **Validate results** against expectations
5. **Report** pass/fail with details

### Example Scenario

```
Parameter: num_rounds = [2, 5, 10, 20]
Expected: Accuracy increases monotonically

Results:
  rounds=2  → accuracy=0.45  ✓
  rounds=5  → accuracy=0.72  ✓ (increased)
  rounds=10 → accuracy=0.85  ✓ (increased)
  rounds=20 → accuracy=0.52  ✗ UNEXPECTED DROP!

Verdict: FAILED - Something is wrong with the framework
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (cli.py)                          │
│            fl-validate run config.yaml                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Config Parser (config/parser.py)                │
│         Loads YAML → Validates with Pydantic                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Grid Generator (core/grid_generator.py)           │
│                                                              │
│   single_sweep mode:     combinatorial mode:                 │
│   [2, 5, 10] rounds  →   rounds × clients =                  │
│   3 experiments          2 × 2 = 4 experiments               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Experiment Runner (core/experiment_runner.py)       │
│                                                              │
│   For each experiment config:                                │
│     1. Check cache (skip if already ran)                     │
│     2. Build Hydra config                                    │
│     3. Call run_fl_simulation(cfg)  ← Your existing code     │
│     4. Collect results                                       │
│     5. Cache results                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Rule Validation (rules/*.py)                    │
│                                                              │
│   MonotonicRule:  "accuracy should increase with rounds"     │
│   BoundedRule:    "accuracy must be between 0 and 1"         │
│   RelativeRule:   "no sudden drops > 50%"                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            JSON Reporter (output/json_reporter.py)           │
│                                                              │
│   Generates structured JSON report for CI/CD integration     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. CLI (`fl_testing/validator/cli.py`)

The command-line interface built with Click. Provides three commands:

| Command | Purpose |
|---------|---------|
| `fl-validate run config.yaml` | Run validation and execute experiments |
| `fl-validate preview config.yaml` | Show what would run without executing |
| `fl-validate list` | List available validation configs |

**Key options for `run`:**
- `--dry-run` / `-d`: Preview without executing
- `--parallel` / `-p`: Run experiments in parallel
- `--workers N`: Number of parallel workers
- `--timeout N`: Seconds before experiment times out
- `--no-cache`: Don't use cached results

### 2. Config Schema (`fl_testing/validator/config/schema.py`)

Defines the structure of validation YAML files using Pydantic models. This ensures your config files are valid before running expensive experiments.

**Key models:**
- `ValidationConfig`: Root configuration
- `ParameterGrid`: Defines how to vary parameters
- `ParameterSpec`: Single parameter's values or range
- `MonotonicExpectation`, `BoundedExpectation`, `RelativeThresholdExpectation`: Rule definitions
- `ExecutionConfig`: How to run (parallel, timeout, caching)

### 3. Config Parser (`fl_testing/validator/config/parser.py`)

Simple YAML loader that:
1. Reads the YAML file
2. Handles the optional `validation:` wrapper key
3. Validates against Pydantic schema
4. Returns a `ValidationConfig` object

### 4. Grid Generator (`fl_testing/validator/core/grid_generator.py`)

Creates experiment configurations from your parameter grid.

**Two modes:**

```
single_sweep mode (ablation study):
  sweep_parameter: num_rounds
  num_rounds: [2, 5, 10]

  Generates: 3 experiments
    exp_001: {num_rounds: 2, ...other params fixed...}
    exp_002: {num_rounds: 5, ...other params fixed...}
    exp_003: {num_rounds: 10, ...other params fixed...}

combinatorial mode (full grid):
  num_rounds: [5, 10]
  num_clients: [3, 5]

  Generates: 2 × 2 = 4 experiments
    exp_001: {num_rounds: 5, num_clients: 3}
    exp_002: {num_rounds: 5, num_clients: 5}
    exp_003: {num_rounds: 10, num_clients: 3}
    exp_004: {num_rounds: 10, num_clients: 5}
```

### 5. Experiment Runner (`fl_testing/validator/core/experiment_runner.py`)

The execution engine. For each experiment:

1. **Check cache**: If we've run this exact config before, skip and use cached result
2. **Build config**: Create Hydra-compatible DictConfig from parameters
3. **Execute**: Call `run_fl_simulation(cfg)` from your existing `main.py`
4. **Collect**: Get the result dict (`Final Round Accuracy`, `Final Round Loss`, etc.)
5. **Cache**: Store result for future runs
6. **Validate**: Pass all results to rule validators

**Features:**
- Sequential or parallel execution
- Timeout handling (kills hung experiments)
- Continue-on-failure (don't abort entire run if one experiment fails)
- Result caching (avoid re-running identical experiments)

### 6. Rules (`fl_testing/validator/rules/`)

Three types of expectation rules:

#### MonotonicRule (`monotonic.py`)
Checks that a metric consistently increases or decreases as a parameter changes.

```python
# Example: Accuracy should increase with more rounds
MonotonicRule(
    metric="Final Round Accuracy",
    parameter="num_rounds",
    direction="increasing",
    tolerance=0.05  # Allow up to 5% noise
)
```

#### BoundedRule (`bounded.py`)
Checks that a metric stays within min/max bounds.

```python
# Example: Accuracy must be between 0 and 1
BoundedRule(
    metric="Final Round Accuracy",
    min_value=0.0,
    max_value=1.0
)
```

#### RelativeThresholdRule (`relative_threshold.py`)
Checks that the metric doesn't change too dramatically between adjacent parameter values.

```python
# Example: Accuracy shouldn't drop more than 50% between steps
RelativeThresholdRule(
    metric="Final Round Accuracy",
    parameter="num_rounds",
    max_change_percent=50
)
```

### 7. JSON Reporter (`fl_testing/validator/output/json_reporter.py`)

Generates a structured JSON report containing:
- Metadata (timestamp, git commit, duration)
- Summary (pass/fail counts)
- All experiment results
- All rule validation results
- Parameter grid info

This format is designed for CI/CD integration - you can parse it programmatically.

---

## YAML Configuration Format

Here's the complete structure of a validation config file:

```yaml
validation:
  # === METADATA ===
  name: "my_validation_test"           # Identifier for this test
  description: "What this test checks" # Optional description
  version: "1.0"                       # Version string
  framework: "flower"                  # flower, flare, or pfl

  # === BASE CONFIGURATION ===
  # These are the default values for ALL experiments
  # They map directly to your existing Hydra config parameters
  base_config:
    seed: 786                          # Random seed for reproducibility
    device: "cpu"                      # cpu or cuda
    num_clients: 10                    # Number of FL clients
    num_rounds: 10                     # Training rounds (may be overridden)
    client_lr: 0.001                   # Learning rate
    client_epochs: 1                   # Local epochs per round
    client_batch_size: 32              # Batch size
    dataset: "mnist"                   # Dataset name
    model_name: "LeNet"                # Model architecture
    deterministic: true                # Deterministic mode
    DATASET_DIVISION_CLIENTS: 1000     # How to divide dataset
    max_test_data_size: 2048           # Test set size

  # === PARAMETER GRID ===
  # Defines which parameters to vary and how
  parameter_grid:
    mode: "single_sweep"               # or "combinatorial"
    sweep_parameter: "num_rounds"      # Required for single_sweep mode
    parameters:
      num_rounds:
        values: [2, 5, 10, 20]         # Explicit list
        # OR use range:
        # range:
        #   start: 2
        #   end: 20
        #   step: 2

  # === EXPECTATIONS ===
  # Rules that define expected behavior
  expectations:
    - type: "monotonic"
      parameter: "num_rounds"
      metric: "Final Round Accuracy"
      direction: "increasing"
      tolerance: 0.05

    - type: "bounded"
      metric: "Final Round Accuracy"
      min_value: 0.0
      max_value: 1.0

    - type: "relative_threshold"
      parameter: "num_rounds"
      metric: "Final Round Loss"
      max_change_percent: 50

  # === EXECUTION SETTINGS ===
  execution:
    mode: "sequential"                 # or "parallel"
    max_workers: 4                     # For parallel mode
    timeout_seconds: 600               # 10 minutes per experiment
    continue_on_failure: true          # Don't abort on failures
    use_cache: true                    # Cache results
    cache_path: "data/validator_cache" # Cache location

  # === OUTPUT SETTINGS ===
  output:
    format: "json"                     # Output format
    path: "validation_results/"        # Output directory
    verbose: true                      # Detailed console output
```

---

## The Three Rule Types

### 1. Monotonic Rule

**Purpose**: Verify that a metric consistently moves in one direction as a parameter changes.

**When to use**:
- More training rounds → better accuracy
- More data → better accuracy
- Higher learning rate → faster convergence (up to a point)

**Configuration**:
```yaml
- type: "monotonic"
  parameter: "num_rounds"        # The parameter being varied
  metric: "Final Round Accuracy" # The metric to check
  direction: "increasing"        # or "decreasing"
  tolerance: 0.05                # Allow 5% tolerance for noise
```

**How it works**:
1. Sort experiments by parameter value
2. For each adjacent pair, check if metric moved in expected direction
3. Allow violations within tolerance
4. Report any violations

**Example**:
```
num_rounds=2  → accuracy=0.45
num_rounds=5  → accuracy=0.72  ✓ (0.72 > 0.45)
num_rounds=10 → accuracy=0.70  ✓ (0.70 > 0.72-0.05, within tolerance)
num_rounds=20 → accuracy=0.50  ✗ (0.50 < 0.70-0.05, violation!)
```

### 2. Bounded Rule

**Purpose**: Verify that a metric stays within acceptable bounds.

**When to use**:
- Accuracy should be between 0 and 1
- Loss should be positive and below some threshold
- Training time should be reasonable

**Configuration**:
```yaml
- type: "bounded"
  metric: "Final Round Accuracy"
  min_value: 0.3    # At least 30% accuracy
  max_value: 1.0    # At most 100%
```

**How it works**:
1. Check every experiment's metric value
2. Flag any values outside [min_value, max_value]
3. Report violations with the actual values

**Example**:
```
exp_001: accuracy=0.45 ✓ (within [0.3, 1.0])
exp_002: accuracy=0.72 ✓
exp_003: accuracy=0.25 ✗ (below 0.3!)
exp_004: accuracy=1.05 ✗ (above 1.0!)
```

### 3. Relative Threshold Rule

**Purpose**: Catch sudden dramatic changes that might indicate bugs.

**When to use**:
- Ensure no catastrophic forgetting
- Catch instability in training
- Detect anomalous results

**Configuration**:
```yaml
- type: "relative_threshold"
  parameter: "num_rounds"
  metric: "Final Round Accuracy"
  max_change_percent: 50         # No more than 50% change between steps
```

**How it works**:
1. Sort experiments by parameter value
2. Calculate percent change between adjacent experiments
3. Flag any changes exceeding the threshold

**Example**:
```
num_rounds=2  → accuracy=0.45
num_rounds=5  → accuracy=0.72  → change=60% ✗ (exceeds 50%!)
num_rounds=10 → accuracy=0.85  → change=18% ✓
num_rounds=20 → accuracy=0.88  → change=4%  ✓
```

---

## How to Run It

### Prerequisites

The dependencies are already installed. Verify with:
```bash
poetry run fl-validate --version
```

### Step 1: Preview (Recommended First)

Always preview before running to see what experiments will execute:

```bash
# Preview the rounds sweep example
poetry run fl-validate preview fl_testing/config/validation/examples/rounds_sweep.yaml
```

Output:
```
Validation: flower_rounds_sweep
Framework: flower
Mode: single_sweep
Total experiments: 3

Experiments to run:
  exp_001: {'num_rounds': 2}
  exp_002: {'num_rounds': 5}
  exp_003: {'num_rounds': 10}

Expectations to validate:
  - monotonic: Final Round Accuracy
  - monotonic: Final Round Loss
  - bounded: Final Round Accuracy
```

### Step 2: Run Validation

```bash
# Run the validation (this will execute actual FL experiments!)
poetry run fl-validate run fl_testing/config/validation/examples/rounds_sweep.yaml
```

This will:
1. Run 3 Flower experiments (rounds=2, 5, 10)
2. Each experiment trains an FL model on MNIST
3. Collect accuracy and loss metrics
4. Validate against the defined rules
5. Generate a JSON report

**Expected time**: ~5-15 minutes depending on your hardware

### Step 3: Review Results

The JSON report is saved to `validation_results/`. You can also see the summary in the console output:

```
============================================================
VALIDATION SUMMARY
============================================================
  Total experiments: 3
  Successful: 3
  Failed: 0
  Cached: 0
  Rules passed: 3
  Rules failed: 0
  Duration: 423.5s

Report saved to: validation_results/flower_rounds_sweep_20260113_120000.json

Overall status: PASSED
============================================================
```

### Available Options

```bash
# Run with parallel execution (faster, uses more resources)
poetry run fl-validate run config.yaml --parallel --workers 4

# Override framework
poetry run fl-validate run config.yaml --framework pfl

# Longer timeout for slow experiments
poetry run fl-validate run config.yaml --timeout 1200

# Skip cache (re-run everything)
poetry run fl-validate run config.yaml --no-cache

# Quiet mode (less output)
poetry run fl-validate run config.yaml --quiet

# List all available configs
poetry run fl-validate list --path fl_testing/config/validation
```

---

## Understanding the Output

### Console Output

During execution, you'll see:
```
[INFO] Loading configuration from config.yaml
[INFO] Generated 3 experiments for flower
[INFO] Running experiment exp_001 (1/3)...
[INFO]   -> success in 142.3s
[INFO] Running experiment exp_002 (2/3)...
[INFO]   -> success in 145.1s
[INFO] Running experiment exp_003 (3/3)...
[INFO]   -> success in 148.7s
[INFO] Completed 3/3 experiments successfully
[OK] Rule [monotonic]: Final Round Accuracy changes increasingly with num_rounds
[OK] Rule [monotonic]: Final Round Loss changes decreasingly with num_rounds
[OK] Rule [bounded]: Final Round Accuracy within bounds [0.0, 1.0]
```

### JSON Report Structure

```json
{
  "meta": {
    "validation_name": "flower_rounds_sweep",
    "framework": "flower",
    "duration_seconds": 436.1,
    "generated_at": "2026-01-13T12:00:00"
  },
  "summary": {
    "total_experiments": 3,
    "successful_experiments": 3,
    "failed_experiments": 0,
    "passed_rules": 3,
    "failed_rules": 0,
    "overall_status": "PASSED"
  },
  "experiments": [
    {
      "id": "exp_001",
      "parameters": {"num_rounds": 2, "num_clients": 5, ...},
      "status": "success",
      "duration_seconds": 142.3,
      "results": {
        "Final Round Accuracy": 0.4523,
        "Final Round Loss": 0.0892
      }
    },
    ...
  ],
  "rule_results": [
    {
      "rule_type": "monotonic",
      "status": "PASSED",
      "message": "Final Round Accuracy changes increasingly with num_rounds",
      "details": {
        "values": [
          {"num_rounds": 2, "Final Round Accuracy": 0.4523},
          {"num_rounds": 5, "Final Round Accuracy": 0.7234},
          {"num_rounds": 10, "Final Round Accuracy": 0.8567}
        ],
        "violations": []
      }
    },
    ...
  ]
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All rules passed |
| 1 | One or more rules failed |
| 2 | Configuration error or crash |

This makes it easy to use in CI/CD pipelines.

---

## File Locations

### New Files Created

```
fl_testing/
├── validator/
│   ├── __init__.py                    # Package exports
│   ├── cli.py                         # CLI entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── parser.py                  # YAML loader
│   │   └── schema.py                  # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── grid_generator.py          # Grid generation
│   │   └── experiment_runner.py       # Execution engine
│   ├── rules/
│   │   ├── __init__.py                # Rule factory
│   │   ├── base.py                    # Abstract base class
│   │   ├── monotonic.py               # Monotonic rule
│   │   ├── bounded.py                 # Bounded rule
│   │   └── relative_threshold.py      # Relative threshold rule
│   └── output/
│       ├── __init__.py
│       └── json_reporter.py           # JSON report generator
│
├── config/
│   └── validation/
│       └── examples/
│           ├── rounds_sweep.yaml      # Single-sweep example
│           ├── lr_sweep.yaml          # Learning rate test
│           ├── full_grid.yaml         # Combinatorial grid
│           └── pfl_test.yaml          # PFL framework test
```

### Modified Files

- `pyproject.toml`: Added `pydantic`, `click` dependencies and `fl-validate` CLI entry point

### Output Locations

- **Validation reports**: `validation_results/*.json`
- **Result cache**: `data/validator_cache/`

---

## Quick Reference

```bash
# Preview what will run
poetry run fl-validate preview <config.yaml>

# Run validation
poetry run fl-validate run <config.yaml>

# Run in parallel
poetry run fl-validate run <config.yaml> --parallel --workers 4

# List available configs
poetry run fl-validate list --path fl_testing/config/validation

# Check version
poetry run fl-validate --version
```

---

## Next Steps

1. **Try the preview**: `poetry run fl-validate preview fl_testing/config/validation/examples/rounds_sweep.yaml`
2. **Run a quick test**: `poetry run fl-validate run fl_testing/config/validation/examples/rounds_sweep.yaml`
3. **Create your own config**: Copy an example and modify for your needs
4. **Integrate with CI/CD**: Use exit codes and JSON output
