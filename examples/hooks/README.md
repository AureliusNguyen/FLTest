# Example hooks

Example hook modules for the fltest lifecycle. Load them via the `FLTEST_HOOKS` environment variable when running the simulation.

From the **repo root**:

```bash
# Load the aggregator (runs A then B handlers)
export FLTEST_HOOKS=examples/hooks/fltest_hooks
poetry run python fltest/main.py ...

# Or load A and B directly in order
export FLTEST_HOOKS=examples/hooks/fltest_hooks_a,examples/hooks/fltest_hooks_b
poetry run python fltest/main.py ...
```

Hook registry lives in `fltest.core.hooks`; you can also use `from fltest import hooks`.

You can also define attack or instrumentation hooks here, for example:

- `fltest_hooks_a.py` / `fltest_hooks_b.py`: simple logging and tmp-file signals.
- `atk_label_flip.py`: label-flipping attack on a single client using
  `before_client_train` and `after_round` hooks.

From the driver and Ray workers, hooks are loaded via `FLTEST_HOOKS`
inside both the simulation and the client processes, so client-side
hooks like `after_client_train` and `before_client_train` defined in
`examples/hooks/...` will run in the workers as long as the working
directory is the repo root when you start the run.
