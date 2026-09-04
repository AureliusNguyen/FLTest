"""The run-matrix table shows what differs between runs and flags failures."""

import types

from fltest.testing.report import print_run_matrix


def _result(name, framework="reference", status="success", error=None, **final):
    """A stand-in for RunResult carrying only the fields the renderer reads."""
    return types.SimpleNamespace(
        run_id=name, run_name=name, framework=framework, status=status, error=error,
        duration_seconds=1.0,
        params={
            "framework": framework, "dataset": "mnist", "data_distribution": "iid",
            "dirichlet_alpha": None, "model_name": "MLP", "num_clients": 4,
            "num_rounds": 3, "seed": 786, "attacks": [], "defenses": [],
        },
        final=final,
    )


def test_shared_parameters_print_once_and_differences_become_columns(capsys):
    a = _result("plain", accuracy=0.9, loss=0.3)
    b = _result("robust", accuracy=0.8, loss=0.4)
    b.params["defenses"] = [{"name": "median", "params": {}}]

    print_run_matrix("demo", [a, b], total_duration=2.0)
    out = capsys.readouterr().out

    assert "same for all runs:" in out
    assert "dataset=mnist" in out          # shared, so stated once above the table
    assert "defense" in out                # differs, so it is a column
    assert "median" in out
    assert "dirichlet_alpha" not in out    # not applicable under IID, so not shown
    assert "2 runs" in out


def test_plugin_metrics_appear_with_short_headers(capsys):
    r = _result("backdoor", accuracy=0.89, loss=0.35, attack_success_rate=0.47,
                per_client_acc_min=0.36, gm_weight_sum=-51.6)
    print_run_matrix("demo", [r])
    out = capsys.readouterr().out

    assert "asr" in out and "0.4700" in out
    assert "pc-min" in out
    assert "gm_weight_sum" not in out      # fingerprint stays in the JSON report only


def test_failed_run_is_reported_with_status_and_reason(capsys):
    ok = _result("good", accuracy=0.9, loss=0.3)
    bad = _result("bad", status="failed", error="RuntimeError: backend exploded\n  stack…")

    print_run_matrix("demo", [ok, bad], total_duration=2.0)
    out = capsys.readouterr().out

    assert "status" in out and "failed" in out
    assert "1 succeeded, 1 failed" in out
    assert "RuntimeError: backend exploded" in out


def test_empty_matrix_does_not_crash(capsys):
    print_run_matrix("demo", [])
    assert "no runs" in capsys.readouterr().out
