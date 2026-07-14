from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_regression_suite import _run_pytest


def _result_tree(tmp_path: Path) -> Path:
    for folder in ("raw_responses", "structured_outputs", "failure_details"):
        (tmp_path / folder).mkdir()
    return tmp_path


def test_pytest_shard_uses_run_local_base_temp(tmp_path):
    root = _result_tree(tmp_path)
    completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
    with patch("scripts.run_regression_suite.subprocess.run", return_value=completed) as run:
        assert _run_pytest(root, "fast", ["tests"]) == 0

    command = run.call_args.args[0]
    base_temp = command[command.index("--basetemp") + 1]
    assert Path(base_temp) == root / "_pytest_tmp" / "fast"


def test_nonzero_pytest_exit_cannot_produce_false_green_report(tmp_path):
    root = _result_tree(tmp_path)
    completed = SimpleNamespace(returncode=1, stdout="setup error", stderr="")
    with patch("scripts.run_regression_suite.subprocess.run", return_value=completed):
        assert _run_pytest(root, "fast", ["tests"]) == 1

    report = (root / "pytest_fast.json").read_text(encoding="utf-8")
    assert '"status": "FAIL"' in report
    assert '"failure_kind": "test_error"' in report
