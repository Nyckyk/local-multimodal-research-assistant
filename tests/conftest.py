from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts.regression_support import ModelAvailabilityError, require_configured_models
from settings import BASE_DIR


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


@pytest.fixture(scope="session")
def regression_cases():
    yaml = pytest.importorskip("yaml", reason="PyYAML is required for regression fixtures")
    fixture_path = BASE_DIR / "tests" / "fixtures" / "regression_cases.yaml"
    return yaml.safe_load(fixture_path.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="session")
def visual_resolution_cases():
    yaml = pytest.importorskip("yaml", reason="PyYAML is required for regression fixtures")
    fixture_path = BASE_DIR / "tests" / "fixtures" / "regression_cases.yaml"
    return yaml.safe_load(fixture_path.read_text(encoding="utf-8"))[
        "visual_resolution_cases"
    ]


@pytest.fixture(scope="session")
def require_ollama_models():
    try:
        return require_configured_models()
    except ModelAvailabilityError as error:
        pytest.fail(str(error), pytrace=False)


@pytest.fixture
def artifact_writer():
    root_value = os.environ.get("REGRESSION_RESULTS_DIR")
    root = Path(root_value) if root_value else BASE_DIR / "test_results" / "manual"
    for folder in ("raw_responses", "structured_outputs", "failure_details"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    def write(case_id: str, *, raw=None, structured=None, failure=None):
        values = {
            "raw_responses": raw,
            "structured_outputs": structured,
            "failure_details": failure,
        }
        for folder, value in values.items():
            if value is None:
                continue
            suffix = ".txt" if isinstance(value, str) else ".json"
            path = root / folder / f"{case_id}{suffix}"
            if isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                path.write_text(
                    json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
                    encoding="utf-8",
                )
    return write


def pytest_configure(config):
    config._regression_started = time.perf_counter()
    config._regression_results = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    status = "PASS" if report.passed else "SKIP" if report.skipped else "FAIL"
    item.config._regression_results.append({
        "nodeid": report.nodeid,
        "status": status,
        "duration_seconds": round(report.duration, 3),
        "detail": str(report.longrepr) if not report.passed else "",
    })


def pytest_sessionfinish(session, exitstatus):
    root_value = os.environ.get("REGRESSION_RESULTS_DIR")
    if not root_value:
        return
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "exit_code": exitstatus,
        "duration_seconds": round(time.perf_counter() - session.config._regression_started, 3),
        "tests": session.config._regression_results,
    }
    shard = os.environ.get("REGRESSION_SHARD", "pytest")
    (root / f"pytest_{shard}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
