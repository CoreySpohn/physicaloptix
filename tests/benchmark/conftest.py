"""Reference-data gate and executed-case accounting for the benchmark tier.

``PHYSICALOPTIX_REQUIRE_REFERENCES=1`` turns every missing reference into a
failure (not a skip) and makes a run in which no benchmark case executed a
failure too. Without it, a missing reference skips with a reason naming the
environment variable to set. The terminal summary always reports how many
cases executed and lists them, so a green run that compared nothing is
visible.
"""

import os
from pathlib import Path

import pytest

REQUIRE_ENV = "PHYSICALOPTIX_REQUIRE_REFERENCES"
EXECUTED = []


def references_required():
    return os.environ.get(REQUIRE_ENV, "") == "1"


def require_path(env_name, *, marker):
    """Directory from ``env_name`` containing ``marker``, or skip / fail."""
    raw = os.environ.get(env_name)
    problem = None
    if not raw:
        problem = f"{env_name} is not set"
    elif not (Path(raw).expanduser() / marker).is_file():
        problem = f"{env_name}={raw} has no {marker}"
    if problem is None:
        return Path(raw).expanduser()
    message = f"benchmark reference data absent: {problem}"
    if references_required():
        pytest.fail(f"{message} ({REQUIRE_ENV}=1 requires it)", pytrace=False)
    pytest.skip(f"{message}; set {env_name} (and {REQUIRE_ENV}=1 to require it)")


BUNDLE_ENV = "PHYSICALOPTIX_NIRCAM_BUNDLE"
REFERENCE_ENV = "PHYSICALOPTIX_NIRCAM_REFERENCE"


@pytest.fixture(scope="session")
def nircam_bundle_dir():
    """The exported NIRCam prescription bundle (``manifest.json`` inside)."""
    return require_path(BUNDLE_ENV, marker="manifest.json")


@pytest.fixture(scope="session")
def nircam_reference_dir():
    """The STPSF references: ``<case>.fits`` and ``band/manifest.json``."""
    return require_path(REFERENCE_ENV, marker="band/manifest.json")


@pytest.fixture
def record_case():
    """Record one executed comparison: ``record_case(case_id, summary)``."""

    def record(case_id, summary):
        EXECUTED.append((case_id, summary))

    return record


def pytest_terminal_summary(terminalreporter):
    gate = "on" if references_required() else "off"
    terminalreporter.write_sep(
        "-", f"benchmark cases executed: {len(EXECUTED)} ({REQUIRE_ENV} {gate})"
    )
    for case_id, summary in EXECUTED:
        terminalreporter.write_line(f"  {case_id}: {summary}")


def pytest_sessionfinish(session, exitstatus):
    if references_required() and not EXECUTED and exitstatus in (0, 5):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print(f"\n{REQUIRE_ENV}=1 but no benchmark case executed: failing the run")
