"""
Tests for terminal_benchmark.py harness guards (A1).
"""
import subprocess
import sys
from pathlib import Path

HARNESS = str(Path(__file__).parent.parent / "scripts" / "terminal_benchmark.py")


def _run(extra_args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, HARNESS] + extra_args,
        capture_output=True,
        text=True,
    )


def test_unlock_test_without_split_test_exits_nonzero():
    """--unlock_test without --split test must exit non-zero (negative guard)."""
    result = _run(["--unlock_test", "--split", "val"])
    assert result.returncode != 0, (
        "--unlock_test --split val must exit non-zero; "
        f"got returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "unlock_test" in result.stderr.lower() or "unlock_test" in result.stdout.lower(), (
        "Expected error message mentioning unlock_test"
    )


def test_split_test_without_unlock_test_exits_nonzero():
    """--split test without --unlock_test must exit non-zero (sealed test guard)."""
    result = _run(["--split", "test"])
    assert result.returncode != 0, (
        "--split test without --unlock_test must exit non-zero; "
        f"got returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "unlock_test" in result.stderr.lower() or "unlock_test" in result.stdout.lower()
