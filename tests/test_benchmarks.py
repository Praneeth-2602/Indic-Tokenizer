from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_throughput_guard_exits_nonzero_when_threshold_is_too_high():
    env = os.environ.copy()
    python_path = str(ROOT / "python")
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_tokenizers.py",
            "--assert-min-chars-per-second",
            "999999999999",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "throughput below threshold" in result.stdout
