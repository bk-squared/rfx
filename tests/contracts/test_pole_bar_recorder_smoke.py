"""Exercise the diagnostic script through labrun's named-pytest route."""
import json
from pathlib import Path
import subprocess
import sys


def test_pole_bar_recorder_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "result.json"
    subprocess.run([sys.executable, "-m", "scripts.diagnostics.pole_increment_bar_record",
                    "--nu", "10630000", "--smoke-steps", "4", "--output", str(output)],
                   cwd=root, check=True, timeout=60)
    record = json.loads(output.read_text())
    assert record["smoke_only"] and record["n_steps"] == 4
    assert record["source_commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
