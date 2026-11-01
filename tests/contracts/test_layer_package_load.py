"""Each layer package loads in a fresh interpreter, with rfx.stepping's runners bound eagerly (rfx#1818).

The layer-structure checks read ASTs only; this one runs the imports.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
EAGER = ("core_step_invariants", "make_core_step", "run_nonuniform", "run_nonuniform_until_decay",
         "run_uniform_loop", "run_uniform_until_decay", "scan_with_progress")


@pytest.mark.parametrize("module", ["rfx.stepping", "rfx.model", "rfx.measurement", "rfx.boundaries"])
def test_layer_package_loads_in_a_fresh_interpreter(module):
    code = (f"import {module}, rfx.stepping as s\n"
            f"missing = [n for n in {EAGER!r} if n not in vars(s)]\n"
            "assert not missing, missing\n")
    env = {**os.environ, "PYTHONPATH": str(REPO), "JAX_PLATFORMS": "cpu"}
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
