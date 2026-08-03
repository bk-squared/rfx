#!/usr/bin/env python3
"""Re-baseline ``tests/fixtures/msl_s_matrix_golden.npy`` (the E2E drift lock).

Replicates ``test_compute_msl_s_matrix_end_to_end_matches_historical_base``'s
EXACT invocation — fixture, frequency grid, ``num_periods=12``, DEFAULT
``enforce_passivity`` — runs the full FDTD once, and saves the resulting S as
the new golden. One run, ~7 min on CPU.

Re-baselining is legitimate ONLY with a written reason in that test's
docstring (see its RE-BASELINED block and PR #516 review finding F4): the
golden is a cross-version drift lock, so regenerating it silently converts a
real change into a non-event. This script prints the old-vs-new deviation so
the reason can be written with its number.

Provenance: first used for the 2026-07-30 re-baseline (PR #516; deviation
history vs the retired S1 golden recorded in issue #509).

Run from anywhere::

    python scripts/capture_msl_e2e_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Script-file invocation puts the script's dir on sys.path, never the cwd —
# without this, `import rfx` resolves to whatever is pip-installed.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import importlib

import jax.numpy as jnp
import numpy as np


def main() -> None:
    t = importlib.import_module("test_msl_sparam_ad")
    golden_old = np.load(t.E2E_GOLDEN_PATH)
    sim = t._build_thru_line_sim()
    freqs = jnp.linspace(
        t.F_MAX / 10, t.F_MAX, golden_old.shape[-1], dtype=jnp.float32
    )
    res = sim.compute_msl_s_matrix(freqs=freqs, num_periods=12)
    S = np.asarray(res.S).astype(np.complex128)

    print(f"old-vs-new max|dS| = {np.max(np.abs(S - golden_old)):.4e}")
    for lbl, X in (("old", golden_old), ("new", S)):
        cp = np.abs(X[0, 0]) ** 2 + np.abs(X[1, 0]) ** 2
        print(
            f"  {lbl}: mean|S11|={np.abs(X[0, 0]).mean():.4f} "
            f"mean|S21|={np.abs(X[1, 0]).mean():.4f} "
            f"colP {cp.min():.4f}..{cp.max():.4f}"
        )
    np.save(t.E2E_GOLDEN_PATH, S)
    print(f"golden re-baselined at {t.E2E_GOLDEN_PATH}")
    print(
        "REMINDER: write the reason (with the deviation above) into the "
        "test's RE-BASELINED docstring block, or revert this."
    )


if __name__ == "__main__":
    main()
