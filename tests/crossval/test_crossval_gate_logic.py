"""Fast, no-simulation regression tests for crossval gate LOGIC.

Covers ``validation/crossval/11_waveguide_port_wr90.py`` (per-freq band +
ceiling gate, issue #340).

That script's actual FDTD gate runs in no automated CI workflow (confirmed
2026-07-14: no ``.github/workflows/*.yml`` invokes
``scripts/run_crossval_cpu.py`` or the script directly;
``tests/contracts/test_crossval_manifest_contract.py`` only unit-tests the runner's
classification logic against synthetic/mocked subprocess results, and the
manifest's structural self-consistency — never the scripts themselves). This
file pins the GATE MATH against synthetic arrays so a future edit to the
script's ceiling logic reds in the fast CI lane, without paying for a full
FDTD run.

cv11 is properly guarded (``if __name__ == "__main__":`` at
validation/crossval/11_waveguide_port_wr90.py:837) and its gate helper is a
pure function, so it is imported directly here.

2026-09-21: the second half of this file pinned the per-bin conservation
ceiling, the settling-tail witness and the fringe-resolved gate of the slab
Fresnel case, whose constants it held as a hand-copied table. That case was
removed; the table and its arms went with it.
"""

from __future__ import annotations

import json
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSVAL_DIR = REPO_ROOT / "validation" / "crossval"


def _load_cv11():
    """Import cv11 as a module without executing its __main__ block."""
    path = CROSSVAL_DIR / "11_waveguide_port_wr90.py"
    spec = importlib.util.spec_from_file_location("_cv11_gate_logic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cv11_per_freq_band_check_rejects_single_bin_spike():
    """A single-bin 1.5 spike in an otherwise all-ones |S| array must FAIL
    the per-freq band gate (mirrors cv11's own selftest, issue #340)."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    spike = np.ones(21)
    spike[10] = 1.5
    assert not cv11.per_freq_band_check(
        "test-spike", f_fake, spike, 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_per_freq_band_check_accepts_healthy_curve():
    """An all-ones |S| array (within the band) must PASS."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    assert cv11.per_freq_band_check(
        "test-healthy", f_fake, np.ones(21), 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_per_freq_band_check_rejects_ceiling_violation_within_band():
    """A value inside [lo, hi] can still violate the SEPARATE passivity
    ceiling — the ceiling must be checked independently of the band."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    mag = np.ones(21)
    mag[5] = 1.06   # inside [0.93, 1.07] but above ceiling=1.05
    assert not cv11.per_freq_band_check(
        "test-ceiling", f_fake, mag, 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_selftest_runs_without_aborting():
    """cv11's own _selftest_per_freq_gate (validation/crossval/
    11_waveguide_port_wr90.py:357-377) calls sys.exit(1) if either of its two
    synthetic checks fails to bite. A normal return here means the gate is
    genuinely live on the version of the code under test."""
    cv11 = _load_cv11()
    cv11._selftest_per_freq_gate()

