"""Fast, no-simulation regression tests for crossval gate LOGIC.

Covers ``validation/crossval/11_waveguide_port_wr90.py`` (per-freq band +
ceiling gate, issue #340) and ``validation/crossval/04_multilayer_fresnel.py``
(per-bin conservation ceiling + settling-tail witness, issue #341).

Neither crossval script's actual FDTD gate runs in any automated CI workflow
(confirmed 2026-07-14: no ``.github/workflows/*.yml`` invokes
``scripts/run_crossval_cpu.py`` or either script directly;
``tests/test_crossval_manifest_contract.py`` only unit-tests the runner's
classification logic against synthetic/mocked subprocess results, and the
manifest's structural self-consistency — never the scripts themselves). This
file pins the GATE MATH against synthetic arrays so a future edit to either
script's ceiling/tail logic reds in the fast CI lane, without paying for a
full (and, for cv04, optional-Meep-dependent) FDTD run.

cv11 is properly guarded (``if __name__ == "__main__":`` at
validation/crossval/11_waveguide_port_wr90.py:837) and its gate helper is a
pure function, so it is imported directly here. cv04 runs its FDTD and gate
computation entirely at MODULE level with no ``__main__`` guard (confirmed
2026-07-14) — importing it would execute the full simulation, so its
ceiling/tail logic and constants are replicated inline instead, with exact
source-line citations.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
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


# ---------------------------------------------------------------------------
# cv04 gate logic, replicated with exact source-line citations (see module
# docstring for why this can't be a direct import).
# ---------------------------------------------------------------------------

# validation/crossval/04_multilayer_fresnel.py:314 (issue #341)
CV04_CONS_MAX_LIMIT = 0.06
# validation/crossval/04_multilayer_fresnel.py:208-210 (issue #341)
CV04_TAIL_WINDOW = 50
CV04_TAIL_PURITY_LIMIT = 1e-3
CV04_TAIL_LIMIT = 0.10


def _cv04_cons_max_ok(r_plus_t_minus_1: np.ndarray) -> bool:
    """Replicates validation/crossval/04_multilayer_fresnel.py:292,315:
    ``cons_rfx = np.abs(R_rfx + T_rfx - 1)``;
    ``cons_max_ok = bool(cons_rfx.max() <= CONS_MAX_LIMIT)``."""
    cons = np.abs(r_plus_t_minus_1)
    return bool(cons.max() <= CV04_CONS_MAX_LIMIT)


def test_cv04_conservation_ceiling_rejects_single_bin_spike():
    """A single out-of-band |R+T-1| bin above the ceiling must FAIL —
    this is the exact class of silent single-bin spike issue #341 closed."""
    healthy = np.full(21, 0.01)
    spike = healthy.copy()
    spike[10] = 0.10  # > CV04_CONS_MAX_LIMIT
    assert not _cv04_cons_max_ok(spike)


def test_cv04_conservation_ceiling_accepts_healthy_curve():
    healthy = np.full(21, 0.01)
    assert _cv04_cons_max_ok(healthy)


def _cv04_tail_ok(inc_tail: np.ndarray, refl_tail: np.ndarray,
                  trans_tail: np.ndarray, inc_peak: float) -> bool:
    """Replicates validation/crossval/04_multilayer_fresnel.py:212-219:
    the settling-tail witness (issue #341) — the last TAIL_WINDOW samples of
    the incident/reflected/transmitted time series must be clean (incident
    tail negligible relative to its own peak = pulse has passed) and settled
    (reflected/transmitted tails below TAIL_LIMIT of the incident peak)."""
    tail_inc_rel = np.max(np.abs(inc_tail)) / inc_peak
    tail_refl_rel = np.max(np.abs(refl_tail)) / inc_peak
    tail_trans_rel = np.max(np.abs(trans_tail)) / inc_peak
    tail_window_clean = tail_inc_rel < CV04_TAIL_PURITY_LIMIT
    return bool(
        tail_window_clean
        and tail_refl_rel < CV04_TAIL_LIMIT
        and tail_trans_rel < CV04_TAIL_LIMIT
    )


def test_cv04_tail_witness_rejects_contaminated_window():
    """A tail window still carrying the direct pulse (incident tail not
    negligible) must FAIL, regardless of the reflected/transmitted levels."""
    inc_peak = 1.0
    contaminated_inc = np.full(CV04_TAIL_WINDOW, 0.5)  # >> purity limit
    settled = np.zeros(CV04_TAIL_WINDOW)
    assert not _cv04_tail_ok(contaminated_inc, settled, settled, inc_peak)


def test_cv04_tail_witness_rejects_unsettled_reflected_or_transmitted():
    """A clean incident window with a reflected/transmitted tail still above
    TAIL_LIMIT (ringing, not yet settled) must FAIL."""
    inc_peak = 1.0
    clean_inc = np.zeros(CV04_TAIL_WINDOW)
    unsettled = np.full(CV04_TAIL_WINDOW, 0.5)  # >> TAIL_LIMIT
    assert not _cv04_tail_ok(clean_inc, unsettled, unsettled, inc_peak)


def test_cv04_tail_witness_accepts_clean_settled_window():
    inc_peak = 1.0
    clean_inc = np.zeros(CV04_TAIL_WINDOW)
    settled = np.full(CV04_TAIL_WINDOW, 0.01)  # << TAIL_LIMIT
    assert _cv04_tail_ok(clean_inc, settled, settled, inc_peak)
