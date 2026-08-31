"""Falsifiers for the cv14 rectangular-cavity gates (issue #812 re-gate).

``validation/crossval/14_rect_cavity_pozar.py`` is claims-bearing and is not
invoked by any CI workflow (same situation documented in
``tests/test_crossval_gate_logic.py``). This file pins its GATE LOGIC so a
future edit that re-loosens it reds in CI.

WHAT WAS WRONG (measured, issue #812)
-------------------------------------
Gate 2 took ``min()`` over the six higher modes. Every axis has a declared
target with a zero index on that axis (n=0 for TE101/TE201/TE102, m=0 for
TE011, l=0 for TM110/TM210), and ``f_mnl`` depends on an extent only through
``m/a``, so such a mode scores identically 0.000% for ANY single-axis
dimensional error -- and ``min()`` selected exactly it. Separately, a mode that
could not be extracted removed itself from ``errs`` instead of failing the gate.

The auditor's measurement, reproduced by ``test_minus_50_percent_shrink_*``
below: shrinking ``a`` 50 -> 25 mm (-50%) failed Gate 1 while Gate 2 still read
PASS on TE011 at 0.0154%.

The two-sided acceptance criterion for the re-gate is that the case still
passes on correct code (``test_correct_cavity_passes_every_gate``) AND fails on
that defect (``test_minus_50_percent_shrink_now_fails_gate_2``). A re-gate that
only did the first would be cosmetic; only the second would have broken the case.

Thresholds and their derivations are pre-declared in
``docs/design_notes/cv14_rect_cavity_gate_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CV14_PATH = REPO_ROOT / "validation" / "crossval" / "14_rect_cavity_pozar.py"


def _load_cv14():
    """Import cv14 as a module without executing its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location("_cv14_gate_logic", CV14_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CV14 = _load_cv14()

# The exactly-registered effective wall separation of the gated leg.
EXACT_EFF = (CV14.A, CV14.B, CV14.D)
# Rayleigh limit of the gated leg's harminv record (8000 samples x 1.9066e-12 s).
FRES = 65.56e6


def _rows(overrides=None, dx=1e-3, dt=1.9066e-12):
    """Synthetic ``run_leg`` rows: every mode measured EXACTLY at its own
    discrete-Yee eigenvalue, i.e. a perfect solver. ``overrides`` maps a mode
    name to the measured frequency (or ``None`` for NOT FOUND)."""
    overrides = overrides or {}
    rows = []
    for name, (m, n, l), _hint in CV14.TARGET_MODES:
        fa = CV14.pozar_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l)
        fy = CV14.yee_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l, dx, dx, dx, dt)
        f = overrides.get(name, fy) if name in overrides else fy
        rows.append((name, (m, n, l), fa, fy, f, 1e4, 1.0, 1e-6, "ey"))
    return rows


# --------------------------------------------------------------------------
# The oracle itself
# --------------------------------------------------------------------------

def test_yee_oracle_reproduces_the_predeclared_prediction_table():
    """``yee_cavity_freq`` must reproduce the table published in the
    pre-declaration note BEFORE any run was made. This pins the second oracle
    against an accidental re-derivation."""
    predicted_ghz = {
        "TE101": 4.798621, "TM110": 5.825889, "TE011": 6.244728,
        "TM111": 6.927523, "TE201": 7.068848, "TM210": 7.803196,
        "TE102": 8.067964,
    }
    for name, (m, n, l), _hint in CV14.TARGET_MODES:
        fy = CV14.yee_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l,
                                  1e-3, 1e-3, 1e-3, 1.9066e-12)
        assert fy / 1e9 == pytest.approx(predicted_ghz[name], abs=1e-6)


def test_yee_oracle_reduces_to_pozar_as_the_mesh_and_step_vanish():
    """First-principles check with no run: sin(x)/x -> 1 and arcsin(x)/x -> 1,
    so the discrete eigenvalue must converge to the continuum one."""
    for name, (m, n, l), _hint in CV14.TARGET_MODES:
        fa = CV14.pozar_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l)
        fy = CV14.yee_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l,
                                  1e-6, 1e-6, 1e-6, 1e-15)
        assert fy == pytest.approx(fa, rel=1e-9), name


def test_yee_oracle_is_second_order_in_cell_size():
    """Halving dx must quarter the Yee-minus-Pozar offset (2nd-order scheme).
    Derivation-level check of the oracle, independent of any simulation."""
    for name, (m, n, l), _hint in CV14.TARGET_MODES:
        fa = CV14.pozar_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l)
        d1 = abs(CV14.yee_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l,
                                      1e-3, 1e-3, 1e-3, 1e-15) - fa)
        d2 = abs(CV14.yee_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l,
                                      0.5e-3, 0.5e-3, 0.5e-3, 1e-15) - fa)
        assert d1 / d2 == pytest.approx(4.0, rel=0.02), name


# --------------------------------------------------------------------------
# Gate 2 -- the audited defect, as synthetic gate math
# --------------------------------------------------------------------------

def test_axis_blind_mode_no_longer_carries_the_whole_gate():
    """THE audited defect. TE011 has m = 0, so it is blind to the x extent.
    Give it a perfect score and put every other mode 10% off: the committed
    ``min()``-over-six gate passed on TE011 alone; ``max()`` over all seven
    must FAIL."""
    bad = {name: CV14.pozar_cavity_freq(CV14.A, CV14.B, CV14.D, m, n, l) * 1.10
           for name, (m, n, l), _h in CV14.TARGET_MODES if name != "TE011"}
    rows = _rows(bad)

    # what the pre-#812 gate computed, replicated here so the regression is
    # anchored to the defect and not merely to the new code
    errs = {n_: abs(f - fa) / fa * 100.0
            for n_, _i, fa, _fy, f, *_r in rows if f is not None}
    higher = {k: v for k, v in errs.items() if k != "TE101"}
    assert min(higher.values()) < 2.0, "old min-gate must be the thing that passed"

    ok, lines = CV14.evaluate_gates(rows, EXACT_EFF, FRES)
    assert not ok
    assert any("Gate 2" in ln and "FAIL" in ln for ln in lines)


def test_not_found_is_a_hard_failure_not_a_silent_drop():
    """A mode that cannot be extracted used to remove itself from ``errs``.
    One missing mode, everything else perfect, must FAIL."""
    ok, lines = CV14.evaluate_gates(_rows({"TM210": None}), EXACT_EFF, FRES)
    assert not ok
    assert any("Gate 2" in ln and "FAIL" in ln and "TM210" in ln for ln in lines)


def test_gate_2_threshold_is_unchanged_at_two_percent():
    """The re-gate moved the AGGREGATOR, never the number. 1.9% on every mode
    still passes; 2.1% on a single mode fails."""
    for pct, expect in ((1.9, True), (2.1, False)):
        # perturb exactly one mode off its POZAR value by pct; the Gate-3
        # budget is inflated so this test isolates Gate 2's threshold
        rows = [(nm, i, fa, fy, (fa * (1 + pct / 100.0) if nm == "TM210" else f),
                 q, a_, e, c)
                for nm, i, fa, fy, f, q, a_, e, c in _rows()]
        ok, _ = CV14.evaluate_gates(rows, EXACT_EFF, FRES * 1e9)
        assert ok is expect, (pct, expect)


# --------------------------------------------------------------------------
# Gate 0 -- wall registration, the quantity the script printed but never gated
# --------------------------------------------------------------------------

def test_gate_0_passes_on_exact_registration():
    ok, lines = CV14.evaluate_gates(_rows(), EXACT_EFF, FRES)
    assert ok
    assert any("Gate 0" in ln and "PASS" in ln for ln in lines)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gate_0_fires_on_a_one_cell_registration_error_on_any_axis(axis):
    """One cell (1 mm) is the smallest geometric error this grid can express.
    Neither Gate 1 (TE101 moves 0.78% for a one-cell x error) nor Gate 2
    (worst mode moves 1.44%) can see it; Gate 0 must."""
    eff = list(EXACT_EFF)
    eff[axis] += 1e-3
    ok, lines = CV14.evaluate_gates(_rows(), tuple(eff), FRES)
    assert not ok
    assert any("Gate 0" in ln and "FAIL" in ln for ln in lines)


def test_gate_3_is_void_not_passing_when_gate_0_fails():
    """The discrete-Yee prediction is exact only for walls on grid planes. When
    Gate 0 fails, Gate 3 must report N/A -- never a PASS that would look like
    corroboration."""
    ok, lines = CV14.evaluate_gates(_rows(), (CV14.A / 2, CV14.B, CV14.D), FRES)
    assert not ok
    g3 = [ln for ln in lines if "Gate 3" in ln]
    assert len(g3) == 1 and "N/A" in g3[0] and "PASS" not in g3[0]


# --------------------------------------------------------------------------
# Gate 3 -- the discrete-Yee residual budget
# --------------------------------------------------------------------------

def test_gate_3_budget_is_one_tenth_of_a_fourier_bin():
    """The budget must follow the record length, not be a frozen MHz number."""
    rows = _rows()
    for fres in (FRES, FRES / 2.0):
        budget = CV14.YEE_BUDGET_BINS * fres
        # a uniform shift of 0.9 * budget on every mode passes ...
        near = {nm: fy + 0.9 * budget for nm, _i, _fa, fy, *_r in rows}
        ok, _ = CV14.evaluate_gates(_rows(near), EXACT_EFF, fres)
        assert ok, fres
        # ... and 1.1 * budget on a SINGLE mode fails
        one = {"TE101": rows[0][3] + 1.1 * budget}
        ok, lines = CV14.evaluate_gates(_rows(one), EXACT_EFF, fres)
        assert not ok, fres
        assert any("Gate 3" in ln and "FAIL" in ln for ln in lines)


def test_gate_3_catches_a_permittivity_error_that_gates_0_1_2_cannot():
    """Detection-power claim, as synthetic gate math (the FDTD version is
    ``test_permittivity_defect_*`` below): a 0.2% cavity-fill permittivity
    error shifts every frequency by -0.0999%. Geometry is untouched, so Gate 0
    passes; 0.1% is far inside Gate 1's 1% and Gate 2's 2%. Only Gate 3 fires."""
    scale = 1.0 / np.sqrt(1.002)
    rows = _rows()
    shifted = {nm: fy * scale for nm, _i, _fa, fy, *_r in rows}
    ok, lines = CV14.evaluate_gates(_rows(shifted), EXACT_EFF, FRES)
    assert not ok
    assert any("Gate 0" in ln and "PASS" in ln for ln in lines)
    assert any("Gate 1" in ln and "PASS" in ln for ln in lines)
    assert any("Gate 2" in ln and "PASS" in ln for ln in lines)
    assert any("Gate 3" in ln and "FAIL" in ln for ln in lines)


# --------------------------------------------------------------------------
# End-to-end falsifiers: real FDTD, both sides of the acceptance criterion
# --------------------------------------------------------------------------

def _leg(build):
    """Run one cv14 leg with ``build`` substituted for ``build_cavity``."""
    import contextlib
    import io
    orig = CV14.build_cavity
    CV14.build_cavity = build
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _pf, gi, _wall = CV14.run_leg(1e-3, 200.0)
    finally:
        CV14.build_cavity = orig
    _shape, _dx, _dz, eff, _dt, _nsteps, fres = gi
    return rows, eff, fres


@pytest.mark.slow
def test_correct_cavity_passes_every_gate():
    """Acceptance criterion (A): the case still passes on today's correct code,
    and with margin. If this is ever close to firing the threshold is wrong,
    not the physics -- the #812 audit measured this solver reproducing all
    seven discrete-Yee eigenvalues to <= 0.15 MHz."""
    rows, eff, fres = _leg(CV14.build_cavity)
    ok, lines = CV14.evaluate_gates(rows, eff, fres)
    assert ok, "\n".join(lines)
    resid = max(abs(f - fy) for _n, _i, _fa, fy, f, *_r in rows)
    assert resid < 0.5 * CV14.YEE_BUDGET_BINS * fres, (
        f"worst discrete-Yee residual {resid/1e6:.3f} MHz is over half the "
        f"{CV14.YEE_BUDGET_BINS * fres/1e6:.3f} MHz budget -- margin lost")


@pytest.mark.slow
def test_minus_50_percent_shrink_now_fails_gate_2():
    """Acceptance criterion (B): the exact defect the #812 audit measured Gate 2
    blind to. ``a`` 50 -> 25 mm with the oracle constants untouched. Before the
    re-gate this failed Gate 1 and still read Gate 2 PASS on the x-blind TE011."""
    from rfx.api import Simulation

    def shrunk(dx):
        sim = Simulation(freq_max=CV14.FREQ_MAX,
                         domain=(CV14.A / 2, CV14.B, CV14.D),
                         boundary="pec", dx=dx)
        sim.add_source((0.0065, 0.011, 0.017), component="ex")
        sim.add_source((0.0095, 0.023, 0.013), component="ey")
        sim.add_source((0.0155, 0.013, 0.023), component="ez")
        sim.add_vector_probe((0.0185, 0.017, 0.029))
        return sim

    rows, eff, fres = _leg(shrunk)

    # the pre-#812 gate, replicated: it PASSED on this defect
    errs = {n_: abs(f - fa) / fa * 100.0
            for n_, _i, fa, _fy, f, *_r in rows if f is not None}
    higher = {k: v for k, v in errs.items() if k != "TE101"}
    assert higher and min(higher.values()) < 2.0, (
        "the old min-over-six gate must still pass here, else this test is no "
        "longer reproducing the audited defect")

    ok, lines = CV14.evaluate_gates(rows, eff, fres)
    assert not ok, "\n".join(lines)
    assert any("Gate 2" in ln and "FAIL" in ln for ln in lines), "\n".join(lines)
    assert any("Gate 0" in ln and "FAIL" in ln for ln in lines), "\n".join(lines)


@pytest.mark.slow
def test_permittivity_defect_is_caught_by_gate_3_alone():
    """Detection power beyond the audited defect, end to end: a 0.2% cavity-fill
    permittivity error leaves the geometry exact and moves every frequency by
    only ~0.1%. Gates 0, 1 and 2 all pass; Gate 3 fails. This is why Gate 3
    exists rather than a tighter percentage in Gate 2."""
    import rfx
    from rfx.api import Simulation

    def defective(dx):
        sim = Simulation(freq_max=CV14.FREQ_MAX, domain=(CV14.A, CV14.B, CV14.D),
                         boundary="pec", dx=dx)
        sim.add_material("air_defect", eps_r=1.002)
        sim.add(rfx.Box((0.0, 0.0, 0.0), (CV14.A, CV14.B, CV14.D)),
                material="air_defect")
        sim.add_source((0.013, 0.011, 0.017), component="ex")
        sim.add_source((0.019, 0.023, 0.013), component="ey")
        sim.add_source((0.031, 0.013, 0.023), component="ez")
        sim.add_vector_probe((0.037, 0.017, 0.029))
        return sim

    rows, eff, fres = _leg(defective)
    ok, lines = CV14.evaluate_gates(rows, eff, fres)
    assert not ok, "\n".join(lines)
    assert any("Gate 0" in ln and "PASS" in ln for ln in lines), "\n".join(lines)
    assert any("Gate 1" in ln and "PASS" in ln for ln in lines), "\n".join(lines)
    assert any("Gate 2" in ln and "PASS" in ln for ln in lines), "\n".join(lines)
    assert any("Gate 3" in ln and "FAIL" in ln for ln in lines), "\n".join(lines)
