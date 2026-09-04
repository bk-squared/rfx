"""An extremum must be LIT before its position is a measurement (#888 lane).

WHAT THIS CLOSES. cv04's spectral mask admits a bin to the analysis band at
2 percent of the incident power peak -- the right floor for a band mean, and
nowhere near enough for an extremum POSITION. The case shipped gating its third
fringe lit at 3.4 percent, reading it 320 MHz away from where the Yee lattice
puts it, and that reading swung 193 MHz when the auxiliary absorber was fixed
underneath it. The gate had an edge check (the analytic extremum plus its own
window plus a bin must fit in the band) and no brightness check.

WHAT WAS MEASURED. cv04's own rig, source bandwidth swept, nothing else changed:

    inc power at fringe 3   0.0336  0.0735  0.1313  0.2037  0.2855  0.4571  0.8414
    position error (MHz)     319.7   306.5   187.2    80.9    22.0     2.7     0.0

against the measurement's own resolution, df_bin/2 = 26.1 MHz. The error falls
under it at an incident power of 0.277; the admission bar is that carried
through the shared gate policy's multiplier, 0.277 * 1.5 = 0.416 -> 0.42.

WHAT IS NOT THE CRITERION, tested and rejected: containment of the half-fringe
search cell. At bw = 0.55 the third fringe's cell is 94 percent inside the band
and the gate still fails by +245.8 MHz; from bw = 0.65 to 1.10 the band top is
identical and the position still moves 81 MHz as the illumination rises.

Derivation and every number:
``docs/design_notes/20260904_aux_absorber_depth_derivation.md`` section 10.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_CV = _REPO / "validation" / "crossval"


def _load(name: str, rel: Path):
    spec = importlib.util.spec_from_file_location(name, rel)
    mod = importlib.util.module_from_spec(spec)
    # frozen dataclasses resolve their own module out of sys.modules at class
    # creation, so register before exec or FringeRow/FringeVerdict blow up
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FG = _load("cv04_fringe_gate_meas", _CV / "comparators" / "fringe_gate.py")
GP = _load("cv04_gate_policy", _REPO / "tests" / "_gate_policy.py")

# The sweep above, as data: (inc power at fringe 3, |position - converged| in Hz).
MEASURED_SWEEP = (
    (0.0336, 319.7e6), (0.0735, 306.5e6), (0.1313, 187.2e6), (0.2037, 80.9e6),
    (0.2855, 22.0e6), (0.4571, 2.7e6), (0.8414, 0.0),
)
DF_BIN_HZ = 52.276988636458154e6
# cv04's realized incident power at each analytic extremum on the DECLARED rig
# (bw = 0.8), printed by the case itself.
CV04_LIT_AT_EXTREMA = {3.7475e9: 0.8761, 7.4950e9: 0.9093, 11.2425e9: 0.4571}
# ... and on the rig that shipped (bw = 0.5).
SHIPPED_LIT_AT_EXTREMA = {3.7475e9: 0.9965, 7.4950e9: 0.3704, 11.2425e9: 0.0336}


def test_the_bar_is_the_measured_threshold_through_the_shared_policy():
    """The number is derived, not chosen, and it is derived from the sweep."""
    below = [p for p, err in MEASURED_SWEEP if err > DF_BIN_HZ / 2.0]
    above = [p for p, err in MEASURED_SWEEP if err <= DF_BIN_HZ / 2.0]
    assert max(below) < FG.FRINGE_INC_POWER_MEASURABLE <= min(above), (
        "the declared measurable threshold must sit in the gap the sweep leaves "
        f"between {max(below)} (error over df_bin/2) and {min(above)} (under it)")
    assert FG.FRINGE_INC_POWER_MIN == pytest.approx(
        math.ceil(FG.FRINGE_INC_POWER_MEASURABLE * GP.ENVELOPE_GATE_MULTIPLIER * 100) / 100)
    assert FG.FRINGE_INC_POWER_MIN > FG.FRINGE_INC_POWER_MEASURABLE


def test_the_bar_is_far_above_the_masks_own_floor():
    """The two floors answer different questions and the gate needs both: 2 % of
    peak admits a bin to a band MEAN, 42 % admits an extremum to a POSITION."""
    assert FG.FRINGE_INC_POWER_MIN > 14.0 * 0.02


def test_the_declared_rig_lights_every_extremum_and_the_shipped_one_did_not():
    for f, lit in CV04_LIT_AT_EXTREMA.items():
        assert lit >= FG.FRINGE_INC_POWER_MIN, (f, lit)
    dim = {f: p for f, p in SHIPPED_LIT_AT_EXTREMA.items() if p < FG.FRINGE_INC_POWER_MIN}
    assert set(dim) == {7.4950e9, 11.2425e9}, dim


def _synthetic_band(f_lo=3.0e9, f_hi=15.0e9, n=229, eps_r=4.0, d=10e-3):
    f = np.linspace(f_lo, f_hi, n)
    n_idx = math.sqrt(eps_r)
    # a lossless slab's R(f), exactly the shape the gate anchors on
    delta = 2.0 * math.pi * f * n_idx * d / FG.C0_DEFAULT
    r = (1 - n_idx) / (1 + n_idx)
    R = np.abs((r * (1 - np.exp(-2j * delta))) / (1 - r ** 2 * np.exp(-2j * delta))) ** 2
    return f, R


def test_a_dim_extremum_is_refused_not_judged():
    """FALSIFIER: the same curve, judged twice, differing only in illumination.

    Brightly lit everywhere the gate judges all three extrema. With the third
    one dimmed under the bar it is REPORTED N/A -- not silently dropped, and not
    read off anyway."""
    f, R = _synthetic_band()
    bright = np.ones_like(f)
    v_bright = FG.compare_fringes(f, R, eps_r=4.0, d=10e-3, n_index=2.0, dx=1e-3,
                                  dt=2.335067793382187e-12, df_bin_hz=DF_BIN_HZ,
                                  inc_power_rel=bright, label="bright")
    assert len(v_bright.rows) == 3 and not v_bright.not_applicable

    dim = np.where(f > 10.5e9, 0.05, 1.0)
    v_dim = FG.compare_fringes(f, R, eps_r=4.0, d=10e-3, n_index=2.0, dx=1e-3,
                               dt=2.335067793382187e-12, df_bin_hz=DF_BIN_HZ,
                               inc_power_rel=dim, label="dim")
    assert len(v_dim.rows) == 2
    assert [k for k, _f, _p in v_dim.not_applicable] == ["max"]
    kind, f_na, lit = v_dim.not_applicable[0]
    assert f_na == pytest.approx(11.2425e9, rel=1e-3)
    assert lit < FG.FRINGE_INC_POWER_MIN
    assert "too dim to measure" in FG.format_fringe_table(v_dim, "dim")
    assert "N/A" in FG.format_fringe_table(v_dim, "dim")


def test_omitting_the_illumination_keeps_the_old_behaviour():
    """The argument is optional, so every other consumer of this comparator is
    unchanged until it passes one."""
    f, R = _synthetic_band()
    v = FG.compare_fringes(f, R, eps_r=4.0, d=10e-3, n_index=2.0, dx=1e-3,
                           dt=2.335067793382187e-12, df_bin_hz=DF_BIN_HZ, label="no-inc")
    assert len(v.rows) == 3 and not v.not_applicable


def test_the_case_treats_a_refused_extremum_as_a_FAIL_not_a_smaller_gate():
    """The starvation hole: refusing an extremum makes the GATE smaller, and a
    smaller gate must not read as a greener case. cv04 declares which extrema
    its rig measures and fails when one goes N/A."""
    src = (_CV / "04_multilayer_fresnel.py").read_text(encoding="utf-8")
    assert "CV04_DECLARED_EXTREMA" in src
    assert "fringe_admission_ok" in src
    assert "fringe_ok = bool(fringe_verdict.ok) and fringe_admission_ok" in src
    assert "inc_power_rel=inc_power[mask] / inc_power.max()" in src
    assert "bw = 0.8" in src


# ==========================================================================
# cv04's exact-lattice witness, GATED on R (#888 lane, PI decision 2026-09-04)
# ==========================================================================
# It was REPORTED, not gated, and the reason given in
# docs/design_notes/20260903_lattice_witness_standard.md section 5.3 was that a
# gate which cannot reject the continuum model is not a gate: the continuum
# falsifier separated 0.099 of the window on 0 of 115 bins. The derived absorber
# and the derived rig moved that to 2.68 on 65 of 115, and W_witness_R from
# above its own ceiling to 2.4x inside it. R is gated here; T and A are not,
# because W_witness_T still exceeds its ceiling and the same falsifier separates
# only 0.38 there.

_LW = _load("cv04_lattice_witness", _CV / "comparators" / "lattice_witness.py")
_W04 = _CV / "_04_fresnel_results" / "lattice_witness.json"


def _cv04_rung() -> dict:
    import json
    return json.loads(_W04.read_text(encoding="utf-8"))


def test_the_artifact_says_R_is_gated_and_T_is_not():
    doc = _cv04_rung()
    assert doc["gated_here"] is True
    assert doc["gated_channels"] == ["R"]
    assert doc["reported_channels"] == ["T", "A"]
    assert "PI decision 2026-09-04" in doc["gated_here_reason"]


def test_cv04s_lattice_witness_passes_on_R():
    r = _cv04_rung()["rungs"]["slab_eps4"]
    for key in ("precond_cpml_gate", "precond_tail_witness", "precond_aux_echo_record",
                "GL1_R", "GL2_R"):
        assert r["gates"][key] is True, (key, r["gates"])
    assert r["W_exceeds_ceiling_R"] is False
    assert r["n_bins_R_over_window"] == 0
    assert r["mean_dR_lattice_gated"] < r["mean_W_witness_R_gated"]


def test_the_R_window_is_inside_its_ceiling_and_tighter_than_the_cases_own():
    """The two conditions that were false when the witness was declared
    non-discriminating, and are the reason it now decides."""
    r = _cv04_rung()["rungs"]["slab_eps4"]
    assert r["mean_W_witness_R_gated"] < r["mean_W_ceiling_R_gated"]
    W_MEAN_R = 0.010   # cv22_dispersive_gates.py, the family's band-mean window
    assert r["mean_W_witness_R_gated"] < W_MEAN_R / 4.0


def test_T_is_still_the_loose_channel_and_is_NOT_gated():
    """Turning R on must not quietly turn T on: T's window still exceeds its own
    ceiling, which is exactly the condition that disqualified R before."""
    r = _cv04_rung()["rungs"]["slab_eps4"]
    assert r["W_exceeds_ceiling_T"] is True
    assert r["mean_W_witness_T_gated"] > r["mean_W_ceiling_T_gated"]
    assert "T" in _cv04_rung()["reported_channels"]


def test_the_case_exits_nonzero_when_the_R_witness_fails():
    """The gate has teeth in the case, not only in this file."""
    src = (_CV / "04_multilayer_fresnel.py").read_text(encoding="utf-8")
    assert "lattice_R_ok" in src
    assert "if not lattice_R_ok:" in src
    assert 'W_exceeds_ceiling_R' in src
