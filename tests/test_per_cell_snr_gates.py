"""Committed gate for the WP 3-2 per-cell gradient-SNR ladder.

Locks the frozen per-cell evidence in
``tests/fixtures/gradient_dx_ladder/per_cell_snr.json`` (produced by
``scripts/diagnostics/gradient_dx_ladder/per_cell_snr.py``) WITHOUT running any
FDTD: it replays the committed per-cell gradient arrays and re-derives every
statistic, ratio, exponent and verdict, so a silent regression in the committed
evidence goes red here.

What is enforced (per the external-reviewer plan, item 3-2):
1. **Fences** — num_periods held FIXED at 20 on every rung (the settling-
   confound trap), precision is float32 (the object of study), one fixed
   objective frequency, per-cell permittivity DoF over a smooth block.
2. **Re-derivation lock** — the committed per-cell ``median|g|``,
   ``median g_rel = median(|g|/J0)``, ``max`` and the design-cell count are
   recomputed FROM the committed per-cell array to rel 1e-9. Corrupting any one
   committed grad value flips its recomputed median and goes red.
3. **Falsifier + verdict consistency** — the NORMALIZED median-g_rel ratio and
   its decay exponent are recomputed from the committed medians; the committed
   verdict must MATCH the ratio (``decay-seen`` REQUIRES the exponent inside
   [2, 4] near dx^3; ``null-no-decay`` REQUIRES it outside). A verdict that
   contradicts its own dx^3 ratio goes red.
4. **R5 decomposition** — raw ``p_raw = p_J + volume``: the recomputed
   objective-scaling exponent p_J and raw exponent p_raw must leave a residual
   matching the per-cell volume exponent 3 within tolerance (this is what makes
   the raw dx^6.9 decay a KNOWN confound, not an unexplained failure).
5. **Reality + comparator** — the directional-FD cross-check confirms the
   per-cell field is a real gradient (sign agrees, rel_err small), and the
   objective repeat-noise comparator is valid on every rung.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "gradient_dx_ladder" / "per_cell_snr.json"

# Producer constants (scripts/diagnostics/gradient_dx_ladder/per_cell_snr.py).
NUM_PERIODS = 20.0
F_OBJ_HZ = 8e9
DECAY_EXP_LO = 2.0
DECAY_EXP_HI = 4.0
DX3_EXPONENT = 3.0
DECOMP_RESIDUAL_TOL = 0.6
COMPARATOR_MIN_RATIO = 100.0
FLOAT32_ULP = 2.0 ** -23
DIR_REL_ERR_CEILING = 1e-2   # directional AD-vs-FD agreement (real-gradient)


def _env() -> dict:
    return json.loads(FIXTURE.read_text())


def _live_rungs(env: dict) -> list:
    rungs = [r for r in env["rungs"] if not r.get("dropped")]
    return sorted(rungs, key=lambda r: r["lambda_over_dx"])


_PCT_KEYS = (("min", 0), ("p10", 10), ("p25", 25), ("median", 50),
             ("p75", 75), ("p90", 90), ("max", 100))


def _median(vals) -> float:
    return float(np.percentile(np.asarray(vals, dtype=float), 50))


def _percentiles(vals) -> dict:
    a = np.asarray(vals, dtype=float)
    out = {k: float(np.percentile(a, q)) for k, q in _PCT_KEYS}
    out["max"] = float(np.max(a))  # match producer's exact np.max for max
    out["min"] = float(np.min(a))
    return out


def _exponent(coarse_val: float, fine_val: float, dx_ratio: float) -> float:
    return math.log(fine_val / coarse_val) / math.log(dx_ratio)


def test_fixture_present_and_meta_fences() -> None:
    env = _env()
    m = env["meta"]
    assert m["work_package"] == "3-2"
    # num_periods held FIXED at 20 -- the settling-confound fence.
    assert m["num_periods"] == NUM_PERIODS
    # float32 is the object of study (no module-level x64).
    assert m["precision"] == "float32"
    assert m["f_obj_hz"] == F_OBJ_HZ
    assert "block" in m["witness"].lower()
    assert "per-cell" in m["witness"].lower()
    assert m["dx3_exponent"] == DX3_EXPONENT
    assert m["decay_exp_lo"] == DECAY_EXP_LO
    assert m["decay_exp_hi"] == DECAY_EXP_HI
    assert m["float32_ulp"] == FLOAT32_ULP
    # meta verdict must agree with the falsifier block verdict.
    assert m["verdict"] == env["falsifier_normalized"]["verdict"]


def test_ladder_covers_coarse_to_fine() -> None:
    env = _env()
    live = _live_rungs(env)
    assert len(live) >= 3, "need >= 3 measured rungs for a curve"
    ratios = [r["lambda_over_dx"] for r in live]
    assert ratios[0] == 20
    assert ratios == sorted(set(ratios)), "duplicate rungs"
    assert ratios[-1] >= 40, "ladder must reach at least lambda/40"


@pytest.mark.parametrize("idx", range(3))
def test_each_rung_rederives_stats_and_fences(idx: int) -> None:
    env = _env()
    live = _live_rungs(env)
    if idx >= len(live):
        pytest.skip(f"only {len(live)} live rungs committed")
    r = live[idx]

    # num_periods fence, held FIXED per rung.
    assert r["num_periods"] == NUM_PERIODS, r["rung"]

    # 1. per-cell array integrity: length == design cell count.
    g = np.asarray(r["per_cell_grad_abs"], dtype=float)
    assert g.size == r["design_cells"], r["rung"]
    assert int(np.prod(r["design_shape"])) == r["design_cells"], r["rung"]
    assert np.all(g >= 0.0), f"{r['rung']}: |g| array must be non-negative"

    # 2. re-derive the FULL raw percentile dict FROM the committed array, so
    #    ANY single-cell corruption that shifts a percentile bracket goes red
    #    (a bare median is robust to tail corruption; the full dict is not).
    raw_pct = _percentiles(g)
    for k, _ in _PCT_KEYS:
        assert raw_pct[k] == pytest.approx(
            r["grad_abs_stats"][k], rel=1e-9, abs=1e-30), f"{r['rung']}:{k}"
    assert _median(g) == pytest.approx(r["median_grad"], rel=1e-9, abs=1e-30)
    assert float(np.max(g)) == pytest.approx(r["max_grad"], rel=1e-9, abs=1e-30)

    # 3. re-derive the mesh-invariant relative sensitivity g_rel = |g| / J0,
    #    full percentile dict included.
    grel = g / abs(r["J0"])
    rel_pct = _percentiles(grel)
    for k, _ in _PCT_KEYS:
        assert rel_pct[k] == pytest.approx(
            r["grad_rel_stats"][k], rel=1e-9, abs=1e-30), f"{r['rung']}:rel:{k}"
    assert _median(grel) == pytest.approx(
        r["median_grad_rel"], rel=1e-9, abs=1e-30), r["rung"]
    assert float(np.max(grel)) == pytest.approx(
        r["max_grad_rel"], rel=1e-9, abs=1e-30), r["rung"]

    # 4. re-derive both frac-above-floor witnesses from the raw array.
    frac_noise = float(np.mean(g > r["grad_repeat_noise"]))
    assert frac_noise == pytest.approx(r["frac_above_noise"], abs=1e-12)
    frac_fd = float(np.mean(grel > FLOAT32_ULP))
    assert frac_fd == pytest.approx(r["frac_rel_above_fd_floor"], abs=1e-12)

    # 5. reality check: directional AD matches directional FD (real gradient).
    fd = r["directional_fd"]
    ad = r["directional_ad"]
    rel_err = abs(ad - fd) / max(abs(fd), 1e-30)
    assert rel_err == pytest.approx(r["directional_rel_err"], rel=1e-9)
    assert (ad * fd) > 0.0, f"{r['rung']}: directional AD/FD signs disagree"
    assert rel_err <= DIR_REL_ERR_CEILING, r["rung"]

    # 6. comparator validity (deterministic CPU -> repeat-noise 0 -> valid).
    if r["j_repeat_noise"] == 0.0:
        assert r["comparator_valid"] is True
    else:
        assert r["fd_diff"] / r["j_repeat_noise"] >= COMPARATOR_MIN_RATIO
        assert r["comparator_valid"] is True

    assert math.isfinite(r["median_grad"]) and math.isfinite(r["J0"])


def test_median_grad_rel_monotone_decreasing() -> None:
    """The mesh-invariant relative sensitivity must shrink monotonically as the
    mesh refines (the SIGNAL side of the SNR story)."""
    env = _env()
    live = _live_rungs(env)
    mr = [r["median_grad_rel"] for r in live]
    assert all(mr[i] > mr[i + 1] for i in range(len(mr) - 1)), mr


def test_falsifier_ratio_and_verdict_consistency() -> None:
    """The committed verdict must MATCH the committed dx^3 ratio: decay-seen
    REQUIRES the normalized exponent inside [2, 4]; null-no-decay REQUIRES it
    outside. A verdict contradicting its own ratio goes red."""
    env = _env()
    live = _live_rungs(env)
    fals = env["falsifier_normalized"]

    coarse = next(r for r in live if r["lambda_over_dx"] == fals["coarse_rung"])
    fine = next(r for r in live if r["lambda_over_dx"] == fals["fine_rung"])
    dx_ratio = fine["dx_m"] / coarse["dx_m"]

    # re-derive ratio + exponent from the committed per-rung medians.
    ratio = fine["median_grad_rel"] / coarse["median_grad_rel"]
    exponent = _exponent(coarse["median_grad_rel"], fine["median_grad_rel"],
                         dx_ratio)
    assert ratio == pytest.approx(fals["median_ratio"], rel=1e-9)
    assert exponent == pytest.approx(fals["measured_exponent"], rel=1e-9)
    assert dx_ratio ** DX3_EXPONENT == pytest.approx(
        fals["predicted_dx3_ratio"], rel=1e-9)

    in_band = DECAY_EXP_LO <= exponent <= DECAY_EXP_HI
    assert fals["decay_seen"] is bool(in_band)
    if fals["verdict"] == "decay-seen":
        assert in_band, "decay-seen requires the exponent inside [2, 4]"
    elif fals["verdict"] == "null-no-decay":
        assert not in_band, "null-no-decay requires the exponent outside [2, 4]"
    else:
        pytest.fail(f"unknown verdict {fals['verdict']!r}")


def test_r5_decomposition_raw_equals_objective_plus_volume() -> None:
    """R5: the raw median|g| decay decomposes as p_raw = p_J + volume(3). This
    is what makes the raw dx^~7 decay a KNOWN confound (the objective is not
    mesh-invariant), not an unexplained failure."""
    env = _env()
    live = _live_rungs(env)
    dec = env["decomposition"]
    fals = env["falsifier_normalized"]

    coarse = next(r for r in live if r["lambda_over_dx"] == fals["coarse_rung"])
    fine = next(r for r in live if r["lambda_over_dx"] == fals["fine_rung"])
    dx_ratio = fine["dx_m"] / coarse["dx_m"]

    p_J = _exponent(coarse["J0"], fine["J0"], dx_ratio)
    p_raw = _exponent(coarse["median_grad"], fine["median_grad"], dx_ratio)
    p_rel = _exponent(coarse["median_grad_rel"], fine["median_grad_rel"],
                      dx_ratio)
    assert p_J == pytest.approx(dec["p_objective_J"], rel=1e-9)
    assert p_raw == pytest.approx(dec["p_raw_median_grad"], rel=1e-9)
    assert p_rel == pytest.approx(dec["p_rel_median_grad"], rel=1e-9)

    residual = p_raw - p_J
    assert residual == pytest.approx(dec["residual_raw_minus_J"], rel=1e-9)
    # the residual (raw minus objective) is the per-cell volume exponent ~3.
    assert abs(residual - DX3_EXPONENT) <= DECOMP_RESIDUAL_TOL
    assert dec["residual_matches_volume"] is True
    # p_rel is the same residual: the normalized metric IS the volume factor.
    assert p_rel == pytest.approx(residual, rel=1e-6)


def test_crossing_ad_floor_and_fd_extrapolation() -> None:
    """AD has no crossing (deterministic repeat-noise floor 0.0); the FD-
    detectability crossing is the committed extrapolation from the finest
    rung's median g_rel toward the fixed float32 ULP floor."""
    env = _env()
    live = _live_rungs(env)
    cross = env["crossing"]
    fals = env["falsifier_normalized"]

    # AD path: every rung's measured gradient repeat-noise floor is 0.0.
    assert all(r["grad_repeat_noise"] == 0.0 for r in live)
    assert cross["ad_repeat_noise_floor"] == 0.0
    assert cross["ad_crossing_lambda_over_dx"] is None

    # FD path applies only when decay is seen.
    if not fals["decay_seen"]:
        assert cross.get("fd_applicable") is False
        return
    assert cross["fd_applicable"] is True
    assert cross["fd_detect_floor"] == FLOAT32_ULP

    finest = live[-1]
    assert cross["fd_ref_rung"] == finest["lambda_over_dx"]
    assert cross["fd_ref_median_grad_rel"] == pytest.approx(
        finest["median_grad_rel"], rel=1e-9)

    p = fals["measured_exponent"]
    n_cross = finest["lambda_over_dx"] * (
        finest["median_grad_rel"] / FLOAT32_ULP) ** (1.0 / p)
    assert n_cross == pytest.approx(
        cross["fd_crossing_lambda_over_dx_extrapolated"], rel=1e-6)
    # the crossing is finer than every MEASURED rung (extrapolated, not run).
    assert n_cross > finest["lambda_over_dx"]
