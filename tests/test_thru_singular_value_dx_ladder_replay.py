"""Replay gate for the wire-THRU singular-value dx ladder (fast lane, no FDTD).

Locks the frozen record in ``tests/fixtures/thru_singular_value_dx_ladder/``
(three rung JSONs written by ``scripts/diagnostics/thru_singular_value_dx_ladder.py``
on VESSL run 369367257803, plus the harvest round's ``verdict.json``) against the
pre-declaration ``docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md``:

1. **Ladder fences** (note section 2) — the only intended variable is dx; CPML
   physical thickness, physical run time, overhang, port extent and the
   one-cell sheet are held, and the recorded rung parameters say so.
2. **Validity gates G1–G5** (note section 4) — re-checked from the stored
   witnesses (delta vs the battery number, preflight code multiset, settling
   per drive, rasterization counts, provenance).
3. **Re-derivation lock** — sv(f) recomputed from the stored complex S matrices
   must match the stored per-bin singular values; the outcome table of note
   section 3 re-applied to the stored 3 GHz excesses must reproduce
   ``verdict.json`` (verdict C, non-closing).
4. **Gate untouched** — the live ``_THRU_MAX_SINGULAR_VALUE`` is 1.01 and every
   rung's sv_max sits below it.
5. **The note carries the verdict** — the results section quotes the same
   headline numbers, so the prose and the record cannot drift apart silently.

Nothing here runs a solver; a red test means the committed evidence, the
adjudication or the note changed, not the physics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.test_lumped_twoport_vi_validation_battery import _THRU_MAX_SINGULAR_VALUE

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "thru_singular_value_dx_ladder"
NOTE = REPO / "docs" / "design_notes" / "thru_singular_value_dx_ladder_predeclaration.md"

DIVISORS = (1, 2, 4)
DX0_M = 0.5e-3
CPML_THICKNESS_M = 4.0e-3
N_STEPS0 = 4000
OVERHANG_M = 0.5e-3
PORT_EXTENT_M = 1.0e-3
N_BINS = 9

# Pre-declaration constants (sections 3 and 4).
FLOOR = 1e-5
BATTERY_SV_MAX = 1.003227
G1_TOL = 1e-5
BATTERY_CODES = ["pec_faces_finite_pec",
                 "wire_port_dead_extent_cells",
                 "wire_port_dead_extent_cells"]
SETTLING_GATE_DB = -40.0
RFX_SHA = "088281899727fcc644814f0ae9451b6b89a26af8"

# Headline numbers of the record (note section 9.2); a silent edit to the
# JSONs goes red here even if the adjudication is recomputed consistently.
SV_MAX_3GHZ = {1: 1.0032274714899068, 2: 1.0003216974938964, 4: 0.9991541764781098}
E1_OVER_E2 = 10.032628637591465
E2_OVER_E4 = 0.38033642429039277
RECIPROCITY_ABS_MAX = {1: 2.668571372691985e-4, 2: 5.664125618630794e-5,
                       4: 1.2401096935056067e-5}


def _rung(divisor: int) -> dict:
    return json.loads((FIXTURE_DIR / f"rung_dx_over_{divisor}.json").read_text())


def _rungs() -> dict[int, dict]:
    return {d: _rung(d) for d in DIVISORS}


def _verdict() -> dict:
    return json.loads((FIXTURE_DIR / "verdict.json").read_text())


def _floored(x: float) -> float:
    return max(abs(x), FLOOR)


# ---------------------------------------------------------------------------
# 1. Ladder fences and provenance
# ---------------------------------------------------------------------------
def test_ladder_fences_hold_the_physical_quantities() -> None:
    rungs = _rungs()
    t_phys = {d: rungs[d]["rung"]["physical_time_s"] for d in DIVISORS}
    for d in DIVISORS:
        r = rungs[d]["rung"]
        assert r["dx_divisor"] == d
        assert math.isclose(r["dx_m"], DX0_M / d, rel_tol=1e-12)
        assert r["cpml_layers"] == 8 * d
        assert math.isclose(r["cpml_thickness_m"], CPML_THICKNESS_M, rel_tol=1e-12)
        assert r["n_steps"] == N_STEPS0 * d
        assert r["n_steps_fixture"] == N_STEPS0
        assert math.isclose(r["dt_s"], r["dt0_s"] / d, rel_tol=1e-12)
        assert math.isclose(t_phys[d], t_phys[1], rel_tol=1e-12)
        assert r["trace_thickness_cells"] == 1
        assert math.isclose(r["overhang_m"], OVERHANG_M, rel_tol=1e-12)
        assert r["overhang_cells"] == d
        assert math.isclose(r["port_extent_m"], PORT_EXTENT_M, rel_tol=1e-12)
        # The sheet's top face sits one cell above the trace height.
        lo, hi = r["trace_box_lo_m"], r["trace_box_hi_m"]
        assert math.isclose(hi[2] - lo[2], DX0_M / d, rel_tol=1e-9)
        assert len(rungs[d]["freqs_hz"]) == N_BINS
        assert rungs[d]["fixture"]["battery_sv_max"] == BATTERY_SV_MAX


def test_g5_provenance_present_and_pinned() -> None:
    for d in DIVISORS:
        r = _rung(d)
        assert r["study"] == "thru_singular_value_dx_ladder"
        assert r["predeclaration"] == (
            "docs/design_notes/thru_singular_value_dx_ladder_predeclaration.md")
        p = r["provenance"]
        assert p["git_sha"] == RFX_SHA
        assert p["x64_enabled"] is False
        assert p["jax_backend"] == "gpu"
        for key in ("jax_version", "python", "timestamp_utc"):
            assert p[key]
        assert r["wall_time_s"]["run"] > 0.0
        assert r["wall_time_s"]["total"] >= r["wall_time_s"]["run"]
        assert r["preflight"]["messages_verbatim"]
        assert r["warnings_verbatim"]


# ---------------------------------------------------------------------------
# 2. Validity gates G1-G4
# ---------------------------------------------------------------------------
def test_g1_dx_rung_reproduces_the_battery_number() -> None:
    r = _rung(1)
    sv = r["singular_values"]
    assert abs(sv["delta_vs_battery_sv_max"]) < G1_TOL
    assert math.isclose(sv["sv_max"] - BATTERY_SV_MAX, sv["delta_vs_battery_sv_max"],
                        rel_tol=0.0, abs_tol=1e-12)
    assert r["preflight"]["codes"] == sorted(BATTERY_CODES)
    assert r["preflight"]["battery_codes_present"] is True
    assert r["preflight"]["extra_codes"] == []


def test_g2_every_rung_carries_the_battery_advisory_multiset() -> None:
    for d in DIVISORS:
        pf = _rung(d)["preflight"]
        assert pf["codes"] == sorted(BATTERY_CODES), d
        assert pf["battery_codes_present"] is True, d
        assert pf["extra_codes"] == [], d
        assert len(pf["messages_verbatim"]) == len(BATTERY_CODES), d
        # The wire-port advisory must state this rung's own rasterization.
        n_cells, n_live = 2 * d + 1, 2 * d
        port_msgs = [m for m in pf["messages_verbatim"] if m.startswith("Wire port at")]
        assert len(port_msgs) == 2, d
        for m in port_msgs:
            assert f"rasterizes to n={n_cells} cells" in m, (d, m[:80])
            assert f"(n_live/n = {n_live}/{n_cells})" in m, (d, m[:80])


def test_g3_settling_below_minus_40_db_at_every_rung() -> None:
    for d in DIVISORS:
        s = _rung(d)["settling"]
        assert len(s["per_drive"]) == 2, d
        for drive in s["per_drive"]:
            assert drive["settling_db"] <= SETTLING_GATE_DB, (d, drive)
            assert drive["settling_db"] == max(drive["settling_db_per_probe"])
            assert drive["record_shape"] == [N_STEPS0 * d, 3], (d, drive)
        assert s["main_pass_settling_db"] <= SETTLING_GATE_DB, d
        assert s["probe_labels"] == ["port1_gap", "mid_line", "port2_gap"]


def test_g4_rasterization_scales_as_a_sheet() -> None:
    rungs = _rungs()
    pec = [rungs[d]["rasterization"]["finite_pec_cells"] for d in DIVISORS]
    assert pec == [340, 1360, 5440]
    assert pec[1] == 4 * pec[0] and pec[2] == 4 * pec[1]
    for d in DIVISORS:
        for port in rungs[d]["rasterization"]["wire_ports"]:
            assert port["n_cells"] == 2 * d + 1, d
            assert port["n_live"] == 2 * d, d
            flags = port["live_flags"]
            assert flags[-1] is False and all(flags[:-1]), (d, flags)
        assert rungs[d]["rasterization"]["grid_shape"] == {
            1: [81, 57, 29], 2: [161, 113, 57], 4: [321, 225, 113]}[d]


# ---------------------------------------------------------------------------
# 3. Re-derivation lock: singular values and the outcome table
# ---------------------------------------------------------------------------
def _stored_s(r: dict) -> np.ndarray:
    return np.asarray(r["s_matrix"]["re"]) + 1j * np.asarray(r["s_matrix"]["im"])


@pytest.mark.parametrize("divisor", DIVISORS)
def test_singular_values_rederive_from_the_stored_s_matrix(divisor: int) -> None:
    r = _rung(divisor)
    S = _stored_s(r)
    assert S.shape == (2, 2, N_BINS)
    sv = r["singular_values"]
    for k in range(N_BINS):
        s_max, s_min = np.linalg.svd(S[:, :, k], compute_uv=False)
        assert math.isclose(s_max, sv["max_per_bin"][k], rel_tol=1e-9), (divisor, k)
        assert math.isclose(s_min, sv["min_per_bin"][k], rel_tol=1e-9), (divisor, k)
    assert sv["sv_max"] == max(sv["max_per_bin"])
    assert sv["sv_max_freq_hz"] == 3e9
    assert math.isclose(sv["excess_3ghz"], sv["max_per_bin"][0] - 1.0,
                        rel_tol=0.0, abs_tol=1e-15)
    assert sv["monotone_decreasing_in_f"] is True
    assert all(np.diff(sv["max_per_bin"]) < 0)
    assert math.isclose(sv["max_per_bin"][0], SV_MAX_3GHZ[divisor], rel_tol=1e-12)
    # abs_s and the column-power / reciprocity witnesses derive from the same S.
    a = r["abs_s"]
    np.testing.assert_allclose(a["s11"], np.abs(S[0, 0]), rtol=1e-9)
    np.testing.assert_allclose(a["s22"], np.abs(S[1, 1]), rtol=1e-9)
    np.testing.assert_allclose(a["s21"], np.abs(S[1, 0]), rtol=1e-9)
    np.testing.assert_allclose(a["s12"], np.abs(S[0, 1]), rtol=1e-9)
    col = [np.abs(S[0, 0]) ** 2 + np.abs(S[1, 0]) ** 2,
           np.abs(S[1, 1]) ** 2 + np.abs(S[0, 1]) ** 2]
    np.testing.assert_allclose(r["column_power"], col, rtol=1e-9)
    np.testing.assert_allclose(r["reciprocity_abs"], np.abs(S[1, 0] - S[0, 1]), rtol=1e-9)
    assert math.isclose(r["reciprocity_abs_max"], max(r["reciprocity_abs"]), rel_tol=1e-12)


def test_outcome_table_reapplied_gives_verdict_c() -> None:
    rungs = _rungs()
    v = _verdict()
    e = {d: rungs[d]["singular_values"]["excess_3ghz"] for d in DIVISORS}
    assert v["floor"] == FLOOR
    for d, key in zip(DIVISORS, ("dx", "dx_over_2", "dx_over_4")):
        assert e[d] == v["excess_3ghz"][key], d
    # Signs: + + - (the flip on the second halving).
    assert e[1] > 0 and e[2] > 0 and e[4] < 0
    assert abs(e[4]) > FLOOR
    assert v["abs_e4_above_floor"] is True
    r12 = _floored(e[1]) / _floored(e[2])
    r24 = _floored(e[2]) / _floored(e[4])
    assert math.isclose(r12, v["ratio_floored_magnitudes"]["e1_over_e2"], rel_tol=1e-12)
    assert math.isclose(r24, v["ratio_floored_magnitudes"]["e2_over_e4"], rel_tol=1e-12)
    assert math.isclose(r12, E1_OVER_E2, rel_tol=1e-9)
    assert math.isclose(r24, E2_OVER_E4, rel_tol=1e-9)
    assert r12 >= 2.0          # the first pair alone meets the A condition
    assert r24 < 1.25          # the second pair does not (and flips sign)
    same_sign = len({math.copysign(1.0, e[d]) for d in DIVISORS}) == 1
    outcome_a = same_sign and r12 >= 2.0 and r24 >= 2.0
    spread = max(e.values()) - min(e.values())
    outcome_b = spread < 0.20 * e[1]
    assert outcome_a is False and v["outcome_a_discretization"] is False
    assert outcome_b is False and v["outcome_b_refuted"] is False
    assert math.isclose(spread, v["outcome_b_spread"], rel_tol=1e-12)
    assert v["verdict"] == "C"
    assert "non-closing" in v["verdict_text"]
    assert "STOP" in v["verdict_text"]
    assert v["rfx_sha"] == RFX_SHA
    assert v["vessl_run_id"] == "369367257803"


def test_every_bin_and_the_unity_crossing() -> None:
    rungs = _rungs()
    v = _verdict()
    for d, key in zip(DIVISORS, ("dx", "dx_over_2", "dx_over_4")):
        sv = rungs[d]["singular_values"]["max_per_bin"]
        f = rungs[d]["freqs_hz"]
        crossings = [[f[k], f[k + 1]] for k in range(N_BINS - 1)
                     if (sv[k] - 1.0) * (sv[k + 1] - 1.0) < 0]
        assert crossings == v["unity_crossing_hz"][key], (d, crossings)
    assert v["unity_crossing_hz"]["dx"] == [[4.5e9, 5.0e9]]
    assert v["unity_crossing_hz"]["dx_over_2"] == [[3.0e9, 3.5e9]]
    assert v["unity_crossing_hz"]["dx_over_4"] == []
    assert all(x < 1.0 for x in rungs[4]["singular_values"]["max_per_bin"])
    assert v["all_bins_below_unity_at_dx_over_4"] is True
    # The whole sv_max(f) curve moves down with each halving (record fact).
    sv1, sv2, sv4 = (np.asarray(rungs[d]["singular_values"]["max_per_bin"]) for d in DIVISORS)
    np.testing.assert_allclose(sv1 - sv2, v["successive_difference_sv_max_per_bin"]["dx_to_dx2"],
                               rtol=1e-12)
    np.testing.assert_allclose(sv2 - sv4, v["successive_difference_sv_max_per_bin"]["dx2_to_dx4"],
                               rtol=1e-12)
    assert np.all(sv1 > sv2) and np.all(sv2 > sv4)


def test_column_power_and_reciprocity_moved_with_dx_as_recorded() -> None:
    rungs = _rungs()
    v = _verdict()
    for d, key in zip(DIVISORS, ("dx_over_1", "dx_over_2", "dx_over_4")):
        cp = rungs[d]["column_power"][0] + rungs[d]["column_power"][1]
        assert 0.95 < min(cp) and max(cp) < 1.0, d
        assert math.isclose(max(cp), v["column_power_max"][key], rel_tol=1e-12)
        assert math.isclose(min(cp), v["column_power_min"][key], rel_tol=1e-12)
        assert math.isclose(rungs[d]["reciprocity_abs_max"], v["reciprocity_abs_max"][key],
                            rel_tol=1e-12)
        assert math.isclose(rungs[d]["reciprocity_abs_max"], RECIPROCITY_ABS_MAX[d],
                            rel_tol=1e-9)
    rec = [rungs[d]["reciprocity_abs_max"] for d in DIVISORS]
    assert rec[0] / rec[1] > 4.0 and rec[1] / rec[2] > 4.0


# ---------------------------------------------------------------------------
# 4. The gate stays; 5. the note carries the verdict
# ---------------------------------------------------------------------------
def test_gate_untouched_and_never_binding_on_the_ladder() -> None:
    assert _THRU_MAX_SINGULAR_VALUE == 1.01
    v = _verdict()
    assert v["gate_thru_max_singular_value"] == _THRU_MAX_SINGULAR_VALUE
    for d in DIVISORS:
        assert _rung(d)["singular_values"]["sv_max"] < _THRU_MAX_SINGULAR_VALUE, d
    assert v["sv_max_below_gate_at_every_rung"] is True


def test_note_results_section_quotes_the_record() -> None:
    text = NOTE.read_text()
    head, sep, results = text.partition("## 9. RESULTS")
    assert sep, "results section missing from the pre-declaration note"
    assert "## 3. Outcome table" in head           # the table predates the results
    assert "Verdict: C" in results
    assert "non-closing" in results
    for d in DIVISORS:
        assert f"{SV_MAX_3GHZ[d]:.7f}" in results, d
    assert "10.03" in results and "8.46e-4" in results
    assert "_THRU_MAX_SINGULAR_VALUE = 1.01" in results
    assert "369367257803" in results
