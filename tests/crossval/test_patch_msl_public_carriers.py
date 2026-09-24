"""Current patch/MSL prose must quote committed measurements, not retired runs.

These checks only read artifacts and source text. Historical measurements may
remain when explicitly scoped to the old run; the current-result sections must
follow the committed artifacts without changing any physics gate tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _json(path):
    return json.loads((ROOT / path).read_text())


def _text(path):
    return (ROOT / path).read_text()


@pytest.mark.docs_consistency
def test_patch_demo_directivity_comment_uses_current_measurement():
    artifact = _json("tests/fixtures/patch_canonical_farfield_e4/"
                     "canonical_farfield_e4_measured_369367259302.json")
    current = _text("examples/tutorials/patch_antenna_demo.py").split(
        "# NTFF box.", 1)[1].split("    pad =", 1)[0]
    assert f"{artifact['measured']['d_abs_diff_db']:.4f} dB" in current


@pytest.mark.docs_consistency
def test_msl_producer_current_geometry_quotes_committed_metadata():
    artifact = _json("tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json")
    current = _text("scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py").split(
        "CURRENT COMMITTED REALIZATION", 1)[1].split("Usage::", 1)[0]
    meta = artifact["meta"]
    lo, hi = (meta[f"trace_y_{edge}_realized_m"] * 1e6 for edge in ("lo", "hi"))
    assert f"{lo:.0f}/{hi:.0f}um (centre {(lo + hi) / 2:.0f}um)" in current
    walls = meta["trace_wall_planes_realized_z_m"]
    assert f"z={walls[0] * 1e6:.0f}/{walls[1] * 1e6:.0f}um" in current
    spacings = {port["n_probe_spacing"] for name, port in artifact["reference_plane_geometry"].items()
                if name.startswith("msl_")}
    assert len(spacings) == 1
    assert f"n_probe_spacing={spacings.pop()} for both ports" in current


# The case script, its validation README row and its public benchmarks row
# carried these numbers too; they left with the MSL thru-line phase case
# (2026-09-23). The two pages below still state them.
@pytest.mark.parametrize("carrier", [
    "docs/guides/sparameter_support_matrix.md",
    "docs/design_notes/issue812_phase_identity_predeclaration.md",
])
@pytest.mark.docs_consistency
def test_msl_current_replay_quotes_fixture_and_labels_historical_openems(carrier):
    cv20 = _json("validation/crossval/_issue812_phase_identity/regate_evidence.json")["cv20"]
    source = _text(carrier)
    assert f"{cv20['blindness']['audit_construction_e1_max_phase_dev_deg']:.4f}" in source
    replay = cv20["run2_openems_with_current_rfx_fixture"]
    for field, scale in [("cross_solver_max_abs_raw_phase_diff_deg", 1),
                         ("analytic_beta_rfx_max_abs_dev_frac", 100),
                         ("analytic_beta_openems_max_abs_dev_frac", 100)]:
        assert f"{replay[field] * scale:.4f}" in source
    normalized = " ".join(source.split())
    # The solver's name is matched case-insensitively: the public pages spell
    # it "OpenEMS" and the scripts "openEMS". Until 2026-09-22 the exact
    # lower-case spelling happened to be present in benchmarks.mdx only in the
    # coax thru-line case's row, so removing that row -- which says nothing
    # about this case -- red this check on a spelling, not on the label it is
    # here to enforce.
    assert "historical run-2" in normalized and "openems" in normalized.lower()


@pytest.mark.docs_consistency
def test_patch_farfield_beam_peak_prose_quotes_committed_cut_angles():
    cuts = _json("tests/fixtures/patch_canonical_farfield_e4/"
                 "canonical_farfield_e4_measured_369367259302.json")["measured"]["cuts_deg"]
    source = _text("tests/crossval/test_patch_canonical_farfield_e4.py")
    e, h = cuts["E_peak_deg"], cuts["H_peak_deg"]
    assert f"measured {e:.0f} deg / {h:.0f} deg" in source
    assert f"measured: rfx {e:.0f}/{h:.0f} deg" in source


# The four checks above ask whether prose -- a comment, a docstring, a guide --
# still quotes the committed records. That is documentation, run by the
# non-required docs-consistency workflow (PI, 2026-09-22). They were also the
# only tests that held these record values, and a number still blocks, so the
# values are pinned here to half a unit in the place the prose prints them.
def test_the_records_hold_the_numbers_the_prose_quotes():
    far = _json("tests/fixtures/patch_canonical_farfield_e4/"
                "canonical_farfield_e4_measured_369367259302.json")["measured"]
    assert far["d_abs_diff_db"] == pytest.approx(0.0659, abs=0.5e-4)
    assert far["cuts_deg"]["E_peak_deg"] == pytest.approx(-1.0, abs=0.5)
    assert far["cuts_deg"]["H_peak_deg"] == pytest.approx(-4.0, abs=0.5)

    artifact = _json("tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json")
    meta = artifact["meta"]
    assert meta["trace_y_lo_realized_m"] == pytest.approx(900e-6, abs=0.5e-6)
    assert meta["trace_y_hi_realized_m"] == pytest.approx(1500e-6, abs=0.5e-6)
    assert meta["trace_wall_planes_realized_z_m"] == pytest.approx(
        [250e-6, 300e-6], abs=0.5e-6)
    assert {name: port["n_probe_spacing"]
            for name, port in artifact["reference_plane_geometry"].items()
            if name.startswith("msl_")} == {"msl_0": 11, "msl_1": 11}

    cv20 = _json("validation/crossval/_issue812_phase_identity/"
                 "regate_evidence.json")["cv20"]
    assert cv20["blindness"]["audit_construction_e1_max_phase_dev_deg"] == (
        pytest.approx(0.0647, abs=0.5e-4))
    replay = cv20["run2_openems_with_current_rfx_fixture"]
    assert replay["cross_solver_max_abs_raw_phase_diff_deg"] == pytest.approx(
        0.5308, abs=0.5e-4)
    # The pages print these two as percentages, to four decimals.
    assert replay["analytic_beta_rfx_max_abs_dev_frac"] * 100 == pytest.approx(
        1.4122, abs=0.5e-4)
    assert replay["analytic_beta_openems_max_abs_dev_frac"] * 100 == pytest.approx(
        0.3068, abs=0.5e-4)
