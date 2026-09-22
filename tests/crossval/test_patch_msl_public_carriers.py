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


def test_patch_demo_directivity_comment_uses_current_measurement():
    artifact = _json("tests/fixtures/patch_canonical_farfield_e4/"
                     "canonical_farfield_e4_measured_369367259302.json")
    current = _text("examples/tutorials/patch_antenna_demo.py").split(
        "# NTFF box.", 1)[1].split("    pad =", 1)[0]
    assert f"{artifact['measured']['d_abs_diff_db']:.4f} dB" in current


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


@pytest.mark.parametrize("carrier", [
    "validation/crossval/20_msl_phase_referee.py",
    "docs/guides/sparameter_support_matrix.md",
    "docs/design_notes/issue812_phase_identity_predeclaration.md",
    "validation/README.md",
    "docs/public/guide/benchmarks.mdx",
])
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


def test_patch_farfield_beam_peak_prose_quotes_committed_cut_angles():
    cuts = _json("tests/fixtures/patch_canonical_farfield_e4/"
                 "canonical_farfield_e4_measured_369367259302.json")["measured"]["cuts_deg"]
    source = _text("tests/crossval/test_patch_canonical_farfield_e4.py")
    e, h = cuts["E_peak_deg"], cuts["H_peak_deg"]
    assert f"measured {e:.0f} deg / {h:.0f} deg" in source
    assert f"measured: rfx {e:.0f}/{h:.0f} deg" in source
