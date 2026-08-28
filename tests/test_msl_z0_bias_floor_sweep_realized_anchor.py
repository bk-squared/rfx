"""Issue #752: guards for the msl_z0_bias_floor_sweep realized-board anchor.

The pre-declared sweep script (scripts/diagnostics/msl_z0_bias_floor_sweep.py)
and its committed JSON score Z0 against Hammerstad-Jensen on the DECLARED
600/254um board at every dx, even though the misaligned points (dx=80,
60um) rasterize a thicker substrate (320, 300um -- the half-open
rasterizer rounds h_sub/dx UP). The sibling script
scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py adds
Hammerstad-Jensen on each point's REALIZED h/W (via
sim.fidelity_report(), no solve) as an ADDITIONAL column, without
touching the pre-declared script or its JSON.

This test module:
  1. Freezes the pre-declared JSON's sha256 -- it must never be edited
     (auditable-because-criteria-predate-the-data property).
  2. Re-derives the realized h/W directly via fidelity_report() (fast,
     no FDTD solve) and checks it agrees with the committed sibling
     artifact to a stated tolerance -- catching drift between the two
     without re-running the (expensive) FDTD sweep.
  3. Pins the two summary tolerances the corrected preflight advisories
     now cite (0.4% over all six points, 0.25% over the misaligned
     pair) so those numbers cannot silently drift unbound (the
     #494->#502 coverage-hole class named in test_msl_port_preflight.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep.json"
)
ANCHOR_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep_realized_anchor.json"
)
_SOURCE_SHA256 = (
    "f56f6b17691613d8782c1d5ce1241c1cd9bc10ef61715b203ed5cd6d4ab18362"
)


def test_pre_declared_sweep_json_is_frozen():
    """The pre-declared JSON and its as-run verdict block must never be
    edited -- they are auditable BECAUSE their criteria predate the
    data. This guards against exactly the mistake issue #752 warns
    against: fixing the reading of the data by rewriting the data."""
    digest = hashlib.sha256(SOURCE_JSON.read_bytes()).hexdigest()
    assert digest == _SOURCE_SHA256, (
        "scripts/diagnostics/msl_z0_bias_floor_sweep/"
        "msl_z0_bias_floor_sweep.json changed -- this file and its "
        "as-run verdict block must stay untouched; add corrections "
        "alongside it (see msl_z0_bias_floor_sweep_realized_anchor.py), "
        "never by editing it"
    )


def test_realized_anchor_json_exists_and_cites_source():
    assert ANCHOR_JSON.exists(), (
        f"{ANCHOR_JSON} missing -- regenerate with "
        "python3 scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py"
    )
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    assert out["source_json_sha256"] == _SOURCE_SHA256, (
        "the sibling artifact's recorded source sha256 no longer matches "
        "the live pre-declared JSON -- regenerate the sibling artifact"
    )
    assert len(out["rows"]) == 6


def test_realized_anchor_matches_fidelity_report_directly():
    """Re-derive h_sub/W_trace realized extents from fidelity_report()
    (no FDTD solve -- fast) for each of the six pre-declared dx points
    and check they agree with the committed sibling artifact. This is
    the "sweep's realized-board column agrees with fidelity_report to a
    stated tolerance" regression test."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msl_z0_bias_floor_sweep_realized_anchor",
        REPO / "scripts" / "diagnostics"
        / "msl_z0_bias_floor_sweep_realized_anchor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in out["rows"]}

    for label, dx in mod.DX_GRID:
        h_real_um, w_real_um = mod.realized_h_w_um(dx)
        committed = by_label[label]
        assert h_real_um == pytest.approx(
            committed["h_sub_realized_um"], abs=0.01
        ), f"{label}: h_sub realized drifted vs committed artifact"
        assert w_real_um == pytest.approx(
            committed["w_trace_realized_um"], abs=0.01
        ), f"{label}: W_trace realized drifted vs committed artifact"


def test_realized_board_deviation_tolerances_hold():
    """Pin the exact tolerances the corrected preflight advisories now
    cite ("within 0.4%" for all six points, used implicitly to justify
    the class docstring's correction). Measured max|dev_real| = 0.377%
    over all six, 0.197% over the misaligned pair (dx=80,60um) -- both
    asserted here with a stated derivation, not an unbound constant."""
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    rows = out["rows"]
    devs_all = [abs(r["dev_vs_realized_board_pct"]) for r in rows]
    devs_misaligned = [
        abs(r["dev_vs_realized_board_pct"])
        for r in rows if r["label"].startswith("misaligned")
    ]
    assert len(devs_misaligned) == 2
    max_all = max(devs_all)
    max_misaligned = max(devs_misaligned)
    assert max_all <= 0.4, (
        f"max|dev_vs_realized_board_pct| over all six points = {max_all}%, "
        "exceeds the 0.4% bound the preflight advisory cites"
    )
    assert max_misaligned <= 0.25, (
        f"max|dev_vs_realized_board_pct| over the misaligned pair = "
        f"{max_misaligned}%, exceeds the 0.25% bound"
    )
    # And the aggregate fields the artifact itself reports must match.
    assert out["max_abs_dev_vs_realized_board_pct_all_six"] == pytest.approx(
        max_all, abs=1e-9)
    assert out[
        "max_abs_dev_vs_realized_board_pct_misaligned_pair"
    ] == pytest.approx(max_misaligned, abs=1e-9)


def test_declared_board_column_unchanged_from_source():
    """z0_measured_ohm and z0_hj_declared_board_ohm in the sibling
    artifact must be copied VERBATIM from the pre-declared JSON, never
    re-solved or retyped by hand."""
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    out = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    src_by_label = {r["label"]: r for r in source["rows"]}
    for row in out["rows"]:
        src = src_by_label[row["label"]]
        assert row["z0_measured_ohm"] == src["z0_measured_ohm"]
        assert row["z0_hj_declared_board_ohm"] == src["z0_hj_ohm"]
