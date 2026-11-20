"""Compare realized board extents directly with the live fidelity report.

The historical anchor remains frozen. The live expectations account for
the realized PEC volume sampler and, since #2349, dielectric faces going
to the nearest node at the six declared sweep points.
Refreshing the artifact itself requires RE-SOLVING the sweep on the new rasterization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ANCHOR_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep_realized_anchor.json"
)

_LIVE_W_UM = {
    "aligned h_sub/3": 592.667,
    "aligned h_sub/4": 571.5,
    "misaligned 80um": 640.0,
}

_LIVE_H_UM = {
    "misaligned 80um": 240.0,
    "misaligned 60um": 240.0,
}


def test_realized_anchor_matches_fidelity_report_directly():
    """Re-derive h_sub/W_trace realized extents from fidelity_report()
    (no FDTD solve -- fast) for each of the six pre-declared dx points
    and check they agree with the live expectations, using the committed
    sibling artifact where unchanged. This is
    the "sweep's realized-board column agrees with fidelity_report to a
    stated tolerance" regression test.

    #766 review N3, re-adjudicated at #802/#834: the committed artifact
    declares its own precision --
    ``"jax_enable_x64=False, inferred from Z0 agreement with the
    realized-board Hammerstad-Jensen anchor (max |dev| = 0.377% over all
    six points, vs 8.6/5.6/4.4% at the alternative (x64) rasterization of
    the aligned class's trace width)"``. This test used to SKIP under
    JAX_ENABLE_X64=1 because the aligned points' trace width then
    rasterized differently (h_sub/3 read W = 592.667µm where the artifact
    has 677.333µm). #834 made realized geometry flag-independent, so that
    justification is gone and the skip is removed: the live re-derivation
    lands on the module-level expectations under BOTH flags (measured
    2026-09-01: this file passes under x64=0 and x64=1). The committed
    artifact itself stays the pre-#802 as-solved record, exactly as the
    re-pin note below explains.

    #931 (2026-09-07): the PEC trace is a one-cell VOLUME, so its width is
    now read from cell CENTRES (§1.1); ``_LIVE_W_UM`` carries the three
    points where that differs from the committed artifact (h_sub/3,
    h_sub/4, 80 um) and the arithmetic. The other three read the committed
    width again. Since #2349 a dielectric face goes to the nearest node:
    ``_LIVE_H_UM`` carries 240/240 um for the misaligned points, while the
    four aligned h_sub values remain at the committed 254 um.
    A volume face rounds to the nearest cell plane (W/dx = 7.087/9.449/11.811/14.173 to 7/9/12/14
    aligned cells, 80 um width faces 14.35/21.85 to 14/22, 60 um width faces 16.467/26.467 to 16/26,
    and substrate 254/80 = 3.175 to 3 cells or 254/60 = 4.233 to 4 cells).
    """
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
        expected_h = _LIVE_H_UM.get(
            label, committed["h_sub_realized_um"])
        assert h_real_um == pytest.approx(expected_h, abs=0.01), (
            f"{label}: h_sub realized drifted vs the live (#2349) "
            "expectation (committed artifact = pre-#802 as-solved record)")
        expected_w = _LIVE_W_UM.get(
            label, committed["w_trace_realized_um"])
        assert w_real_um == pytest.approx(expected_w, abs=0.01), (
            f"{label}: W_trace realized drifted vs the live (#931 §1.1) "
            "expectation (committed artifact = pre-#802 as-solved record)")
