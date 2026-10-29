"""Compare realized board extents directly with the live fidelity report.

The historical anchor remains frozen. The live expectations account for
the realized PEC volume sampler at the six declared sweep points.
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

# Realized trace width, in um, that the lattice ownership contract's
# centre-sampled PEC VOLUME (#931 §1.1; ``fidelity_report`` reads that
# sampler since 20a8ccfc) produces at the three sweep points where it
# differs from the committed (pre-#802 f32) artifact. Re-derived live by
# ``realized_h_w_um()`` in both tests below -- these are the CURRENT
# expectation, so a future rasterizer move reds them; the committed
# artifact keeps the pre-#802 values as its as-solved record.
#
# Arithmetic (W = 600 um; a volume face rounds to the NEAREST cell plane;
# the trace's lower face is on a node at every aligned point):
#   h_sub/3  W/dx =  7.087 ->  7 cells  592.667  (committed 677.333: the f32
#                                                 node sampler took the hi face)
#   h_sub/4  W/dx =  9.449 ->  9 cells  571.5    (committed 635.0 = 10 nodes)
#   h_sub/5  W/dx = 11.811 -> 12 cells  609.6    = committed (558.8 under the
#                                                 #802/#834 node sampler)
#   h_sub/6  W/dx = 14.173 -> 14 cells  592.667  = committed (635.0 under it)
#   80 um    faces at 14.35 / 21.85 dx -> planes 14 / 22, 8 cells, 640.0
#                                                (committed 560.0 = 7 nodes)
#   60 um    faces at 16.467 / 26.467 dx -> 16 / 26, 10 cells, 600.0 = committed
# History: the #802/#834 exact-coordinate NODE sampler moved h_sub/3, /5
# and /6 (592.667 / 558.8 / 635.0) and left h_sub/4 and the misaligned
# pair alone; the centre sampler moves h_sub/3, /4 and 80 um instead and
# lands back on the committed widths at h_sub/5, /6 and 60 um. Measured
# 2026-09-07, fidelity_report(), no solve, JAX_PLATFORMS=cpu; h_sub is a
# dielectric (node-sampled, §1.1 unchanged) and reads the committed
# 254 / 254 / 254 / 254 / 320 / 300 um at every point.
_LIVE_W_UM = {
    "aligned h_sub/3": 592.667,
    "aligned h_sub/4": 571.5,
    "misaligned 80um": 640.0,
}


def test_realized_anchor_matches_fidelity_report_directly():
    """Re-derive h_sub/W_trace realized extents from fidelity_report()
    (no FDTD solve -- fast) for each of the six pre-declared dx points
    and check they agree with the committed sibling artifact. This is
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
    width again. h_sub (a dielectric) is unchanged at every point.
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

    # Re-pinned at the exact-coordinate fix (#802) and again at #931
    # §1.1 (centre-sampled PEC volume). The committed artifact stays
    # untouched — it is the as-solved record of the frozen sweep (its Z0
    # numbers were measured on the pre-#802 f32 rasterization, and its
    # 0.4% bound test reads it as such). The live re-derivation therefore
    # matches the committed rows where the realization lands on the same
    # width (h_sub/5, h_sub/6, 60 um) and ``_LIVE_W_UM`` where it does not.
    # Refreshing the artifact itself requires RE-SOLVING the sweep on the
    # new rasterization — a re-measurement lane, not a value edit here.
    for label, dx in mod.DX_GRID:
        h_real_um, w_real_um = mod.realized_h_w_um(dx)
        committed = by_label[label]
        assert h_real_um == pytest.approx(
            committed["h_sub_realized_um"], abs=0.01
        ), f"{label}: h_sub realized drifted vs committed artifact"
        expected_w = _LIVE_W_UM.get(
            label, committed["w_trace_realized_um"])
        assert w_real_um == pytest.approx(expected_w, abs=0.01), (
            f"{label}: W_trace realized drifted vs the live (#931 §1.1) "
            "expectation (committed artifact = pre-#802 as-solved record)")
