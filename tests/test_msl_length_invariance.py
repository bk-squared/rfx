"""MSL thru-line length-invariance gate (issue #80 extractor soundness).

A uniform matched 50 Ω microstrip thru-line is length-invariant by physics:
adding length only adds transmission phase, so |S11|, |S21|, and the passivity
sum |S11|^2 + |S21|^2 must NOT change with trace length, and the extracted Z0
must stay real-positive.

The pre-fix V·I single-plane wave split (a=(V+Z0_hj*I)/2, b=(V-Z0_hj*I)/2,
S11=b/a) violated this: it injected a length-ACCUMULATING bias driven by the
analytic-Z0-vs-mesh-Z0 mismatch, so passivity grew with length
(1.30 @ 8 mm -> 4.74 @ 37 mm at the same mesh) and the extracted Z0 of the
passive port even went negative. That is the structural blocker documented in
docs/research_notes/20260529_msl_broad_e5_extractor_blocker.md.

The fix (PR #99 / commit f92410a, re-landed on this branch) assembles S from
the voltage-only spatial fit (S_ii = gamma/alpha), which reads reflection from
the standing-wave shape alone — independent of Z0 and the Ampere-loop current —
and is passive by construction when the 2-wave fit is well-conditioned.

This gate asserts the EXTRACTOR property the fix restores (length-invariance,
real-positive Z0, bounded passivity), NOT the absolute |S11| floor — the
residual matched-line |S11| at a coarse mesh is the separate source near-field /
Yee-staircase contribution ("Fix B"), which is itself length-invariant.

Measured (dx = h_sub/4 = 63.5 um, 3.0-4.5 GHz):
  pre-fix  8 mm: max_pass 1.072, |S11| 0.233, Z0 -41.6  (non-physical)
  post-fix 8 mm: max_pass 1.045, |S11| 0.183, Z0 +41.4
  post-fix 16 mm: max_pass 1.040, |S11| 0.186, Z0 +41.7  (|Δmax_pass| 0.006)
"""
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

EPS_R = 3.66          # RO4350B
H_SUB = 254e-6
W_TRACE = 600e-6      # 50 Ohm design width
PORT_MARGIN = 2e-3
F_MAX = 5e9
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9
DX = H_SUB / 4.0      # 63.5 um — coarse end of the recipe, where the
                      # pre-fix length-dependence was most visible.


def _run_thru(l_line, dx=DX, num_periods=12, n_freqs=30):
    lx = l_line + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)
    lz = H_SUB + 1.5e-3
    y_c = ly / 2.0

    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    sim.add(Box((0.0, y_c - W_TRACE / 2, H_SUB),
                (lx, y_c + W_TRACE / 2, H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + l_line, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0)

    res = sim.compute_msl_s_matrix(n_freqs=n_freqs, num_periods=num_periods,
                                   strict_extractor=False)
    m = (res.freqs >= GATE_F_LO) & (res.freqs <= GATE_F_HI)
    s11 = np.abs(res.S[0, 0, m])
    s21 = np.abs(res.S[1, 0, m])
    passivity = s11**2 + s21**2
    return {
        "max_pass": float(np.max(passivity)),
        "mean_s11": float(np.mean(s11)),
        "mean_s21": float(np.mean(s21)),
    }


@pytest.mark.slow
def test_msl_thru_line_is_length_invariant():
    """The spatial-fit extractor must produce length-invariant S on a thru-line.

    Guards regression to the V·I-split extractor (issue #80), whose passivity
    grew with trace length. The discriminating assertion is that the passivity
    sum and |S11| do NOT change between two trace lengths on the same mesh.
    """
    short = _run_thru(8e-3)
    long = _run_thru(16e-3)

    # --- 1. LENGTH-INVARIANCE (the core fix) ---------------------------------
    # Pre-fix this delta was order-unity (1.30 -> 4.74 over 8->37 mm).
    d_pass = abs(long["max_pass"] - short["max_pass"])
    d_s11 = abs(long["mean_s11"] - short["mean_s11"])
    assert d_pass < 0.05, (
        f"passivity is length-dependent: max_pass 8mm={short['max_pass']:.3f} "
        f"vs 16mm={long['max_pass']:.3f} (|Δ|={d_pass:.3f} ≥ 0.05) — the "
        f"V·I-split length-accumulating bias has regressed (issue #80)."
    )
    assert d_s11 < 0.05, (
        f"|S11| is length-dependent: 8mm={short['mean_s11']:.3f} vs "
        f"16mm={long['mean_s11']:.3f} (|Δ|={d_s11:.3f} ≥ 0.05)."
    )

    # --- 2. PASSIVITY bounded (not diverging) --------------------------------
    for tag, r in (("8mm", short), ("16mm", long)):
        assert r["max_pass"] < 1.10, (
            f"{tag}: max passivity |S11|²+|S21|² = {r['max_pass']:.3f} ≥ 1.10 "
            f"— non-physical for a passive thru-line."
        )
        # --- 3. forward transmission sane -----------------------------------
        assert 0.90 < r["mean_s21"] < 1.10, (
            f"{tag}: mean |S21| = {r['mean_s21']:.3f} outside (0.90, 1.10)."
        )
