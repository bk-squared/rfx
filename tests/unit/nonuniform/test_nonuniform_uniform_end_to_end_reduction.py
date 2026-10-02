"""Uniform-profile NU and uniform Yee solves use the same current drive (#1373).

The five PEC/CPML plain/slab cases retain their waveform, timing, stability,
and trace-residual checks. The cross-path amplitude ratio is now one.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.geometry.csg import Box
from rfx.grid import C0

NA, NB, NZ = 45, 39, 4
STEPS = 1200

# Deliberately NOT marked slow: the whole point is that this runs in the default
# lane. An anchor the fast suite deselects would have missed #562's F1 exactly
# the way the existing suite did. Measured 38.6 s for the six cases here (it
# was 29.8 s for five before staircase-slab-cpml was added, #582 E2) — worth
# keeping in view against the fast-lane time pressure #545 tracks.


def _f110(dx: float) -> float:
    return (C0 / 2) * np.sqrt((1 / (NA * dx)) ** 2 + (1 / (NB * dx)) ** 2)


# Measured max |ratio - 1| = 3.041e-7.
# Allow accumulation headroom over 1200 steps; still below the old 1e-5 (#1373).
_RATIO_TOL = 2e-6


def _run(dx: float, *, nonuniform: bool, smoothing: bool, boundary: str = "pec",
         slab: bool | None = None):
    """``slab`` defaults to ``smoothing`` (legacy behaviour: the dielectric
    slab was only ever exercised together with the smoother). Pass it
    explicitly to decouple the two — the staircase-slab+cpml combo (#582 E2)
    needs the slab WITHOUT subpixel_smoothing to isolate the CPML pad
    material extension from the smoother.
    """
    if slab is None:
        slab = smoothing
    f0 = _f110(dx)
    profiles = (dict(dx_profile=np.full(NA, dx), dy_profile=np.full(NB, dx))
                if nonuniform else {})
    sim = Simulation(freq_max=2.5 * f0, domain=(NA * dx, NB * dx, NZ * dx),
                     dx=dx, boundary=boundary,
                     cpml_layers=(0 if boundary == "pec" else 8), **profiles)
    if slab:
        sim.add_material("slab", eps_r=4.0)
        # interface deliberately inside a voxel so the smoother engages
        # (when subpixel_smoothing=True; a no-op when it is False, giving
        # the plain staircase rasterization of the same geometry)
        sim.add(Box((0.0, 0.0, 0.3 * dx), (NA * dx, NB * dx, 2.0 * dx)),
                material="slab")
    # z = 3*dx, NOT NZ*dx/2: the latter is EXACTLY the slab's upper face, so the
    # factor prediction would ride on Box's exclusive-upper-bound convention and
    # a future inclusive rule would red this test for an unrelated reason
    # (#570 review, finding 10). 3*dx is unambiguously vacuum.
    sim.add_source((NA * dx / 3, NB * dx / 3, 3.0 * dx), "ez",
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((2 * NA * dx / 3, 2 * NB * dx / 3, 3.0 * dx), "ez")
    result = sim.run(n_steps=STEPS, compute_s_params=False,
                     skip_preflight=True, subpixel_smoothing=smoothing)
    return np.asarray(result.time_series, dtype=float).ravel(), float(result.dt)


_CASES = [
    pytest.param(False, "pec", False, id="plain-pec"),
    pytest.param(True, "pec", True, id="subpixel-pec"),
    pytest.param(False, "cpml", False, id="plain-cpml"),
    pytest.param(
        False, "cpml", True, id="staircase-slab-cpml",
    ),
    pytest.param(True, "cpml", True, id="subpixel-cpml"),
]


@pytest.mark.parametrize("smoothing,boundary,slab", _CASES)
def test_nu_solve_reduces_to_uniform_solve_with_current_drive(
        smoothing, boundary, slab):
    """Same mesh, same geometry, both builders: equal current-drive amplitudes.

    Run for both PEC and CPML boundary conditions.

    ``staircase-slab-cpml`` (slab=True, smoothing=False) isolates the CPML
    pad material extension from the smoother (#582 E2): it was never run
    before this fix — the slab used to be added only under ``if
    smoothing:`` — and pre-fix it already carried ~90% of the
    ``subpixel-cpml`` divergence (residual 7.7261e-3 of the 1.9890e-2
    total, both measured on the pre-fix tree, commit 31395e0), because the
    missing NU pad replication is a material-assembly gap, not a smoother
    interaction.
    """
    dx = 1e-3
    uni, dt_u = _run(dx, nonuniform=False, smoothing=smoothing, boundary=boundary,
                     slab=slab)
    nu, dt_n = _run(dx, nonuniform=True, smoothing=smoothing, boundary=boundary,
                    slab=slab)

    assert dt_u == pytest.approx(dt_n, rel=1e-12), (dt_u, dt_n)
    assert np.abs(uni).max() > 1e-6, "uniform leg produced no field"

    # the two traces are the SAME waveform: identical shape, identical timing.
    # Threshold accommodates both boundaries — measured deviation from 1 is
    # <1e-10 (pec) and 3.7e-9 (cpml, whose absorber implementations differ in
    # detail between the two paths); 1e-7 leaves ~27x margin on the worse case.
    assert np.corrcoef(uni, nu)[0, 1] > 1 - 1e-7
    assert int(np.argmax(np.abs(nu))) == int(np.argmax(np.abs(uni)))

    # neither leg grows: a diverging run would break this, and #565 read this
    # amplitude ratio as divergence
    for tag, trace in (("uniform", uni), ("nonuniform", nu)):
        head = np.abs(trace[:len(trace) // 4]).max()
        tail = np.abs(trace[3 * len(trace) // 4:]).max()
        assert tail <= 5.0 * max(head, 1e-30), (
            f"{tag} leg grows: head {head:.4g} -> tail {tail:.4g}")

    scale = float(np.dot(uni, nu) / np.dot(uni, uni))
    assert scale == pytest.approx(1.0, rel=_RATIO_TOL), (boundary, scale)

    residual = float(np.abs(nu - scale * uni).max() / np.abs(nu).max())
    # measured (post-#582 fix, 1200 steps): 2.0e-5 (pec), 9.5e-6
    # (pec + smoothing), 1.1e-4 (cpml, no slab). staircase-slab-cpml and
    # subpixel-cpml used to be untested/xfail (#582: the NU material
    # assembler never replicated eps/sigma/mu_r into the CPML pads the way
    # the uniform assembler does, rfx/api/_compile.py:188-231 vs
    # rfx/runners/nonuniform.py's assemble_materials_nu — an edge-touching
    # slab therefore saw a different absorber medium per path: 736 pad cells
    # eps 4.0-vs-1.0 at the slab's k=9 layer). Fixed by adding the same pad
    # replication to assemble_materials_nu; post-fix measured 8.7e-5
    # (staircase-slab-cpml, was 7.7261e-3 pre-fix — carried ~90% of the
    # subpixel-cpml divergence on its own) and 1.1e-4 (subpixel-cpml, was
    # 1.9890e-2 pre-fix, record-length-independent at 600/1200/2400 steps).
    # Pre-fix numbers measured on the pre-fix tree (commit 31395e0).
    assert residual < 3e-4, (
        f"after fitting the measured near-unit amplitude ratio the two paths "
        f"still differ by {residual:.3e} of full scale — the NU solve is not "
        f"reducing to the uniform solve on an identical mesh")


def test_cross_path_current_ratio_is_one_at_both_cell_sizes():
    """Halving dx preserves the ratio of NU to uniform current drives."""
    ratios = {}
    for dx in (1e-3, 0.5e-3):
        uni, _ = _run(dx, nonuniform=False, smoothing=False)
        nu, _ = _run(dx, nonuniform=True, smoothing=False)
        ratios[dx] = float(np.dot(uni, nu) / np.dot(uni, uni))
        assert ratios[dx] == pytest.approx(1.0, rel=_RATIO_TOL), (dx, ratios[dx])
    assert len(set(round(v, 6) for v in ratios.values())) == 1
