"""MSL launch fixture derives from REGISTERED materials (issue #483).

Pre-#483, the auto-``eps_r_sub`` branch sampled the (possibly overridden)
``materials`` through ``stop_gradient``: finite differences re-derived the
mode profile / sigma loading / source amplitude at ``alpha ± h`` while the
AD tape saw them frozen at the linearization point — FD and ``jax.grad``
differentiated DIFFERENT functions. Measured on the f64 referee
(np=3 raw-forward objective): 61.7% gradient deficit pre-fix,
0.04-0.19% post-fix; converged np=20 through the full extraction:
13.7% pre-fix. The fix samples ``self._assemble_materials(grid)`` (the
registered, concrete materials) so the fixture is the same constant on
both sides for every caller (forward / topology / sparam_driver).

The gate below is an f32 mini-referee: np=1, h=1e-2 central FD, probes
in the air above the trace. Measured endpoints on THIS configuration
(PR #486 review cycle — the first version put the probes inside the PEC
trace and claimed unmeasured margins; both corrected here):
pre-fix main 0.126, post-fix 0.0002; at np=3 the same referee reads
0.358 vs 0.0000. The 0.03 gate therefore discriminates with ~4x margin
on the fail side and ~100x on the pass side at np=1 cost (~40 s), well
above the f32 comparator noise at h=1e-2 (sub-1%).
The exact h_sub/dx ratio of 3 places the foil sheet on the laminate face.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from rfx.sources.msl_port import (
    msl_cell, msl_cross_section_span, msl_port_from_entry)
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
SUBSTRATE_NORMAL_CELLS = 3
_DX = 254e-6 / SUBSTRATE_NORMAL_CELLS
_L_LINE = 6e-3
_MARGIN = 2e-3


def _msl_sim_auto_eps():
    """MSL thru with AUTO eps_r_sub (the branch #483 fixed)."""
    lx = _L_LINE + 2 * _MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * _DX)
    lz = _H_SUB + 0.5e-3
    sim = Simulation(snap="declared",
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB)), material="pec")
    sim.add_msl_port(position=(_MARGIN, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    for x in (0.004, 0.005, 0.006):
        sim.add_probe(position=(x, y_c, _H_SUB + 2 * _DX), component="ez")
    assert all(
        grid.position_to_index(probe.position)[2]
        > grid.position_to_index((0.0, 0.0, _H_SUB))[2]
        for grid in [sim._build_grid()] for probe in sim._probes
    ), "Air probes above the trace preserve the referee's discriminating power"
    return sim


@pytest.mark.highmem
def test_auto_eps_msl_gradient_matches_fd_mini_referee():
    """f32 mini-referee (np=1, h=1e-2) on the auto-eps_r_sub MSL forward:
    AD within 3% of central FD. Measured: pre-#483 main reads 0.126 (the
    fixture followed alpha under FD only); post-fix reads 0.0002.
    An earlier run measured +8.1 GB RSS high-water delta."""
    sim = _msl_sim_auto_eps()
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(alpha):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sim.forward(eps_override=eps_base * alpha, num_periods=1.0)
        return jnp.sum(jnp.asarray(r.time_series) ** 2)

    _, g = jax.value_and_grad(objective)(jnp.float32(1.0))
    g_ad = float(g)
    assert np.isfinite(g_ad) and abs(g_ad) > 0.0

    h = 1e-2
    f_p = float(objective(jnp.float32(1.0 + h)))
    f_m = float(objective(jnp.float32(1.0 - h)))
    g_fd = (f_p - f_m) / (2.0 * h)
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 0.03, (
        f"MSL auto-eps fixture gradient deficit is back: rel_err={rel:.3f} "
        f"(g_ad={g_ad:.4e}, g_fd={g_fd:.4e}) — the launch fixture must "
        "derive from registered materials on BOTH the FD and AD sides "
        "(issue #483)."
    )


def test_explicit_and_auto_eps_build_the_same_fixture():
    """With the substrate eps matching, the auto branch must resolve the
    same eps_r_sub the explicit branch is given — the forward results are
    identical (same fixture, same fields). Cheap concrete-run check.
    The 1e-6*peak bar allows the explicit float64 3.66 to differ from the directly checked registered float32(3.66), measured earlier as 8.9e-08 on a 0.134 peak (6.7e-07 relative)."""
    import dataclasses

    sim_a = _msl_sim_auto_eps()
    sim_b = _msl_sim_auto_eps()
    sim_b._msl_ports[:] = [
        dataclasses.replace(pe, eps_r_sub=_EPS_R) for pe in sim_b._msl_ports
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = sim_a.forward(num_periods=1.0)
        rb = sim_b.forward(num_periods=1.0)
    a = np.asarray(ra.time_series)
    b = np.asarray(rb.time_series)
    assert a.shape == b.shape
    _mp = msl_port_from_entry(sim_a._msl_ports[0])
    _span = msl_cross_section_span(sim_a._build_grid(), _mp)
    _cell = msl_cell(sim_a._msl_ports[0].direction, _span["i_feed"],
                     _span["w_centre"], (_span["n_lo"] + _span["n_hi"]) // 2)
    _mats = sim_a._assemble_materials(sim_a._build_grid(), pec_sheets=[],
                                      pec_wires=[])[0]
    assert float(np.asarray(_mats.eps_r)[_cell]) == float(np.float32(_EPS_R)), (
        "the auto branch did not sample the registered substrate material at "
        f"the trace-centre cell {_cell}")
    assert np.allclose(a, b, rtol=0.0, atol=1e-6 * float(np.abs(a).max())), (
        "auto-resolved eps_r_sub built a different launch fixture than the "
        "explicit value — the auto branch is not sampling the registered "
        "materials (issue #483)"
    )
