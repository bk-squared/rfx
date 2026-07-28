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

The gate below is an f32 mini-referee: np=1, h=1e-2 central FD. The f32
comparator noise (#477: +/-3-5% at h=1e-3, smaller at h=1e-2) and the
np=1 truncation are both tiny against the 60%-class pre-fix defect, so
rel_err < 0.15 discriminates with >4x margin on both sides (measured
endpoints: pre-fix ~0.6, post-fix <0.05).
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6
_L_LINE = 6e-3
_MARGIN = 2e-3


def _msl_sim_auto_eps():
    """MSL thru with AUTO eps_r_sub (the branch #483 fixed)."""
    lx = _L_LINE + 2 * _MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * _DX)
    lz = _H_SUB + 0.5e-3
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    # NOTE: no eps_r_sub kwarg — exercises the auto branch.
    sim.add_msl_port(position=(_MARGIN, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    # raw observables for the referee objective (forward() records
    # time_series only for registered point probes)
    for x in (0.004, 0.005, 0.006):
        sim.add_probe(position=(x, y_c, 0.0003), component="ez")
    return sim


def test_auto_eps_msl_gradient_matches_fd_mini_referee():
    """f32 mini-referee (np=1, h=1e-2) on the auto-eps_r_sub MSL forward:
    AD within 15% of central FD. Pre-#483 this read ~60% (the fixture
    followed alpha under FD only); post-fix it reads <5%."""
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
    assert rel < 0.15, (
        f"MSL auto-eps fixture gradient deficit is back: rel_err={rel:.3f} "
        f"(g_ad={g_ad:.4e}, g_fd={g_fd:.4e}) — the launch fixture must "
        "derive from registered materials on BOTH the FD and AD sides "
        "(issue #483)."
    )


def test_explicit_and_auto_eps_build_the_same_fixture():
    """With the substrate eps matching, the auto branch must resolve the
    same eps_r_sub the explicit branch is given — the forward results are
    identical (same fixture, same fields). Cheap concrete-run check."""
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
    assert np.array_equal(a, b), (
        "auto-resolved eps_r_sub built a different launch fixture than the "
        "explicit value — the auto branch is not sampling the registered "
        "materials (issue #483)"
    )
