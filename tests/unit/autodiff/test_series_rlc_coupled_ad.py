"""d|S11|^2/dR through a series R + C load, AD against a float64 difference (#1163).

The series RLC update is now one implicit solve of the element current
together with its edge field (``rfx.lumped._update_series``). This checks that
the gradient ``forward(rlc_values_override=...)`` returns through that solve
agrees with a central finite difference computed with float64 FIELDS, on a load
whose resistance (50 ohm) is below the edge's own impedance d/(D0*A) = 215 ohm
-- the regime where the replaced update was a negative resistance and the run
was not finite.

Fixture: the one-cell-wide parallel-plate line of
``tests/unit/ports/test_series_rlc_edge_coupling.py`` (Zc = eta0, lumped port
on node 1), series 50 ohm + 1 pF across the gap 16 cells away, one bin at
5 GHz, objective |S11|^2. x64 is scoped to the test, never module-level.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

try:  # modern JAX: scoped x64 promoted to top-level
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 20
LOAD_NODE = N_NODES - 3
R0 = 50.0
C0 = 1e-12
FREQ = np.array([5e9])
N_STEPS = 1200
#: Relative FD step. The FD side runs float64 fields, so round-off is ~1e-16
#: relative and the truncation error (h^2/6)|f'''| dominates; 1e-3 puts it
#: near 1e-7 relative -- far below the 5 % bar (the chain-closure contract's
#: AD/FD bar, also used by the lumped/wire battery).
H_REL = 1e-3
BAR = 0.05


def _sim(precision: str) -> Simulation:
    sim = Simulation(
        freq_max=10e9, domain=((N_NODES - 1) * DX, DX, DX), dx=DX,
        precision=precision,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")))
    sim.add_port(position=(DX, 0.0, 0.0), component="ez", impedance=ETA0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    sim.add_lumped_rlc(position=(LOAD_NODE * DX, 0.0, 0.0), component="ez",
                       R=R0, C=C0, topology="series")
    return sim


def _objective(sim):
    def f(r):
        res = sim.forward(port_s11_freqs=jnp.asarray(FREQ), n_steps=N_STEPS,
                          skip_preflight=True,
                          rlc_values_override={0: {"R": r}})
        return jnp.sum(jnp.abs(res.s_params.reshape(-1)) ** 2)
    return f


def test_dS11sq_dR_series_rc_ad_matches_float64_fd():
    with _enable_x64(True):
        sim32 = _sim("float32")
        sim64 = _sim("float64")
        g32, g64 = sim32._build_grid(), sim64._build_grid()
        assert (tuple(g32.shape), g32.dt) == (tuple(g64.shape), g64.dt), (
            "the float64 referee is not the same discrete rig")

        r = jnp.asarray(R0, dtype=jnp.float64)
        val, g_ad = jax.value_and_grad(_objective(sim32))(r)
        val, g_ad = float(val), float(g_ad)

        f64 = _objective(sim64)
        h = R0 * H_REL
        g_fd = (float(f64(r + h)) - float(f64(r - h))) / (2.0 * h)

    assert np.isfinite(val) and 0.0 < val < 1.0 + 1e-6
    assert np.isfinite(g_ad) and g_ad != 0.0
    rel = abs(g_ad - g_fd) / abs(g_fd)
    assert rel < BAR, (
        f"d|S11|^2/dR at R={R0} ohm, C={C0} F: AD {g_ad:.6e} vs float64 FD "
        f"{g_fd:.6e} (rel {rel:.3%})")
