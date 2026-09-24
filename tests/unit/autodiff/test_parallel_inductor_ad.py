"""d|S11|^2/dL through a parallel lumped inductor, AD against a float64
difference (#1245).

The parallel inductor is solved together with its edge field in one
trapezoidal step (``rfx.lumped._update_parallel``, #1245), and its value can
be a tracer (``forward(rlc_values_override={0: {"L": ...}})``). This checks
that the gradient ``forward()`` returns through that update agrees with a
central finite difference computed with float64 FIELDS.

Fixture: the one-cell-wide parallel-plate line of
``tests/unit/autodiff/test_series_rlc_coupled_ad.py`` (Zc = eta0, lumped port
on node 1), 10 nH (314 ohm at 5 GHz) in parallel with a folded 200 ohm across
the gap 16 cells away, one bin at 5 GHz, objective |S11|^2. The resistor is
there because a pure inductor ends this line without loss: |S11|^2 is then 1
for every L and its true gradient is zero (measured 1.000000; what AD and FD
return there is the truncated record's residual). With it, |S11|^2 = 0.187
and d|S11|^2/dL = -2.065e7 per henry, the same at twice the record (measured
AD in float32 fields against float64 FD: agreement to 6 digits). x64 is
scoped to the test, never module-level.
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
L0 = 10e-9
R0 = 200.0
FREQ = np.array([5e9])
N_STEPS = 1200
#: Relative FD step; the FD side runs float64 fields, so truncation
#: (h^2/6)|f'''| dominates, near 1e-7 relative -- far below the 5 % bar (the
#: chain-closure contract's AD/FD bar, as in test_series_rlc_coupled_ad.py).
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
                       R=R0, L=L0, topology="parallel")
    return sim


def _objective(sim):
    def f(l_h):
        res = sim.forward(port_s11_freqs=jnp.asarray(FREQ), n_steps=N_STEPS,
                          skip_preflight=True,
                          rlc_values_override={0: {"L": l_h}})
        return jnp.sum(jnp.abs(res.s_params.reshape(-1)) ** 2)
    return f


def test_dS11sq_dL_parallel_inductor_ad_matches_float64_fd():
    with _enable_x64(True):
        sim32 = _sim("float32")
        sim64 = _sim("float64")
        g32, g64 = sim32._build_grid(), sim64._build_grid()
        assert (tuple(g32.shape), g32.dt) == (tuple(g64.shape), g64.dt), (
            "the float64 referee is not the same discrete rig")

        l_h = jnp.asarray(L0, dtype=jnp.float64)
        val, g_ad = jax.value_and_grad(_objective(sim32))(l_h)
        val, g_ad = float(val), float(g_ad)

        f64 = _objective(sim64)
        h = L0 * H_REL
        g_fd = (float(f64(l_h + h)) - float(f64(l_h - h))) / (2.0 * h)

    assert np.isfinite(val) and 0.0 < val < 1.0 + 1e-6
    assert np.isfinite(g_ad) and g_ad != 0.0
    rel = abs(g_ad - g_fd) / abs(g_fd)
    assert rel < BAR, (
        f"d|S11|^2/dL at L={L0} H || R={R0} ohm: AD {g_ad:.6e} vs float64 FD "
        f"{g_fd:.6e} (rel {rel:.3%})")
