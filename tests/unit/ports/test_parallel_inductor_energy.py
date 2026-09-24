"""A parallel L || C lumped element in a closed PEC box keeps its energy (#1245).

The physics
-----------
A 2 nH inductor in parallel with 1 pF (the C folded into the edge
permittivity) sits on the centre Ez edge of an 8 mm vacuum box with PEC walls.
The capacitor starts charged and nothing drives the box afterwards, so the
circuit rings near 1/(2 pi sqrt(LC)) = 3.56 GHz (3.48 GHz measured, the box's
own field adds a little capacitance) and nothing in it can dissipate: the
total energy must stay what it was.

The energy is the Yee scheme's conserved quantity (dV = dx^3)::

    W^n = 1/2 sum eps_c (E_c^n)^2 dV + 1/2 mu0 sum H^{n-1/2}.H^{n+1/2} dV
          + 1/2 L (I^n)^2

``eps_c`` is each edge's own permittivity, ``dt/Cb`` from
:func:`rfx.core.yee.e_component_coeffs` (the folded C included), so the
capacitor's energy is in the first sum and the inductor's in the last.

The step is rfx's own primitives in :func:`rfx.simulation.make_core_step`'s
order (E^n read, H update, E update, PEC, lumped element); the element update
is looked up in :mod:`rfx.lumped` at call time.

MEASURED over 1600 steps (10.6 periods of the LC): float64 fields,
max |W/W0 - 1| = 3.6e-15; float32 fields, 2.3e-6. The inductor holds
0.484 of W0 on average and 0.977 at its peak. The backward-Euler inductor
this replaced (the field loaded by I^{n+1}, half a step late) lost 91.6 % of
the energy over the same run (W/W0 = 0.084): the series resistance w^2 L dt
it carried. Written as (D0*e_std - I/A - (gamma/4)*E^n)/(D0 + gamma/4), the
same trapezoidal algebra gained 1.1e-4 in float32 (the rounded ratio
D0/(D0 + gamma/4) scales the whole edge field every step).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:  # modern JAX: scoped x64 promoted to top-level
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import lumped
from rfx.boundaries.pec import apply_pec
from rfx.core.yee import (MU_0, e_component_coeffs, init_materials, init_state,
                          update_e, update_h)
from rfx.grid import Grid

L_H = 2e-9
C_F = 1e-12
N_STEPS = 1600
#: float64 rounding over 1600 steps measured 3.6e-15; float32 2.3e-6. The
#: float32 bar sits 4x over its measurement and 11x under the ill-conditioned
#: spelling's 1.1e-4.
BARS = {"float64": 1e-10, "float32": 1e-5}
#: 1/2 L I^2 over W0: MEASURED 0.484 on average and 0.977 at its peak (the box's
#: own field keeps the rest); 0 and 0 if the inductor update is a no-op.
INDUCTOR_MEAN_SHARE = (0.45, 0.50)
INDUCTOR_PEAK_SHARE = 0.95


def _energy_trace(field_dtype):
    grid = Grid(freq_max=10e9, domain=(8e-3, 8e-3, 8e-3), dx=1e-3, cpml_layers=0)
    dx, dt = float(grid.dx), float(grid.dt)
    spec = lumped.LumpedRLCSpec(L=L_H, C=C_F, topology="parallel",
                                position=(4e-3, 4e-3, 4e-3), component="ez")
    mats = lumped.setup_rlc_materials(grid, spec, init_materials(grid.shape))
    meta = lumped.build_rlc_meta(grid, spec, mats)
    assert not meta.is_series and meta.has_inductor
    i, j, k = meta.i, meta.j, meta.k
    mats = mats._replace(eps_r=mats.eps_r.astype(field_dtype),
                         sigma=mats.sigma.astype(field_dtype),
                         mu_r=mats.mu_r.astype(field_dtype))
    periodic = (False, False, False)
    _, cbs = e_component_coeffs(mats, dt, periodic)
    eps_c = [dt / cb for cb in cbs]
    dv = dx ** 3

    st0 = init_state(grid.shape, field_dtype=field_dtype)
    st0 = st0._replace(ez=st0.ez.at[i, j, k].set(1.0))
    rlc0 = lumped.init_rlc_state(
        dtype=lumped.rlc_carry_dtype([meta], field_dtype))

    def step(carry, _):
        st, rlc = carry
        e_prev = st.ez[i, j, k]
        h_old = (st.hx, st.hy, st.hz)
        w_e = 0.5 * dv * sum(jnp.sum(eps * f ** 2)
                             for eps, f in zip(eps_c, (st.ex, st.ey, st.ez)))
        w_l = 0.5 * L_H * rlc.inductor_current ** 2
        st = update_h(st, mats, dt, dx, periodic)
        w_h = 0.5 * MU_0 * dv * sum(jnp.sum(a * b) for a, b in
                                    zip(h_old, (st.hx, st.hy, st.hz)))
        st = apply_pec(update_e(st, mats, dt, dx, periodic))
        st, rlc = lumped.update_rlc_element(st, rlc, meta, e_prev)
        return (st, rlc), (w_e + w_h + w_l, w_l)

    _, (w, w_l) = jax.lax.scan(step, (st0, rlc0), None, length=N_STEPS)
    periods = N_STEPS * dt / (2 * np.pi * np.sqrt(L_H * C_F))
    return (np.asarray(w, dtype=np.float64), np.asarray(w_l, dtype=np.float64),
            periods)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_parallel_lc_in_a_pec_box_keeps_its_energy(precision):
    if precision == "float64":
        with _enable_x64(True):
            w, w_l, periods = _energy_trace(jnp.float64)
    else:
        w, w_l, periods = _energy_trace(jnp.float32)
    assert periods >= 10.0
    assert np.all(np.isfinite(w)) and w[0] > 0.0
    # The inductor must take part: an LC ring hands the whole energy to L
    # once per half period and holds half of it on average. An inductor
    # update that did nothing would leave a lossless folded C, and the
    # energy check alone would pass.
    share, peak = w_l.mean() / w[0], w_l.max() / w[0]
    assert INDUCTOR_MEAN_SHARE[0] <= share <= INDUCTOR_MEAN_SHARE[1] and peak >= INDUCTOR_PEAK_SHARE, (
        f"{precision}: the inductor held {share:.3f} of the energy on average and "
        f"{peak:.3f} at most (an L||C ring: ~0.5 and ~1)")
    drift = np.abs(w / w[0] - 1.0)
    n = int(np.argmax(drift))
    assert drift.max() <= BARS[precision], (
        f"{precision}: a lossless L||C changed its energy by {w[n] / w[0] - 1.0:+.3e} "
        f"at step {n} of {N_STEPS} ({periods:.1f} LC periods); bar {BARS[precision]:.0e}")
