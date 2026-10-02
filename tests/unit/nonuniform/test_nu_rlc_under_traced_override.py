"""A graded-mesh board with a lumped RLC element can be differentiated
through ``eps_override`` (#1373).

The board of ``test_nu_drive_sees_override_1267.py``: a 50 ohm wire port
feeding a patch through two substrate layers (eps_r 3.38 under 10.2) on a
graded z mesh, here with a lumped element on the port's top Ez edge -- a
0.3 pF capacitor in parallel, or a 20 ohm + 0.5 nH series branch one cell
up. An element enters Ampere's law through the E-update denominator of its
own edge, ``D0 = eps/dt + sigma/2`` (#1163), so it reads the permittivity
the override supplies. The graded lane builds its element records with the
concrete builder for ``run()`` and ``forward()`` alike, and that builder
turned the edge's eps and sigma into Python floats: under ``jax.grad``
through the override they are tracers, and the run raised
``ConcretizationTypeError`` before the first step. A user could not take
the gradient of a graded board that carried a lumped element. A concrete
material still takes the Python-float arithmetic, byte for byte as before;
a traced one keeps the arrays' dtype, as the uniform lane's traced builder
does.

What is checked, per element:
1. the value under the trace equals the eager value of the same override,
   within the cross-trace bar for a summed quantity (1e-4 of its size);
2. d|S11|^2/d(ln eps_r of the 10.2 layer) at 9 GHz from ``jax.grad``
   through the override equals a Richardson central difference of boards
   BUILT with the perturbed eps_r (h = 0.02, 0.01; 0.04 as the ladder
   check) -- a route that never touches the override -- within 1e-3.
Mutation (b): restore the unconditional ``float()`` in
``rfx.lumped.edge_update_denominator`` (every builder call kept) and both
cases raise again.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._x64_compat import enable_x64
from tests.unit.nonuniform import test_nu_drive_sees_override_1267 as B

GRAD_RTOL = 1e-3
VALUE_TOL = 1e-4


def _board(eps_lo, eps_hi, element):
    if element == "parallel_C":
        return B._board(eps_lo, eps_hi, port="wire", cap=0.3e-12)
    sim = B._board(eps_lo, eps_hi, port="wire")
    # a series R-L branch on the Ez edge one cell above the port's top edge
    sim.add_lumped_rlc((B.I_PORT * B.DX, B.J_PORT * B.DX, B.Z_G + 3 * B.DX),
                       "ez", R=20.0, L=0.5e-9, topology="series")
    return sim


def _mask():
    eps0 = np.asarray(B._drawn(B._board(B.EPS_LO, B.EPS_HI)).eps_r)
    return (eps0 > 0.5 * (B.EPS_LO + B.EPS_HI)).astype(float)  # the 10.2 layer


def _s11sq(sim, eps_override=None):
    return B._run_obs(sim, "upper", eps_override)


@functools.lru_cache(maxsize=None)
def _ad(element):
    """(eager value, value under the trace, jax.grad) through the override."""
    with enable_x64():
        sim = _board(B.EPS_LO, B.EPS_HI, element)
        eps0 = jnp.asarray(B._drawn(sim).eps_r)
        mask = jnp.asarray(_mask())

        def f(a):
            return _s11sq(sim, eps0 * jnp.exp(a * mask))

        eager = float(f(jnp.asarray(0.0)))
        traced, g = jax.value_and_grad(f)(jnp.asarray(0.0))
        return eager, float(traced), float(g)


@functools.lru_cache(maxsize=None)
def _fd_built(element):
    """Central differences of boards BUILT with the 10.2 layer x e^(+-h)."""
    with enable_x64():
        def obs(a):
            return float(_s11sq(_board(B.EPS_LO, B.EPS_HI * np.exp(a), element)))
        d = {h: (obs(h) - obs(-h)) / (2 * h) for h in B.FD_LADDER}
    h0, h1, h2 = B.FD_LADDER
    return d, (4 * d[h2] - d[h1]) / 3, (4 * d[h1] - d[h0]) / 3


@pytest.mark.parametrize("element", ["parallel_C", "series_RL"])
def test_a_graded_board_with_a_lumped_element_differentiates_through_the_override(
        element):
    eager, traced, g = _ad(element)
    assert abs(traced - eager) <= VALUE_TOL * abs(eager), (traced, eager)
    d, rich, rich_coarse = _fd_built(element)
    rel = abs(g - rich) / abs(rich)
    print(f"[{element}] |S11|^2 {eager:.6f} (traced {traced:.6f}); AD {g:.6f}; "
          f"FD built {d}; Richardson {rich:.6f} (coarse {rich_coarse:.6f}); "
          f"|AD/FD - 1| = {rel:.2e}")
    assert abs(rich_coarse - rich) / abs(rich) < 5 * GRAD_RTOL
    assert rel <= GRAD_RTOL, (g, rich)
