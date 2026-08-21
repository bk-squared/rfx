"""#689 site 1 — the NU wire-port Ampere loop must not wrap at index 0.

Defect (external review against 4eb7fa4).
``rfx.nonuniform.wire_port_current`` spelled each backward H read as a raw
``h[mi - 1, ...]``. At ``mi == 0`` that is Python's negative index, i.e. H
at the OPPOSITE face of the domain. Two things are wrong with it at once:

  * a 4-term discrete Ampere loop is a LOCAL operator — its value cannot
    depend on a field cell it does not enclose;
  * the port current is supposed to be the loop integral of the very curl
    the NU E-update integrates, and ``_curl_h_nu`` builds its backward
    differences from ``rfx.core.yee._shift_bwd``, which pads with ZERO.

A wire port lands at index 0 whenever it sits flush against a PEC / PMC /
periodic face — those faces get no CPML pad (``rfx/nonuniform.py``).

ORACLE 1 — LOCALITY. Perturb H by +1000 at the far face and re-read the
current. Expected delta 0.0 under EVERY convention (wrap, zero-pad,
PEC-pad), so the oracle refutes non-locality only and cannot be gamed by
choosing a convention. Measured on the 8x8x8 fixture below, dual = 1e-3:

    comp  index      perturbed        4eb7fa4     fixed
    ez    (0,4,4)    hy(7,4,4)        -1.000000   +0.000000
    ez    (4,0,4)    hx(4,7,4)        +1.000000   +0.000000
    ex    (4,4,0)    hy(4,4,7)        +1.000000   +0.000000
    ex    (4,0,4)    hz(4,7,4)        -1.000000   +0.000000
    ey    (4,4,0)    hx(4,4,7)        -1.000000   +0.000000
    ey    (0,4,4)    hz(7,4,4)        +1.000000   +0.000000

All six branches were non-local, one per (component, wrapping axis) pair.

ORACLE 2 — SOLVER-STENCIL CONSISTENCY. Independent expression built from
``_shift_bwd``, not from the port code. On the same fixture:

    (hy - _shift_bwd(hy, 0))[0,4,4] = 0.8767164349555969   <- the oracle
    hy[0,4,4] - hy[7,4,4]           = 2.4202959537506104   <- 4eb7fa4
    hy[0,4,4]                       = 0.8767164349555969   <- fixed

Interior indices are untouched by either oracle, which is why no committed
#672 number moves (that module tests at LOOP_IDX = (K_STEP, K_STEP,
K_STEP) and at (6,6,6) / (2,2,2), all interior). The interior-agreement
test below pins that explicitly.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import _shift_bwd
from rfx.nonuniform import wire_port_current

N = 8
DUAL = (1.0e-3, 1.0e-3, 1.0e-3)


def _fields(seed=689):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal((N, N, N)).astype(np.float32))
                 for _ in range(3))


# (component, port index, H component perturbed, far-face index)
NONLOCAL_CASES = [
    ("ez", (0, 4, 4), 1, (N - 1, 4, 4)),
    ("ez", (4, 0, 4), 0, (4, N - 1, 4)),
    ("ex", (4, 4, 0), 1, (4, 4, N - 1)),
    ("ex", (4, 0, 4), 2, (4, N - 1, 4)),
    ("ey", (4, 4, 0), 0, (4, 4, N - 1)),
    ("ey", (0, 4, 4), 2, (N - 1, 4, 4)),
]


@pytest.mark.parametrize("comp,idx,h_i,far", NONLOCAL_CASES)
def test_wire_port_current_is_local_at_index_zero(comp, idx, h_i, far):
    """ORACLE 1 — a 4-term loop cannot see the opposite domain face."""
    h = list(_fields())
    before = float(wire_port_current(*h, comp, *idx, *DUAL))
    h2 = list(h)
    h2[h_i] = h[h_i].at[far].add(1000.0)
    after = float(wire_port_current(*h2, comp, *idx, *DUAL))
    assert after - before == 0.0, (comp, idx, after - before)


def test_wire_port_current_matches_the_solver_curl_at_index_zero():
    """ORACLE 2 — the loop legs equal the E-update's own backward stencil.

    Built from ``_shift_bwd`` directly; it is not a copy of the fix.
    """
    hx, hy, hz = _fields()
    dx, dy, dz = DUAL

    # ez at mi == 0: leg 1 differences hy along x, leg 2 differences hx
    # along y (interior, so it is a control on the untouched leg).
    idx = (0, 4, 4)
    expect = (float((hy - _shift_bwd(hy, 0))[idx]) * dy
              - float((hx - _shift_bwd(hx, 1))[idx]) * dx)
    got = float(wire_port_current(hx, hy, hz, "ez", *idx, *DUAL))
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-12), (got, expect)

    # ex at mk == 0
    idx = (4, 4, 0)
    expect = (float((hz - _shift_bwd(hz, 1))[idx]) * dz
              - float((hy - _shift_bwd(hy, 2))[idx]) * dy)
    got = float(wire_port_current(hx, hy, hz, "ex", *idx, *DUAL))
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-12), (got, expect)

    # ey at mi == 0
    idx = (0, 4, 4)
    expect = (float((hx - _shift_bwd(hx, 2))[idx]) * dx
              - float((hz - _shift_bwd(hz, 0))[idx]) * dz)
    got = float(wire_port_current(hx, hy, hz, "ey", *idx, *DUAL))
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-12), (got, expect)


def test_interior_indices_are_bit_identical_to_the_pre_689_spelling():
    """No committed #672 number moves: every interior index is untouched."""
    hx, hy, hz = _fields()
    dx, dy, dz = DUAL

    def old(comp, mi, mj, mk):
        if comp == "ez":
            return ((hy[mi, mj, mk] - hy[mi - 1, mj, mk]) * dy
                    - (hx[mi, mj, mk] - hx[mi, mj - 1, mk]) * dx)
        if comp == "ex":
            return ((hz[mi, mj, mk] - hz[mi, mj - 1, mk]) * dz
                    - (hy[mi, mj, mk] - hy[mi, mj, mk - 1]) * dy)
        return ((hx[mi, mj, mk] - hx[mi, mj, mk - 1]) * dx
                - (hz[mi, mj, mk] - hz[mi - 1, mj, mk]) * dz)

    n = 0
    for comp in ("ex", "ey", "ez"):
        for mi in range(1, N):
            for mj in range(1, N):
                for mk in range(1, N):
                    a = float(old(comp, mi, mj, mk))
                    b = float(wire_port_current(hx, hy, hz, comp,
                                                mi, mj, mk, *DUAL))
                    assert a == b, (comp, mi, mj, mk, a, b)
                    n += 1
    assert n == 3 * (N - 1) ** 3


def test_length_one_axis_difference_is_zero_not_a_zero_pad():
    """A z-invariant (2-D-shaped) field must give a zero z-difference, the
    way the uniform lane's wrap does — not h[k] against a zero pad."""
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.standard_normal((N, N, 1)).astype(np.float32))
    b = jnp.asarray(rng.standard_normal((N, N, 1)).astype(np.float32))
    c = jnp.asarray(rng.standard_normal((N, N, 1)).astype(np.float32))
    dx, dy, dz = DUAL
    idx = (4, 4, 0)
    # ex: leg 2 differences hy along z -> must contribute exactly 0
    got = float(wire_port_current(a, b, c, "ex", *idx, *DUAL))
    expect = float((c[idx] - c[4, 3, 0])) * dz
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-12), (got, expect)
