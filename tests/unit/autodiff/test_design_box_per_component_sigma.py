"""A design box whose conductivity is per E COMPONENT (sheet lane, #1179).

``materials.sigma`` is one array per cell index, multiplied into all three E
components, and the design box inherited that: one ``sigma`` array built one
``(Ca, Cb)`` pair for ex, ey and ez alike. A conducting SHEET is not that
material. A zero-thickness sheet carries current only along its two in-plane
edges; the edge through the sheet carries none. A design variable per
in-plane edge therefore has to leave the third component on the background
coefficient, or the "sheet" is a conducting slab one cell thick.

What is pinned here:

* the LOCK — ``sigma=(s, s, s)`` runs bit for bit as ``sigma=s`` does, fields
  and S-parameters, so the existing single-array lane is untouched;
* the routing — ``sigma_x`` alone suppresses ex inside the box and leaves ey
  alone. Both halves are asserted, so swapping the component index in
  ``update_e_box`` reds the test rather than moving the suppression to the
  other component unnoticed;
* the same routing at the coefficient level, by calling ``update_e_box``
  directly on a known state, where the expected array is written out rather
  than produced by the code under test.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import GaussianPulse, Simulation
from rfx.core.yee import FDTDState, e_update_coeffs, update_e_box

F0 = 8e9
DX = 2e-3
CPML = 5
DOMAIN = (24e-3, 20e-3, 16e-3)
BOX_LO = (10e-3, 8e-3, 6e-3)
BOX_HI = (14e-3, 12e-3, 10e-3)
N_STEPS = 60


def _sim(with_probes=False):
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_port(position=(4e-3, 10e-3, 8e-3), component="ez",
                 impedance=50.0,
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    if with_probes:
        # Both inside the box, one per in-plane component, and OFF the
        # y = 10 mm plane the ez port makes a symmetry plane (ey is
        # identically zero on it, which would make half this test vacuous).
        sim.add_probe((12e-3, 11e-3, 8e-3), "ex")
        sim.add_probe((12e-3, 11e-3, 8e-3), "ey")
    return sim


def _box_shape(sim):
    grid = sim._build_grid()
    lo = grid.position_to_index(BOX_LO)
    hi = grid.position_to_index(BOX_HI)
    return tuple(hi[d] - lo[d] + 1 for d in range(3))


def _run(sim, eps, sigma, n_steps=N_STEPS):
    return sim.forward(design_box=(BOX_LO, BOX_HI),
                       design_eps_override=eps,
                       design_sigma_override=sigma,
                       n_steps=n_steps, checkpoint=False,
                       skip_preflight=True,
                       port_s11_freqs=jnp.asarray([F0], jnp.float32))


# ---------------------------------------------------------------------------
# the lock
# ---------------------------------------------------------------------------

def test_three_equal_conductivities_reproduce_the_single_array_lane():
    """``(s, s, s)`` is the single-array run, bit for bit."""
    sim = _sim(with_probes=True)
    shape = _box_shape(sim)
    rng = np.random.default_rng(3)
    eps = jnp.asarray(2.0 + 2.0 * rng.random(shape), jnp.float32)
    s = jnp.asarray(0.01 + 0.05 * rng.random(shape), jnp.float32)

    one = _run(sim, eps, s)
    three = _run(sim, eps, (s, s, s))

    a = np.asarray(one.time_series)
    b = np.asarray(three.time_series)
    assert a.shape == b.shape
    assert np.abs(a).max() > 0.0, "fixture excites nothing"
    assert np.array_equal(a, b), (
        f"time series differ by {np.abs(a - b).max():.3e}")
    sa = np.asarray(one.s_params)
    sb = np.asarray(three.s_params)
    assert np.abs(sa).max() > 0.0
    assert np.array_equal(sa, sb), (
        f"S-parameters differ by {np.abs(sa - sb).max():.3e}")


def test_a_conductivity_tuple_of_the_wrong_length_is_refused():
    sim = _sim()
    shape = _box_shape(sim)
    eps = jnp.ones(shape, jnp.float32)
    s = jnp.full(shape, 0.02, jnp.float32)
    with pytest.raises(ValueError, match="3-tuple"):
        _run(sim, eps, (s, s))


def test_a_conductivity_component_of_the_wrong_shape_is_refused():
    sim = _sim()
    shape = _box_shape(sim)
    eps = jnp.ones(shape, jnp.float32)
    s = jnp.full(shape, 0.02, jnp.float32)
    bad = jnp.full((shape[0] + 1,) + shape[1:], 0.02, jnp.float32)
    with pytest.raises(ValueError, match="sigma_y"):
        _run(sim, eps, (s, bad, s))


# ---------------------------------------------------------------------------
# the routing
# ---------------------------------------------------------------------------

def test_a_conductivity_on_x_alone_suppresses_ex_and_leaves_ey_alone():
    """The component the design conductivity names is the one that is loaded.

    A swap of the component index in ``update_e_box`` keeps the first half
    of this test (something IS suppressed) and breaks the second (ey moves
    instead of ex), which is why both are asserted.
    """
    sim = _sim(with_probes=True)
    shape = _box_shape(sim)
    eps = jnp.ones(shape, jnp.float32)
    quiet = jnp.zeros(shape, jnp.float32)
    loud = jnp.full(shape, 1.0e3, jnp.float32)   # sigma*dt/2eps >> 1

    base = np.asarray(_run(sim, eps, (quiet, quiet, quiet)).time_series)
    x_only = np.asarray(_run(sim, eps, (loud, quiet, quiet)).time_series)
    y_only = np.asarray(_run(sim, eps, (quiet, loud, quiet)).time_series)

    # time_series is (n_steps, n_probe): column 0 is ex, column 1 is ey,
    # both at the same point inside the box
    assert base.shape[1] == 2
    ex0, ey0 = np.abs(base[:, 0]).max(), np.abs(base[:, 1]).max()
    assert ex0 > 0.0 and ey0 > 0.0, "fixture drives neither in-plane edge"

    ex_x, ey_x = np.abs(x_only[:, 0]).max(), np.abs(x_only[:, 1]).max()
    ex_y, ey_y = np.abs(y_only[:, 0]).max(), np.abs(y_only[:, 1]).max()

    # sigma_x loads ex hard and leaves ey close to where it was
    assert ex_x < 0.05 * ex0, f"ex only fell to {ex_x / ex0:.3f} of itself"
    assert ey_x > 0.5 * ey0, f"ey fell to {ey_x / ey0:.3f} under sigma_x"
    # and the mirror image
    assert ey_y < 0.05 * ey0, f"ey only fell to {ey_y / ey0:.3f} of itself"
    assert ex_y > 0.5 * ex0, f"ex fell to {ex_y / ex0:.3f} under sigma_y"


def test_update_e_box_applies_each_component_its_own_coefficients():
    """The routing at the kernel, against coefficients written out by hand."""
    shape = (4, 4, 4)
    box = (1, 3, 1, 3, 1, 3)
    bs = (2, 2, 2)
    rng = np.random.default_rng(7)

    def f(x):
        return jnp.asarray(x, jnp.float32)

    prev = FDTDState(
        ex=f(rng.random(shape)), ey=f(rng.random(shape)), ez=f(rng.random(shape)),
        hx=f(rng.random(shape)), hy=f(rng.random(shape)), hz=f(rng.random(shape)),
        step=0,
    )
    dt, dx = 1.0e-12, 1.0e-3
    eps = jnp.full(bs, 2.0, jnp.float32)
    sig = (jnp.full(bs, 1.0e2, jnp.float32),
           jnp.full(bs, 1.0e-2, jnp.float32),
           jnp.full(bs, 3.0e1, jnp.float32))
    ca, cb = zip(*(e_update_coeffs(eps, s, dt) for s in sig))

    out = update_e_box(prev, prev, box, tuple(ca), tuple(cb), dx)

    from rfx.core.yee import curl_h
    cx, cy, cz = curl_h(prev.hx, prev.hy, prev.hz, dx, (False, False, False))
    sl = (slice(1, 3), slice(1, 3), slice(1, 3))
    for c, (field, curl) in enumerate((("ex", cx), ("ey", cy), ("ez", cz))):
        want = ca[c] * getattr(prev, field)[sl] + cb[c] * curl[sl]
        got = getattr(out, field)[sl]
        assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-6), (
            f"{field} did not use coefficient {c}")
    # the three coefficient pairs are genuinely different, so a swap moves a
    # number: without this the loop above would pass on equal coefficients
    assert not np.allclose(np.asarray(ca[0]), np.asarray(ca[1]))
    assert not np.allclose(np.asarray(ca[1]), np.asarray(ca[2]))
