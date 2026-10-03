"""``compute_far_field`` under ``jax.jit`` (#1364, the jit half).

``compute_far_field`` chose its implementation by checking three face arrays
for tracers, inside a bare ``try/except Exception: pass``. A box built inside
the caller's jit (``make_ntff_box`` converts the numpy frequencies with
``jnp.asarray(..., float32)``, which is recorded) carries traced frequencies,
so the check sent concrete face data to the numpy path, which then failed on
``np.asarray(box.freqs)``; with traced faces the jnp path failed on the same
read, the ``except`` swallowed it and the numpy path failed again, reported at
the wrong line. The dispatcher now checks every input leaf, with no
``except``, and ``compute_far_field_jax`` reads the frequency count from the
shape. The forward's own box is concrete under jit since #1367; this pins
the transform on its own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MM = 1e-3
THETA = np.linspace(0.2, 2.9, 4)
PHI = np.array([0.0, 1.5])
#: Far fields are sums over the surface and the record (PI 2026-09-25).
MAX_REL_SUMMED = 1.0e-4


def _rel_at_peak(plain, other):
    """``max|plain - other| / max|plain|``."""
    plain = np.asarray(plain).astype(np.complex128)
    other = np.asarray(other).astype(np.complex128)
    return float(np.max(np.abs(plain - other)) / np.max(np.abs(plain)))


def _far_field_inputs():
    from rfx.farfield import NTFFData, make_ntff_box
    from rfx.grid import Grid

    grid = Grid(freq_max=16e9, domain=(16 * MM, 16 * MM, 12 * MM), dx=1.0 * MM,
                cpml_layers=4)
    lo, hi = (1 * MM, 1 * MM, 1 * MM), (15 * MM, 15 * MM, 11 * MM)
    box = make_ntff_box(grid, lo, hi, np.array([7e9, 8e9]))
    rng = np.random.default_rng(7)
    faces = {}
    for name in NTFFData._fields:
        if name.startswith("c_"):
            continue
        shape = np.shape(getattr(_zero_data(box), name))
        faces[name] = jnp.asarray(rng.standard_normal(shape)
                                  + 1j * rng.standard_normal(shape), jnp.complex64)
    data = _zero_data(box)._replace(**faces)
    return grid, lo, hi, box, data


def _zero_data(box):
    from rfx.farfield import init_ntff_data
    return init_ntff_data(box)


def test_compute_far_field_dispatches_on_a_traced_frequency_list():
    """A box built inside the caller's jit carries traced frequencies while
    the face data are concrete: the dispatcher must take the jnp path (it
    checked three face arrays and hid the failure under a bare ``except``),
    and the pattern must equal the host path's."""
    from rfx.farfield import compute_far_field, make_ntff_box

    grid, lo, hi, box, data = _far_field_inputs()
    host = compute_far_field(data, box, grid, THETA, PHI)

    def pattern():
        traced_box = make_ntff_box(grid, lo, hi, np.array([7e9, 8e9]))
        ff = compute_far_field(data, traced_box, grid, THETA, PHI)
        return ff.E_theta, ff.E_phi

    e_th, e_ph = jax.jit(pattern)()
    assert _rel_at_peak(host.E_theta, e_th) <= MAX_REL_SUMMED
    assert _rel_at_peak(host.E_phi, e_ph) <= MAX_REL_SUMMED


def test_compute_far_field_jax_under_jit_of_the_gradient_with_numpy_freqs():
    """``compute_far_field_jax`` reads the frequency count statically, so a
    box built from numpy frequencies inside ``jax.jit(jax.grad(...))`` (the
    issue's workaround, moved into the objective) traces and matches the
    plain gradient."""
    from rfx.farfield import compute_far_field_jax, make_ntff_box

    grid, lo, hi, _box, data = _far_field_inputs()

    def loss(scale):
        box = make_ntff_box(grid, lo, hi, np.array([7e9, 8e9]))
        scaled = data._replace(**{f: getattr(data, f) * scale
                                  for f in ("x_lo", "x_hi", "y_lo", "y_hi",
                                            "z_lo", "z_hi")})
        ff = compute_far_field_jax(scaled, box, grid, THETA, PHI)
        return jnp.sum(jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2)

    v, g = jax.value_and_grad(loss)(jnp.float32(1.5))
    vj, gj = jax.jit(jax.value_and_grad(loss))(jnp.float32(1.5))
    assert float(g) > 0.0
    assert _rel_at_peak(v, vj) <= MAX_REL_SUMMED
    assert _rel_at_peak(g, gj) <= MAX_REL_SUMMED
