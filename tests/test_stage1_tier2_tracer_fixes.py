"""Stage 1 (Tier-2) — differentiable-path tracer-break regression tests.

Three rfx functions used to force a traced value to a Python ``float``,
raising ``TracerArrayConversionError`` / ``ConcretizationError`` on the
differentiable path:

* GEO-C3 — ``nonuniform.make_current_source`` did
  ``float(materials.eps_r[...])`` — broke the differentiable-material path.
* ``geometry.csg.Box.mask_on_coords`` did ``np.asarray(coords)`` — broke
  the differentiable-mesh path.
* ``probes.update_sparam_probe`` did ``float(state.step)`` — a tracer
  leak when the probe update runs inside the jitted scan.

Each fix keeps the forward output bit-identical for non-traced inputs
and stays traceable for traced inputs. The fourth test pins the
``kottke_inv_eps_from_occupancy`` hard ``f<1e-3`` clamp behavior, which
was deliberately KEPT (2026-05-17 decision — the clamp's docstring
defends it as functionally smooth).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import MaterialArrays, init_materials, init_state
from rfx.geometry.csg import Box
from rfx.geometry.smoothing import kottke_inv_eps_from_occupancy
from rfx.grid import Grid
from rfx.nonuniform import make_current_source, make_nonuniform_grid
from rfx.probes.probes import init_sparam_probe, update_sparam_probe
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port


def test_geoc3_make_current_source_traceable_with_traced_materials():
    """GEO-C3: make_current_source must not float() traced materials.

    jax.grad through the source-waveform normalisation (which depends on
    eps_r via Cb) raised TracerArrayConversionError pre-fix.
    """
    dz_profile = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
    grid = make_nonuniform_grid((0.02, 0.02), dz_profile, 0.5e-3, 8)
    shape = (grid.nx, grid.ny, grid.nz)
    pulse = GaussianPulse(f0=2.4e9, bandwidth=0.8)

    def waveform_energy(eps_scale):
        materials = MaterialArrays(
            eps_r=jnp.ones(shape) * eps_scale,   # traced
            sigma=jnp.zeros(shape),
            mu_r=jnp.ones(shape),
        )
        src = make_current_source(grid, (10, 10, 5), "ez", pulse, 40, materials)
        return jnp.sum(jnp.abs(src[4]))

    grad = jax.grad(waveform_energy)(4.4)
    assert np.isfinite(float(grad))
    # Cb ~ 1/eps, so the waveform energy genuinely depends on eps_scale.
    assert float(grad) != 0.0


def test_csg_box_mask_on_coords_traceable_with_traced_coords():
    """Box.mask_on_coords must trace — np.asarray(coords) raised
    TracerArrayConversionError on the differentiable-mesh path."""
    box = Box((0.002, 0.0, 0.0), (0.008, 0.01, 0.01))
    y = jnp.linspace(0.0, 0.01, 6)
    z = jnp.linspace(0.0, 0.01, 6)

    def masked_count(x):
        m = box.mask_on_coords(x, y, z)
        return jnp.sum(m.astype(jnp.float32))

    x = jnp.linspace(0.0, 0.01, 10)
    cnt = jax.jit(masked_count)(x)
    assert np.isfinite(float(cnt))


def test_csg_box_mask_on_coords_forward_bit_identical():
    """The traceable rewrite is byte-identical to a direct volume mask
    on concrete coordinates (the non-thin-sheet case)."""
    box = Box((0.002, 0.0, 0.0), (0.0085, 0.01, 0.01))
    x = jnp.linspace(0.0, 0.012, 24)
    y = jnp.linspace(0.0, 0.01, 6)
    z = jnp.linspace(0.0, 0.01, 6)
    m = np.asarray(box.mask_on_coords(x, y, z))
    # x-extent (6.5 mm) spans many cells -> volume path, half-open [lo, hi).
    mx = np.asarray((x >= 0.002) & (x < 0.0085))
    expected = mx[:, None, None] & np.asarray(
        (y >= 0.0) & (y < 0.01))[None, :, None] & np.asarray(
        (z >= 0.0) & (z < 0.01))[None, None, :]
    np.testing.assert_array_equal(m, expected)


def test_probes_update_sparam_probe_traceable_with_traced_step():
    """probes: update_sparam_probe must not float() the traced
    state.step (a tracer inside the jitted scan)."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    port = LumpedPort(
        position=(0.025, 0.025, 0.0125), component="ez",
        impedance=50.0, excitation=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    materials = setup_lumped_port(grid, port, materials)
    freqs = jnp.linspace(1e9, 5e9, 8)
    sprobe = init_sparam_probe(grid, port, freqs, dft_total_steps=100)

    def run_one(step):
        st = state._replace(step=step)
        sp = update_sparam_probe(sprobe, st, grid, port, grid.dt)
        return jnp.sum(jnp.abs(sp.v_dft))

    out = jax.jit(run_one)(jnp.asarray(5, dtype=jnp.int32))
    assert np.isfinite(float(out))


def test_kottke_clamp_tail_gradient_is_pinned_zero():
    """kottke_inv_eps_from_occupancy keeps the hard f<1e-3 clamp
    (smoothing.py:856). Pin the documented behavior: an occupancy cell
    in the clamped tail (f<1e-3) has exactly zero gradient; the output
    stays finite. (2026-05-17 decision: keep the clamp.)"""
    grid = Grid(freq_max=1e9, domain=(8e-4, 8e-4, 8e-4), dx=1e-4,
                cpml_layers=0)

    def loss(occ):
        inv_xx, _, _ = kottke_inv_eps_from_occupancy(grid, occ)
        return jnp.sum(inv_xx)

    occ = jnp.zeros((8, 8, 8), dtype=jnp.float32)
    occ = occ.at[2, 2, 2].set(5e-4)   # deep in the clamped tail (f < 1e-3)
    occ = occ.at[5, 5, 5].set(0.5)    # well above the threshold

    grad = jax.grad(loss)(occ)
    assert np.all(np.isfinite(np.asarray(grad)))
    # Tail cell: the f<1e-3 clamp zeroes its gradient — by design.
    assert float(grad[2, 2, 2]) == 0.0
