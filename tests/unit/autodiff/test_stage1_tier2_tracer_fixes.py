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
from rfx.geometry.rasterize_grid import (
    GridCoords, _material_cell_mask, axis_cell_sizes, cell_centres_from_nodes,
)
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


def test_material_box_traceable_with_traced_coords():
    """The material centre window accepts differentiable mesh coordinates."""
    box = Box((0.002, 0.0, 0.0), (0.008, 0.01, 0.01))
    y = jnp.linspace(0.0, 0.01, 6)
    z = jnp.linspace(0.0, 0.01, 6)

    def masked_count(x):
        co = GridCoords(x, y, z, (len(x), len(y), len(z)))
        widths = tuple(axis_cell_sizes(c) for c in co[:3])
        m = _material_cell_mask(box, co, cell_centres_from_nodes(co, widths),
                                cell_sizes=widths)
        return jnp.sum(m.astype(jnp.float32))

    x = jnp.linspace(0.0, 0.01, 10)
    cnt = jax.jit(masked_count)(x)
    assert np.isfinite(float(cnt))


def test_material_box_forward_bit_identical():
    """Material occupancy agrees with a direct centre window on the drawing."""
    box = Box((0.002, 0.0, 0.0), (0.0085, 0.01, 0.01))
    x = jnp.linspace(0.0, 0.012, 24)
    y = jnp.linspace(0.0, 0.01, 6)
    z = jnp.linspace(0.0, 0.01, 6)
    co = GridCoords(x, y, z, (len(x), len(y), len(z)))
    widths = tuple(axis_cell_sizes(c) for c in co[:3])
    m = np.asarray(_material_cell_mask(
        box, co, cell_centres_from_nodes(co, widths), cell_sizes=widths))
    # x-extent (6.5 mm) spans many cells -> volume path, half-open [lo, hi).
    xh = np.asarray(x, dtype=np.float64)
    yh = np.asarray(y, dtype=np.float64)
    zh = np.asarray(z, dtype=np.float64)
    def direct_centre_mask(coords, lo, hi):
        widths = np.diff(coords)
        widths = np.concatenate((widths, widths[-1:]))

        def band(face):
            return 1e-9 * widths + np.minimum(
                16 * float(np.finfo(np.float32).eps) * abs(face), 5e-4 * widths)

        centres = coords + widths / 2
        return (centres >= lo - band(lo)) & (centres < hi - band(hi))

    mx = direct_centre_mask(xh, 0.002, 0.0085)
    my = direct_centre_mask(yh, 0.0, 0.01)
    mz = direct_centre_mask(zh, 0.0, 0.01)
    expected = mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    np.testing.assert_array_equal(m, expected)


def test_pec_and_dielectric_share_the_volume_sampler():
    """Both material and PEC cell ownership round off-node faces alike."""
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import (
        GridCoords, cell_centres_from_nodes, classify_pec_entry,
    )

    dx = 1e-3
    axis = np.arange(11, dtype=np.float64) * dx      # nodes 0 .. 10 mm
    co = GridCoords(x=axis, y=axis, z=axis, shape=(11, 11, 11))
    box = Box((0.0023, 0.0023, 0.0023), (0.0063, 0.0063, 0.0063))

    node = np.asarray(_material_cell_mask(box, co, cell_centres_from_nodes(co)))
    assert np.where(node.any(axis=(1, 2)))[0].tolist() == [2, 3, 4, 5]

    cells, sheet, wire = classify_pec_entry(
        box, co, cell_centres_from_nodes(co), name="body")
    assert sheet is None and wire is None
    cells_np = np.asarray(cells)
    assert np.where(cells_np.any(axis=(1, 2)))[0].tolist() == [2, 3, 4, 5]

    edges = realized_pec_edge_masks(cells)
    for a in range(3):
        assert realized_wall_planes(edges, a) == [2, 3, 4, 5, 6], (
            "walls must sit on BOTH drawn faces, 2 mm and 6 mm")


def test_probes_update_sparam_probe_traceable_with_traced_step():
    """probes: update_sparam_probe must not float() the traced
    state.step (a tracer inside the jitted scan)."""
    grid = Grid(freq_max=5e9, domain=(17 * (299792458.0 / 5e9 / 20.0), 17 * (299792458.0 / 5e9 / 20.0), 9 * (299792458.0 / 5e9 / 20.0)), cpml_layers=0)  # main's solved length; #1887
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
