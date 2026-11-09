"""Opaque ownership, hand-counted edges and high-face ghost contracts."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_wall_planes, wall_edge_masks
from rfx.model.conductors import realized_conductors
from rfx.model.wall_loss import build_wall_operator


def model(**kwargs):
    return Simulation(freq_max=10e9, domain=(.006,)*3, dx=.0015,
                      boundary="pec", snap="declared", **kwargs)


def operator(sim, graded=False):
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    root = realized_conductors(sim, grid, nonuniform=graded)
    return grid, root, build_wall_operator(sim, root, None, lane="contract")


@pytest.mark.parametrize("selection", ["domain", "solid"])
def test_realized_masks_planes_conductor_and_port_admission_unchanged(selection):
    sim = model()
    solid = Box((0., 0., 0.), (.006, .0015, .006))
    sim.add(solid, material="pec")
    sim.add_port((.003, .003, .003), component="ey", impedance=50)
    grid, before, _ = operator(sim)
    edges = tuple(np.array(a) for a in before.edges)
    cells = np.array(sim.conductor_mask(grid))
    findings = tuple(str(f) for f in sim.preflight())
    declaration = {"shape": solid} if selection == "solid" else {"faces": ("y_lo", "y_hi")}
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, **declaration)
    _, after, ctx = operator(sim)
    for old, new in zip(edges, after.edges):
        np.testing.assert_array_equal(old, new)
    for axis in range(3):
        assert realized_wall_planes(edges, axis) == realized_wall_planes(after.edges, axis)
    np.testing.assert_array_equal(cells, sim.conductor_mask(grid))
    assert tuple(str(f) for f in sim.preflight()) == findings
    domain = wall_edge_masks(grid.shape, sim._boundary_spec.pec_faces())
    for mask, owned, wall in zip((ctx.mask_ex, ctx.mask_ey, ctx.mask_ez), after.edges, domain):
        assert not np.any(np.asarray(mask) & ~(np.asarray(owned) | wall))


def test_tiny_face_edges_and_rim_sum():
    sim = model()
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    _, _, ctx = operator(sim)
    # Four physical cells per axis: each y face owns 4*3 Ex and 3*4 Ez.
    assert [int(np.asarray(m).sum()) for m in (ctx.mask_ex, ctx.mask_ey, ctx.mask_ez)] == [24, 0, 24]
    assert not np.asarray(ctx.mask_ex)[:, :, 0].any()  # undeclared z rim stays PEC
    rim = model()
    rim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("x_lo", "y_lo"))
    _, _, ctx = operator(rim)
    assert bool(ctx.mask_ez[0, 0, 1])
    # Independent SI closed form, two full 1.5 mm E-curl spacings.
    expected = 2 / np.sqrt(np.pi * 8e9 * (4*np.pi*1e-7) / 5.8e5) / .0015
    assert float(ctx.sigma_sheet[2, 0, 0, 1]) == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform-forward-step", "graded-forward-step"])
@pytest.mark.parametrize("axis", [0, 1], ids=["x_hi", "y_hi"])
def test_declared_high_normal_ghost_stays_exactly_zero(monkeypatch, graded, axis):
    from rfx.stepping import uniform_setup, uniform_loop, graded_setup, graded_loop
    from rfx.core.yee import curl_h_nu
    kwargs = {"dy_profile": np.full(4, .0015)} if graded else {}
    sim = model(**kwargs)
    face = "xyz"[axis] + "_hi"
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=(face,))
    grid, root, ctx = operator(sim, graded)
    setup = graded_setup if graded else uniform_setup
    original = setup.init_state
    seeded_states = []

    def seeded(*args, **kwargs):
        state = original(*args, **kwargs)
        index = [2, 2, 2]
        index[axis] = -2
        hz = state.hz.at[tuple(index)].set(1.)
        # Graded inv_d*_h[-1] is zero: an interior seed cannot drive the
        # stored high H row. Seed that row too so its transverse curl drives
        # the normal E ghost, whose initial value remains zero.
        index[axis] = -1
        state = state._replace(hz=hz.at[tuple(index)].set(1.))
        seeded_states.append(state)
        return state

    monkeypatch.setattr(setup, "init_state", seeded)
    if graded:
        result = graded_loop.run_nonuniform(grid, root.materials, 4, pec_faces=set(sim._boundary_spec.pec_faces()), sheet_impedance=ctx)
        state = result["state"]
    else:
        state = uniform_loop.run(grid, root.materials, 4, sheet_impedance=ctx).state
    ghost = np.take(np.asarray((state.ex, state.ey)[axis]), -1, axis=axis)
    assert len(seeded_states) == 1
    if graded:
        assert float((grid.inv_dx_h, grid.inv_dy_h)[axis][-1]) == 0
        seed = seeded_states[0]
        curls = curl_h_nu(seed.hx, seed.hy, seed.hz, grid.inv_dx, grid.inv_dy, grid.inv_dz)
        assert np.max(np.abs(np.take(np.asarray(curls[axis]), -1, axis=axis))) > 0
    np.testing.assert_array_equal(ghost, np.zeros_like(ghost))
    assert np.max(np.abs(np.asarray(state.hz))) > 0


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("selection", ["domain", "solid"])
def test_run_forward_share_wall_operator(graded, selection):
    from rfx import GaussianPulse
    kwargs = {"dy_profile": np.full(4, .0015)} if graded else {}
    sim = model(**kwargs)
    if selection == "solid":
        slab = Box((0., 0., 0.), (.006, .0015, .006))
        sim.add(slab, material="pec")
        sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, shape=slab)
    else:
        sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    sim.add_source((.003, .003, .003), "ey", waveform=GaussianPulse(f0=8e9, bandwidth=.25), amplitude_kind="current")
    sim.add_probe((.003, .003, .003), "ey")
    run = np.asarray(sim.run(n_steps=64, compute_s_params=False, skip_preflight=True).time_series)
    forward = np.asarray(sim.forward(n_steps=64, skip_preflight=True).time_series)
    assert np.max(np.abs(run)) > 0
    assert np.max(np.abs(run-forward)) <= 9*np.spacing(np.float32(np.max(np.abs(run))))


def test_wall_parameters_keep_closed_form_derivatives():
    import jax
    sim = model()
    grid = sim._build_grid()
    root = realized_conductors(sim, grid)

    def density(sigma, f0):
        current = model()
        current.add_wall_loss(sigma_bulk=sigma, f0=f0, faces=("y_lo", "y_hi"))
        return build_wall_operator(current, root, None, lane="gradient contract").sigma_sheet[0, 1, 0, 1]

    ds, df = jax.grad(density, argnums=(0, 1))(5.8e5, 8e9)
    g = 1 / np.sqrt(np.pi * 8e9 * (4*np.pi*1e-7) / 5.8e5)
    assert float(ds) == pytest.approx(g / (.0015 * 2 * 5.8e5), rel=2e-6)
    assert float(df) == pytest.approx(-g / (.0015 * 2 * 8e9), rel=2e-6, abs=1e-15)
