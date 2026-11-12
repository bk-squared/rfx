"""Opaque ownership, hand-counted edges and high-face ghost contracts."""
import json

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


def test_solid_face_conductivity_closed_form():
    sim = model()
    solid = Box((.0015,) * 3, (.0045,) * 3)
    sim.add(solid, material="pec")
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, shape=solid)
    _, _, ctx = operator(sim)
    # The solid spans cells 1,2 and node planes 1..3 on each axis.
    # x_lo: Ey[1,1,2] runs y=1..2 at x=1,z=2; z=2 is between rim planes 1,3.
    # z_lo: Ey[2,1,1] runs y=1..2 at x=2,z=1; x=2 is between rim planes 1,3.
    # Each edge's open segment is interior to its face, not along a face rim.
    indices = ((1, 1, 2), (2, 1, 1))
    rs = np.sqrt(np.pi * 8e9 * (4*np.pi*1e-7) / 5.8e5)
    expected = 1 / (rs * .0015)  # One full E-curl spacing, independent SI formula.
    actual = [float(ctx.sigma_sheet[(1,) + index]) for index in indices]
    print(json.dumps(dict(component="ey", faces=["x_lo", "z_lo"], indices=indices,
                          sigma_sheet=actual, expected=expected)))
    for index in indices:
        assert bool(ctx.mask_ey[index])
    assert actual == pytest.approx([expected, expected], rel=2e-6)


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


def test_uniform_graded_high_face_row_parity(monkeypatch):
    """Pin lossy, lossless and padded high rows on identical dyadic cells.

    bk-squared/rfx#2613 owns the 4.553e-8 lane amplitude scale: the uniform
    current-source coefficient forms eps_r * EPS_0 in float32 at
    rfx/core/yee.py:1014, even in a float64 run. Its fix PR must tighten
    these record bars to the rounding-only control (exactly 0 on this box
    in Stage A) and write the cause.
    """
    import math
    import time

    import jax

    from rfx import GaussianPulse
    from rfx.ringdown import identify
    from rfx.stepping import uniform_setup

    started = time.monotonic()
    h, f101 = 2**-10, 8.190329e9
    realized = []
    original = uniform_setup.core_step_invariants

    def capture(ctx, **kwargs):
        invariants = original(ctx, **kwargs)
        realized.append((tuple(ctx.grid.shape), invariants["h_row_factors"]))
        return invariants

    monkeypatch.setattr(uniform_setup, "core_step_invariants", capture)
    with jax.enable_x64():
        for scene in ("S1", "S1_lossless", "S2"):
            records, dts, qs = [], [], []
            for graded in (False, True):
                run_started = time.monotonic()
                kwargs = {"dy_profile": np.full(12, h)} if graded else {}
                sim = Simulation(freq_max=10e9, domain=(30*h, 12*h, 24*h),
                                 dx=h, boundary="cpml" if scene == "S2" else "pec",
                                 snap="declared", precision="float64", **kwargs)
                if scene == "S1":
                    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9,
                                      faces=("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"))
                sim.add_source((.0073, .0041, .0059), "ey",
                               waveform=GaussianPulse(f0=8e9, bandwidth=.25), amplitude_kind="current")
                sim.add_probe((.0217, .0083, .0151), "ey")
                grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
                assert bool(sim._uses_nonuniform_mesh) == graded
                dt = float(grid.dt)
                record = np.asarray(sim.run(n_steps=math.ceil((5e-9 if scene == "S2" else 20e-9)/dt),
                                            compute_s_params=False).time_series).ravel()
                assert record.dtype == np.float64
                records.append(record)
                dts.append(dt)
                if scene == "S1":
                    modes = identify(record, dt, math.ceil(2e-9/dt), len(record), freq_max=10e9)
                    pole = modes.s[np.argmin(np.abs(modes.s.imag/(2*np.pi)-f101))]
                    qs.append(float(pole.imag / (-2*pole.real)))
                print(f"{scene} {'G' if graded else 'U'} wall_s={time.monotonic()-run_started:.6f}")
            u, g = records
            assert u.shape == g.shape
            per_step = float(np.max(np.abs(u-g)) / np.max(np.abs(u)))
            du, dg = [np.sum(x * np.exp(-2j*np.pi*f101*np.arange(len(x))*dt))
                      for x, dt in zip(records, dts)]
            r = float(abs(du-dg) / abs(du))
            print(f"{scene} per_step={per_step:.17g} r={r:.17g}")
            assert per_step <= 2e-7, (scene, "per_step", per_step)
            assert r <= 2e-7, (scene, "r", r)
            if scene == "S1":
                q_relative = abs(qs[0]/qs[1]-1)
                print(f"S1 Q_U={qs[0]:.17g} Q_G={qs[1]:.17g} Q_relative={q_relative:.17g}")
                assert q_relative <= 1.7e-6, ("S1", "Q_relative", q_relative)

        assert [shape for shape, _ in realized] == [(31, 13, 25), (31, 13, 25), (63, 45, 57)]
        for shape, factors in realized:
            for axis, factor in enumerate(factors):
                assert factor is not None, ("realized-factor", shape, axis, factor)
                expected = np.ones(shape[axis], dtype=np.float32)
                expected[-1] = 0.
                np.testing.assert_array_equal(factor, expected,
                                              err_msg=f"realized-factor shape={shape} axis={axis}")
        print(f"realized-factor rows={len(realized)*3} wall_s={time.monotonic()-started:.6f}")


def test_withheld_cpml_high_h_rows():
    """A live outer E face must not drive the six outside tangential H rows."""
    import math
    import jax
    import jax.numpy as jnp
    from rfx import GaussianPulse
    from rfx.core.yee import init_materials
    from rfx.simulation import run, SourceSpec

    with jax.enable_x64():
        h = 2**-10
        sim = Simulation(freq_max=10e9, domain=(30*h, 12*h, 24*h),
                         dx=h, boundary="cpml", snap="declared", precision="float64")
        grid = sim._build_grid()
        assert grid.shape == (63, 45, 57)
        steps = math.ceil(5e-9 / grid.dt)
        # S2's Ey current location, independently rounded onto its Yee edge.
        node = (round(.0073/h) + 16, round(.0041/h - .5) + 16,
                round(.0059/h) + 16)
        pulse = GaussianPulse(f0=8e9, bandwidth=.25)
        wave = jax.vmap(pulse)(jnp.arange(steps) * grid.dt)
        wave = wave * (grid.dt / (8.8541878128e-12 * h**3))
        result = run(grid, init_materials(grid.shape), steps, boundary="cpml",
                     pec_axes="", return_state=True, field_dtype=jnp.float64,
                     sources=[SourceSpec(*node, "ey", wave)])
        state = result.state
        for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
            field = np.asarray(getattr(state, name))
            assert field.dtype == np.float64
            assert np.isfinite(field).all(), ("withheld-finite", name)
        for axis in range(3):
            row = [slice(None)] * 3
            row[axis] = -1
            for component in range(3):
                if component == axis:
                    continue
                h_row = np.asarray(getattr(state, "h" + "xyz"[component]))[tuple(row)]
                e_row = np.asarray(getattr(state, "e" + "xyz"[component]))[tuple(row)]
                h_max, e_max = float(np.max(np.abs(h_row))), float(np.max(np.abs(e_row)))
                print(f"withheld axis={axis} component={component} H_max={h_max:.17g} E_max={e_max:.17g}")
                assert h_max == 0., ("withheld-H", axis, component, h_max)
                assert e_max > 0., ("withheld-E-live", axis, component, e_max)


def test_graded_cpml_faces_have_realized_pec_backing():
    """Graded's CPML writer relies on backing; a new withhold route must fail."""
    from rfx.boundaries.setup import nonuniform_boundaries

    h = 2**-10
    sim = Simulation(freq_max=10e9, domain=(30*h, 12*h, 24*h), dx=h,
                     boundary="cpml", snap="declared", dy_profile=np.full(12, h))
    grid = sim._build_nonuniform_grid()
    realized = nonuniform_boundaries(grid, "xyz", (), ())
    assert realized[0]
    assert realized[4] == "xyz"
    expected = frozenset(("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"))
    print(f"graded realized PEC backing={sorted(realized[6])}")
    assert realized[6] == expected, ("graded-PEC-backing", realized[6])
