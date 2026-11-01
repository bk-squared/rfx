"""The traced assembly must solve the concrete conductor, or reject stale nominal cells."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.boundaries.spec import BoundarySpec
from rfx.core.jax_utils import is_tracer
from rfx.geometry import Cylinder, OrientedBox
from rfx.geometry.smoothing import continued_conductor_shape
from rfx.model.conductors import solve_conductors
from tests.unit.nonuniform import test_traced_graded_axis_nominal_profile as board


def board_scene(delta, *, stale=False, inset=False, nominal=True, cpml=4):
    with patch.object(board, "CPML", cpml):
        sim = board._board(delta * 0 if stale else delta)
    if stale:
        if is_tracer(delta):
            sim._dx_profile = jnp.asarray(board.PROF, dtype=jnp.float32).at[-2].add(delta)
        else:
            sim._dx_profile = board.PROF.copy()
            sim._dx_profile[-2] += float(delta)
    if not nominal:
        sim._nominal_profiles = None
    lo = 2 * board.DX if inset else 0.
    hi = board.LX - (2 * board.DX if inset else .1 * board.PROF[-1] if stale else 0.)
    sim.add(Box((lo, 2 * board.DX, board.Z_G - board.DX),
                (hi, 18 * board.DX, board.Z_G)), material="pec")
    return sim


def arrays(sim):
    grid = sim._build_nonuniform_grid()
    product = solve_conductors(sim, grid, nonuniform=True)
    zero = jnp.zeros(grid.shape, dtype=bool)
    return (product.pec_cells if product.pec_cells is not None else zero,
            *(product.pec_edges if product.pec_edges is not None else (zero,) * 3),
            product.materials.eps_r, product.materials.sigma)


def equal_arrays(actual, expected):
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_a1_board_eager():
    expected = arrays(board_scene(0.))
    actual, _ = jax.jvp(lambda d: arrays(board_scene(d)), (jnp.float32(0),), (jnp.float32(1),))
    equal_arrays(actual, expected)


def ring_scene(profile):
    nominal = {"z": np.ones(8)} if is_tracer(profile) else None
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                     boundary="cpml", cpml_layers=6, dz_profile=profile,
                     nominal_profiles=nominal, dt=1e-12, dt_min_cell=.5)
    sim.add(Box((0., 0., 2.), (7.9, 7.9, 3.)), material="pec")
    return sim


def test_a1_ring_jit():
    import rfx
    from tests.oracle import test_lossless_open_domain_ringdown_does_not_grow as ring

    profile = np.full(48, ring.H / 3)
    constructor = rfx.Simulation

    def go(d):
        kw = {"nominal_profiles": {"z": profile}} if is_tracer(d) else {}
        with patch.object(rfx, "Simulation", lambda **args: constructor(**args, dz_profile=d, **kw)):
            return arrays(ring._build(n=3, pad_h=10, cpml=6))

    equal_arrays(jax.jit(go)(profile), go(profile))


def test_a4_clear_shape_is_identity_without_check(monkeypatch):
    import rfx.geometry._traced_conductor as traced

    def forbidden(*args, **kwargs):
        pytest.fail("a clear conductor added a continuation callback")
    monkeypatch.setattr(traced, "traced_check", forbidden)

    def go(d):
        sim = board_scene(d, inset=True)
        shape = sim._geometry[-1].shape
        assert continued_conductor_shape(sim, sim._build_nonuniform_grid(), shape) is shape
        return arrays(sim)
    expected = arrays(board_scene(0., inset=True))
    equal_arrays(jax.jvp(go, (jnp.float32(0),), (jnp.float32(1),))[0], expected)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_a7_stale_outermost_layer(jit):
    def go(d):
        sim = board_scene(d, stale=True)
        sim._ports = []
        return arrays(sim)
    call = jax.jit(go) if jit else lambda d: jax.jvp(go, (d,), (jnp.float32(1),))[0]
    jax.block_until_ready(call(jnp.float32(0)))
    with pytest.raises(Exception, match="outermost layer at x-hi.*nominal_profiles.*axis concrete"):
        jax.block_until_ready(call(jnp.float32(3e-4)))


def test_a8_missing_nominal():
    with pytest.raises(TracedStructureRefusal, match="Declare nominal_profiles for axis x"):
        jax.jvp(lambda d: arrays(board_scene(d, nominal=False)),
                (jnp.float32(0),), (jnp.float32(1),))


@pytest.mark.parametrize("kind", ["sheet", "oriented"])
def test_a8_shape_refusals(kind):
    def go(d):
        sim = board_scene(d)
        sim._geometry.pop()
        if kind == "sheet":
            sim.add_thin_conductor(Cylinder((.002, .005, board.Z_G), .002, 0., axis="z"))
        else:
            sim.add(OrientedBox((.002, .005, board.Z_G - board.DX / 2),
                                (.004, .008, board.DX)), material="pec")
        return arrays(sim)
    with pytest.raises((ValueError, NotImplementedError), match="sheet's plane|OrientedBox"):
        jax.jvp(go, (jnp.float32(0),), (jnp.float32(1),))


@pytest.mark.parametrize("nominal", [False, True])
def test_unpadded_traced_axis_needs_nominal_for_candidate(nominal):
    profile = np.array([1., .5, .5, 2., 2., .5, .5, 1.])
    def scene(d):
        sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,
                         boundary=BoundarySpec(x="cpml", y="pec", z="cpml"),
                         cpml_layers=2, dy_profile=d, dt=1e-12, dt_min_cell=.5,
                         nominal_profiles={"y": profile} if nominal and is_tracer(d) else None)
        sim.add(Box((0., 2., 2.), (8., 6., 4.)), material="pec")
        return sim
    call = jax.jit(lambda d: arrays(scene(d)))
    if nominal:
        equal_arrays(call(profile), arrays(scene(profile)))
    else:
        with pytest.raises(TracedStructureRefusal, match="nominal_profiles for axis y"):
            call(profile)


def test_continued_coordinates_are_concrete():
    def go(d):
        sim = ring_scene(d)
        sim._geometry.clear()
        sim.add(Box((2., 2., 0.), (6., 6., 7.9)), material="pec")
        shape = sim._geometry[-1].shape
        solved = continued_conductor_shape(sim, sim._build_nonuniform_grid(), shape)
        assert solved is not shape
        assert all(not is_tracer(v) for bound in solved.bounding_box() for v in bound)
        return jnp.asarray(solved.bounding_box())
    _, tangent = jax.jvp(go, (jnp.ones(8),), (jnp.ones(8),))
    np.testing.assert_array_equal(tangent, 0.)


def test_windowed_nominal_continues():
    from rfx.model import NominalProfile

    profile = np.ones(8)

    def go(d):
        sim = ring_scene(d)
        if is_tracer(d):
            sim._nominal_profiles = {"z": NominalProfile(profile, traced_cells=(2, 6))}
        return arrays(sim)

    equal_arrays(jax.jit(go)(profile), go(profile))


def test_lattice_wrapper_keeps_refusal():
    from tests.unit.materials.test_sheet_impedance import PlanarSheet
    def go(d):
        sim = board_scene(d)
        sim._geometry.pop()
        shape = PlanarSheet(2, board.Z_G, (0., .002), (board.LX, .006))
        sim.add_thin_conductor(shape, sigma_bulk=5.8e7)
        continued_conductor_shape(sim, sim._build_nonuniform_grid(), shape)
        return d
    with pytest.raises(TracedStructureRefusal, match="PlanarSheet.*x-lo.*Box volume.*concrete"):
        jax.jvp(go, (jnp.float32(0),), (jnp.float32(1),))


@pytest.mark.parametrize("path", ["nonuniform", "distributed_nu", "uniform", "subgridded"])
def test_a6_paths(path):
    def go(d):
        sim = board_scene(d)
        if path == "distributed_nu":
            from rfx.runners.distributed_nu import build_sharded_nu_grid
            build_sharded_nu_grid(sim._build_nonuniform_grid(), 2)
        elif path == "uniform":
            sim._require_uniform_mesh("traced conductor test")
        elif path == "subgridded":
            sim.add_refinement(z_range=(.003, .006), ratio=2)
            sim._dispatch_plan(mode="forward", n_steps=1, num_periods=1.)
        else:
            assert sim._dispatch_plan(mode="forward", n_steps=1, num_periods=1.).lane == "fwd_nonuniform"
            return arrays(sim)
        return d
    if path == "nonuniform":
        equal_arrays(jax.jvp(go, (jnp.float32(0),), (jnp.float32(1),))[0], arrays(board_scene(0.)))
    else:
        error = TracedStructureRefusal if path == "distributed_nu" else NotImplementedError
        match = {"distributed_nu": "traced mesh metrics", "uniform": "non-uniform mesh",
                 "subgridded": "subgridding on a non-uniform mesh"}[path]
        with pytest.raises(error, match=match):
            jax.jvp(go, (jnp.float32(0),), (jnp.float32(1),))


@pytest.mark.parametrize("axis", ["x", "z"])
def test_axial_cylinder_continues(axis):
    profile = np.ones(8)
    def scene(d):
        sim = ring_scene(d)
        sim._geometry.clear()
        center = (4., 4., 4.)
        sim.add(Cylinder(center, 1., 8., axis=axis), material="pec")
        return sim
    equal_arrays(jax.jit(lambda d: arrays(scene(d)))(profile), arrays(scene(profile)))
