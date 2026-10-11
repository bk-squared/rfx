"""Declared-box NumPy oracles for the public realized cell snapshot."""

from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.model import RealizedView, realized_view


U = 2.0**-10
BOXES = (
    ((2 * U,) * 3, (4 * U,) * 3, 4., 2., 1.5),
    ((5 * U,) * 3, (7 * U,) * 3, 9., 4., 2.),
)


def _sim(lane, *, pad=2, periodic_axes=""):
    # Unequal refined edges make both a nominal/global spacing and the
    # smallest edge disagree with the independent wavelength oracle.
    profiles = ({f"d{a}_profile": np.array([U, U] + [U / div] * (5 * div) + [U])
                 for a, div in zip("xyz", (2, 4, 2))}
                if lane == "nonuniform" else {})
    boundary = (BoundarySpec(**{a: "periodic" if a in periodic_axes else "cpml"
                                for a in "xyz"}) if periodic_axes else "cpml")
    sim = Simulation(freq_max=1e9, domain=(8 * U,) * 3, dx=U,
                     cpml_layers=pad, boundary=boundary, **profiles)
    for i, (lo, hi, eps, sigma, mu) in enumerate(BOXES):
        sim.add_material(f"box{i}", eps_r=eps, sigma=sigma, mu_r=mu)
        sim.add(Box(lo, hi), material=f"box{i}")
    return sim


def _declared_cells(nodes):
    # The final stored node has no successor: all fixtures finish with U.
    # Interior widths and all box occupancy come only from the public nodes.
    widths = [np.append(np.diff(line), U) for line in nodes]
    centres = np.meshgrid(*(line + d / 2 for line, d in zip(nodes, widths)),
                          indexing="ij")
    shape = tuple(len(line) for line in nodes)
    eps, sigma, mu = np.ones(shape), np.zeros(shape), np.ones(shape)
    for lo, hi, er, sig, mr in BOXES:
        inside = np.ones(shape, dtype=bool)
        for a in range(3):
            assert lo[a] in nodes[a] and hi[a] in nodes[a]
            inside &= (centres[a] >= lo[a]) & (centres[a] < hi[a])
        eps[inside], sigma[inside], mu[inside] = er, sig, mr
    return eps, sigma, mu, widths


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_w1_declared_boxes_and_cell_wavelength(lane):
    sim = _sim(lane)
    view = realized_view(sim)
    eps, sigma, mu, widths = _declared_cells(view.nodes)
    np.testing.assert_array_equal(view.eps_r, eps)
    np.testing.assert_array_equal(view.sigma, sigma)
    interior = tuple(tuple(int(i) for i in np.searchsorted(line, [0., 8 * U]))
                     for line in view.nodes)
    assert view.interior == interior
    region = tuple(slice(lo, hi) for lo, hi in interior)
    edges = np.meshgrid(*(d[s] for d, s in zip(widths, region)), indexing="ij")
    expected = np.min(299792458. / (1e9 * np.sqrt(eps[region] * mu[region])
                                  * np.maximum.reduce(edges)))
    assert view.min_cells_per_wavelength == expected
    assert view.lane == lane
    assert isinstance(view.dt, float) and view.dt > 0
    assert tuple(len(line) for line in view.nodes) == view.shape
    ctx = sim._campaign_ctx()
    assert view.dt == float(ctx.grid.dt)
    for actual, original in zip(view.nodes, ctx.nodes):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("preflight_first", [False, True])
def test_w2_one_shared_assembly(lane, preflight_first, monkeypatch):
    sim = _sim(lane)
    method = "_assemble_materials_nu" if lane == "nonuniform" else "_assemble_materials"
    assembly = Mock(wraps=getattr(sim, method))
    monkeypatch.setattr(sim, method, assembly)
    if preflight_first:
        sim.preflight()
        assert assembly.call_count == 1
    ctx = sim._campaign_ctx()
    grid = ctx.grid
    first = realized_view(sim)
    second = realized_view(sim)
    assert sim._campaign_ctx() is ctx
    assert ctx.grid is grid
    assert ctx.realized().grid is grid
    assert assembly.call_count == 1
    assert assembly.call_args.args[0] is grid
    np.testing.assert_array_equal(first.eps_r, second.eps_r)
    np.testing.assert_array_equal(first.sigma, second.sigma)


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
@pytest.mark.parametrize("pad", [0, 2])
def test_w3_every_node_round_trips(lane, pad):
    """Every node round-trips through the lane's own lookup. The graded lookup clamps
    absorber-pad positions to the interior by design, so on that lane W3 covers the interior
    nodes (decision record 1 of the declaration)."""
    sim = _sim(lane, pad=pad)
    view = realized_view(sim)
    grid = sim._campaign_ctx().grid
    lookup = (grid.position_to_index if lane == "uniform"
              else lambda pos: sim._pos_to_nu_index(grid, pos))
    misses = []
    for a, line in enumerate(view.nodes):
        lo, hi = view.interior[a] if lane == "nonuniform" else (0, len(line))
        for i, x in enumerate(line):
            if not lo <= i <= hi:
                continue
            pos = [0., 0., 0.]
            pos[a] = float(x)
            actual = lookup(tuple(pos))[a]
            if actual != i:
                misses.append((a, i, float(x), actual))
    assert not misses, f"(axis, index, coordinate, lookup): {misses}"


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_snapshot_is_frozen_keyword_only_and_detached(lane):
    sim = _sim(lane)
    view = realized_view(sim)
    assert all(field.kw_only for field in fields(RealizedView))
    with pytest.raises(FrozenInstanceError):
        view.dt = 0
    ctx = sim._campaign_ctx()
    originals = (*ctx.nodes, ctx.realized().materials.eps_r,
                 ctx.realized().materials.sigma)
    for arr, original in zip((*view.nodes, view.eps_r, view.sigma), originals):
        assert type(arr) is np.ndarray
        assert arr.dtype == np.float64
        assert not arr.flags.writeable
        assert not np.shares_memory(arr, np.asarray(original))
        with pytest.raises(ValueError):
            arr.flat[0] = 99
        arr.setflags(write=True)
        arr.flat[0] = 99
        assert np.asarray(original).flat[0] != 99


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_configuration_change_invalidates_shared_context(lane):
    sim = _sim(lane)
    first = realized_view(sim)
    ctx = sim._campaign_ctx()
    sim.add(Box((0.,) * 3, (U,) * 3), material="box0")
    second = realized_view(sim)
    assert sim._campaign_ctx() is not ctx
    index = tuple(lo for lo, _ in second.interior)
    assert first.eps_r[index] == 1.
    assert second.eps_r[index] == 4.


@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_lumped_and_wire_operands_are_excluded(lane):
    sim = _sim(lane)
    ctx = sim._campaign_ctx()
    root = ctx.realized()
    base = root.materials
    one = jnp.ones(root.grid.shape)
    # Inject recorded stamps into the SAME cached assembly, including two
    # components. The expected volume remains the declared-box NumPy oracle.
    stamped = base._replace(
        eps_r=base.eps_r + 5 * one, sigma=base.sigma + 12 * one,
        eps_r_lumped=(2 * one, None, 3 * one),
        sigma_lumped=(None, 5 * one, 7 * one),
        mu_r_wire=(20 * one, 30 * one, None),
    )
    ctx._realized = replace(root, materials=stamped)
    view = realized_view(sim)
    eps, sigma, mu, widths = _declared_cells(view.nodes)
    np.testing.assert_array_equal(view.eps_r, eps)
    np.testing.assert_array_equal(view.sigma, sigma)
    region = tuple(slice(lo, hi) for lo, hi in view.interior)
    edges = np.meshgrid(*(d[s] for d, s in zip(widths, region)), indexing="ij")
    assert view.min_cells_per_wavelength == np.min(
        299792458. / (1e9 * np.sqrt(eps[region] * mu[region])
                     * np.maximum.reduce(edges)))
    assert {field.name for field in fields(view)} == {
        "nodes", "interior", "eps_r", "sigma", "shape", "dt", "lane",
        "min_cells_per_wavelength",
    }


def test_periodic_interior_has_no_bounding_duplicate():
    view = realized_view(_sim("uniform", periodic_axes="xyz"))
    assert view.shape == (8, 8, 8)
    assert view.interior == ((0, 8),) * 3


def test_malformed_context_raises_its_error(monkeypatch):
    sim = _sim("uniform")
    monkeypatch.setattr(sim, "_build_grid", Mock(side_effect=ValueError("bad mesh")))
    with pytest.raises(ValueError, match="^ValueError: bad mesh$"):
        realized_view(sim)


def test_assembly_error_refuses_partial_view(monkeypatch):
    sim = _sim("uniform")
    monkeypatch.setattr(sim, "_assemble_materials", Mock(side_effect=ValueError("bad cells")))
    with pytest.raises(ValueError, match="^ValueError: bad cells$"):
        realized_view(sim)


@pytest.mark.parametrize("mesh", ["dx", "profile"])
def test_traced_mesh_raises_context_error(mesh):
    sim = _sim("uniform")

    def traced(value):
        if mesh == "dx":
            sim._dx = value
        else:
            sim._dz_profile = value
        with pytest.raises(ValueError) as error:
            realized_view(sim)
        assert str(error.value) == "traced-mesh"
        return value

    jax.make_jaxpr(traced)(U if mesh == "dx" else np.full(8, U))
