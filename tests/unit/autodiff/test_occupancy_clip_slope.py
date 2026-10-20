"""#1926: full clip slope at binary PEC designs, with unchanged forward bits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.model import occupancy


SHAPE = (23, 17, 13)
CELLS = ((10, 8, 6), (7, 8, 6))


def _scene(mesh):
    kwargs = {}
    if mesh == "graded":
        kwargs["dy_profile"] = np.array([.001] + [.0009, .0011] * 7 + [.001])
    sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
                     boundary="pec", **kwargs)
    sim.add_source((.004, .007, .005), "ez", amplitude_kind="field")
    sim.add_probe((.016, .009, .006), "ez")
    grid = sim._build_nonuniform_grid() if mesh == "graded" else sim._build_grid()
    assert tuple(grid.shape) == SHAPE
    return sim


def _binary():
    return jnp.zeros(SHAPE, jnp.float32).at[CELLS[0]].set(1.0)


def _forward(sim, cells):
    return sim.forward(pec_occupancy_override=cells, n_steps=200,
                       skip_preflight=True).time_series


@pytest.mark.parametrize("x, expected", [
    (-0.1, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.1, 0.0),
])
def test_t1_helper_slope(x, expected):
    assert float(jax.grad(occupancy.clip_occupancy)(
        jnp.asarray(x, jnp.float32))) == expected


def test_t1_helper_primal():
    x = jnp.asarray([-0.1, 0.0, 0.3, 1.0, 1.1, -jnp.inf, jnp.inf],
                    jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(occupancy.clip_occupancy(x)).view(np.uint32),
        np.asarray(jnp.clip(x, 0.0, 1.0)).view(np.uint32))


@pytest.fixture(scope="module", params=["uniform", "graded"])
def binary_derivatives(request):
    mesh = request.param
    sim, cells = _scene(mesh), _binary()

    def loss(m):
        return jnp.sum(_forward(sim, m)[:, 0] ** 2)

    value, ad = jax.value_and_grad(loss)(cells)
    baseline = float(value)
    readings = {}
    for cell, step in zip(CELLS, (-1e-2, 1e-2)):
        first = float(loss(cells.at[cell].add(step)))
        second = float(loss(cells.at[cell].add(2 * step)))
        fd = (-3 * baseline + 4 * first - second) / (2 * step)
        readings[cell] = (float(ad[cell]), fd)
    return mesh, readings


@pytest.mark.parametrize("cell", CELLS, ids=["occupied", "empty"])
def test_t2_binary_ad_matches_one_sided_fd(binary_derivatives, cell):
    mesh, readings = binary_derivatives
    ad, fd = readings[cell]
    assert np.isfinite(ad) and np.isfinite(fd) and fd != 0.0, (mesh, cell, ad, fd)
    ratio = ad / fd
    print(f"T2 mesh={mesh} cell={cell} AD={ad:.12g} FD={fd:.12g} AD/FD={ratio:.12g}")
    # 5e-2 is the comparator's own noise: the float32
    # second-order one-sided FD scatters 0.964-1.015 of AD over 8 readings (#1926 body), measured before the bar; the defect
    # it guards is a factor 0.5, ten times outside. The exact check is the helper slope test.
    assert abs(ratio - 1.0) <= 5e-2, (
        f"mesh={mesh} cell={cell} AD={ad:.12g} FD={fd:.12g} AD/FD={ratio:.12g}")


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("fractional", [False, True], ids=["binary", "fractional"])
def test_t3_forward_bits_unchanged(mesh, fractional, monkeypatch):
    sim, cells = _scene(mesh), _binary()
    if fractional:
        cells = cells.at[11, 8, 6].set(0.4)
    actual = np.asarray(_forward(sim, cells))
    # All external sites import locally from this namespace on each call.
    monkeypatch.setattr(occupancy, "clip_occupancy",
                        lambda x: jnp.clip(x, 0.0, 1.0))
    # _edge_factors is also traced through a cached jit for a slab view.
    jax.clear_caches()
    expected = np.asarray(_forward(sim, cells))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
