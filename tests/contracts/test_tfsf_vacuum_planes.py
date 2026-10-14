"""The slab TF/SF vacuum rule applies to concrete and traced public entries."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


VACUUM_MESSAGE = (
    "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x "
    "boundaries; non-vacuum material found at x_lo-1"
)
TRACED_MESSAGE = (
    "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x "
    "boundaries; non-vacuum material found at x_lo"
)


def _scene(graded):
    profile = 1e-3 * np.array([1.2, 1.1, 1., .9, .8, .7, .6,
                              .6, .7, .8, .9, 1., 1.1, .6])
    mesh = {"dz_profile": profile} if graded else {}
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                     cpml_layers=4, **mesh)
    sim.add_tfsf_source(margin=3)
    sim.add_probe((.023, .012, .007), "ez")
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_box_on_vacuum_planes_refused(graded, entry):
    sim = _scene(graded)
    sim.add_material("fill", eps_r=4.)
    sim.add(Box((0., 0., 0.), (.030, .020, .012)), material="fill")
    with pytest.raises(ValueError) as caught:
        getattr(sim, entry)(n_steps=8)
    assert str(caught.value) == VACUUM_MESSAGE


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_forward_override_on_vacuum_planes_refused(graded):
    sim = _scene(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    with pytest.raises(ValueError) as caught:
        sim.forward(n_steps=8, eps_override=jnp.full(grid.shape, 4.))
    assert str(caught.value) == VACUUM_MESSAGE


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_traced_forward_override_on_x_lo_refused(graded):
    sim = _scene(graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()

    def energy(eps):
        # Four pad cells plus margin three: x_lo is node seven.
        # Keep its neighbours vacuum so the literal must name x_lo.
        material = jnp.ones(grid.shape).at[7, :, :].set(eps)
        return jnp.sum(sim.forward(n_steps=8, eps_override=material).time_series ** 2)

    with pytest.raises(Exception) as caught:
        jax.grad(energy)(4.)
    assert TRACED_MESSAGE in str(caught.value)
    print(f"traced graded={graded}: {type(caught.value).__name__}: {caught.value}")


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_interior_dielectric_with_vacuum_planes_admitted(graded, entry):
    sim = _scene(graded)
    sim.add_material("slab", eps_r=4.)
    sim.add(Box((.013, 0., 0.), (.017, .020, .012)), material="slab")
    result = getattr(sim, entry)(n_steps=120)
    values = np.asarray(result.time_series)
    assert np.isfinite(values).all()
    assert np.max(np.abs(values)) > 0


@pytest.mark.parametrize("crossing", [True, False], ids=["crossing", "interior"])
def test_graded_design_box_vacuum_planes(crossing):
    sim = _scene(graded=True)
    x_lo, x_hi = (.001, .029) if crossing else (.013, .017)
    box = ((x_lo, .008, .005), (x_hi, .012, .007))
    grid = sim._build_nonuniform_grid()
    bounds = sim._design_box_bounds_from_corners(grid, box)
    shape = tuple(bounds[2 * d + 1] - bounds[2 * d] for d in range(3))
    eps = jnp.full(shape, 4.)
    if crossing:
        with pytest.raises(ValueError) as caught:
            sim.forward(n_steps=120, design_box=box, design_eps_override=eps)
        assert str(caught.value) == VACUUM_MESSAGE
    else:
        result = sim.forward(n_steps=120, design_box=box, design_eps_override=eps)
        values = np.asarray(result.time_series)
        assert np.isfinite(values).all()
        assert np.max(np.abs(values)) > 0


def test_traced_x_cell_sizes_refused():
    def energy(profile):
        sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                         dx_profile=profile, cpml_layers=4)
        sim.add_tfsf_source()
        sim.add_probe((.023, .012, .007), "ez")
        return jnp.sum(sim.forward(n_steps=8, skip_preflight=True).time_series ** 2)

    with pytest.raises(ValueError) as caught:
        jax.grad(energy)(jnp.full(30, .001))
    assert str(caught.value) == (
        "a plane wave on a graded mesh needs a fixed, uniform x axis; "
        "the x cell sizes are traced")
