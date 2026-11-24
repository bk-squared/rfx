"""Traced geometry/mesh admission precedes local two-device time loops."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.runners.distributed_nu import build_sharded_nu_grid
from tests.unit.autodiff.test_graded_wire_live_cell_admission import _board
from rfx._traced_refusal import TracedStructureRefusal
from rfx.errors import AccuracyWarning


def _distributed(value, kind):
    if kind == "wire":
        sim = _board(value)
    else:
        dz = (jnp.full(8, .001).at[3].add(value).at[4].add(-value)
              if kind == "flux" else np.full(8, .001))
        sim = Simulation(freq_max=10e9, domain=(.008, .008, .008),
                         dx=.001, boundary="pec", dz_profile=dz)
        if kind == "flux":
            sim.add_flux_monitor(axis="z", coordinate=.002, freqs=[1e9])
        else:
            shape = (Box((.002, .002, .002), (.005 + value, .005, .005))
                     if kind == "Box" else Sphere((.004, .004, .004), .002 + value))
            sim.add(shape, material="pec")
    return jnp.sum(sim.forward(
        distributed=True, devices=jax.devices("cpu")[:2], n_steps=2,
        skip_preflight=True).time_series)


@pytest.mark.parametrize("kind", ["wire", "flux", "Box", "Sphere"])
@pytest.mark.parametrize("route", ["jit", "vmap", "jvp", "grad"])
def test_traced_distributed_setup_refuses_before_loop(kind, route):
    assert len(jax.devices("cpu")) >= 2
    fn = lambda value: _distributed(value, kind)
    refusal = TracedStructureRefusal
    if kind in ("wire", "flux"):
        message = "graded multi-device solve.*traced mesh metrics.*concrete mesh"
    elif route in ("jvp", "grad"):
        refusal, message = NotImplementedError, "gradient.*declared shape dimension"
    else:
        message = f"{kind}.*traced.*concrete"
    with pytest.raises(refusal, match=message):
        if route == "jit":
            jax.jit(fn)(jnp.float32(0.))
        elif route == "vmap":
            jax.vmap(fn)(jnp.zeros(2))
        elif route == "jvp":
            jax.jvp(fn, (jnp.float32(0.),), (jnp.float32(1.),))
        else:
            jax.grad(fn)(jnp.float32(0.))


def test_concrete_distributed_mesh_still_builds():
    grid = _board(0.)._build_nonuniform_grid()
    sharded = build_sharded_nu_grid(grid, 2)
    np.testing.assert_array_equal(sharded.inv_dz, grid.inv_dz)
    assert sharded.dt == float(grid.dt)


TRACED_MESH_MESSAGE = "graded multi-device solve.*traced mesh metrics.*concrete mesh"
GRADING_REFUSAL_MESSAGE = r"grading_ratio=\d+\.\d\d exceeds 5\.0.*run on one device"
GRADING_CAUTION_MESSAGE = r"grading ratio \d+\.\d\d:1 exceeds the 3:1 stability caution"
JAX_CONVERSION_ERRORS = (jax.errors.TracerArrayConversionError,
                         jax.errors.ConcretizationTypeError,
                         jax.errors.TracerBoolConversionError)


def _graded_box(axis, cells, freq_max=10e9):
    """A PEC box with one source and one probe, graded on ``axis`` by ``cells``."""
    extent = float(np.sum(cells)) if not isinstance(cells, jax.core.Tracer) else .008
    domain = tuple(extent if a == axis else .008 for a in "xyz")
    sim = Simulation(freq_max=freq_max, domain=domain, dx=.001, boundary="pec",
                     **{f"d{axis}_profile": cells})
    centre = [.004, .004, .004]
    centre["xyz".index(axis)] = extent / 2
    sim.add_source(position=(centre[0] - .0005, centre[1], centre[2] + .0005), component="ez",
                   amplitude_kind="current")
    sim.add_probe(position=(centre[0] + .0015, centre[1], centre[2] + .0005), component="ez")
    return sim


def _solve(sim, entry, skip_preflight, n_steps=2):
    devices = jax.devices("cpu")[:2]
    if entry == "forward":
        out = sim.forward(distributed=True, devices=devices, n_steps=n_steps,
                          skip_preflight=skip_preflight)
    else:
        out = sim.run(devices=devices, n_steps=n_steps, skip_preflight=skip_preflight)
    return jnp.sum(jnp.asarray(out.time_series) ** 2)


def _traced_cells(value):
    return jnp.full(8, .001).at[3].add(value).at[4].add(-value)


@pytest.mark.parametrize("axis", ["z", "x"])
@pytest.mark.parametrize("route", ["jit", "grad", "vmap"])
@pytest.mark.parametrize("skip_preflight", [False, True], ids=["preflight_on", "preflight_off"])
@pytest.mark.parametrize("entry", ["forward", "run"])
def test_traced_profile_gets_the_named_refusal_on_both_graded_entries(entry, skip_preflight,
                                                                      route, axis):
    assert len(jax.devices("cpu")) >= 2
    fn = lambda value: _solve(_graded_box(axis, _traced_cells(value)), entry, skip_preflight)
    with pytest.raises(TracedStructureRefusal, match=TRACED_MESH_MESSAGE) as caught:
        if route == "vmap":
            jax.vmap(fn)(jnp.zeros(2))
        else:
            (jax.jit(fn) if route == "jit" else jax.grad(fn))(jnp.float32(0.))
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert not any(isinstance(e, JAX_CONVERSION_ERRORS) for e in chain), (
        "the named refusal must come before any host read of the traced profile: "
        f"{[type(e).__name__ for e in chain]}")


def _named_refusal_without_conversion_error(caught):
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert not any(isinstance(e, JAX_CONVERSION_ERRORS) for e in chain), (
        "the named refusal must come before any host read of the traced profile: "
        f"{[type(e).__name__ for e in chain]}")


@pytest.mark.parametrize("entry", ["forward", "run"])
def test_traced_profile_without_n_steps_is_refused_before_the_step_count(entry):
    """With ``n_steps=None`` the entry counts steps from the grid's time step; the traced
    profile is refused by name before that count reads it."""
    assert len(jax.devices("cpu")) >= 2
    fn = lambda value: _solve(_graded_box("z", _traced_cells(value)), entry, False, n_steps=None)
    with pytest.raises(TracedStructureRefusal, match=TRACED_MESH_MESSAGE) as caught:
        jax.jit(fn)(jnp.float32(0.))
    _named_refusal_without_conversion_error(caught)


def _geometric_cells(n_cells):
    return .001 * 1.3 ** np.arange(n_cells)


@pytest.mark.parametrize("skip_preflight", [False, True], ids=["preflight_on", "preflight_off"])
@pytest.mark.parametrize("entry", ["forward", "run"])
def test_grading_ratio_above_five_is_refused_on_both_graded_entries(entry, skip_preflight):
    cells = _geometric_cells(8)
    assert cells.max() / cells.min() > 5.0
    sim = _graded_box("z", cells, freq_max=3e9)
    with pytest.raises(ValueError, match=GRADING_REFUSAL_MESSAGE):
        _solve(sim, entry, skip_preflight)


@pytest.mark.parametrize("skip_preflight", [False, True], ids=["preflight_on", "preflight_off"])
@pytest.mark.parametrize("entry", ["forward", "run"])
def test_grading_ratio_between_three_and_five_warns_on_both_graded_entries(entry, skip_preflight):
    cells = _geometric_cells(7)
    assert 3.0 < cells.max() / cells.min() <= 5.0
    sim = _graded_box("z", cells, freq_max=3e9)
    with pytest.warns(AccuracyWarning, match=GRADING_CAUTION_MESSAGE):
        _solve(sim, entry, skip_preflight)
