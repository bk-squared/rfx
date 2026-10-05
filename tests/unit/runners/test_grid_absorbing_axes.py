"""Runner axes belong to the grid; feature-specific subsets remain supported."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Grid, run
from rfx.boundaries.axes import resolve_cpml_axes
from rfx.core.yee import init_materials
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform, run_nonuniform_until_decay
from rfx.simulation import SourceSpec, run_until_decay


def _grid(graded, axes="z"):
    if graded:
        dz = np.full(40, 0.001)
        dz[19:21] = (0.0009, 0.0011)
        return make_nonuniform_grid((0.02, 0.02), dz, 0.001, 8, cpml_axes=axes)
    return Grid(freq_max=10e9, domain=(0.02, 0.02, 0.04), dx=0.001,
                cpml_layers=8, cpml_axes=axes)


def _pulse_run(grid, graded, decay=False, *, reference=False, **kwargs):
    n = 6
    pulse = jnp.asarray([1., 0., 0., 0., 0., 0.], dtype=jnp.float32)
    source = (2, 2, grid.nz // 2, "ez", pulse)
    mats = init_materials(grid.shape)
    if graded:
        runner = run_nonuniform_until_decay if decay else run_nonuniform
        options = dict(max_steps=n, min_steps=n, check_interval=n) if decay else dict(n_steps=n)
        if reference:
            kwargs["cpml_axes"] = ""
        return runner(grid, mats, sources=[source], **options, **kwargs)["state"]
    runner = run_until_decay if decay else run
    options = dict(max_steps=n, min_steps=n, check_interval=n) if decay else dict(n_steps=n)
    return runner(grid, mats, boundary="pec" if reference else "cpml",
                  sources=[SourceSpec(*source)], **options, **kwargs).state


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("decay", [False, True], ids=["fixed", "decay"])
def test_omitted_axes_do_not_absorb_inside_unpadded_faces(graded, decay):
    """A pulse two cells from x/y must match a no-absorber PEC control.

    Six Yee steps reach both unpadded faces, inside the old eight-layer
    x/y profiles, but cannot reach either z profile from the midplane.
    Thus exact field equality detects unwanted transverse absorption without
    depending on profile metadata or allowing an all-zero-field false pass.
    """
    grid = _grid(graded)
    reference = _pulse_run(grid, graded, decay, reference=True)
    implicit = _pulse_run(grid, graded, decay)
    assert np.max(np.abs(np.asarray(reference.ez)[1:4, 1:4, grid.nz // 2])) > 0
    for actual, expected in zip(implicit, reference):
        np.testing.assert_array_equal(actual, expected)
    explicit = _pulse_run(grid, graded, decay, cpml_axes="z")
    for actual, expected in zip(explicit, implicit):
        assert np.asarray(actual).dtype == np.asarray(expected).dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("decay", [False, True], ids=["fixed", "decay"])
def test_extra_absorbing_axis_is_refused(graded, decay):
    with pytest.raises(ValueError, match="cpml_axes='xyz'.*cpml_axes='z'.*authoritative"):
        _pulse_run(_grid(graded), graded, decay, cpml_axes="xyz")


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_explicit_subset_still_runs(graded):
    """A subset is honoured: dropping x/y absorption changes the field near those faces.

    The TF/SF and oblique-RCS paths pass a subset on purpose. On an all-axis
    grid, a z-only run must differ from the default run where the pulse meets
    the x/y profiles (about 0.7 of the peak, measured on both paths), so a
    path that ignored the explicit subset would fail here.
    """
    grid = _grid(graded, "xyz")
    default = np.asarray(_pulse_run(grid, graded).ez)
    subset = np.asarray(_pulse_run(grid, graded, cpml_axes="z").ez)
    assert np.any(subset)
    near = (slice(0, 12), slice(0, 12), grid.nz // 2)
    assert np.max(np.abs(subset[near] - default[near])) > 0.1 * np.max(np.abs(default[near]))


@pytest.mark.parametrize("mode,periodic,explicit,expected", [
    ("3d", "", "zyxx", "xyz"),
    ("2d_tmz", "", "xyz", "xy"),
    ("2d_tez", "x", "xyz", "y"),
    ("3d", "y", "xyz", "xz"),
])
def test_grid_filtering_and_axis_set_semantics(mode, periodic, explicit, expected):
    grid = Grid(freq_max=10e9, domain=(0.02,) * 3, dx=0.001,
                mode=mode, periodic_axes=periodic)
    assert resolve_cpml_axes(grid, explicit) == expected
    assert resolve_cpml_axes(grid) == expected


@pytest.mark.parametrize("name", [
    "extract_s_matrix", "extract_s_matrix_wire",
    "extract_waveguide_s_matrix", "extract_waveguide_s_matrix_flux",
    "extract_waveguide_s_params_normalized", "extract_multimode_s_matrix",
    "extract_multimode_s_matrix_flux",
])
def test_port_extractors_refuse_extra_axes_before_assembling_ports(name):
    from rfx.probes import probes
    from rfx.sources import waveguide_port

    module = probes if name in ("extract_s_matrix", "extract_s_matrix_wire") else waveguide_port
    runner = getattr(module, name)
    import inspect
    kwargs = {}
    for parameter in inspect.signature(runner).parameters.values():
        if parameter.default is inspect.Parameter.empty:
            kwargs[parameter.name] = None
    kwargs.update(grid=_grid(False), cpml_axes="xyz")
    with pytest.raises(ValueError, match="cpml_axes='xyz'.*cpml_axes='z'"):
        runner(**kwargs)
