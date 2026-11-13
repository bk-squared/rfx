"""Grid and run walls determine refusal; cell-span spectra match the merge base."""

import base64
import json
from pathlib import Path
import re
import warnings
import zlib

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.auto_config import smooth_grading
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.measurement import waveguide_s
from rfx.model.graded_grid import make_nonuniform_grid
from rfx.preflight._common import PreflightWarning
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, init_multimode_waveguide_port,
)
from rfx.stepping import graded_loop, uniform_loop


EXTRACTORS = (
    "extract_waveguide_s_matrix", "extract_waveguide_s_matrix_flux",
    "extract_waveguide_s_params_normalized", "extract_multimode_s_matrix",
    "extract_multimode_s_matrix_flux",
)


@pytest.fixture(scope="module")
def scene():
    grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                cpml_layers=10, cpml_axes="x")
    return grid, jnp.linspace(4e9, 10e9, 15), grid.num_timesteps(num_periods=10)


def _ports(grid, whole_axis=None):
    y_hi = grid.ny if whole_axis == "y" else grid.ny - 1
    z_hi = grid.nz if whole_axis == "z" else grid.nz - 1
    return [WaveguidePort(
        x_index=index, y_slice=(0, y_hi), z_slice=(0, z_hi),
        a=(grid.ny - 1) * grid.dx, b=(grid.nz - 1) * grid.dx,
        mode=(1, 0), mode_type="TE", direction=direction,
    ) for index, direction in ((15, "+x"), (grid.nx - 16, "-x"))]


def _configs(scene, whole_axis=None, multimode=False):
    grid, freqs, steps = scene
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    return [initializer(
        port, grid.dx, freqs, f0=6e9, probe_offset=15, ref_offset=3,
        dft_total_steps=steps, **({"n_modes": 2} if multimode else {}),
    ) for port in _ports(grid, whole_axis)]


def _x_profile():
    return smooth_grading(
        np.concatenate([np.full(20, 0.002), np.full(40, 0.001), np.full(20, 0.002)]),
        max_ratio=1.3,
    )


def _graded_grid():
    profile = _x_profile()
    return make_nonuniform_grid(
        (float(np.sum(profile)), 0.04), np.full(10, 0.002), 0.002,
        dx_profile=profile, cpml_layers=10, cpml_axes="x",
        pec_faces={"y_lo", "y_hi", "z_lo", "z_hi"},
    )


@pytest.mark.parametrize("axis", ["y", "z", "yz"])
@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("graded", [False, True])
def test_initializers_warn_without_uniform_grid(scene, axis, multimode, graded):
    grid, freqs, _ = scene
    if graded:
        grid = _graded_grid()
    port = _ports(grid, axis)[0]
    if axis == "yz":
        port = port._replace(y_slice=(0, grid.ny), z_slice=(0, grid.nz))
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    with pytest.warns(PreflightWarning, match="aperture reaches past") as caught:
        initializer(port, grid if graded else grid.dx, freqs,
                    **({"n_modes": 2} if multimode else {}))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert f"on {', '.join(axis)};" in message
    assert "the walls are not checked on this path" in message
    assert "build with a uniform Grid" in message
    assert "cell span `(lo, hi - 1)`" in message


@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("graded", [False, True])
def test_initializers_cell_span_without_uniform_grid_stays_quiet(scene, multimode, graded):
    grid, freqs, _ = scene
    if graded:
        grid = _graded_grid()
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    with warnings.catch_warnings():
        warnings.simplefilter("error", PreflightWarning)
        initializer(_ports(grid)[0], grid if graded else grid.dx, freqs,
                    **({"n_modes": 2} if multimode else {}))


@pytest.mark.parametrize("graded", [False, True])
def test_simulation_cell_span_build_and_run_stays_quiet(graded):
    profile = _x_profile() if graded else None
    domain_x = float(np.sum(profile)) if graded else 0.12
    with warnings.catch_warnings():
        warnings.simplefilter("error", PreflightWarning)
        sim = Simulation(
            freq_max=12e9, domain=(domain_x, 0.04, 0.02), dx=0.002,
            boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=10,
            dx_profile=profile,
        )
        sim.add_waveguide_port(0.01, direction="+x", freqs=jnp.linspace(5e9, 7e9, 3),
                              f0=6e9, name="guide")
        result = sim.run(n_steps=3, compute_s_params=False)
    cfg = result.waveguide_ports["guide"]
    assert (cfg.u_lo, cfg.u_hi) == (0, result.grid.ny - 1)
    assert (cfg.v_lo, cfg.v_hi) == (0, result.grid.nz - 1)


def _extract(scene, name, whole_axis=None):
    grid, _, steps = scene
    material = init_materials(grid.shape)
    args = [grid, material]
    if name.endswith("_flux") or name.endswith("_normalized"):
        args.append(material)
    args.extend((_configs(scene, whole_axis, "multimode" in name), steps))
    result = getattr(waveguide_s, name)(
        *args, boundary="cpml", cpml_axes="x", pec_axes="yz",
    )
    return np.asarray(result[0] if isinstance(result, tuple) else result)


def _refusal(grid, axis, port_name):
    count = grid.shape["xyz".index(axis)]
    return re.escape(
        f"Waveguide port {port_name}: {axis}_slice={(0, count)} includes the "
        f"node row outside the high electric wall {axis}_hi; use "
        f"cell span {axis}_slice={(0, count - 1)}, as Simulation builds it."
    )


class _SetupReached(Exception):
    pass


def _accepted_setup(*args, **kwargs):
    raise _SetupReached


def _no_step(*args, **kwargs):
    raise AssertionError("The whole-node aperture reached stepping")


@pytest.mark.parametrize("axis", ["y", "z"])
@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("grid_kind", ["explicit", "grid_dx"])
@pytest.mark.parametrize("conformal", [False, True])
def test_initializers_refuse_uniform_node_plane(scene, axis, multimode, grid_kind, conformal):
    _, freqs, steps = scene
    grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                cpml_layers=10, cpml_axes="x",
                conformal_faces={"y_hi", "z_hi"} if conformal else set())
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    kwargs = {"n_modes": 2} if multimode else {}
    metric = grid if grid_kind == "grid_dx" else grid.dx
    if grid_kind == "explicit":
        kwargs["grid"] = grid
    with pytest.raises(ValueError, match=_refusal(grid, axis, "+x")) as error:
        initializer(_ports(grid, axis)[0], metric, freqs, f0=6e9,
                    dft_total_steps=steps, **kwargs)
    assert str(error.value).endswith(
        "These walls are the Grid's declared faces; overrides by a run "
        "or extractor (an axis made periodic, a PMC face, or an axis withheld "
        "by pec_axes) must be declared on the Grid."
    )
    # Expected bounds come directly from node counts, independently of validation.
    valid = initializer(_ports(grid)[0], metric, freqs, f0=6e9,
                        dft_total_steps=steps, **kwargs)
    for cfg in valid if multimode else [valid]:
        assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny - 1)
        assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz - 1)


@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("grid_kind", ["explicit", "grid_dx"])
def test_2d_initializers_and_loop_accept_whole_z_plane(scene, monkeypatch, multimode, grid_kind):
    _, freqs, steps = scene
    grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.002), dx=0.002,
                cpml_layers=10, cpml_axes="x", mode="2d_tez")
    assert grid.nz == 1
    assert "z" not in grid.periodic_axes
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    kwargs = {"n_modes": 2} if multimode else {}
    metric = grid if grid_kind == "grid_dx" else grid.dx
    if grid_kind == "explicit":
        kwargs["grid"] = grid
    port = _ports(grid, "z")[0]._replace(b=grid.dx)
    whole = initializer(port, metric, freqs, f0=6e9, dft_total_steps=steps, **kwargs)
    configs = whole if multimode else [whole]
    for cfg in configs:
        assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny - 1)
        assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz)
    monkeypatch.setattr(uniform_loop, "_build_step_setup", _accepted_setup)
    with pytest.raises(_SetupReached):
        uniform_loop.run(grid, init_materials(grid.shape), 1, boundary="cpml",
                         cpml_axes="x", waveguide_ports=configs)


@pytest.mark.parametrize("walls", ["periodic", "pmc_hi"])
@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("grid_kind", ["explicit", "grid_dx"])
@pytest.mark.parametrize("conformal", [False, True])
def test_initializers_accept_declared_periodic_or_pmc_high_face(scene, walls, multimode,
                                                               grid_kind, conformal):
    _, freqs, steps = scene
    grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                cpml_layers=10, cpml_axes="x",
                periodic_axes="y" if walls == "periodic" else "",
                pmc_faces={"y_hi"} if walls == "pmc_hi" else set(),
                conformal_faces={"y_hi", "z_hi"} if conformal else set())
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    kwargs = {"n_modes": 2} if multimode else {}
    metric = grid if grid_kind == "grid_dx" else grid.dx
    if grid_kind == "explicit":
        kwargs["grid"] = grid
    whole = initializer(_ports(grid, "y")[0], metric, freqs, f0=6e9,
                        dft_total_steps=steps, **kwargs)
    for cfg in whole if multimode else [whole]:
        assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny)
        assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz - 1)


@pytest.mark.parametrize("axis", ["y", "z"])
@pytest.mark.parametrize("multimode", [False, True])
@pytest.mark.parametrize("graded", [False, True])
def test_initializers_without_uniform_grid_defer_to_loop(scene, monkeypatch, axis, multimode, graded):
    grid, freqs, steps = scene
    if graded:
        grid = make_nonuniform_grid((0.12, 0.04), np.full(10, 0.002), 0.002,
                                    cpml_layers=10, cpml_axes="x")
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    kwargs = {"n_modes": 2} if multimode else {}
    metric = grid if graded else grid.dx
    whole = initializer(_ports(grid, axis)[0], metric, freqs, f0=6e9,
                        dft_total_steps=steps, **kwargs)
    configs = whole if multimode else [whole]
    for cfg in configs:
        assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny if axis == "y" else grid.ny - 1)
        assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz if axis == "z" else grid.nz - 1)
    if graded:
        run = graded_loop.run_nonuniform
        monkeypatch.setattr(graded_loop, "_build_nu_scan", _no_step)
        run_kwargs = {"pec_faces": {"y_lo", "y_hi", "z_lo", "z_hi"}}
    else:
        run = uniform_loop.run
        monkeypatch.setattr(uniform_loop, "_build_step_setup", _no_step)
        run_kwargs = {"boundary": "cpml"}
    with pytest.raises(ValueError, match=_refusal(grid, axis, "0 (+x)")):
        run(grid, init_materials(grid.shape), 1, cpml_axes="x",
            waveguide_ports=configs, **run_kwargs)
    # Expected bounds come directly from node counts, independently of validation.
    valid = initializer(_ports(grid)[0], metric, freqs, f0=6e9,
                        dft_total_steps=steps, **kwargs)
    for cfg in valid if multimode else [valid]:
        assert (cfg.u_hi, cfg.v_hi) == (grid.ny - 1, grid.nz - 1)


@pytest.mark.parametrize("name", EXTRACTORS)
@pytest.mark.parametrize("axis", ["y", "z"])
def test_extractors_refuse_before_stepping(scene, monkeypatch, name, axis):
    monkeypatch.setattr(uniform_loop, "run", _no_step)
    with pytest.raises(ValueError, match=_refusal(scene[0], axis, "0 (+x)")):
        _extract(scene, name, axis)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("axis", ["y", "z"])
def test_loops_refuse_prebuilt_node_plane(scene, monkeypatch, graded, decay, axis):
    grid, freqs, steps = scene
    if graded:
        grid = make_nonuniform_grid((0.12, 0.04), np.full(10, 0.002), 0.002,
                                    cpml_layers=10, cpml_axes="x")
        module = graded_loop
        name = "run_nonuniform_until_decay" if decay else "run_nonuniform"
        monkeypatch.setattr(module, "_build_nu_scan", _no_step)
        kwargs = {"pec_faces": {"y_lo", "y_hi", "z_lo", "z_hi"}}
    else:
        module = uniform_loop
        name = "run_until_decay" if decay else "run"
        monkeypatch.setattr(module, "_build_step_setup", _no_step)
        kwargs = {"boundary": "cpml", "pec_axes": "yz"}
    kwargs.update(cpml_axes="x", waveguide_ports=_configs((grid, freqs, steps), axis))
    args = [grid, init_materials(grid.shape)]
    if decay:
        kwargs.update(max_steps=1, n_end=0, decay_by=0)
    else:
        args.append(1)
    with pytest.raises(ValueError, match=_refusal(grid, axis, "0 (+x)")):
        getattr(module, name)(*args, **kwargs)
    monkeypatch.setattr(module, "_build_nu_scan" if graded else "_build_step_setup", _accepted_setup)
    kwargs["waveguide_ports"] = _configs((grid, freqs, steps))
    with pytest.raises(_SetupReached):
        getattr(module, name)(*args, **kwargs)


@pytest.mark.parametrize("entry", ["run", "run_until_decay", "run_nonuniform",
                                   "run_nonuniform_until_decay", "extract_waveguide_s_matrix"])
@pytest.mark.parametrize("walls", ["pec_lo_pmc_hi", "pmc_lo_pec_hi", "periodic"])
def test_whole_plane_uses_the_runs_high_wall(scene, monkeypatch, entry, walls):
    _, freqs, steps = scene
    pec_faces = {"z_lo", "z_hi"}
    pmc_faces = set()
    periodic_axes = "y" if walls == "periodic" else ""
    if walls == "pec_lo_pmc_hi":
        pec_faces.add("y_lo")
        pmc_faces.add("y_hi")
    elif walls == "pmc_lo_pec_hi":
        pmc_faces.add("y_lo")
        pec_faces.add("y_hi")
    declarations = dict(cpml_layers=10, cpml_axes="x", pec_faces=pec_faces,
                        pmc_faces=pmc_faces, periodic_axes=periodic_axes)
    graded = "nonuniform" in entry
    if graded:
        grid = make_nonuniform_grid((0.12, 0.04), np.full(10, 0.002), 0.002,
                                    **declarations)
    else:
        grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                    **declarations)
    configs = [init_waveguide_port(
        port, grid.dx, freqs, f0=6e9, dft_total_steps=steps,
    ) for port in _ports(grid, "y")]
    for cfg in configs:
        assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny)
        assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz - 1)

    args = [grid, init_materials(grid.shape)]
    kwargs = {"cpml_axes": "x"}
    if graded:
        module = graded_loop
        monkeypatch.setattr(module, "_build_nu_scan", _accepted_setup)
        kwargs.update(pec_faces=pec_faces, pmc_faces=pmc_faces)
    else:
        module = uniform_loop
        monkeypatch.setattr(module, "_build_step_setup", _accepted_setup)
        kwargs.update(boundary="cpml", periodic=(False, walls == "periodic", False))
    if entry.startswith("extract"):
        module = waveguide_s
        # Reaching run proves the extractor itself admitted the aperture.
        monkeypatch.setattr(uniform_loop, "run", _accepted_setup)
        args.extend((configs, 1))
    else:
        kwargs["waveguide_ports"] = configs
        if entry.endswith("until_decay"):
            kwargs.update(max_steps=1, n_end=0, decay_by=0)
        else:
            args.append(1)
    if walls == "pmc_lo_pec_hi":
        with pytest.raises(ValueError, match=_refusal(grid, "y", "0 (+x)")):
            getattr(module, entry)(*args, **kwargs)
    else:
        with pytest.raises(_SetupReached):
            getattr(module, entry)(*args, **kwargs)


@pytest.mark.parametrize("normal_axis,transverse_axis", [("y", "z"), ("z", "y")])
def test_non_x_normal_loop_refuses_node_span_and_accepts_cells(scene, monkeypatch,
                                                              normal_axis, transverse_axis):
    grid, freqs, steps = scene
    count = grid.shape["xyz".index(transverse_axis)]
    port = WaveguidePort(
        x_index=3, y_slice=None, z_slice=None, normal_axis=normal_axis,
        direction=f"+{normal_axis}", u_slice=(0, grid.nx - 1), v_slice=(0, count),
        a=(grid.nx - 1) * grid.dx, b=(count - 1) * grid.dx,
    )
    cfg = init_waveguide_port(port, grid.dx, freqs, f0=6e9,
                              probe_offset=2, ref_offset=1, dft_total_steps=steps)
    assert (cfg.v_lo, cfg.v_hi) == (0, count)
    material = init_materials(grid.shape)
    monkeypatch.setattr(uniform_loop, "_build_step_setup", _no_step)
    with pytest.raises(ValueError, match=_refusal(grid, transverse_axis, f"0 (+{normal_axis})")):
        uniform_loop.run(grid, material, 1, boundary="cpml", waveguide_ports=[cfg])
    cell_cfg = init_waveguide_port(port._replace(v_slice=(0, count - 1)), grid.dx,
                                   freqs, f0=6e9, grid=grid, probe_offset=2,
                                   ref_offset=1, dft_total_steps=steps)
    assert (cell_cfg.u_lo, cell_cfg.u_hi) == (0, grid.nx - 1)
    assert (cell_cfg.v_lo, cell_cfg.v_hi) == (0, count - 1)
    monkeypatch.setattr(uniform_loop, "_build_step_setup", _accepted_setup)
    with pytest.raises(_SetupReached):
        uniform_loop.run(grid, material, 1, boundary="cpml", waveguide_ports=[cell_cfg])


def test_simulation_builds_the_accepted_cell_span():
    sim = Simulation(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=10)
    sim.add_waveguide_port(0.01, direction="+x")
    grid = sim._build_grid()
    cfg = sim._build_waveguide_port_config(sim._waveguide_ports[0], grid,
                                          jnp.linspace(4e9, 10e9, 15), n_steps=200)
    assert (cfg.u_lo, cfg.u_hi) == (0, grid.ny - 1)
    assert (cfg.v_lo, cfg.v_hi) == (0, grid.nz - 1)


@pytest.mark.parametrize("name", EXTRACTORS)
def test_cell_span_s_equals_merge_base(scene, name):
    """Compiled float32 sums vary across machines, so comparison is not bitwise."""
    reference = json.loads((Path(__file__).parents[2] / "data" /
                            "waveguide_cell_span_66be9e1f7.json").read_text())
    assert reference["merge_base"] == "66be9e1f77e121f5b40d83b0dcd330ebc7f239cc"
    expected = reference["extractors"][name]
    actual = _extract(scene, name)
    assert actual.dtype.str == expected["dtype"]
    assert list(actual.shape) == expected["shape"]
    shape = expected["shape"]
    expected_array = np.frombuffer(
        zlib.decompress(base64.b64decode(expected["zlib_base64"])),
        dtype=expected["dtype"],
    ).reshape(shape)
    nan_mismatches = np.count_nonzero(np.isnan(actual) != np.isnan(expected_array))
    finite = np.isfinite(expected_array)
    # Each S entry uses its own peak over finite frequency bins.
    # Exactly-zero entries must stay zero; all-NaN entries use only the mask.
    scale = np.max(np.where(finite, np.abs(expected_array), 0.0),
                   axis=-1, keepdims=True)
    delta_complex = np.zeros_like(actual)
    np.subtract(actual, expected_array, out=delta_complex,
                where=finite)
    delta = np.abs(delta_complex)
    ratio = np.divide(delta, scale, out=np.where(delta > 0, np.inf, 0.0),
                      where=scale > 0)
    ratio = np.where(finite & ~np.isfinite(actual), np.inf, ratio)
    worst_ratio = np.max(ratio)
    message = (f"max|delta| / own peak = {worst_ratio:.9g}; "
               f"NaN-mask mismatch count = {nan_mismatches}")
    assert nan_mismatches == 0, message
    assert np.all(ratio <= 1e-4), message
