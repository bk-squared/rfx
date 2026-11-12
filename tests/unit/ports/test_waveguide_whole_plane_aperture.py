"""Whole-node apertures are refused; cell-span spectra retain every stored bit."""

import base64
import json
from pathlib import Path
import re
import zlib

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.measurement import waveguide_s
from rfx.model.graded_grid import make_nonuniform_grid
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
@pytest.mark.parametrize("grid_kind,conformal", [("explicit", False), ("explicit", True), ("grid_dx", False)])
def test_initializers_refuse_whole_node_plane(scene, axis, multimode, grid_kind, conformal):
    grid, freqs, steps = scene
    if grid_kind == "grid_dx":
        grid = make_nonuniform_grid((0.12, 0.04), np.full(10, 0.002), 0.002,
                                    cpml_layers=10, cpml_axes="x")
    else:
        # The refusal has the same meaning with or without conformal faces.
        grid = Grid(freq_max=12e9, domain=(0.12, 0.04, 0.02), dx=0.002,
                    cpml_layers=10, cpml_axes="x")
        grid.conformal_faces = {"y_hi", "z_hi"} if conformal else set()
    initializer = init_multimode_waveguide_port if multimode else init_waveguide_port
    kwargs = {"n_modes": 2} if multimode else {}
    metric = grid if grid_kind == "grid_dx" else grid.dx
    if grid_kind == "explicit":
        kwargs["grid"] = grid
    with pytest.raises(ValueError, match=_refusal(grid, axis, "+x")):
        initializer(_ports(grid, axis)[0], metric, freqs, f0=6e9,
                    dft_total_steps=steps, **kwargs)
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
def test_cell_span_s_is_bitwise_equal_to_merge_base(scene, name):
    reference = json.loads((Path(__file__).parents[2] / "data" /
                            "waveguide_cell_span_70f259a45.json").read_text())
    assert reference["merge_base"] == "70f259a454ba662e51c1dadd2e21f5b8ac867f5c"
    expected = reference["extractors"][name]
    actual = _extract(scene, name)
    assert actual.dtype.str == expected["dtype"]
    assert list(actual.shape) == expected["shape"]
    assert actual.tobytes() == zlib.decompress(base64.b64decode(expected["zlib_base64"]))
