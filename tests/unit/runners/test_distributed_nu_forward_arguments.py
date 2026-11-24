"""NU forward passes all arrays (including spacings) as time-loop jit arguments."""

from functools import partial
import re
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.materials.lorentz import lorentz_pole
from rfx.runners import distributed_nu
from rfx.stepping import slab_graded_legacy_loop
from tests.unit.runners.test_distributed_nu_forward_staging import _select_legacy_poles_cpml

pytestmark = pytest.mark.distributed


def _model(case):
    """The permeability contrast keeps an all-equal captured array from becoming a broadcast invisible to the HLO-literal gate."""
    cpml = case in ("cpml", "debye", "lorentz", "mixed")
    sim = Simulation(
        freq_max=15e9, domain=(15e-3, 7e-3, 7e-3), dx=1e-3,
        dz_profile=np.array([0.8, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2]) * 1e-3,
        boundary="cpml" if cpml else "pec", cpml_layers=2 if cpml else 0,
    )
    if case != "uniform":
        poles = {}
        if case in ("debye", "mixed"):
            poles["debye_poles"] = [DebyePole(delta_eps=1., tau=1e-11)]
        if case in ("lorentz", "mixed"):
            poles["lorentz_poles"] = [lorentz_pole(1., 2 * np.pi * 3e9, 1e9)]
        sim.add_material("block", eps_r=4.4, sigma=0.02, mu_r=1.5, **poles)
        sim.add(Box((5e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material="block")
        sim.add(Box((5e-3, 2e-3, 2e-3), (7e-3, 3e-3, 5e-3)), material="pec")
    sim.add_source((8e-3, 4e-3, 4e-3), "ez", amplitude_kind="field")
    sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
    return sim


def _per_cell_captures(body, *, observe=None):
    captured, seen = [], set()

    def walk(value, path):
        if isinstance(value, jax.core.Tracer) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (jax.Array, np.ndarray)):
            captured.append((path, value.shape))
            if observe is not None:
                observe(path, value)
            return
        if isinstance(value, FunctionType):
            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    continue
                walk(child, f"{path}.{name}")
        if isinstance(value, (tuple, list)):
            for name, child in zip(getattr(value, "_fields", range(len(value))), value):
                walk(child, f"{path}[{name}]")
        if isinstance(value, dict):
            for name, child in value.items():
                walk(child, f"{path}[{name}]")
        if isinstance(value, partial):
            walk(value.func, f"{path}.func")
            walk(value.args, f"{path}.args")
            walk(value.keywords, f"{path}.keywords")
        if hasattr(value, "__wrapped__"):
            walk(value.__wrapped__, f"{path}.__wrapped__")

    walk(body, "scan")
    return captured


def _loop_scene(name, case):
    """M1 scenes: keep case2 exactly as the original nondispersive fixture."""
    if name in (None, "case2"):
        return _model(case)
    if name in ("dft_plane", "flux_monitor", "ntff_box", "current_moment"):
        from tests.unit.runners._graded_records_scene import radiating_model
        sim, _, _ = radiating_model("cpml", 2, grading="z")
        sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
        if name == "dft_plane":
            sim.add_dft_plane_probe(axis="y", coordinate=4e-3, freqs=[3e9, 7e9])
        elif name == "flux_monitor":
            sim.add_flux_monitor(axis="y", coordinate=4e-3, freqs=[3e9, 7e9])
        elif name == "ntff_box":
            sim.add_ntff_box((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3), freqs=[3e9, 7e9])
        else:
            sim.add_current_moment_monitor((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3),
                                          block_size=2e-3, freqs=[3e9, 7e9])
        return sim
    if name in ("lumped_port", "wire_port"):
        sim = Simulation(freq_max=10e9, domain=(.016, .006, .006), dx=.001,
                         boundary="pec", dz_profile=np.array([.001]*3 + [.0005]*6))
        sim.add_port((.008, .003, .003), "ez", impedance=50,
                     **({"extent": .002} if name == "wire_port" else {}))
        sim.add_probe((.008, .003, .003), "ez")
        return sim
    if name == "graded_pmc":
        from tests.contracts.test_multidevice_setup_boundary import scene
        return scene("pmc", "graded")
    raise ValueError(name)


@pytest.fixture(scope="module",
                params=[("mixed", None, 0), ("mixed", 2, 0), ("mixed", 2, 2),
                        ("cpml", None, 0)],
                ids=["plain", "segmented", "warmup_segmented", "nondispersive"])
def loop_program(request, scene=None, diagnostic=None):
    if len(jax.devices("cpu")) < 2:
        pytest.skip("needs two CPU devices")
    case, checkpoint, warmup = request.param
    sim = _loop_scene(scene, case)
    grid = sim._build_nonuniform_grid()
    occupancy = jnp.zeros(grid.shape, dtype=jnp.float32).at[8:10, 4:6, 4:6].set(0.2)
    slab_cells = int(np.prod(grid.shape)) // 2
    records = {"constants": [], "captures": [], "scans": 0, "slab_cells": slab_cells}
    scan = slab_graded_legacy_loop.lax.scan
    observe = None if diagnostic is None else diagnostic.capture

    def traced_scan(body, *args, **kwargs):
        records["scans"] += 1
        records["captures"].extend(_per_cell_captures(body, observe=observe))
        return scan(body, *args, **kwargs)

    def traced_jit(fn, *args, **kwargs):
        entry = jax.jit(fn, *args, **kwargs)
        if fn.__name__ != "run_fn":
            return entry

        def run(*args, **kwargs):
            records["captures"].extend(_per_cell_captures(fn, observe=observe))
            lowered = entry.lower(*args, **kwargs)
            hlo = lowered.compile().as_text()
            if diagnostic is not None:
                diagnostic.program(jax.make_jaxpr(fn)(*args, **kwargs).consts,
                                   lowered.as_text(debug_info=True), hlo)
            for match in re.finditer(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+constant\(", hlo):
                dims = tuple(int(d) for d in match.group(2).split(",") if d)
                records["constants"].append((match.group(1), dims))
            return entry(*args, **kwargs)

        return run

    with pytest.MonkeyPatch.context() as patch:
        _select_legacy_poles_cpml(patch)
        for owner in (distributed_nu, slab_graded_legacy_loop):
            patch.setattr(owner, "jax", SimpleNamespace(**{**vars(jax), "jit": traced_jit}))
            patch.setattr(owner, "lax", SimpleNamespace(**{**vars(jax.lax), "scan": traced_scan}))
        result = sim.forward(
            distributed=True, devices=jax.devices("cpu")[:2], n_steps=8,
            checkpoint_every=checkpoint, n_warmup=warmup,
            pec_occupancy_override=(occupancy if scene in
                (None, "case2", "dft_plane", "flux_monitor", "ntff_box") else None),
            skip_preflight=True,
        )
        assert np.isfinite(result.time_series).all()
        assert np.max(np.abs(result.time_series)) > 0
    return records


def test_compiled_program_has_no_per_cell_constants(loop_program):
    constants = loop_program["constants"]
    assert constants, "no HLO constants parsed; the check would be vacuous"
    large = [(dtype, dims) for dtype, dims in constants
             if dims and int(np.prod(dims)) >= loop_program["slab_cells"]]
    assert not large, f"per-cell HLO literals: {large}"
