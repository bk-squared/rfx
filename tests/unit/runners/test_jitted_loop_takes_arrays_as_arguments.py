"""The single-device time loops compile no per-cell array into a program.

``run(until_decay=...)`` drives the uniform time loop through a jitted step.
When that function closed over the material, PEC-mask and coefficient arrays,
XLA compiled each one into the program as a grid-sized constant and held
copies of it while compiling: on a large non-uniform board (~1e8 cells) the
first chunk's compile took +27.8 GiB of host memory and 67.6 s, against
+0.16 GiB and 5.1 s with the arrays as arguments, with the same outputs bit
for bit. #1211 and #1217 fixed the same class on the distributed runners
(``test_distributed_nu_forward_arguments.py``); this pins it for the
single-device loops.

Uniform rows: the compiled loop program holds no constant with as many
elements as half the grid.

Graded rows (``until_decay``, ``report_every``, ``until_identified``): every
chunk is the eager scan the single scan runs, which takes the arrays its body
closes over as operands. The hazard is only a ``jax.jit`` whose function
closes over them. Two views:
* (i) every scan the graded loop runs is ``recorded_scan`` of the setup's own
  ``step_fn`` (the function the single scan runs), and the scans cover every
  recorded step;
* (ii) no ``jax.jit`` created in the graded loop, and not the jitted energy
  measure of the decay stop, wraps a function whose closure holds a concrete,
  non-uniform volume array; on the rows that reach the energy check, that
  program also holds no grid-sized constant.

The model has a graded z mesh, a lossy dielectric slab, two PEC sheets and a
mu_r = 1.5 block, so no captured array is all-equal (an all-equal array is
lowered as one scalar broadcast and would hide a capture). A Debye slab
covers the dispersive E update. The energy-step rows run 3000 steps so the
energy check is reached; the 24-step rows end before it.

Mutations that turn this red: return no arrays from
``split_loop_invariants`` or let the jitted uniform loop step with
``make_core_step(_step_ctx)`` (uniform); wrap the graded chunk scan in a
``jax.jit`` closing over ``setup.step_fn`` (view ii); run the graded energy
stop's one step on that eager chunk scan (view ii, energy-step rows).
"""
from __future__ import annotations

import dataclasses
from functools import partial
import re
from types import FunctionType, SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.materials.lorentz import lorentz_pole
import rfx.measurement.decay as decay_mod
import rfx.stepping.graded_loop as nonuniform_mod
import rfx.stepping.uniform_loop as simulation_mod

_LOOPS = {simulation_mod: "_single_step"}
_CONSTANT = re.compile(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+constant\(")


def _model(kind: str, dispersive: bool = False):
    """kind: "graded" (3-D, graded z), "uniform" (3-D), "thin" (3-D, one cell
    of interior in z), "2d" (mode="2d_tmz")."""
    dx = 1e-3
    kw = {}
    if kind == "graded":
        kw["dz_profile"] = np.array(
            [1.0, 1.0, 0.8, 0.5, 0.35, 0.35, 0.35, 0.5, 0.8, 1.0, 1.0, 1.0]) * 1e-3
    zdom = {"graded": 8.65e-3, "uniform": 9e-3, "thin": 1e-3, "2d": 1e-3}[kind]
    if kind == "2d":
        kw["mode"] = "2d_tmz"
    # #1138: the stub sheet is solved -15 % / +7.8 % off its drawing; this test
    # checks what the compiled loop closes over, not sheet size.
    sim = Simulation(freq_max=12e9, domain=(14e-3, 11e-3, zdom), dx=dx,
                     boundary="cpml", cpml_layers=4, snap="declared", **kw)
    if kind in ("thin", "2d"):
        z = 0.0 if kind == "2d" else 0.5e-3
        sim.add_material("sub", eps_r=3.66, sigma=0.004,
                         **({"debye_poles": [DebyePole(delta_eps=1.0, tau=1e-11)]}
                            if dispersive else {}))
        sim.add(Box((3e-3, 2e-3, 0.0), (9e-3, 8e-3, zdom)), material="sub")
        sim.add_material("mag", eps_r=2.0, mu_r=1.5)
        sim.add(Box((10e-3, 2e-3, 0.0), (12e-3, 5e-3, zdom)), material="mag")
        sim.add(Box((5e-3, 9e-3, 0.0), (7e-3, 10e-3, zdom)), material="pec")
        sim.add_source((4e-3, 5.5e-3, z), "ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
        sim.add_probe((11e-3, 5.5e-3, z), "ez")
        return sim
    z0, z1 = 2.8e-3, 3.85e-3
    if dispersive:
        # Debye slab and a Lorentz block: both dispersive E updates
        sim.add_material("sub", eps_r=3.66, sigma=0.004,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add_material("lor", eps_r=2.0, mu_r=1.5,
                         lorentz_poles=[lorentz_pole(1.0, 2 * np.pi * 9e9, 1e9)])
        sim.add(Box((8e-3, 1e-3, 5e-3), (10e-3, 3e-3, 7e-3)), material="lor")
    else:
        sim.add_material("sub", eps_r=3.66, sigma=0.004)
        sim.add_material("mag", eps_r=2.0, mu_r=1.5)
        sim.add(Box((8e-3, 1e-3, 5e-3), (10e-3, 3e-3, 7e-3)), material="mag")
        # a Leontovich (surface-impedance) sheet and a lumped port
        sim.add_thin_conductor(Box((3e-3, 8e-3, z1), (12e-3, 9e-3, z1)),
                               sigma_bulk=5.8e7, thickness=35e-6,
                               surface_impedance_f0=6e9)
        sim.add_port((12e-3, 5.5e-3, z0), "ez", impedance=50.0, excite=False,
                     extent=z1 - z0)
    sim.add(Box((0.0, 0.0, z0), (14e-3, 11e-3, z1)), material="sub")
    sim.add(Box((0.0, 0.0, z0), (14e-3, 11e-3, z0)), material="pec")
    sim.add(Box((3e-3, 4.5e-3, z1), (12e-3, 6.5e-3, z1)), material="pec")
    sim.add_source((4e-3, 5.5e-3, 4.8e-3), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
    sim.add_probe((11e-3, 5.5e-3, 4.8e-3), "ez")
    return sim


def _is_cell_array(x, grid_axes):
    return (isinstance(x, (jax.Array, np.ndarray))
            and not isinstance(x, jax.core.Tracer)
            and sum(1 for n in x.shape if n > 1) >= grid_axes)


def _captured_cell_arrays(fn, grid_axes):
    """Concrete, not-all-equal per-cell arrays reachable from ``fn``'s closure."""
    found, seen = [], set()

    def walk(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if _is_cell_array(value, grid_axes):
            a = np.asarray(value)
            if not np.all(a == a.reshape(-1)[0]):
                found.append((path, value.shape))
            return
        if isinstance(value, FunctionType):
            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
                try:
                    walk(cell.cell_contents, f"{path}.{name}")
                except ValueError:
                    continue
        elif isinstance(value, partial):
            walk(value.func, f"{path}.func")
            walk(value.args, f"{path}.args")
            walk(value.keywords, f"{path}.keywords")
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                walk(child, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}[{key!r}]")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                walk(getattr(value, f.name), f"{path}.{f.name}")

    walk(fn, fn.__name__)
    return found


def _hlo_constants(lowered):
    hlo = lowered.compile().as_text()
    return [(m.group(1), tuple(int(d) for d in m.group(2).split(",") if d))
            for m in _CONSTANT.finditer(hlo)]


def _record(sim, graded, run_kwargs):
    records = {"constants": [], "captures": [], "loops": 0}
    shape = (sim._build_nonuniform_grid().shape if graded
             else sim._build_grid().shape)
    records["cells"] = int(np.prod(shape))
    records["grid_axes"] = sum(1 for n in shape if n > 1)

    def traced_jit(fn, *args, **kwargs):
        entry = jax.jit(fn, *args, **kwargs)
        if getattr(fn, "__name__", "") not in _LOOPS.values():
            return entry
        compiled = {}

        def call(*a, **k):
            if not compiled:
                records["loops"] += 1
                records["captures"].extend(_captured_cell_arrays(fn, records["grid_axes"]))
                records["constants"].extend(_hlo_constants(entry.lower(*a, **k)))
                compiled["done"] = True
            return entry(*a, **k)

        return call

    with pytest.MonkeyPatch.context() as patch:
        for mod in _LOOPS:
            patch.setattr(mod, "jax", SimpleNamespace(**{**vars(jax), "jit": traced_jit}))
        result = sim.run(**({"compute_s_params": False} | run_kwargs))
    ts = np.asarray(result.time_series)
    assert ts.shape[0] > 0 and np.isfinite(ts).all()
    return records


_NU_DECAY = dict(until_decay=1e-30, decay_min_steps=24, decay_max_steps=24,
                 decay_check_interval=12, skip_preflight=True)
# long enough that the decay stop forms the energy after the drive (12 checks)
_NU_ENERGY = dict(until_decay=1e-30, decay_min_steps=250, decay_max_steps=3000,
                  decay_check_interval=250, skip_preflight=True)
_U_DECAY = dict(until_decay=1e-30, decay_min_steps=6, decay_max_steps=6,
                decay_check_interval=3)

# (model kind, dispersive, run kwargs)
_CASES = {
    "uniform-until_decay": ("uniform", False, _U_DECAY),
    "uniform-dispersive-until_decay": ("uniform", True, _U_DECAY),
    "uniform-thin_z-until_decay": ("thin", True, _U_DECAY),
    "uniform-2d_tmz-until_decay": ("2d", False, _U_DECAY),
}

_GRADED = {
    "graded-until_decay": ("graded", False, _NU_DECAY),
    "graded-dispersive-until_decay": ("graded", True, _NU_DECAY),
    "graded-report_every": ("graded", False, dict(
        n_steps=24, report_every=12, skip_preflight=True)),
    "graded-energy_step": ("graded", False, _NU_ENERGY),
    "graded-dispersive-energy_step": ("graded", True, _NU_ENERGY),
    # run(until_identified=True) drives the same chunked loop with a stop_fn
    "graded-until_identified": ("ringdown", False, None),
}


@pytest.fixture(scope="module", params=sorted(_CASES))
def loop_program(request):
    kind, dispersive, run_kwargs = _CASES[request.param]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _record(_model(kind, dispersive), False, run_kwargs)


def test_compiled_loop_has_no_grid_sized_constants(loop_program):
    assert loop_program["loops"], "the jitted time loop was never compiled here"
    constants = loop_program["constants"]
    assert constants, "no HLO constants parsed; the check would be vacuous"
    half = loop_program["cells"] // 2
    large = [(dtype, dims) for dtype, dims in constants
             if dims and int(np.prod(dims)) >= half]
    assert not large, f"grid-sized constants in the compiled loop: {large}"


def _record_graded(sim, run_kwargs):
    shape = sim._build_nonuniform_grid().shape
    grid_axes = sum(1 for n in shape if n > 1)
    rec = {"cells": int(np.prod(shape)), "setups": [], "scans": [], "jits": [],
           "energy": [], "energy_calls": 0, "energy_constants": []}
    build, scan = nonuniform_mod._build_nu_scan, nonuniform_mod.recorded_scan

    def built(*a, **k):
        setup = build(*a, **k)
        rec["setups"].append(setup)
        return setup

    def scanned(f, init, xs=None, *a, **k):
        rec["scans"].append((f, int(np.shape(jax.tree_util.tree_leaves(xs)[0])[0])))
        return scan(f, init, xs, *a, **k)

    def graded_jit(fn, *a, **k):
        rec["jits"].append(fn)
        return jax.jit(fn, *a, **k)

    def decay_jit(fn, *a, **k):
        entry = jax.jit(fn, *a, **k)
        if getattr(fn, "__name__", "") != "stepped_energy":
            return entry
        rec["energy"].append(fn)

        def call(*aa, **kk):
            if not rec["energy_calls"]:
                rec["energy_constants"].extend(_hlo_constants(entry.lower(*aa, **kk)))
            rec["energy_calls"] += 1
            return entry(*aa, **kk)

        return call

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nonuniform_mod, "_build_nu_scan", built)
        patch.setattr(nonuniform_mod, "recorded_scan", scanned)
        patch.setattr(nonuniform_mod, "jax", SimpleNamespace(**{**vars(jax), "jit": graded_jit}))
        patch.setattr(decay_mod, "jax", SimpleNamespace(**{**vars(jax), "jit": decay_jit}))
        result = sim.run(**({"compute_s_params": False} | run_kwargs))
    ts = np.asarray(result.time_series)
    assert ts.shape[0] > 0 and np.isfinite(ts).all()
    rec["steps"] = int(ts.shape[0])
    # closures are walked after the run, when every cell they read is filled
    rec["jit_captures"] = [c for fn in rec["jits"] for c in _captured_cell_arrays(fn, grid_axes)]
    rec["energy_captures"] = [c for fn in rec["energy"] for c in _captured_cell_arrays(fn, grid_axes)]
    return rec


@pytest.fixture(scope="module", params=sorted(_GRADED))
def graded_run(request):
    kind, dispersive, run_kwargs = _GRADED[request.param]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "ringdown":
            from rfx.ringdown import RingdownSpec
            from tests.unit.sparams.test_ringdown_run import FREQS, _box
            rec = _record_graded(_box("graded"), dict(
                n_steps=1500, compute_s_params=True, s_param_freqs=FREQS,
                skip_preflight=True, ringdown=RingdownSpec(),
                until_identified=True))
        else:
            rec = _record_graded(_model(kind, dispersive), run_kwargs)
    rec["case"] = request.param
    return rec


def test_graded_chunks_are_the_single_scan(graded_run):
    """View (i): each chunk is recorded_scan of the setup's own step_fn."""
    scans = graded_run["scans"]
    assert scans, "the graded loop ran no recorded_scan"
    own = {id(setup.step_fn) for setup in graded_run["setups"]}
    foreign = [f for f, _ in scans if id(f) not in own]
    assert not foreign, f"a chunk scans another step function: {foreign}"
    if graded_run["case"] != "graded-until_identified":
        covered = sum(n for _, n in scans)
        assert covered == graded_run["steps"], (
            f"the recorded scans cover {covered} of {graded_run['steps']} steps")


def test_graded_jits_close_over_no_cell_array(graded_run):
    """View (ii): no jit in the graded loop or the energy measure closes over a per-cell array."""
    assert not graded_run["jit_captures"], (
        f"a jit in the graded loop closes over per-cell arrays: {graded_run['jit_captures'][:4]}")
    assert not graded_run["energy_captures"], (
        f"the energy measure closes over per-cell arrays: {graded_run['energy_captures'][:4]}")
    if graded_run["case"].endswith("energy_step"):
        assert graded_run["energy_calls"], "the energy check was never reached; the row is vacuous"
        assert graded_run["energy_constants"], "no HLO constants parsed; the check would be vacuous"
        half = graded_run["cells"] // 2
        large = [(dtype, dims) for dtype, dims in graded_run["energy_constants"]
                 if dims and int(np.prod(dims)) >= half]
        assert not large, f"grid-sized constants in the energy measure: {large}"
