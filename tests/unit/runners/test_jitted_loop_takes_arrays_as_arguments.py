"""The jitted single-device time loops take the per-cell arrays as arguments.

``run(until_decay=...)`` (and ``run(report_every=...)`` on a graded mesh)
drives the time loop through a jitted chunk (non-uniform) or a jitted step
(uniform). When that function closed over the material, PEC-mask and
coefficient arrays, XLA compiled each one into the program as a grid-sized
constant and held copies of it while compiling: on a large non-uniform board
(~1e8 cells) the first chunk's compile took +27.8 GiB of host memory and
67.6 s, against +0.16 GiB and 5.1 s with the arrays as arguments, with the
same outputs bit for bit. #1211 and #1217 fixed the same class on the
distributed runners (``test_distributed_nu_forward_arguments.py``); this
pins it for the single-device loops.

Two views of one invariant, per path:
* the compiled loop program holds no constant with as many elements as half
  the grid;
* the jitted function's closure holds no concrete, non-uniform volume array.

The model has a graded z mesh, a lossy dielectric slab, two PEC sheets and a
mu_r = 1.5 block, so no captured array is all-equal (an all-equal array is
lowered as one scalar broadcast and would hide a capture). A Debye slab
covers the dispersive E update.

Mutations that turn this red: return no arrays from
``split_loop_invariants``; or keep the call and its argument but let the
jitted loop step with the closure-bound ``setup.step_fn`` (non-uniform) or
``make_core_step(_step_ctx)`` (uniform).
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
import rfx.nonuniform as nonuniform_mod
import rfx.simulation as simulation_mod

_LOOPS = {nonuniform_mod: "_run_chunk", simulation_mod: "_single_step"}


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
                hlo = entry.lower(*a, **k).compile().as_text()
                for m in re.finditer(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+constant\(", hlo):
                    dims = tuple(int(d) for d in m.group(2).split(",") if d)
                    records["constants"].append((m.group(1), dims))
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
_U_DECAY = dict(until_decay=1e-30, decay_min_steps=6, decay_max_steps=6,
                decay_check_interval=3)

# (model kind, dispersive, run kwargs)
_CASES = {
    "graded-until_decay": ("graded", False, _NU_DECAY),
    "graded-dispersive-until_decay": ("graded", True, _NU_DECAY),
    "graded-report_every": ("graded", False, dict(
        n_steps=24, report_every=12, skip_preflight=True)),
    "uniform-until_decay": ("uniform", False, _U_DECAY),
    "uniform-dispersive-until_decay": ("uniform", True, _U_DECAY),
    "uniform-thin_z-until_decay": ("thin", True, _U_DECAY),
    "uniform-2d_tmz-until_decay": ("2d", False, _U_DECAY),
    # run(until_identified=True) drives the same chunked loop with a stop_fn
    "graded-until_identified": ("ringdown", False, None),
}


@pytest.fixture(scope="module", params=sorted(_CASES))
def loop_program(request):
    kind, dispersive, run_kwargs = _CASES[request.param]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "ringdown":
            from rfx.ringdown import RingdownSpec
            from tests.unit.sparams.test_ringdown_run import FREQS, _box
            return _record(_box("graded"), True, dict(
                n_steps=1500, compute_s_params=True, s_param_freqs=FREQS,
                skip_preflight=True, ringdown=RingdownSpec(),
                until_identified=True))
        return _record(_model(kind, dispersive), kind == "graded", run_kwargs)


def test_compiled_loop_has_no_grid_sized_constants(loop_program):
    assert loop_program["loops"], "the jitted time loop was never compiled here"
    constants = loop_program["constants"]
    assert constants, "no HLO constants parsed; the check would be vacuous"
    half = loop_program["cells"] // 2
    large = [(dtype, dims) for dtype, dims in constants
             if dims and int(np.prod(dims)) >= half]
    assert not large, f"grid-sized constants in the compiled loop: {large}"


def test_jitted_loop_closes_over_no_per_cell_array(loop_program):
    assert loop_program["loops"], "the jitted time loop was never compiled here"
    assert not loop_program["captures"], loop_program["captures"]
