"""NU forward's compiled time loop receives per-cell data as jit arguments."""

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

pytestmark = pytest.mark.distributed


def _model(case):
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
        # mu_r != 1: XLA folds an all-equal captured array into a broadcast,
        # which the HLO-literal gate could not see (review, 2026-09-23).
        sim.add_material("block", eps_r=4.4, sigma=0.02, mu_r=1.5, **poles)
        sim.add(Box((5e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material="block")
        sim.add(Box((5e-3, 2e-3, 2e-3), (7e-3, 3e-3, 5e-3)), material="pec")
    sim.add_source((8e-3, 4e-3, 4e-3), "ez", amplitude_kind="field")
    sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
    return sim


def _per_cell_captures(body):
    captured, seen = [], set()

    def walk(value, path):
        if isinstance(value, jax.core.Tracer) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, jax.Array):
            if value.ndim >= 3:
                captured.append((path, value.shape))
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


# "nondispersive" traces the E update without Debye/Lorentz poles, a branch the
# dispersive model never reaches (review, 2026-09-23: a capture there passed).
@pytest.fixture(scope="module",
                params=[("mixed", None, 0), ("mixed", 2, 0), ("mixed", 2, 2),
                        ("cpml", None, 0)],
                ids=["plain", "segmented", "warmup_segmented", "nondispersive"])
def loop_program(request):
    if len(jax.devices("cpu")) < 2:
        pytest.skip("needs two CPU devices")
    case, checkpoint, warmup = request.param
    sim = _model(case)
    grid = sim._build_nonuniform_grid()
    occupancy = jnp.zeros(grid.shape, dtype=jnp.float32).at[8:10, 4:6, 4:6].set(0.2)
    slab_cells = int(np.prod(grid.shape)) // 2
    records = {"constants": [], "captures": [], "scans": 0, "slab_cells": slab_cells}
    scan = distributed_nu.lax.scan

    def traced_scan(body, *args, **kwargs):
        records["scans"] += 1
        records["captures"].extend(_per_cell_captures(body))
        return scan(body, *args, **kwargs)

    def traced_jit(fn, *args, **kwargs):
        entry = jax.jit(fn, *args, **kwargs)
        if fn.__name__ != "run_fn":
            return entry

        def run(*args, **kwargs):
            hlo = entry.lower(*args, **kwargs).compile().as_text()
            for match in re.finditer(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+constant\(", hlo):
                dims = tuple(int(d) for d in match.group(2).split(",") if d)
                records["constants"].append((match.group(1), dims))
            return entry(*args, **kwargs)

        return run

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(distributed_nu, "jax", SimpleNamespace(**{**vars(jax), "jit": traced_jit}))
        patch.setattr(distributed_nu, "lax", SimpleNamespace(**{**vars(distributed_nu.lax),
                                                            "scan": traced_scan}))
        result = sim.forward(
            distributed=True, devices=jax.devices("cpu")[:2], n_steps=8,
            checkpoint_every=checkpoint, n_warmup=warmup,
            pec_occupancy_override=occupancy, skip_preflight=True,
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


def test_scan_closures_have_no_concrete_per_cell_arrays(loop_program):
    assert loop_program["scans"], "must inspect the actual time loop"
    assert not loop_program["captures"], loop_program["captures"]
