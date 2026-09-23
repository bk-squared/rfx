"""Differentiable NU setup: slab ownership, legacy bits, and memory lifetime.

Fresh subprocesses isolate live-array accounting and the virtual-device count.
The reference below deliberately keeps main cca8ee5b's init-then-split order.
"""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.core.yee import EPS_0, MaterialArrays
from rfx.materials.debye import init_debye
from rfx.materials.lorentz import init_lorentz, lorentz_pole
from rfx.nonuniform import make_current_source, position_to_index
from rfx.runners import distributed_nu as nu
from rfx.runners.distributed import _split_materials
from rfx.runners._distributed_common import shard_stacked
from rfx.simulation import ProbeSpec, SourceSpec

pytestmark = pytest.mark.distributed
ROOT = Path(__file__).resolve().parents[3]


def _model(case="mixed", large=False):
    # CPML plus the endpoint adds five cells on each axis: the memory gate's
    # actual grid is (45, 29, 29), including uneven two-device x padding.
    nx, ny, nz = (40, 24, 24) if large else (12, 7, 7)
    dz = np.full(nz, 1e-3)
    dz[:nz // 3] *= .8
    sim = Simulation(
        freq_max=15e9, domain=(nx * 1e-3, ny * 1e-3, float(dz.sum())),
        dx=1e-3, dz_profile=dz, boundary="cpml", cpml_layers=2,
    )
    poles = {}
    if case in ("mixed", "occupancy", "sigma", "debye"):
        poles["debye_poles"] = [DebyePole(1., 1e-11)]
    if case in ("mixed", "occupancy", "sigma", "lorentz"):
        poles["lorentz_poles"] = [lorentz_pole(1., 2 * np.pi * 3e9, 1e9)]
    sim.add_material("block", eps_r=4.4, sigma=.02, mu_r=1.5, **poles)
    sim.add(Box((.3 * nx * 1e-3, .2 * ny * 1e-3, .2 * dz.sum()),
                (.7 * nx * 1e-3, .8 * ny * 1e-3, .8 * dz.sum())), material="block")
    sim.add(Box((.45 * nx * 1e-3, .4 * ny * 1e-3, .4 * dz.sum()),
                (.55 * nx * 1e-3, .6 * ny * 1e-3, .6 * dz.sum())), material="pec")
    # Off the PEC block, close enough for nonzero traces/gradients in 8 steps;
    # the source and probe straddle the two-device seam in this 17-cell grid.
    sim.add_source((.45 * nx * 1e-3, .7 * ny * 1e-3, .5 * dz.sum()),
                   "ez", amplitude_kind="field")
    sim.add_probe((.6 * nx * 1e-3, .7 * ny * 1e-3, .5 * dz.sum()), "ez")
    return sim


def _legacy_forward(self, *, eps_override=None, sigma_override=None,
                    pec_mask_override=None, pec_occupancy_override=None,
                    n_steps, devices, checkpoint_every=None, n_warmup=0,
                    **unused):
    """Main's full-domain setup, frozen as an independent staging oracle.

    Only preflight and unsupported-port checks are omitted. Source normalization,
    padding, initializer order and all runner arguments follow cca8ee5b.
    Locals intentionally survive the runner call for the before measurement.
    """
    grid = self._build_nonuniform_grid()
    materials, db_spec, lr_spec, mask = self._assemble_materials_nu(grid)
    concrete = materials
    materials = materials._replace(
        eps_r=materials.eps_r if eps_override is None else eps_override,
        sigma=materials.sigma if sigma_override is None else sigma_override,
        eps_r_lumped=materials.eps_r_lumped if eps_override is None else None,
        sigma_lumped=materials.sigma_lumped if sigma_override is None else None,
    )
    if pec_mask_override is not None:
        mask = pec_mask_override if mask is None else mask | pec_mask_override
    debye = None if db_spec is None else init_debye(db_spec[0], materials, grid.dt, mask=db_spec[1])
    lorentz = None if lr_spec is None else init_lorentz(lr_spec[0], materials, grid.dt, mask=lr_spec[1])
    sg = nu.build_sharded_nu_grid(grid, len(devices))
    mesh = Mesh(np.array(devices), ("x",))
    shd = NamedSharding(mesh, P("x"))
    padded = materials
    if sg.pad_x:
        widths = ((0, sg.pad_x), (0, 0), (0, 0))
        padded = MaterialArrays(*(jnp.pad(a, widths, constant_values=v)
                                  for a, v in zip(materials[:3], (1., 0., 1.))))
    slabs = _split_materials(padded, len(devices), sg.ghost_width)
    placed = MaterialArrays(*(shard_stacked(a, shd) for a in slabs[:3]))
    pec = nu.shard_pec_mask_x_slab(mask, sg)
    occupancy = nu.shard_pec_occupancy_x_slab(pec_occupancy_override, sg)
    cpml_params, cpml_stacked = nu.init_cpml_for_sharded_nu(
        sg, len(devices), pec_faces=getattr(self, "_pec_faces", None))
    cpml = nu.shard_cpml_state_x_slab(cpml_stacked, sg, mesh)
    db = None if debye is None else (
        nu.shard_debye_coeffs_x_slab(debye[0], sg, mesh),
        nu.shard_debye_state_x_slab(debye[1], sg, mesh))
    lr = None if lorentz is None else (
        nu.shard_lorentz_coeffs_x_slab(lorentz[0], sg, mesh),
        nu.shard_lorentz_state_x_slab(lorentz[1], sg, mesh))
    sources = []
    for pe in self._ports:
        i, j, k, c, wf = make_current_source(
            grid, position_to_index(grid, pe.position), pe.component,
            pe.waveform, n_steps, concrete, amplitude_kind=pe.amplitude_kind)
        sources.append(SourceSpec(i=int(i), j=int(j), k=int(k), component=c,
                                  waveform=jnp.asarray(wf)))
    probes = [ProbeSpec(*map(int, position_to_index(grid, pe.position)), pe.component)
              for pe in self._probes]
    result = nu.run_nonuniform_distributed_pec(
        sg, placed, pec, n_steps, sources=sources, probes=probes,
        n_devices=len(devices), exchange_interval=1, debye=db, lorentz=lr,
        devices=devices, cpml_params=cpml_params, cpml_state=cpml,
        sharded_pec_occupancy=occupancy, checkpoint_every=checkpoint_every,
        n_warmup=n_warmup, emit_time_series=True,
        pmc_faces=frozenset(self._boundary_spec.pmc_faces()))
    return self._pack_nu_forward_result(
        time_series=result["time_series"], grid=grid, ntff_data=None,
        ntff_box=None, s_params=None, freqs=None, dft_planes=None,
        wire_port_sparams=None)


def _inputs(sim, case):
    shape = sim._build_nonuniform_grid().shape
    rng = np.random.default_rng(1217)
    eps = jnp.asarray(rng.uniform(1.2, 4.8, shape).astype(np.float32))
    kwargs = {}
    if case == "occupancy":
        kwargs["pec_occupancy_override"] = jnp.asarray(rng.uniform(0, .2, shape).astype(np.float32))
    if case == "sigma":
        kwargs["sigma_override"] = jnp.asarray(rng.uniform(0, .04, shape).astype(np.float32))
    return eps, kwargs


def _forward(sim, eps, kwargs):
    return sim.forward(eps_override=eps, distributed=True, devices=jax.devices("cpu"),
                       n_steps=8, skip_preflight=True, **kwargs).time_series


def _bits(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, label
    assert np.isfinite(a).all() and np.isfinite(b).all(), label + " nonfinite"
    assert a.tobytes() == b.tobytes(), (label, int(np.count_nonzero(a != b)),
                                      float(np.max(np.abs(a - b))))


def _bit_case(case, checkpoint, warmup, transform):
    sim = _model(case)
    eps, kwargs = _inputs(sim, case)
    kwargs.update(checkpoint_every=checkpoint, n_warmup=warmup)
    f = lambda e: _forward(sim, e, kwargs)

    def evaluate():
        if transform == "bits":
            trace = f(eps)
            gradient = jax.grad(lambda e: jnp.sum(f(e) ** 2))(eps)
            assert np.any(np.asarray(trace)) and np.any(np.asarray(gradient)), "vacuous bits"
            return trace, gradient
        plain = f(eps)
        if transform == "jvp":
            primal = jax.jvp(f, (eps,), (jnp.ones_like(eps),))[0]
        elif transform == "vjp":
            primal = jax.vjp(f, eps)[0]
        elif transform == "value_and_grad":
            def loss(e):
                trace = f(e)
                return jnp.sum(trace ** 2), trace

            (_, primal), _ = jax.value_and_grad(loss, has_aux=True)(eps)
        elif transform == "vmap":
            primal = jax.vmap(f)(jnp.stack([eps, eps * 1.1]))[0]
        elif transform == "stop_gradient":
            primal = f(jax.lax.stop_gradient(eps))
        else:
            raise AssertionError(transform)
        return plain, primal

    actual = evaluate()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sim, "_forward_distributed_nonuniform_from_materials",
                      MethodType(_legacy_forward, sim))
        expected = evaluate()
    for i, (a, b) in enumerate(zip(actual, expected)):
        _bits(a, b, f"{case}/{transform}/{i}")
    pairs = None
    if transform != "bits":
        pairs = [np.asarray(a).tobytes() == np.asarray(b).tobytes() for a, b in (actual, expected)]
        assert pairs[0] == pairs[1], (transform, pairs)
    print("RESULT " + json.dumps({"version": jax.__version__, "case": case,
                                  "devices": len(jax.devices()), "transform": transform,
                                  "bits": True, "cross_equal_new_main": pairs,
                                  "checkpoint": checkpoint, "warmup": warmup}))


def _per_shard():
    sim = _model("occupancy")
    eps, kwargs = _inputs(sim, "occupancy")
    captures = []

    def capture(*args, **kw):
        bound = inspect.signature(runner).bind(*args, **kw).arguments
        mesh = Mesh(np.array(jax.devices("cpu")), ("x",))
        shd = NamedSharding(mesh, P("x"))
        # The old runner received masks before placement; compare on the
        # same destination sharding it uses on entry to its compiled loop.
        selected = {name: bound[name] for name in (
            "sharded_materials", "sharded_pec_mask", "sharded_pec_occupancy",
            "debye", "lorentz", "cpml_state")}
        captures.append(jax.tree.map(lambda a: jax.device_put(a, shd), selected))
        return {"time_series": jnp.zeros((8, 1))}

    runner = nu.run_nonuniform_distributed_pec
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nu, "run_nonuniform_distributed_pec", capture)
        _forward(sim, eps, kwargs)
        patch.setattr(sim, "_forward_distributed_nonuniform_from_materials",
                      MethodType(_legacy_forward, sim))
        _forward(sim, eps, kwargs)
    checked = 0
    for (path, a), (_, b) in zip(*[jax.tree_util.tree_flatten_with_path(c)[0] for c in captures]):
        for x, y in zip(a.addressable_shards, b.addressable_shards):
            assert x.device == y.device and x.index == y.index
            _bits(x.data, y.data, str(path))
            checked += 1
    assert checked == 49 * len(jax.devices()), checked
    print("RESULT " + json.dumps({"shard_arrays": checked, "version": jax.__version__}))


def _memory(mode, mutation="none", legacy=False):
    sim = _model(large=True)
    grid = sim._build_nonuniform_grid()
    cells = int(np.prod(grid.shape))
    eps = jnp.full(grid.shape, 1.5, dtype=jnp.float32)
    devices = jax.devices("cpu")
    nx_local = nu.build_sharded_nu_grid(grid, len(devices)).nx_local
    records, extents, retained = [], {"debye": [], "lorentz": []}, []
    scan = jax.lax.scan
    codes = {init_debye.__code__: "debye", init_lorentz.__code__: "lorentz"}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            arrays = (frame.f_locals["materials"], frame.f_locals["mask"])
            for arr in jax.tree.leaves(arrays):
                if isinstance(arr, jax.core.Tracer):
                    extent = arr.shape[-3]  # shard_map traces local inputs
                else:
                    extent = max(s.data.shape[-3] for s in arr.addressable_shards)
                extents[codes[frame.f_code]].append(extent)

    def traced_scan(*args, **kwargs):
        sys.setprofile(None)
        if not records:
            per_device = {str(d): 0 for d in devices}
            whole = []
            for arr in jax.live_arrays():
                for shard in arr.addressable_shards:
                    per_device[str(shard.device)] += shard.data.nbytes
                if len(arr.sharding.device_set) == 1 and arr.size >= cells:
                    whole.append({"device": str(arr.addressable_shards[0].device),
                                  "shape": arr.shape, "dtype": str(arr.dtype),
                                  "caller": arr is eps})
            records.append({"bytes": per_device, "whole": whole, "cells": cells,
                            "grid": grid.shape, "extents": extents, "nx_local": nx_local})
        return scan(*args, **kwargs)

    stage = nu.stage_forward_dispersion_x_slab

    def mutate_stage(mat, dt, spec, sg, mesh, kind):
        if mutation == "whole_init" and kind == "debye":
            # Reintroduce the old whole-domain initialization, process-locally.
            materials, db, _, _ = sim._assemble_materials_nu(grid)
            retained.append(init_debye(db[0], materials, dt, mask=db[1]))
        if mutation == "retain" and kind == "debye":
            retained.append(jnp.full(grid.shape, 2., dtype=jnp.float32))
        return stage(mat, dt, spec, sg, mesh, kind)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax.lax, "scan", traced_scan)
        patch.setattr(nu, "stage_forward_dispersion_x_slab", mutate_stage)
        if legacy:
            patch.setattr(sim, "_forward_distributed_nonuniform_from_materials",
                          MethodType(_legacy_forward, sim))
        sys.setprofile(profile)
        try:
            f = lambda e: _forward(sim, e, {})
            result = f(eps) if mode == "plain" else jax.grad(lambda e: jnp.sum(f(e) ** 2))(eps)
            jax.block_until_ready(result)
        finally:
            sys.setprofile(None)
    assert records
    record = records[0]
    record.update(mode=mode, version=jax.__version__, legacy=legacy)
    print("RESULT " + json.dumps(record), flush=True)
    if legacy:
        return
    for kind, seen in extents.items():
        assert seen and max(seen) <= nx_local, f"whole-domain {kind} initialization: {seen} > {nx_local}"
    unexpected = [a for a in record["whole"] if not a["caller"]]
    assert not unexpected, f"whole-domain setup arrays: {unexpected}"
    sizes = list(record["bytes"].values())
    # The user's A1 bound: 30% covers small replicated profiles/scalars and
    # transient shard views; exclude the one caller-owned eps buffer.
    assert sizes[0] - eps.nbytes <= 1.3 * min(sizes), record


def _multipole_shards():
    """Different pole masks expose a rank/pole transpose hidden by one pole."""
    grid = _model()._build_nonuniform_grid()
    devices = jax.devices("cpu")
    sg = nu.build_sharded_nu_grid(grid, len(devices))
    assert sg.pad_x > 0
    mesh = Mesh(np.array(devices), ("x",))
    rng = np.random.default_rng(1208)
    materials = MaterialArrays(*(
        jnp.asarray(rng.uniform(lo, hi, grid.shape).astype(np.float32))
        for lo, hi in ((1., 6.), (0., .1), (1., 2.))))
    placed = MaterialArrays(*(nu.stage_forward_array_x_slab(a, sg, mesh, pad)
                             for a, pad in zip(materials[:3], (1., 0., 1.))))
    masks = [jnp.asarray(rng.random(grid.shape) > .5) for _ in range(2)]
    checked = 0
    for kind, poles, init, split_c, split_s in (
        ("debye", [DebyePole(1., 1e-11), DebyePole(.5, 3e-11)], init_debye,
         nu.shard_debye_coeffs_x_slab, nu.shard_debye_state_x_slab),
        ("lorentz", [lorentz_pole(1., 2 * np.pi * 3e9, 1e9),
                     lorentz_pole(.5, 2 * np.pi * 5e9, 2e9)], init_lorentz,
         nu.shard_lorentz_coeffs_x_slab, nu.shard_lorentz_state_x_slab),
    ):
        actual = nu.stage_forward_dispersion_x_slab(
            placed, grid.dt, (poles, masks), sg, mesh, kind)
        coeffs, state = init(poles, materials, grid.dt, mask=masks)
        expected = split_c(coeffs, sg, mesh), split_s(state, sg, mesh)
        for (path, a), (_, b) in zip(*[jax.tree_util.tree_flatten_with_path(c)[0]
                                      for c in (actual, expected)]):
            for x, y in zip(a.addressable_shards, b.addressable_shards):
                assert x.device == y.device and x.index == y.index
                _bits(x.data, y.data, str(path))
                checked += 1
    assert checked == 20 * len(devices)
    print("RESULT " + json.dumps({"multipole_shard_arrays": checked,
                                  "version": jax.__version__}))


def _zero_cc_stage(original):
    def mutate(*args):
        result = original(*args)
        if args[-1] == "lorentz" and result is not None:
            coeffs, state = result
            cc = coeffs.cc.at[0].set(0.).at[-1].set(0.)
            result = coeffs._replace(cc=cc), state
        return result
    return mutate


def _mutation_cc():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nu, "stage_forward_dispersion_x_slab",
                      _zero_cc_stage(nu.stage_forward_dispersion_x_slab))
        try:
            _per_shard()
        except AssertionError as error:
            assert "cc" in str(error), str(error)
            print("MUTATION_RED cc " + str(error))
        else:
            raise AssertionError("cc mutation survived per-shard gate")
        sim = _model()
        eps, kw = _inputs(sim, "mixed")
        gradient = jax.grad(lambda e: jnp.sum(_forward(sim, e, kw) ** 2))(eps)
        print("RESULT " + json.dumps({"cc_mutation_nan_gradient": bool(np.isnan(gradient).any())}))


def _child(mode, *args, devices=2, expected_error=None):
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    # Retain a requested JAX overlay while removing paths to other RFX clones.
    pythonpath = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                  if p and not (Path(p) / "rfx").is_dir()]
    env["PYTHONPATH"] = os.pathsep.join([*pythonpath, str(ROOT)])
    run = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).resolve()),
                          mode, *map(str, args)], env=env, text=True,
                         capture_output=True, timeout=90)
    if expected_error:
        assert run.returncode != 0 and expected_error in run.stderr, run.stdout + run.stderr
        print(f"MUTATION_RED {args[-1]}: {expected_error}")
    else:
        assert run.returncode == 0, run.stdout + run.stderr
    for line in run.stdout.splitlines():
        if line.startswith(("RESULT ", "MUTATION_RED ")):
            print(line)
    return run


@pytest.mark.parametrize("mode", ["plain", "grad"])
def test_live_arrays_and_slab_initialization(mode):
    _child("memory", mode)


@pytest.mark.parametrize("devices", [2, 3])
def test_every_staged_shard_matches_legacy(devices):
    _child("shards", devices=devices)


# Each case is a fresh process (~6-30 s). The fast lane keeps the memory gate, the
# legacy shard equality, one bit case and the transforms the CI build is judged
# on; the full matrix and the mutations run with the slow suite.
_slow = pytest.mark.slow


@pytest.mark.parametrize("devices", [2, pytest.param(3, marks=_slow)])
def test_multiple_pole_shards_match_legacy(devices):
    _child("multipole", devices=devices)


@pytest.mark.parametrize("case,devices", [("mixed", 2)] + [
    pytest.param(case, devices, marks=_slow)
    for case in ("mixed", "debye", "lorentz", "lossy", "occupancy", "sigma")
    for devices in (2, 3, 4) if (case, devices) != ("mixed", 2)
])
def test_trace_and_eps_gradient_bits(case, devices):
    _child("bits", case, "None", 0, "bits", devices=devices)


@_slow
@pytest.mark.parametrize("checkpoint,warmup", [(2, 0), (2, 2), (None, 2)])
def test_checkpoint_warmup_bits(checkpoint, warmup):
    _child("bits", "mixed", checkpoint, warmup, "bits")


@pytest.mark.parametrize("transform", ["jvp", "vjp", "value_and_grad", "vmap",
                                       pytest.param("stop_gradient", marks=_slow)])
def test_cross_trace_pattern(transform):
    _child("bits", "mixed", "None", 0, transform)


@_slow
@pytest.mark.parametrize("mutation,error", [
    ("whole_init", "whole-domain debye initialization"),
    ("retain", "whole-domain setup arrays"),
])
def test_mutations_trip_gate(mutation, error):
    _child("memory", "plain", mutation, expected_error=error)


@_slow
def test_zero_lorentz_cc_mutation():
    _child("cc")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "memory":
        _memory(*args)
    elif command == "before":
        _memory(args[0], legacy=True)
    elif command == "shards":
        _per_shard()
    elif command == "multipole":
        _multipole_shards()
    elif command == "cc":
        _mutation_cc()
    elif command == "bits":
        _bit_case(args[0], None if args[1] == "None" else int(args[1]), int(args[2]), args[3])
    else:
        raise AssertionError(command)
