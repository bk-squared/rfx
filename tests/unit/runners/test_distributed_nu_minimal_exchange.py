"""NU live Yee halos: frozen reference, poisoned ghosts, AD and compiled loops.

Run this file with ``-m distributed`` to include the slow topology/AD matrix.
The script modes emit peak-ULP and bitwise measurements for each JAX version.
"""

from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import jax

# Distributed initialization must precede any backend use in a fresh worker.
if __name__ == "__main__" and sys.argv[1] == "worker" and sys.argv[2] != "reference":
    jax.distributed.initialize(sys.argv[2], 2, int(sys.argv[3]), initialization_timeout=30)

import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
import matplotlib

from rfx import Box, DebyePole, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.materials.lorentz import lorentz_pole
from rfx.nonuniform import position_to_index
from rfx.runners import distributed_nu as runner
# These are frozen test-local copies, independent of production exchange and
# sampling helpers. Share #1222's oracle rather than freezing a second version.
from tests.unit.runners.test_distributed_minimal_exchange import (
    _FIELDS, _full_e, _full_h, _full_sample, _poison,
)

pytestmark = pytest.mark.distributed
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
N_STEPS = 8


def _model():
    dx = np.full(14, 1e-3)
    dx[3:6] *= .8
    dx[6:9] *= 1.2
    sim = Simulation(
        freq_max=15e9, domain=(float(dx.sum()), 4e-3, 4e-3), dx=1e-3,
        dx_profile=dx, boundary=BoundarySpec(
            x="cpml", y=Boundary(lo="pmc", hi="cpml"),
            z=Boundary(lo="pec", hi="cpml")), cpml_layers=1,
    )
    grid = sim._build_nonuniform_grid()
    n_devices = len(jax.devices("cpu"))
    assert grid.nx == 17 and grid.nx % n_devices, grid.shape
    width = (grid.nx + n_devices - 1) // n_devices
    edges = np.r_[0., np.cumsum(np.asarray(grid.dx_arr))]
    indices = sorted({i for rank in range(1, n_devices)
                      for i in (rank * width - 1, rank * width)})
    for i in indices:
        pos = (float(edges[i] - edges[grid.pad_x_lo]), 2e-3, 2e-3)
        assert position_to_index(grid, pos)[0] == i
        for component in ("ey", "ez"):
            sim.add_source(pos, component, amplitude_kind="field",
                           waveform=lambda t: jnp.cos(t * 2e10))
        for component in _FIELDS:
            sim.add_probe(pos, component)
    sim.add_material("lossy_poles", eps_r=2.5, sigma=.03,
                     debye_poles=[DebyePole(.5, 1e-11)],
                     lorentz_poles=[lorentz_pole(.5, 2 * np.pi * 3e9, 1e9)])
    sim.add(Box((2e-3, 1e-3, 1e-3), (12e-3, 3e-3, 3e-3)), material="lossy_poles")
    sim.add(Box((6e-3, 3e-3, 1e-3), (9e-3, 4e-3, 3e-3)), material="pec")
    return sim


def _inputs(sim, sharded):
    rng = np.random.default_rng(1222)
    shape = sim._build_nonuniform_grid().shape
    host = [rng.uniform(lo, hi, shape).astype(np.float32)
            for lo, hi in ((1.2, 4.8), (0., .04), (0., .2))]
    return tuple(sim.shard_distributed_override(a) if sharded else jnp.asarray(a) for a in host)


def _forward(sim, designs, checkpoint=None, warmup=0, steps=N_STEPS, emit=True):
    return sim.forward(
        eps_override=designs[0], sigma_override=designs[1], pec_occupancy_override=designs[2],
        distributed=True, devices=jax.devices("cpu"), n_steps=steps,
        checkpoint_every=checkpoint, n_warmup=warmup, emit_time_series=emit,
        skip_preflight=True,
    ).time_series


def _evaluate(sim, designs, checkpoint=None, warmup=0, steps=N_STEPS):
    f = lambda *a: _forward(sim, a, checkpoint, warmup, steps)
    trace = f(*designs)
    grads = jax.grad(lambda *a: jnp.sum(f(*a) ** 2), argnums=(0, 1, 2))(*designs)
    assert trace.shape == (steps, len(sim._probes)) and trace.dtype == jnp.float32
    assert trace.sharding.is_fully_replicated
    for grad, design in zip(grads, designs):
        assert grad.sharding == design.sharding
    return dict(zip(("trace", "eps", "sigma", "occupancy"), (trace, *grads)))


def _metric(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite"
    peak = float(np.max(np.abs(b)))
    error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return {"ulp": error / float(np.spacing(np.float32(peak))), "max_abs": error,
            "peak": peak, "bitwise": a.tobytes() == b.tobytes()}


def _within(actual, expected):
    metric = _metric(actual, expected)
    assert metric["ulp"] <= 9, metric
    return metric


def _report(kind, **data):
    print("RESULT " + json.dumps({"kind": kind, "version": jax.__version__,
                                  "devices": len(jax.devices()), **data}), flush=True)


@contextmanager
def _reference(patch, *, probes=True):
    with patch.context() as p:
        p.setattr(runner, "_exchange_h_ghosts_nu", _full_h)
        p.setattr(runner, "_exchange_e_ghosts_nu", _full_e)
        if probes:
            # The singleton axis adapts the old replicated per-step psum to
            # the new post-scan sum, without changing the frozen computation.
            p.setattr(runner, "sample_probes_shmap", lambda *a, **k: _full_sample(*a, **k)[None, :])
        yield


def _parity(patch, sharded=False, checkpoint=None, warmup=0):
    sim = _model()
    designs = _inputs(sim, sharded)
    actual = _evaluate(sim, designs, checkpoint, warmup)
    with _reference(patch):
        expected = _evaluate(sim, designs, checkpoint, warmup)
    nx = sim._build_nonuniform_grid().nx
    width = (nx + len(jax.devices()) - 1) // len(jax.devices())
    for seam in range(width, nx, width):
        assert np.any(np.asarray(expected["occupancy"])[seam - 1:seam + 1]), (
            "occupancy gradient must witness the input halo transpose", seam)
    if sharded:
        for name in ("eps", "sigma", "occupancy"):
            assert np.all(np.asarray(actual[name])[nx:] == 0), name
    metrics = {}
    for name in actual:
        assert np.any(expected[name]), f"vacuous {name}"
        metrics[name] = _within(actual[name], expected[name])
    _report("parity", sharded=sharded, checkpoint=checkpoint, warmup=warmup, **metrics)


def _install_poison(patch, live=False):
    for name in ("_exchange_h_ghosts_nu", "_exchange_e_ghosts_nu"):
        original = getattr(runner, name)

        def exchange(st, mesh, n_devices, original=original):
            # Poison every INTERIOR slot the old exchange wrote and the new
            # one leaves alone; physical-boundary ghosts were never exchanged.
            return _poison(original(st, mesh, n_devices), mesh, n_devices, live=live)

        patch.setattr(runner, name, exchange)


def _poison_case(patch, live=False):
    sim = _model()
    designs = _inputs(sim, True)
    expected = _forward(sim, designs, 3, 1)
    with patch.context() as p:
        _install_poison(p, live)
        actual = _forward(sim, designs, 3, 1)
    _report("poison", **_within(actual, expected))


def _transforms(sharded):
    sim = _model()
    designs = _inputs(sim, sharded)
    f = lambda *a: _forward(sim, a, 3, 1)
    plain = f(*designs)
    primals = {
        "jvp": jax.jvp(f, designs, tuple(jnp.ones_like(a) for a in designs))[0],
        "vjp": jax.vjp(f, *designs)[0],
        # Local staging's device_put batches onto its leading axis on older
        # JAX; use a batch divisible by the mesh size on every topology.
        "vmap": jax.vmap(f)(*(jnp.stack([a * (1 + .01 * i)
                                        for i in range(len(jax.devices()))])
                              for a in designs))[0],
    }

    def loss(*a):
        trace = f(*a)
        return jnp.sum(trace ** 2), trace

    (value, primals["value_and_grad"]), _ = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(*designs)
    _report("transforms", sharded=sharded,
            value=_within(value, jnp.sum(plain ** 2)),
            **{k: _within(v, plain) for k, v in primals.items()})


def _loop_counts(hlo):
    """Count operations in each actual while body and its called computations."""
    computations, current = {}, None
    for line in hlo.splitlines():
        match = re.match(r"^(?:ENTRY )?%([\w.-]+).*\{$", line)
        if match:
            current = match[1]
            computations[current] = []
        elif line == "}":
            current = None
        elif current is not None:
            computations[current].append(line)
    bodies = re.findall(r"\bwhile\([^\n]*\bbody=%([\w.-]+)", hlo)
    assert bodies, "no compiled while bodies"
    result = []
    for body in bodies:
        visited = set()

        def walk(name):
            if name in visited:
                return []
            visited.add(name)
            lines = computations[name]
            children = {n for line in lines for n in re.findall(r"%([\w.-]+)", line)
                        if n in computations}
            return lines + [line for n in children for line in walk(n)]

        lines = walk(body)
        counts = {}
        for op in ("collective-permute", "all-reduce"):
            matches = [line for line in lines if re.search(rf"\s{op}(?:-start|-done)?\(", line)]
            starts = sum(f" {op}-start(" in line for line in matches)
            dones = sum(f" {op}-done(" in line for line in matches)
            assert starts == dones, (body, op, matches)
            counts[op] = len(matches) - dones
        # XLA 0.4.33 also lowers scatters to local while loops. They contain
        # no communication. Require exactly one/two communicating time loops
        # below, so losing an entire forward/transpose exchange still fails.
        if any(counts.values()):
            result.append(counts)
    assert result, "no communicating time loops"
    return result


def _collectives(grad=False):
    sim = _model()
    designs = _inputs(sim, True)
    captured = []

    def capture_jit(fn, *args, **kwargs):
        entry = jax.jit(fn, *args, **kwargs)
        if fn.__name__ != "run_fn":
            return entry

        def run(*a, **k):
            captured.append((entry, a))
            return entry(*a, **k)
        return run

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "jax", SimpleNamespace(**{**vars(jax), "jit": capture_jit}))
        _forward(sim, designs)
    assert len(captured) == 1
    entry, args = captured[0]
    if grad:
        c0, invariants, warmup, opt = args
        materials, mask, occupancy, debye, lorentz, cpml, spacings = invariants

        def loss(eps, sigma, occ, db, lr):
            # These are the loop inputs reached by the three design gradients;
            # replicated spacings/CPML coefficients are constants in that AD.
            inv = (materials._replace(eps_r=eps, sigma=sigma), mask, occ,
                   db, lr, cpml, spacings)
            return jnp.sum(entry(c0, inv, warmup, opt)[1] ** 2)

        compiled = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4))).lower(
            materials.eps_r, materials.sigma, occupancy, debye, lorentz).compile()
    else:
        compiled = entry.lower(*args).compile()
    hlo = compiled.as_text()
    counts = _loop_counts(hlo)
    _report("collectives", grad=grad, loops=counts)
    return counts


def _assert_counts(counts, grad=False):
    # No warmup/remat here: one forward loop and, under grad, its transpose.
    assert len(counts) == (2 if grad else 1), counts
    assert all(c == {"collective-permute": 2, "all-reduce": 0} for c in counts), counts


def _long_run(patch, sharded=True):
    sim = _model()
    designs = _inputs(sim, sharded)
    maxima = np.zeros(6)
    samples = [0]

    def observe(*fields):
        for i, (name, field) in enumerate(zip(_FIELDS, fields)):
            slabs = np.asarray(field).reshape(len(jax.devices()), -1, *field.shape[1:])
            dead = []
            if name in ("ex", "hx", "ey", "ez"):
                dead.append(slabs[1:, 0])
            if name in ("ex", "hx", "hy", "hz"):
                dead.append(slabs[:-1, -1])
            value = max(float(np.max(np.abs(a))) for a in dead)
            assert np.isfinite(value), (name, value)
            maxima[i] = max(maxima[i], value)
        samples[0] += 1

    with patch.context() as p:
        for name in ("_exchange_h_ghosts_nu", "_exchange_e_ghosts_nu"):
            original = getattr(runner, name)

            def measured(st, mesh, n_devices, original=original):
                st = original(st, mesh, n_devices)
                jax.debug.callback(observe, *(getattr(st, c) for c in _FIELDS))
                return st

            p.setattr(runner, name, measured)
        outputs = _evaluate(sim, designs, steps=500)
        jax.block_until_ready(outputs)
        jax.effects_barrier()
    assert samples[0] >= 1000, samples
    for name, a in outputs.items():
        assert np.isfinite(a).all() and np.any(a), name
    _report("long", sharded=sharded, steps=500, samples=samples[0],
            dead_ghost_max=dict(zip(_FIELDS, maxima.tolist())),
            gradient_max={k: float(np.max(np.abs(v))) for k, v in outputs.items() if k != "trace"})


def _env(devices):
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}")
    overlay = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
               if p and not (Path(p) / "rfx").is_dir()]
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *overlay])
    env.setdefault("MPLCONFIGDIR", matplotlib.get_configdir())
    return {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}


def _child(mode, *args, devices=2):
    run = subprocess.run([sys.executable, str(SCRIPT), mode, *map(str, args)],
                         cwd=ROOT, env=_env(devices), capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)


@pytest.mark.parametrize("sharded", [False, True])
def test_trace_and_design_gradients_match_full_exchange(monkeypatch, sharded):
    _parity(monkeypatch, sharded)


def test_dead_ghosts(monkeypatch):
    _poison_case(monkeypatch)


def test_live_ghost_poison_is_detected(monkeypatch):
    with pytest.raises(AssertionError, match="nonfinite"):
        _poison_case(monkeypatch, live=True)


@pytest.mark.parametrize("grad", [False, True])
def test_compiled_loop_collectives(grad):
    _assert_counts(_collectives(grad), grad)


@pytest.mark.slow
@pytest.mark.parametrize("devices", [2, 3, 4])
def test_topology_matrix(devices):
    _child("matrix", devices=devices)


@pytest.mark.slow
@pytest.mark.parametrize("devices", [2, 3, 4])
def test_long_run_finite_gradients_and_dead_ghosts(devices):
    _child("long", devices=devices)


@pytest.mark.slow
def test_mutations_trip_gates(monkeypatch):
    def missing_ez(st, mesh, n_devices):
        @partial(shard_map, mesh=mesh, in_specs=P("x"), out_specs=P("x"), check_rep=False)
        def only_ey(ey):
            received = lax.ppermute(ey[1], "x", [(i, i - 1) for i in range(1, n_devices)])
            return ey.at[-1].set(jnp.where(lax.axis_index("x") < n_devices - 1, received, ey[-1]))
        return st._replace(ey=only_ey(st.ey))

    with monkeypatch.context() as p:
        p.setattr(runner, "_exchange_e_ghosts_nu", missing_ez)
        with pytest.raises(AssertionError, match="ulp") as failure:
            _parity(p, True)
        _report("mutation_red", mutation="drop_ez", error=str(failure.value))
    with pytest.raises(AssertionError, match="nonfinite"):
        _poison_case(monkeypatch, live=True)
    _report("mutation_red", mutation="live_poison")
    with _reference(monkeypatch, probes=False):
        counts = _collectives()
        assert counts == [{"collective-permute": 12, "all-reduce": 0}], counts
        with pytest.raises(AssertionError):
            _assert_counts(counts)
    _report("mutation_red", mutation="full_exchange", loops=counts)


@pytest.mark.parametrize("emit,empty", [(False, False), (True, True)])
def test_empty_trace_shape_and_replication(monkeypatch, emit, empty):
    sim = _model()
    designs = _inputs(sim, True)
    if empty:
        sim._probes.clear()
    captured = []

    def checked_jit(fn, *a, **k):
        entry = jax.jit(fn, *a, **k)
        if fn.__name__ != "run_fn":
            return entry

        def run(*args, **kwargs):
            result = entry(*args, **kwargs)
            captured.append(result[1])
            return result
        return run

    monkeypatch.setattr(runner, "jax", SimpleNamespace(**{**vars(jax), "jit": checked_jit}))
    trace = _forward(sim, designs, 3, 1, emit=emit)
    assert len(captured) == 1
    assert captured[0].shape == (N_STEPS, 0) and captured[0].dtype == jnp.float32
    assert captured[0].sharding.is_fully_replicated
    if not emit:
        assert trace is None  # Public forward preserves the no-emission API.
        return
    assert trace.shape == (N_STEPS, 0) and trace.dtype == jnp.float32
    assert trace.sharding.is_fully_replicated


def _worker(address, rank, output):
    rank = int(rank)
    if address != "reference":
        assert jax.process_count() == 2 and jax.local_device_count() == 1
    sim = _model()
    shape, sharding = sim.distributed_override_layout()

    def design(offset, step):
        def local(index):
            lo, hi, _ = index[0].indices(shape[0])
            indices = np.arange(lo * shape[1] * shape[2], hi * shape[1] * shape[2])
            return (offset + (indices % 71) * step).astype(np.float32).reshape(hi - lo, *shape[1:])
        return jax.make_array_from_callback(shape, sharding, local)

    designs = (design(1.2, .05), design(0., .0005), design(0., .002))
    result = _evaluate(sim, designs, 3, 1)
    np.save(Path(output) / f"trace-{rank}.npy", np.asarray(result.pop("trace")))
    for name, grad in result.items():
        for shard in grad.addressable_shards:
            np.save(Path(output) / f"grad-{name}-{shard.device.id}.npy", np.asarray(shard.data))
    if address != "reference":
        jax.distributed.shutdown()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux: distributed gRPC bind fails on Darwin")
def test_two_process_plain_and_grad(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    _child("worker", "reference", 0, reference)
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        address = f"localhost:{listener.getsockname()[1]}"
    processes, handles = [], []
    deadline = time.monotonic() + 300
    try:
        for rank in range(2):
            handle = (tmp_path / f"rank-{rank}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(
                [sys.executable, str(SCRIPT), "worker", address, str(rank), str(tmp_path)],
                cwd=ROOT, env=_env(1), stdout=handle, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for handle in handles:
            handle.close()
    logs = "\n".join((tmp_path / f"rank-{rank}.log").read_text() for rank in range(2))
    assert all(p.returncode == 0 for p in processes), logs
    expected = np.load(reference / "trace-0.npy")
    assert np.any(expected)
    for rank in range(2):
        _report("two_process_trace", rank=rank,
                **_within(np.load(tmp_path / f"trace-{rank}.npy"), expected))

    def gather(directory, name):
        return np.concatenate([np.load(p) for p in sorted(
            directory.glob(f"grad-{name}-*.npy"), key=lambda p: int(p.stem.split("-")[2]))])

    for name in ("eps", "sigma", "occupancy"):
        expected = gather(reference, name)
        assert np.any(expected), name
        _report("two_process_gradient", design=name, **_within(gather(tmp_path, name), expected))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    with pytest.MonkeyPatch.context() as patch:
        if mode == "matrix":
            for sharded in (False, True):
                for checkpoint, warmup in ((None, 0), (3, 1)):
                    _parity(patch, sharded, checkpoint, warmup)
                _transforms(sharded)
            _poison_case(patch)
            for grad in (False, True):
                _assert_counts(_collectives(grad), grad)
        elif mode == "long":
            _long_run(patch)
        elif mode == "transforms":
            for sharded in (False, True):
                _transforms(sharded)
        elif mode == "worker":
            _worker(*args)
        else:
            raise AssertionError(mode)
