"""Global NU designs: differentiable halos, capacity, and Linux multi-processes.

Fresh children isolate memory accounting and CPU topology. The Linux gate uses
one device per process and a separate one-process, two-device reference. No
host gather is used inside the objective (only owned gradient shards are saved).
"""

import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import jax

# Initialize before importing RFX or any fixture that might initialize JAX's
# backend. Each worker is a fresh interpreter with one local CPU device.
if __name__ == "__main__" and sys.argv[1] == "worker" and sys.argv[2] != "reference":
    jax.distributed.initialize(sys.argv[2], 2, int(sys.argv[3]), initialization_timeout=30)

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.runners import distributed_nu as nu
from tests.unit.runners.test_distributed_nu_forward_arguments import loop_program as _loop_program
from tests.unit.runners.test_distributed_nu_forward_staging import _model

pytestmark = pytest.mark.distributed
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()


def _metric(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.isfinite(a).all() and np.isfinite(b).all(), "ulp comparison requires finite arrays"
    peak = np.max(np.abs(a))
    diff = np.max(np.abs(a - b))
    return {"ulp": float(diff / np.spacing(peak)), "max_abs": float(diff),
            "peak": float(peak), "bitwise": a.tobytes() == b.tobytes()}


def _within(a, b):
    metric = _metric(a, b)
    assert metric["ulp"] <= 9, metric
    return metric


def _host_inputs(sim):
    rng = np.random.default_rng(1217)
    shape = sim._build_nonuniform_grid().shape
    return [rng.uniform(lo, hi, shape).astype(np.float32)
            for lo, hi in ((1.2, 4.8), (0., .04), (0., .2))]


def _trace(sim, eps, sigma=None, occupancy=None):
    return sim.forward(
        eps_override=eps, sigma_override=sigma, pec_occupancy_override=occupancy,
        distributed=True, devices=jax.devices("cpu"), n_steps=8,
        skip_preflight=True,
    ).time_series


def _parity():
    sim = _model()
    host = _host_inputs(sim)
    local = [jnp.asarray(a) for a in host]
    sharded = [sim.shard_distributed_override(a) for a in host]
    traces, gradients = [], []
    for inputs in (local, sharded):
        f = lambda e, s, o: _trace(sim, e, s, o)
        traces.append(f(*inputs))
        # All three designs: eps and sigma ghosts only feed ghost-cell updates
        # that the E exchange overwrites (zero cotangent), but occupancy ghosts
        # enter the seam cells' soft-PEC product, so only a gradient w.r.t.
        # occupancy sees a halo transpose that drops or double-counts a ghost.
        gradients.append(jax.grad(lambda e, s, o: jnp.sum(f(e, s, o) ** 2),
                                  argnums=(0, 1, 2))(*inputs))
    nx = host[0].shape[0]
    assert np.any(traces[0]) and all(np.any(g) for g in gradients[0])
    metrics = {"trace": _within(traces[0], traces[1])}
    for name, want, got, design in zip(("eps", "sigma", "occupancy"),
                                       gradients[0], gradients[1], sharded):
        assert got.sharding == design.sharding, name
        assert np.all(np.asarray(got)[nx:] == 0), name
        metrics[f"gradient_{name}"] = _within(want, np.asarray(got)[:nx])
    # Direct halo equality makes the seam contract observable even when a
    # short trace happens not to encounter a particular material ghost.
    sg = nu.build_sharded_nu_grid(sim._build_nonuniform_grid(), len(jax.devices()))
    for a, b, pad in zip(local, sharded, (1., 0., 0.)):
        expected = nu.stage_forward_array_x_slab(a, sg, b.sharding.mesh, pad)
        actual = nu.stage_forward_array_x_slab(b, sg, b.sharding.mesh, pad)
        _within(expected, actual)
    print("RESULT " + json.dumps({"mode": "parity", "devices": len(jax.devices()),
                                  "version": jax.__version__, **metrics}), flush=True)


def _transforms():
    sim = _model()
    eps = sim.shard_distributed_override(_host_inputs(sim)[0])
    f = lambda e: _trace(sim, e)
    plain = f(eps)
    primals = {
        "jvp": jax.jvp(f, (eps,), (jnp.ones_like(eps),))[0],
        "vjp": jax.vjp(f, eps)[0],
        "vmap": jax.vmap(f)(jnp.stack([eps, eps * 1.1]))[0],
    }

    def loss(e):
        trace = f(e)
        return jnp.sum(trace ** 2), trace

    (_, primals["value_and_grad"]), _ = jax.value_and_grad(loss, has_aux=True)(eps)
    print("RESULT " + json.dumps({"version": jax.__version__, "sharded_transforms": {
        name: _within(plain, primal) for name, primal in primals.items()}}), flush=True)


def _memory(mode):
    sim = _model(large=True)
    grid = sim._build_nonuniform_grid()
    cells = int(np.prod(grid.shape))
    shape, sharding = sim.distributed_override_layout()
    eps = jax.make_array_from_callback(
        shape, sharding, lambda index: np.full(
            tuple(s.stop - (s.start or 0) if s.stop is not None else n
                  for s, n in zip(index, shape)), 1.5, np.float32))
    records = []
    scan = jax.lax.scan

    def inspect_scan(*args, **kwargs):
        if not records:
            sizes = {str(d): 0 for d in jax.devices()}
            whole = []
            for arr in jax.live_arrays():
                for shard in arr.addressable_shards:
                    sizes[str(shard.device)] += shard.data.nbytes
                    if shard.data.size >= cells:
                        whole.append({"shape": arr.shape, "device": str(shard.device)})
            records.append({"bytes": sizes, "bytes_per_global_cell": {
                d: count / cells for d, count in sizes.items()}, "whole": whole,
                "cells": cells, "grid": grid.shape, "mode": mode})
        return scan(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax.lax, "scan", inspect_scan)
        f = lambda e: _trace(sim, e)
        result = f(eps) if mode == "plain" else jax.grad(lambda e: jnp.sum(f(e) ** 2))(eps)
        jax.block_until_ready(result)
    assert records, "must measure the actual loop entry"
    print("RESULT " + json.dumps(records[0]), flush=True)
    assert not records[0]["whole"], f"whole-domain buffer: {records[0]}"


def _mutate(kind):
    """Process-local fault injection; exiting the child discards the mutation."""
    if kind == "spacing":
        source = inspect.getsource(nu.run_nonuniform_distributed_pec)
        source = source.replace(
            '(inv_dx_sharded, inv_dy_rep, inv_dz_rep,\n         inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep) = spacings',
            '(_, inv_dy_rep, inv_dz_rep,\n         inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep) = spacings')
        # Keep the module globals so the closure gate can instrument jit/scan.
        exec(source, vars(nu))
    elif kind == "halo":
        original = nu.stage_sharded_forward_override

        def no_halo(arr, sg, mesh, pad_value):
            result = original(arr, sg, mesh, pad_value)
            return nu.shard_map(lambda slab: slab.at[0].set(0).at[-1].set(0),
                                mesh=mesh, in_specs=nu.P("x"), out_specs=nu.P("x"),
                                check_rep=False)(result)

        nu.stage_sharded_forward_override = no_halo
    elif kind == "whole":
        retained = []

        def copy_whole(arr, sg, mesh, pad_value):
            # Recreate the prohibited staging route and keep its input alive
            # through the loop, as happens when a caller owns the local design.
            local = jnp.asarray(np.asarray(arr), device=jax.devices()[0])
            retained.append(local)
            return nu.stage_forward_array_x_slab(local[:sg.nx], sg, mesh, pad_value)

        nu.stage_sharded_forward_override = copy_whole
    else:
        raise AssertionError(kind)


def _env(devices):
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
           "OMP_NUM_THREADS": "1"}
    overlay = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
               if p and not (Path(p) / "rfx").is_dir()]
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *overlay])
    return {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}


def _child(mode, *args, devices=2):
    run = subprocess.run([sys.executable, str(SCRIPT), mode, *map(str, args)],
                         cwd=ROOT, env=_env(devices), capture_output=True,
                         text=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)


@pytest.mark.parametrize("devices", [2, pytest.param(3, marks=pytest.mark.slow)])
def test_sharded_overrides_match_local_trace_and_gradient(devices):
    _child("parity", devices=devices)


@pytest.mark.slow
def test_sharded_transformed_primals():
    _child("transforms")


@pytest.mark.parametrize("mode", ["plain", "grad"])
def test_no_whole_domain_buffer_at_loop(mode):
    _child("memory", mode)


@pytest.fixture(scope="module")
def loop_program():
    return _loop_program.__wrapped__(SimpleNamespace(param=("mixed", None, 0)))


def test_compiled_loop_has_no_concrete_arrays(loop_program):
    assert loop_program["scans"]
    assert not loop_program["captures"], loop_program["captures"]
    constants = loop_program["constants"]
    assert constants, "HLO check must observe constants"
    assert not [(dtype, dims) for dtype, dims in constants
                if dims and np.prod(dims) >= loop_program["slab_cells"]]


@pytest.mark.slow
@pytest.mark.parametrize("case", [("mixed", 2, 0), ("mixed", 2, 2), ("cpml", None, 0)])
def test_other_loop_branches_have_no_concrete_arrays(case):
    program = _loop_program.__wrapped__(SimpleNamespace(param=case))
    test_compiled_loop_has_no_concrete_arrays(program)


def test_forward_does_not_gather_final_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forward gathered the final fields")

    monkeypatch.setattr(nu, "unstack_and_gather", forbidden)
    sim = _model()
    eps = sim.shard_distributed_override(_host_inputs(sim)[0])
    assert np.isfinite(_trace(sim, eps)).all()


@pytest.mark.slow
def test_checkpoint_tail_with_sharded_design():
    sim = _model()
    eps = sim.shard_distributed_override(_host_inputs(sim)[0])

    def forward(e, chunk):
        return sim.forward(eps_override=e, distributed=True, n_steps=8,
                           checkpoint_every=chunk, n_warmup=1,
                           skip_preflight=True).time_series

    _within(forward(eps, None), forward(eps, 3))
    gradients = [jax.grad(lambda e: jnp.sum(forward(e, chunk) ** 2))(eps)
                 for chunk in (None, 3)]
    _within(*gradients)


def test_layout_validation():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    sim = _model()
    shape, sharding = sim.distributed_override_layout()
    sg = nu.build_sharded_nu_grid(sim._build_nonuniform_grid(), len(jax.devices()))
    wrong = NamedSharding(Mesh(sharding.mesh.devices[::-1], ("x",)), P("x"))
    arr = jax.make_array_from_callback(shape, wrong, lambda i: np.zeros(shape, np.float32)[i])
    with pytest.raises(ValueError, match="same ordered devices"):
        _trace(sim, arr)
    assert sg.nx_padded == shape[0]


def test_padded_layout_as_a_local_array_is_refused():
    """Its pad row would enter as a real cell (eps 0) and NaN the gradient."""
    sim = _model()
    padded = np.asarray(sim.shard_distributed_override(_host_inputs(sim)[0]))
    with pytest.raises(ValueError, match="local override must have the grid shape"):
        _trace(sim, padded)


def test_non_x_sharded_override_matches_the_local_array():
    """Any placement but P('x') on forward's mesh is a local whole-domain array."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    sim = _model()
    host = _host_inputs(sim)[0]
    _, sharding = sim.distributed_override_layout()
    y_split = jax.device_put(host, NamedSharding(sharding.mesh, P(None, "x")))
    want = np.asarray(_trace(sim, jnp.asarray(host)))
    got = np.asarray(_trace(sim, y_split))
    assert np.any(want) and got.tobytes() == want.tobytes()


def test_vmap_over_a_traced_mask_matches_single_calls():
    """A traced pec_mask_override (vmap) takes the traceable split in one process."""
    from rfx.nonuniform import position_to_index

    sim = _model()
    grid = sim._build_nonuniform_grid()
    source = position_to_index(grid, sim._ports[0].position)
    probe = position_to_index(grid, sim._probes[0].position)
    masks = np.zeros((2, *grid.shape), bool)
    # PEC between source and probe: the second member's trace must differ.
    masks[1, source[0] + 1:probe[0], source[1], source[2]] = True
    f = lambda m: sim.forward(pec_mask_override=m, distributed=True,
                              devices=jax.devices("cpu"), n_steps=8,
                              skip_preflight=True).time_series
    batched = np.asarray(jax.vmap(f)(jnp.asarray(masks)))
    assert np.any(batched[0] != batched[1]), "the mask must act on the trace"
    for member, mask in zip(batched, masks):
        _within(np.asarray(f(jnp.asarray(mask))), member)


@pytest.mark.slow
@pytest.mark.parametrize("kind,gate,error", [
    ("spacing", "closure", "captures"), ("halo", "parity", "ulp"),
    ("whole", "memory", "whole-domain buffer"),
])
def test_mutations_trip_gates(kind, gate, error):
    _child("mutation", kind, gate, error)


def _worker(address, rank, output, mode):
    rank = int(rank)
    if address != "reference":
        assert jax.process_count() == 2 and jax.local_device_count() == 1
    sim = _model()
    if mode == "refuse":
        for name in ("eps_override", "sigma_override", "pec_occupancy_override"):
            with pytest.raises(ValueError, match="requires an x-sharded global array"):
                sim.forward(**{name: jnp.asarray(_host_inputs(sim)[0])},
                            distributed=True, n_steps=8, skip_preflight=True)
        print(f"rank={rank} local overrides refused", flush=True)
    else:
        shape, sharding = sim.distributed_override_layout()
        # Deterministic design generated by global indices, independently on
        # each owning device. No rank ever constructs a full design on device.
        def design(offset, step):
            def local(index):
                lo, hi, _ = index[0].indices(shape[0])
                indices = np.arange(lo * shape[1] * shape[2], hi * shape[1] * shape[2])
                return (offset + (indices % 71) * step).astype(np.float32).reshape(
                    hi - lo, *shape[1:])
            return jax.make_array_from_callback(shape, sharding, local)

        designs = (design(1.2, .05), design(0., .0005), design(0., .002))
        f = lambda e, s, o: _trace(sim, e, s, o)
        trace = f(*designs)
        # Occupancy ghosts enter the seam cells' soft-PEC product, so only its
        # gradient sees a halo transpose that drops or double-counts a ghost.
        grads = jax.grad(lambda e, s, o: jnp.sum(f(e, s, o) ** 2),
                         argnums=(0, 1, 2))(*designs)
        assert trace.is_fully_replicated
        np.save(Path(output) / f"trace-{rank}.npy", np.asarray(trace))
        for name, grad, design_ in zip(("eps", "sigma", "occupancy"), grads, designs):
            assert grad.sharding == design_.sharding, name
            for shard in grad.addressable_shards:
                np.save(Path(output) / f"grad-{name}-{shard.device.id}.npy",
                        np.asarray(shard.data))
        print(f"rank={rank} plain+grad ok; trace replicated; gradient={grads[0].sharding}", flush=True)
    if address != "reference":
        jax.distributed.shutdown()


@pytest.fixture(scope="module")
def two_process_results(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("requires Linux: jax.distributed gRPC bind fails on macOS")
    output = tmp_path_factory.mktemp("nu-global")
    reference = output / "reference"
    reference.mkdir()
    _child("worker", "reference", 0, reference, "forward")
    logs = []
    for mode in ("forward", "refuse"):
        with socket.socket() as listener:
            listener.bind(("localhost", 0))
            address = f"localhost:{listener.getsockname()[1]}"
        processes, handles = [], []
        deadline = time.monotonic() + 180
        try:
            for rank in range(2):
                handle = (output / f"{mode}-{rank}.log").open("w")
                handles.append(handle)
                processes.append(subprocess.Popen(
                    [sys.executable, str(SCRIPT), "worker", address, str(rank), str(output), mode],
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
            logs.extend((output / f"{mode}-{rank}.log").read_text() for rank in range(2))
        assert all(p.returncode == 0 for p in processes), "\n".join(logs)
    print("\n".join(logs))
    return output, reference, logs


def test_two_process_plain_and_grad(two_process_results):
    output, reference, _ = two_process_results
    expected_trace = np.load(reference / "trace-0.npy")
    assert np.any(expected_trace), "non-vacuous multi-process trace"
    for rank in range(2):
        print("RESULT trace", rank, _within(expected_trace, np.load(output / f"trace-{rank}.npy")))
    # CPU device IDs in multi-process JAX encode process index (e.g. 0, 2048).
    def gathered(directory, name):
        return np.concatenate([np.load(p) for p in sorted(
            directory.glob(f"grad-{name}-*.npy"), key=lambda p: int(p.stem.split("-")[2]))])

    for name in ("eps", "sigma", "occupancy"):
        expected = gathered(reference, name)
        assert np.any(expected), f"non-vacuous multi-process {name} gradient"
        print("RESULT gradient", name, _within(expected, gathered(output, name)))


def test_two_process_local_override_refused(two_process_results):
    assert sum("local overrides refused" in log for log in two_process_results[2]) == 2


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "parity":
        _parity()
    elif mode == "transforms":
        _transforms()
    elif mode == "memory":
        _memory(*args)
    elif mode == "worker":
        _worker(*args)
    elif mode == "mutation":
        kind, gate, error = args
        _mutate(kind)
        try:
            if gate == "parity":
                _parity()
            elif gate == "memory":
                _memory("plain")
            else:
                records = _loop_program.__wrapped__(SimpleNamespace(param=("mixed", None, 0)))
                assert not records["captures"], f"captures: {records['captures']}"
        except AssertionError as failure:
            assert error in str(failure), str(failure)
            print(f"MUTATION_RED {kind}: {failure}; process-local mutation discarded on exit")
        else:
            raise AssertionError(f"mutation {kind} survived gate {gate}")
    else:
        raise AssertionError(mode)
