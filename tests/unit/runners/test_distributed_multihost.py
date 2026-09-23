"""Topology-independent scan arguments, local bit pins and global JAX arrays."""

from functools import partial
import hashlib
import inspect
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import jax
from jax.experimental import multihost_utils
import jaxlib
import numpy as np
import pytest

from rfx import Simulation
from rfx.runners import distributed_v2

# Included in pr-tests.yml's ubuntu fast suite: no slow/gpu marker. The
# subprocess test has its own shared deadline instead of a pytest plugin.
pytestmark = pytest.mark.distributed
_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
_BASELINE = {
    "pec": {
        "trace": "0ee5a57d7263d66d374b9911b2d735b40b17f4aca406465607d83ea07b33943d",
        "ex": "e56cc2a200d8856de5fe7b727314d2c6d63a0c6dd64c1a6c85f78a505b1ea396",
        "ey": "f3c49f0aeb1eeb7c232ddfa19f3e2d74feb585ca94ae39a4d9703e0716944f87",
        "ez": "f5c4b5e1c7a06799da43663e320270bd7ba4346bc9659dae166d367491dd745f",
        "hx": "f4899f531f073c935462d2e97459484e359b985f994d9b1e2bfe6125fe769e43",
        "hy": "316d373d2880b5ab9a1580fb23ab70489b292b408657a738333db19e8a27a09b",
        "hz": "5e6f7547699614e24e21233865eb46bc71efa2a5bb0554008137f40b4d2b020d",
    },
    "cpml": {
        "trace": "e8a9024ff2b5917bc90e7738b584dbf03aa952d14c992dc6eb27c66a90c5f46e",
        "ex": "4f01aa7ae12de326523bfedced694f2dd2f5f7a8740bda08f8652dd3cbc0c3e9",
        "ey": "f1d80073b0cafe60232da9e43ed76ab2ff65a299e25235b6259e555d710d2d20",
        "ez": "a0ea84740b70365f27f429372c0fe3940f79b22c6221002a60ea69f1a4d85b94",
        "hx": "32967a5083226d72ad4bca2dc9bc6f724b2bd55a24c72c009fa474df1d5a3909",
        "hy": "66026b56e7582d08ec7768e06d089dc418e2ace6d823fe026acc6dd0fd197a1c",
        "hz": "5ca0ad0dccdfe1912f4cdc144b82ebc823d1a3d3988bc63f15e8d6442125b3ae",
    },
}


def _build_box(boundary, *, dz_profile=None):
    sim = Simulation(
        freq_max=15e9, domain=(15e-3, 7e-3, 7e-3), dx=1e-3,
        boundary=boundary, cpml_layers=2 if boundary == "cpml" else 0,
        dz_profile=dz_profile,
    )
    # Preserve the legacy amplitude default used by the origin/main capture.
    sim.add_source((8e-3, 4e-3, 4e-3), "ez")
    sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
    sim.add_probe((10e-3, 4e-3, 4e-3), "ez")
    return sim


def _cpu_devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices (root conftest.py)")
    assert jax.process_count() == 1
    return devices[:2]


def _snapshot(result):
    return {
        "trace": np.asarray(result.time_series),
        **{name: np.asarray(getattr(result.state, name)) for name in _FIELDS},
    }


def _relative_error(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    peak = float(np.max(np.abs(expected)))
    assert peak > 0, "a zero fixture cannot bind the parity check"
    max_diff = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    return max_diff / peak, max_diff


def _scan_body_captures(monkeypatch, run):
    """Run normally, then inspect every scan body seen through the runner's lax."""
    bodies = []
    scan = distributed_v2.lax.scan

    def traced_scan(f, *args, **kwargs):
        bodies.append(f)
        return scan(f, *args, **kwargs)

    # Replace only the runner's namespace; leave JAX's internal lax uses alone.
    runner_lax = SimpleNamespace(**{**vars(distributed_v2.lax), "scan": traced_scan})
    with monkeypatch.context() as patch:
        patch.setattr(distributed_v2, "lax", runner_lax)
        run()
    assert bodies, "the run must trace a scan body for the capture check to bind"

    captured = []

    def walk(value, path, seen):
        # A jit argument may also satisfy isinstance(value, jax.Array).
        if isinstance(value, jax.core.Tracer):
            return
        if isinstance(value, jax.Array):
            captured.append((path, value.shape, str(value.sharding)))
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, FunctionType):
            for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    continue
                walk(child, f"{path}.{name}", seen)
        if isinstance(value, (tuple, list)):
            names = getattr(value, "_fields", range(len(value)))
            for name, child in zip(names, value):
                walk(child, f"{path}[{name}]", seen)
        if isinstance(value, partial):
            walk(value.func, f"{path}.func", seen)
            walk(value.args, f"{path}.args", seen)
            for name, child in (value.keywords or {}).items():
                walk(child, f"{path}.keywords[{name}]", seen)
        if hasattr(value, "__wrapped__"):
            walk(value.__wrapped__, f"{path}.__wrapped__", seen)

    for index, body in enumerate(bodies):
        walk(body, f"scan[{index}]", set())
    return captured


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_single_process_scan_uses_explicit_array_arguments(boundary, monkeypatch):
    devices = _cpu_devices()
    entries = []

    def traced_jit(f, *args, **kwargs):
        entry = Mock(wraps=jax.jit(f, *args, **kwargs))
        entries.append(entry)
        return entry

    runner_jax = SimpleNamespace(**{**vars(jax), "jit": traced_jit})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    gather = Mock(wraps=multihost_utils.process_allgather)
    monkeypatch.setattr(multihost_utils, "process_allgather", gather)
    captured = _scan_body_captures(
        monkeypatch, lambda: _build_box(boundary).run(n_steps=3, devices=devices),
    )
    assert not captured, f"single-process scan captured concrete arrays: {captured}"
    gather.assert_not_called()
    assert len(entries) == 1
    entries[0].assert_called_once()
    # carry, xs, materials, Debye, Lorentz, [CPML parameters], PEC mask.
    assert len(entries[0].call_args.args) == (7 if boundary == "cpml" else 6)
    assert entries[0].call_args.kwargs == {}


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_multi_process_scan_body_has_no_concrete_captures(boundary, monkeypatch):
    devices = _cpu_devices()
    runner_jax = SimpleNamespace(**{**vars(jax), "process_index": lambda: 1})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    assert distributed_v2._spans_other_processes(devices)
    gather = Mock(wraps=multihost_utils.process_allgather)
    monkeypatch.setattr(multihost_utils, "process_allgather", gather)
    captured = _scan_body_captures(
        monkeypatch, lambda: _build_box(boundary).run(n_steps=3, devices=devices),
    )
    assert not captured, f"multi-process scan captured concrete arrays: {captured}"
    gather.assert_called_once()
    assert gather.call_args.kwargs == {"tiled": True}


def test_multi_process_topology_is_decided_by_devices_not_process_count(monkeypatch):
    with monkeypatch.context() as patch:
        runner_jax = SimpleNamespace(**{**vars(jax), "process_index": lambda: 0})
        patch.setattr(distributed_v2, "jax", runner_jax)
        local = SimpleNamespace(process_index=0)
        foreign = SimpleNamespace(process_index=1)
        assert not distributed_v2._spans_other_processes([local, local])
        assert distributed_v2._spans_other_processes([local, foreign])

    devices = _cpu_devices()
    # Only process_count changes: the mesh still contains this process's devices.
    runner_jax = SimpleNamespace(**{**vars(jax), "process_count": lambda: 2})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    assert not distributed_v2._spans_other_processes(devices)
    gather = Mock(wraps=multihost_utils.process_allgather)
    monkeypatch.setattr(multihost_utils, "process_allgather", gather)
    _build_box("pec").run(n_steps=3, devices=devices)
    gather.assert_not_called()


def test_non_uniform_grid_is_refused_across_processes(monkeypatch):
    devices = _cpu_devices()
    runner_jax = SimpleNamespace(**{**vars(jax), "process_index": lambda: 1})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    assert distributed_v2._spans_other_processes(devices)
    # Profiles are constructor arguments; Simulation has no set_dz_profile API.
    sim = _build_box("pec", dz_profile=[1e-3] * 7)
    scan = Mock(wraps=distributed_v2.lax.scan)
    runner_lax = SimpleNamespace(**{**vars(distributed_v2.lax), "scan": scan})
    monkeypatch.setattr(distributed_v2, "lax", runner_lax)
    with pytest.raises(NotImplementedError, match="non-uniform grid") as exc:
        sim.run(n_steps=3, devices=devices)
    assert "more than one JAX process" in str(exc.value)
    scan.assert_not_called()


@pytest.mark.skipif(
    (sys.platform, platform.machine(), jax.__version__, jaxlib.__version__)
    != ("darwin", "arm64", "0.10.2", "0.10.2"),
    reason="SHA256 pins require the measurement toolchain: Darwin arm64 JAX/jaxlib 0.10.2",
)
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_single_process_path_is_bit_identical(boundary):
    """Pin the explicit-argument scan after PI decision A (2026-09-23).

    BEFORE: origin/main c2dbf922bc9230cd6fa4054a7b6d84d118768bde used the
    capturing-lambda path on one process. Decision A passes material, PEC-mask,
    dispersion and CPML arrays as explicit jit arguments on every topology.
    Root cause of the shift (reviewer, measured): XLA's algebraic simplifier
    rewrote arithmetic on the captured, uniform eps_r array of these vacuum
    boxes; with arguments it no longer knows the array is uniform. Disabling
    the algsimp pass makes old and new bit-identical; boxes whose eps_r varies
    anywhere are bit-identical without it. Which models move depends on the
    XLA version.

    Measured on Darwin arm64, JAX/jaxlib 0.10.2, two CPU devices, 20 steps:
    max per-value float32 ULP shift (trace / all six fields) was
    PEC 6 / 1,469,981,254 and CPML 3 / 1,459,299,516. These field distances use
    monotonic float32 bit ordering across signs, collapsing signed zero;
    near-zero cancellation values change sign. Restricting to same-sign
    values gives field maxima of 70,132,873 and 461,373,440 ULP, respectively.
    Max absolute field shifts divided by spacing(field peak) are only 0.5
    and 1.0 ULP; a few-ULP bound per field value would be incorrect.

    The pins are what this test computes: _build_box's fixture, 20 steps on two
    CPU devices, SHA256 over np.asarray(array).tobytes() of time_series and
    state.ex/ey/ez/hx/hy/hz. They remain toolchain scoped. CPML adds two cells
    per face: shape 20x12x12.
    """
    result = _build_box(boundary).run(n_steps=20, devices=_cpu_devices())
    arrays = _snapshot(result)
    assert np.max(np.abs(arrays["trace"])) > 0
    actual = {name: hashlib.sha256(array.tobytes()).hexdigest()
              for name, array in arrays.items()}
    assert actual == _BASELINE[boundary]


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_single_and_multi_process_topologies_are_bit_identical(boundary, monkeypatch):
    devices = _cpu_devices()
    expected = _snapshot(_build_box(boundary).run(n_steps=20, devices=devices))
    # Override only the runner's view of its own process index, so every local
    # device looks like another process's and the runner selects the
    # multi-process path. JAX itself and process_allgather still see one real
    # process, so this exercises the same explicit-argument scan under both
    # topology predicates without starting a distributed runtime or mocking
    # numerical operations. The real multi-host collective is tested below.
    runner_jax = SimpleNamespace(**{**vars(jax), "process_index": lambda: 1})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    assert distributed_v2._spans_other_processes(devices)
    actual = _snapshot(_build_box(boundary).run(n_steps=20, devices=devices))
    # Byte equality (np.array_equal would treat +0 and -0 as equal). This pins
    # that both topology predicates compile the same scan and that the
    # cross-process gather is a no-op at one real process; agreement across
    # real hosts is the Linux two-process test below.
    for name in ("trace", *_FIELDS):
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].tobytes() == expected[name].tobytes(), f"{boundary} {name}"


@pytest.mark.skipif(
    sys.platform != "linux",
    reason=("jax.distributed cannot start on macOS: gRPC bind [::] fails"
            if sys.platform == "darwin" else "requires Linux jax.distributed"),
)
def test_two_processes_return_identical_global_results(tmp_path, record_property):
    expected = _snapshot(_build_box("pec").run(n_steps=20, devices=_cpu_devices()))
    assert expected["ez"].shape == (16, 8, 8)
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        address = f"localhost:{listener.getsockname()[1]}"

    script = tmp_path / "worker.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nimport jax\nimport numpy as np\n"
        "address, rank, output = sys.argv[1:]\n"
        "rank = int(rank)\n"
        "jax.distributed.initialize(address, 2, rank, initialization_timeout=20)\n"
        "from rfx import Simulation\n"
        + inspect.getsource(_build_box)
        + "\nassert jax.process_count() == 2\n"
        "assert jax.device_count() == 2 and jax.local_device_count() == 1\n"
        "result = _build_box('pec').run(n_steps=20, devices=jax.devices())\n"
        "np.save(Path(output) / f'trace-{rank}.npy', np.asarray(result.time_series))\n"
        "np.save(Path(output) / f'ez-{rank}.npy', np.asarray(result.state.ez))\n"
        "jax.distributed.shutdown()\n"
    )
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
           "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "1",
           "MPLCONFIGDIR": str(tmp_path / "mpl")}
    # Proxy variables can make localhost rendezvous hang in gRPC.
    env = {key: value for key, value in env.items() if not key.lower().endswith("_proxy")}
    processes, logs = [], []
    timed_out = False
    # Two cold jax.distributed subprocesses on a small CI runner: the
    # coordinator rendezvous alone can take tens of seconds.
    deadline = time.monotonic() + 180
    try:
        for rank in range(2):
            log = (tmp_path / f"worker-{rank}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, str(script), address, str(rank), str(tmp_path)],
                cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            ))
        for process in processes:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait(timeout=5)
        for log in logs:
            log.close()
    output = "\n".join((tmp_path / f"worker-{rank}.log").read_text() for rank in range(2))
    assert not timed_out, f"two-process run exceeded 50 s:\n{output}"
    assert all(process.returncode == 0 for process in processes), output
    for name in ("trace", "ez"):
        first, second = (np.load(tmp_path / f"{name}-{rank}.npy") for rank in range(2))
        assert first.shape == second.shape == expected[name].shape
        assert first.dtype == second.dtype == expected[name].dtype
        assert first.tobytes() == second.tobytes(), f"processes disagree on {name}"
        exact = first.tobytes() == expected[name].tobytes()
        relative, max_diff = _relative_error(first, expected[name])
        record_property(f"{name}_single_process_exact", exact)
        record_property(f"{name}_relative_error", relative)
        record_property(f"{name}_max_abs_diff", max_diff)
        if not exact:
            # Real multi-host compilation may differ from a local mesh. Retain
            # the repo's distributed parity tolerance (test_distributed.py).
            print(f"{name}: max abs diff={max_diff:.9e}, relative={relative:.9e}")
            assert relative <= 1e-4, f"{name}: {max_diff=:.9e}, {relative=:.9e}"
