"""Topology selection preserves one-process bits and permits global JAX arrays."""

import hashlib
import inspect
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import jax
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
        "trace": "cf0259c5ca4f93a0cc9c1ab3399c8b53ccc96fc885d445e6f19fb84cbdf93306",
        "ex": "9d638c756bbbfb57503ec0aea5b80d29db888b5b738bf9000561df097cac3f36",
        "ey": "0616b19095a22a26979e5ed17c289d398dcda4e710944f47fb6c2b6cd33737d2",
        "ez": "647bb085bede68881b625cb77e94a7f8618c7e07f41f34ec2b8b0776b876bc03",
        "hx": "e972a6efa32b43b3a84f41a65c4c9397b004e4930a545f0b84d79cdc5a50388d",
        "hy": "9a8e6e56b763bb07d7235e80a15687d05f670c68160c9fa0d73ab473b8ec26d2",
        "hz": "3b91cd16b2687b94d3bafec9bb2a644a445c4d0b7bce58f545dcbf202aa271d3",
    },
    "cpml": {
        "trace": "1e356ea20d2ada2c5396d4d0e6d3817e729f6da89148ee9ff94bcd111da3cf50",
        "ex": "a29f88f21487d8a372477021925b25d9fc035e87ef08b00c444c7400f7a6e758",
        "ey": "c667e8ac30db1e99a05c8b916e4ec2398699e5acdc0518776f7c45d0f6c5fe91",
        "ez": "c2b468b89d12b5fa9550043d65c941d3d814d4a365169f21c42da5c0869580ef",
        "hx": "d2a994357bccccf2d42e084498982307df4fbabaa60fc49278c7fa4d8b076a11",
        "hy": "5894b9874ed2f87c3a023489d67451d722cbfb2dbe650fa961236e22cde69654",
        "hz": "262f697fce6f33d1bb87c232328fd267106a0bcaaf2baf330b050078b8a20839",
    },
}


def _build_box(boundary):
    sim = Simulation(
        freq_max=15e9, domain=(15e-3, 7e-3, 7e-3), dx=1e-3,
        boundary=boundary, cpml_layers=2 if boundary == "cpml" else 0,
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


@pytest.mark.skipif(
    (sys.platform, platform.machine(), jax.__version__, jaxlib.__version__)
    != ("darwin", "arm64", "0.10.2", "0.10.2"),
    reason="origin/main SHA256 pins require the capture toolchain: Darwin arm64 JAX/jaxlib 0.10.2",
)
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_single_process_path_is_bit_identical(boundary):
    """The multi-process path changes constant folding; the single-process path must not move.

    BEFORE: origin/main e7f7e02704fd46ea7e21f127b19fc81cb66d6148, scratch
    worktree /tmp/rfx-mh/origin-main, macOS arm64, Python 3.11.2, JAX/jaxlib
    0.10.2, two CPU devices. Command (cwd = that scratch worktree)::

        PYTHONPATH=$PWD XLA_FLAGS=--xla_force_host_platform_device_count=2 /Users/byungkwankim/Documents/rfx/.venv/bin/python /tmp/rfx-mh/measure.py /tmp/rfx-mh/before

    measure.py ran _build_box's exact fixture for 20 steps on jax.devices()[:2]
    and printed hashlib.sha256(np.asarray(array).tobytes()).hexdigest() for
    time_series and state.ex/ey/ez/hx/hy/hz. The pins are host/toolchain scoped,
    as in tests/locks/test_runner_split_bit_identity.py; parity below runs on
    other CPU toolchains too. CPML adds two cells per face: shape 20x12x12.
    """
    result = _build_box(boundary).run(n_steps=20, devices=_cpu_devices())
    arrays = _snapshot(result)
    assert np.max(np.abs(arrays["trace"])) > 0
    actual = {name: hashlib.sha256(array.tobytes()).hexdigest()
              for name, array in arrays.items()}
    assert actual == _BASELINE[boundary]


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_multi_process_variant_matches_single_within_parity(
    boundary, monkeypatch, record_property,
):
    devices = _cpu_devices()
    expected = _snapshot(_build_box(boundary).run(n_steps=20, devices=devices))
    # Override only the runner's topology query. JAX itself and process_allgather
    # still see one real process, so this exercises both explicit-argument scan
    # bodies and jit entries without starting a distributed runtime or mocking
    # their numerical operations. The real multi-host collective is tested below.
    runner_jax = SimpleNamespace(**{**vars(jax), "process_count": lambda: 2})
    monkeypatch.setattr(distributed_v2, "jax", runner_jax)
    actual = _snapshot(_build_box(boundary).run(n_steps=20, devices=devices))
    # Normalize the six-component field state by its common peak. In this ez
    # fixture hz is cancellation noise (~1e-12), so its own peak is not a useful
    # denominator for a float32 lane-parity check.
    for name, keys in (("trace", ("trace",)), ("fields", _FIELDS)):
        got = np.concatenate([actual[key].ravel() for key in keys])
        want = np.concatenate([expected[key].ravel() for key in keys])
        relative, max_diff = _relative_error(got, want)
        record_property(f"{boundary}_{name}_relative_error", relative)
        record_property(f"{boundary}_{name}_max_abs_diff", max_diff)
        assert relative <= 1e-4, f"{boundary} {name}: {relative=:.9e}, {max_diff=:.9e}"


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
    deadline = time.monotonic() + 50
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
            # Explicit jit arguments change float32 constant folding. Retain
            # the repo's distributed parity tolerance (test_distributed.py).
            print(f"{name}: max abs diff={max_diff:.9e}, relative={relative:.9e}")
            assert relative <= 1e-4, f"{name}: {max_diff=:.9e}, {relative=:.9e}"
