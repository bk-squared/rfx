"""CPU contract for tracker 1598, including its compiled-program premise."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.stepping import rank
from tests._multi_device_scene import finite, scene


@pytest.fixture
def devices():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("two-host-device rows are exercised by the fresh subprocess")
    return devices[:2]


def test_two_host_device_subprocess_if_needed(tmp_path):
    if len(jax.devices("cpu")) >= 2:
        return
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu",
               PYTHONDONTWRITEBYTECODE="1", JAX_ENABLE_COMPILATION_CACHE="false",
               XLA_FLAGS="--xla_force_host_platform_device_count=2")
    result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__)),
                             "-o", "addopts=", "-rA", "--basetemp", str(tmp_path / "child")],
                            cwd=root, env=env, text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def pretend_gpu(monkeypatch):
    monkeypatch.setattr(rank, "_is_gpu_mesh", lambda mesh: True)
    monkeypatch.setenv("XLA_FLAGS", "")


def assert_refused(call):
    with pytest.raises(NotImplementedError) as exc:
        call()
    assert "1598" in str(exc.value)
    assert "xla_gpu_enable_command_buffer" in str(exc.value)
    assert str(exc.value) == rank.OUTER_JIT_REFUSAL


@pytest.mark.parametrize("case", ["forward", "grad", "no_argument", "uniform_run"])
def test_T1_outer_jit_refused(devices, pretend_gpu, case):
    eps, forward, objective, run = scene(devices, uniform=case == "uniform_run")
    if case == "uniform_run":
        # Measured reachable on CPU: a later tracing error must not hide a regression.
        assert_refused(lambda: jax.jit(run)())
    elif case == "no_argument":
        assert_refused(lambda: jax.jit(lambda: forward(eps))())
    else:
        call = forward if case == "forward" else jax.grad(objective)
        assert_refused(lambda: jax.jit(call)(eps))


@pytest.mark.parametrize("case", ["forward", "grad", "value_and_grad", "vmap",
                                  "checkpoint", "warmup", "uniform_run"])
def test_T2_unjitted_runs(devices, pretend_gpu, case):
    options = {"checkpoint_every": 2} if case == "checkpoint" else {"n_warmup": 1} if case == "warmup" else {}
    eps, forward, objective, run = scene(devices, uniform=case == "uniform_run", **options)
    if case == "uniform_run":
        result = run()
    elif case == "forward":
        result = forward(eps)
    elif case == "vmap":
        result = jax.vmap(forward)(jnp.stack((eps, eps * 1.01)))
    else:
        transform = jax.value_and_grad if case == "value_and_grad" else jax.grad
        result = transform(objective)(eps)
    finite(result)


@pytest.mark.parametrize("flags", ["--xla_gpu_enable_command_buffer=",
    "--xla_force_host_platform_device_count=2 --xla_gpu_enable_command_buffer= --xla_cpu_enable_fast_math=false",
    "--xla_gpu_enable_command_buffer=FUSION"])
@pytest.mark.parametrize("gradient", [False, True], ids=["forward", "grad"])
def test_T3_flag_escape(devices, pretend_gpu, monkeypatch, flags, gradient):
    monkeypatch.setenv("XLA_FLAGS", flags)
    eps, forward, objective, _ = scene(devices)
    call = jax.jit(jax.grad(objective) if gradient else forward)
    if flags.endswith("FUSION"):
        assert_refused(lambda: call(eps))
    else:
        finite(call(eps))


@pytest.mark.parametrize("gradient", [False, True], ids=["forward", "grad"])
def test_T4_real_cpu_runs(devices, monkeypatch, gradient):
    monkeypatch.setenv("XLA_FLAGS", "")
    assert all(d.platform == "cpu" for d in devices)
    eps, forward, objective, _ = scene(devices)
    finite(jax.jit(jax.grad(objective) if gradient else forward)(eps))


def test_T5_compiled_partition_id_premise(devices, monkeypatch):
    """if this goes red because the outer program has no partition-id any more, the refusal of tracker 1598 can be reconsidered"""
    import rfx.runners.distributed_nu as runner
    eps, forward, _, _ = scene(devices)
    assert jax.jit(forward).lower(eps).compile().as_text().count("partition-id(") >= 1
    captures = []
    original_jit = jax.jit

    def capturing_jit(fn, *args, **kwargs):
        compiled = original_jit(fn, *args, **kwargs)
        if getattr(fn, "__name__", "") != "run_fn":
            return compiled

        def run(*a, **k):
            captures.append(compiled.lower(*a, **k).compile().as_text())
            return compiled(*a, **k)
        return run

    monkeypatch.setattr(runner, "jax", SimpleNamespace(**{**vars(jax), "jit": capturing_jit}))
    finite(forward(eps))
    assert captures
    assert all("partition-id(" not in text for text in captures)


@pytest.mark.parametrize("flags,on", [("", True), ("--xla_gpu_enable_command_buffer=", False),
    ("--xla_gpu_enable_command_buffer= --other=1", False),
    ("--xla_gpu_enable_command_buffer=FUSION", True),
    ("prefix--xla_gpu_enable_command_buffer=", True)])
def test_T6_flag_parser(monkeypatch, flags, on):
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert rank._command_buffers_on() is on


@pytest.mark.parametrize("platforms,gpu", [(["gpu", "gpu"], True), (["cpu", "cpu"], False),
    (["cpu", "gpu"], True), (["cpu", None], True)])
def test_T6_gpu_predicate(platforms, gpu):
    devices = [SimpleNamespace(**({} if p is None else {"platform": p})) for p in platforms]
    mesh = SimpleNamespace(devices=np.asarray(devices, dtype=object))
    assert rank._is_gpu_mesh(mesh) is gpu


def test_T6_probe_uncertainty_refuses(monkeypatch):
    def broken(*args):
        raise RuntimeError("probe unavailable")
    monkeypatch.setattr(rank, "jax", SimpleNamespace(lax=SimpleNamespace(add=broken)))
    assert rank._staged_by_enclosing_program() is True


def test_T6_single_device_never_refused(devices, pretend_gpu):
    mesh = jax.sharding.Mesh(np.asarray(devices[:1]), ("x",))
    finite(jax.jit(lambda: rank.mesh_ranks(mesh))())
