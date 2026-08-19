"""CPU-safe tests for backend classification and Metal reporting."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import pytest

from rfx import backends, diagnostics
from rfx import gpu as gpu_module
from rfx import simulation
from rfx.core.yee import init_materials
from rfx.gpu import DeviceInfo
from rfx.grid import Grid


def test_platform_normalization_accepts_names_and_devices():
    assert backends.normalize_platform("  METAL ") == "metal"
    assert backends.normalize_platform(SimpleNamespace(platform="CUDA")) == "cuda"
    assert backends.normalize_platform(None) == "unknown"
    assert backends.normalize_platform("") == "unknown"


def test_backend_capabilities_keep_metal_off_baked_pec_fast_path():
    for platform in ("gpu", "GPU", "cuda", "CUDA", "rocm", "ROCM"):
        assert backends.is_gpu_platform(platform)
        assert backends.is_accelerator_platform(platform)
        assert backends.supports_baked_pec_fast_path(platform)

    assert backends.is_gpu_platform("METAL")
    assert backends.is_accelerator_platform("METAL")
    assert not backends.supports_baked_pec_fast_path("METAL")
    assert backends.is_accelerator_platform("TPU")
    assert not backends.is_gpu_platform("TPU")
    assert not backends.supports_baked_pec_fast_path("TPU")
    assert not backends.is_accelerator_platform("cpu")


def test_current_backend_and_metal_detection_are_case_normalized(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "METAL")

    assert backends.current_backend() == "metal"
    assert backends.is_metal_backend()
    assert not backends.supports_baked_pec_fast_path()


def test_uniform_run_keeps_metal_off_baked_pec_fast_path(monkeypatch):
    """Lock the simulation wiring, not only the pure capability helper."""
    class _StepContextCaptured(RuntimeError):
        pass

    grid = Grid(
        freq_max=1e9,
        domain=(0.03, 0.03, 0.03),
        dx=0.01,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)

    monkeypatch.setattr(backends, "current_backend", lambda: "metal")

    def _unexpected_precompute(*_args, **_kwargs):
        pytest.fail("Metal selected the CUDA/ROCm baked-PEC fast path")

    def _capture_step_context(ctx):
        assert not ctx.use_fast_he
        assert ctx.fast_coeffs is None
        raise _StepContextCaptured

    monkeypatch.setattr(simulation, "precompute_coeffs", _unexpected_precompute)
    monkeypatch.setattr(simulation, "make_core_step", _capture_step_context)

    with pytest.raises(_StepContextCaptured):
        simulation.run(grid, materials, 1, boundary="pec")


def test_device_info_reports_metal_without_changing_tuple_arity(monkeypatch):
    metal_device = SimpleNamespace(
        id=0,
        device_kind="Apple M-series GPU",
        platform="METAL",
    )
    monkeypatch.setattr(gpu_module.jax, "devices", lambda: [metal_device])
    monkeypatch.setattr(gpu_module.jax, "default_backend", lambda: "METAL")

    info = gpu_module.device_info()

    assert isinstance(info, DeviceInfo)
    assert DeviceInfo._fields == ("backend", "gpu_available", "devices")
    assert len(info) == 3
    assert info.backend == "metal"
    assert info.gpu_available
    assert info.metal_available
    assert info.accelerator_available
    assert info.devices[0]["platform"] == "metal"


@pytest.mark.parametrize("x64_enabled", [False, True])
def test_diagnostics_report_metal_as_experimental_not_cpu_only(
    monkeypatch, x64_enabled,
):
    class _MetalDevice:
        platform = "METAL"

        def __str__(self) -> str:
            return "METAL:0"

    monkeypatch.setattr(jax, "devices", lambda: [_MetalDevice()])
    monkeypatch.setattr(jax.config, "read", lambda _name: x64_enabled)

    real_version_of = diagnostics._version_of

    def _version_of(dist: str) -> str:
        if dist == "jax-metal":
            return "0.test"
        return real_version_of(dist)

    monkeypatch.setattr(diagnostics, "_version_of", _version_of)
    report = diagnostics._Report()

    diagnostics._check_jax(report)
    output = str(report)

    assert "Metal backend present" in output
    assert "experimental" in output
    assert "jax-metal 0.test" in output
    assert "CPU-only" not in output
    assert "real float32" in output
    assert "use the CPU backend" in output
    assert "JAX_ENABLE_X64=1" not in output
    assert "float64 available" not in output


def test_waveguide_config_default_aperture_is_host_numpy():
    import numpy as np

    from rfx.sources.waveguide_port import WaveguidePortConfig

    aperture_default = WaveguidePortConfig.__new__.__defaults__[0]
    assert isinstance(aperture_default, np.ndarray)
    assert aperture_default.shape == (0, 0)
    assert aperture_default.dtype == np.float32
