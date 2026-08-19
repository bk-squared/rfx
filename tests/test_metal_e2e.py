"""Opt-in smoke gates that must execute on an actual Apple Metal device."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from rfx import extract_s_matrix_wire, run_adi_2d, run_nonuniform
from rfx.api import Simulation
from rfx.backends import normalize_platform
from rfx.geometry.csg import Box
from rfx.gpu import device_info
from rfx.sources.sources import GaussianPulse
from rfx.vmap_sweep import vmap_material_sweep


if normalize_platform(jax.default_backend()) != "metal":
    pytest.skip("requires the Apple JAX Metal backend", allow_module_level=True)


pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def _assert_real_finite_signal(result, n_steps: int) -> None:
    signal = np.asarray(jax.device_get(result.time_series))
    assert signal.shape == (n_steps, 1)
    assert signal.dtype == np.float32
    assert np.isrealobj(signal)
    assert np.isfinite(signal).all()
    assert float(np.max(np.abs(signal))) > 0.0


def test_metal_device_is_reported_as_an_accelerating_gpu():
    info = device_info()

    assert info.backend == "metal"
    assert info.gpu_available
    assert info.metal_available
    assert info.accelerator_available
    assert all(device["platform"] == "metal" for device in info.devices)


def test_metal_real_float32_pec_forward_smoke():
    n_steps = 32
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        boundary="pec",
    )
    sim.add_source(
        (0.01, 0.01, 0.01),
        "ez",
        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
        amplitude_kind="field",
    )
    sim.add_probe((0.012, 0.01, 0.01), "ez")

    result = sim.run(
        n_steps=n_steps,
        compute_s_params=False,
        skip_preflight=True,
    )

    _assert_real_finite_signal(result, n_steps)


def test_metal_real_float32_cpml_and_dielectric_smoke():
    n_steps = 40
    sim = Simulation(
        freq_max=5e9,
        domain=(0.024, 0.024, 0.024),
        dx=0.002,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_material("dielectric", eps_r=2.5)
    sim.add(
        Box((0.011, 0.007, 0.007), (0.015, 0.017, 0.017)),
        material="dielectric",
    )
    sim.add_source(
        (0.007, 0.012, 0.012),
        "ez",
        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
        amplitude_kind="field",
    )
    sim.add_probe((0.018, 0.012, 0.012), "ez")

    result = sim.run(
        n_steps=n_steps,
        compute_s_params=False,
        skip_preflight=True,
    )

    _assert_real_finite_signal(result, n_steps)


def test_metal_lumped_port_is_time_domain_only():
    n_steps = 32
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        boundary="pec",
    )
    sim.add_port((0.01, 0.01, 0.01), "ez")
    sim.add_probe((0.012, 0.01, 0.01), "ez")

    with pytest.raises(NotImplementedError, match="compute_s_params=False"):
        sim.run(n_steps=2, skip_preflight=True)

    result = sim.run(
        n_steps=n_steps,
        compute_s_params=False,
        skip_preflight=True,
    )

    _assert_real_finite_signal(result, n_steps)


def test_metal_guards_unsupported_complex_and_reverse_mode_paths():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        boundary="pec",
    )
    sim.add_source(
        (0.01, 0.01, 0.01), "ez", amplitude_kind="field",
    )
    sim.add_probe((0.012, 0.01, 0.01), "ez")
    sim.add_dft_plane_probe(
        axis="x",
        coordinate=0.01,
        component="ez",
        freqs=np.asarray([2e9], dtype=np.float32),
    )

    with pytest.raises(NotImplementedError, match="DFT plane probes"):
        sim.run(n_steps=2, compute_s_params=False, skip_preflight=True)

    safe_sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        boundary="pec",
    )
    safe_sim.add_source(
        (0.01, 0.01, 0.01), "ez", amplitude_kind="field",
    )
    safe_sim.add_probe((0.012, 0.01, 0.01), "ez")
    with pytest.raises(NotImplementedError, match="reverse-mode AD"):
        safe_sim.forward(n_steps=2, skip_preflight=True)


def test_metal_advanced_runner_front_doors_fail_fast():
    with pytest.raises(NotImplementedError, match="Non-uniform"):
        run_nonuniform(None, None, 1)
    with pytest.raises(NotImplementedError, match="ADI solver"):
        run_adi_2d(None, None, None, None, None, 0.0, 0.0, 0.0, 1)
    with pytest.raises(NotImplementedError, match="Wire-port S-parameter"):
        extract_s_matrix_wire(None, None, [], None)

    sim = Simulation(
        freq_max=5e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.002,
        boundary="pec",
    )
    with pytest.raises(NotImplementedError, match="vmap/batched"):
        vmap_material_sweep(sim, "eps_r", [1.0])
