"""The material fit replays E and H records on the measurement time base."""

import sys
import time

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.differentiable_material_fit import differentiable_material_fit
from rfx.grid import Grid
from rfx.material_fit import DebyeFitResult
from rfx.materials.debye import DebyePole
from rfx.measurement import dft


@pytest.fixture(scope="module")
def fitted_record():
    started = time.perf_counter()
    freqs = np.array([9e9, 10e9, 11e9], dtype=np.float64)
    captured = []
    original_transform = dft.transform

    def spy(records, frequencies, dt, kind="E", **kwargs):
        def store(values):
            captured.append((np.array(values, dtype=np.float64), float(dt), kind))

        jax.debug.callback(store, records, ordered=True)
        return original_transform(records, frequencies, dt, kind, **kwargs)

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(
            freq_max=12e9, domain=(0.006,) * 3, dx=0.001,
            boundary="pec", cpml_layers=0,
        )
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.0,) * 3, (0.006,) * 3), material="dut")
        sim.add_port(
            (0.002, 0.003, 0.003), "ez",
            waveform=GaussianPulse(f0=10e9, bandwidth=0.8),
        )
        sim.add_probe((0.003, 0.003, 0.003), "ez")
        sim.add_probe((0.003, 0.003, 0.003), "hy")
        return sim

    truth = DebyeFitResult(
        eps_inf=1.5, poles=[DebyePole(delta_eps=0.5, tau=10e-12)], fit_error=0.0,
    )
    kwargs = dict(
        n_debye_poles=1, n_iterations=1, initial_guess=truth,
        learning_rate=0.0, verbose=False, normalization="per_probe_max",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Grid, "num_timesteps", lambda self, num_periods=20.0: 128)
        patch.setattr(dft, "transform", spy)
        first = differentiable_material_fit(
            factory, np.zeros((2, 2, 3), dtype=np.complex128), freqs, **kwargs,
        )
        jax.effects_barrier()
        assert len(captured) == 2, "row (a): the fit must call the measurement DFT per probe"
        expected = np.zeros((2, 2, 3), dtype=np.complex128)
        for i, offset in enumerate((1.0, 0.5)):
            records, dt, _ = captured[i]
            times = (np.arange(len(records), dtype=np.float64) + offset) * np.float64(dt)
            spectrum = np.sum(
                records[:, None] * np.exp(-2j * np.pi * times[:, None] * freqs[None, :]),
                axis=0, dtype=np.complex128,
            ) * np.float64(dt)
            peak = np.max(np.abs(spectrum))
            assert peak > 1e-30
            expected[i, i] = spectrum / max(peak, 1e-30)
        planted = differentiable_material_fit(factory, expected, freqs, **kwargs)
        jax.effects_barrier()
    wall_time = time.perf_counter() - started
    print(f"fixture: 6 cells per side, 128 steps, E=ez H=hy; wall={wall_time:.6f}s", file=sys.stderr)
    print(f"callback calls={len(captured)}; dt={captured[0][1]:.17g}s", file=sys.stderr)
    return first, planted, expected, freqs, captured


def test_a_same_record_two_spectra(fitted_record):
    first, _, expected, freqs, captured = fitted_record
    assert len(captured) == 4
    assert not np.array_equal(captured[0][0], captured[1][0])
    for i in range(2):
        peak = np.max(np.abs(expected[i, i]))
        assert np.all(np.abs(expected[i, i]) >= 0.1 * peak)
        error = np.abs(first.final_s_params[i, i] - expected[i, i])
        assert np.all(error <= 1e-4 * peak), (i, error, 1e-4 * peak)
        np.testing.assert_array_equal(captured[i][0], captured[i + 2][0])
    assert 2.0 * np.pi * freqs[-1] * captured[0][1] >= 0.05
    np.testing.assert_allclose(first.final_s_params[0, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(first.final_s_params[1, 0], 0.0, atol=0.0)


def test_b_planted_target_phase(fitted_record):
    _, planted, target, freqs, captured = fitted_record
    for i in range(2):
        phase = np.angle(planted.final_s_params[i, i]) - np.angle(target[i, i])
        residual = np.arctan2(np.sin(phase), np.cos(phase))
        stamp = 2.0 * np.pi * freqs * np.float64(captured[i][1])
        slope = np.linalg.lstsq(
            np.column_stack((stamp, np.ones_like(stamp))), residual, rcond=None,
        )[0][0]
        print(f"probe {i}: residual={residual.tolist()}, slope={slope:.9g}", file=sys.stderr)
        assert np.all(np.abs(residual) < 1e-3)
        assert abs(slope) < 1e-2
