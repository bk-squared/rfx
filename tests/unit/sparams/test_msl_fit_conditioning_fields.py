"""Physical conditioning anchor and public MSL diagnostic dataflow."""
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from rfx import Simulation
from rfx.errors import AccuracyWarning, RfxRefusal
from rfx.grid import Grid
from rfx.probes import msl_wave_decomp as decomp
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n
from rfx.sparams._msl_stages import _check_msl_extraction
from tests.unit.sparams.test_msl_spatial_driver import (
    FREQS, _build_case, _install_manufactured_backends, _knob,
)


@pytest.mark.parametrize("dx", [1e-4, 2e-4])
@pytest.mark.parametrize("phi", [0.01, 0.03, 0.3])
def test_fabricated_forward_plus_one_percent_backward(dx, phi, capsys):
    """The expected small-phase constant is derived from the five-probe residual."""
    u = np.arange(5) / 4
    residual = np.array([1 / 8, -1 / 16, -1 / 8, -1 / 16, 1 / 8])
    np.testing.assert_allclose(np.linalg.norm(residual), np.sqrt(14) / 16)
    np.testing.assert_allclose(residual @ np.column_stack((np.ones(5), u)), 0)
    constant = np.sqrt(5) / np.linalg.norm(residual)
    length = 8 * dx
    beta = phi / length
    x = (29 + 8 * u) * dx
    v = np.exp(-1j * phi * u) + 0.01j * np.exp(1j * phi * u)
    with enable_x64(False):
        result = decomp.extract_msl_nprobe(v, x, np.array([(1 - 0.01j) / 50]), beta)
    fitted_beta = np.real(np.asarray(result["beta"]))
    np.testing.assert_allclose(result["fit_span_phase"], fitted_beta * length, rtol=1e-5)
    expected = constant / phi**2
    if phi < 0.3:
        np.testing.assert_allclose(result["fit_amplification"], expected, rtol=0.01)
    with capsys.disabled():
        print(f"\nconditioning dx={dx:g} phi={phi:g}: "
              f"span={float(result['fit_span_phase'][0]):.9g}, "
              f"A={float(result['fit_amplification'][0]):.9g}, anchor={expected:.9g}")


def test_exact_projected_derivative_on_unequal_positions():
    """An independent complex128 projection checks the exact, non-series definition."""
    x = np.array([0, 0.3, 0.9, 1.6, 2.0]) * 1e-3
    beta = np.array([200.0, 800.0])
    alpha = np.array([0.7 + 0.1j, 1.0 - 0.4j])
    gamma = np.array([0.2 - 0.3j, -0.6 + 0.1j])
    expected = []
    for b, a, g in zip(beta, alpha, gamma):
        forward, backward = np.exp(-1j * b * x), np.exp(1j * b * x)
        waves = np.column_stack((forward, backward))
        column = -1j * x * a * forward + 1j * x * g * backward
        perpendicular = column - waves @ np.linalg.lstsq(waves, column, rcond=None)[0]
        v_fit = a * forward + g * backward
        expected.append(np.linalg.norm(v_fit) / (b * np.linalg.norm(perpendicular)))
    with enable_x64(False):
        span, amplification = decomp._msl_fit_conditioning(
            *(jnp.asarray(value) for value in (x, beta, alpha, gamma)))
    np.testing.assert_allclose(span, beta * (x[-1] - x[0]), rtol=1e-6)
    np.testing.assert_allclose(amplification, expected, rtol=2e-5)


def test_fabricated_record_amplitude_invariance():
    """A relative voltage departure has dimensionless, amplitude-invariant gain."""
    x = np.arange(5) * 2e-4
    beta = 0.3 / (x[-1] - x[0])
    alpha, gamma = 1.0, 0.01j
    forward, backward = np.exp(-1j * beta * x), np.exp(1j * beta * x)
    v_fit = alpha * forward + gamma * backward
    waves = np.column_stack((forward, backward))
    column = -1j * x * alpha * forward + 1j * x * gamma * backward
    perpendicular = column - waves @ np.linalg.lstsq(waves, column, rcond=None)[0]
    expected = np.linalg.norm(v_fit) / (beta * np.linalg.norm(perpendicular))
    amplifications = []
    with enable_x64(False):
        for scale in (1.0, 1e-3, 1e3):
            _, amplification = decomp._msl_fit_conditioning(
                jnp.asarray(x), jnp.array([beta]),
                jnp.array([scale * alpha]), jnp.array([scale * gamma]))
            amplifications.append(np.asarray(amplification))
    np.testing.assert_allclose(amplifications, expected, rtol=1e-5)
    np.testing.assert_allclose(amplifications, np.broadcast_to(amplifications[0], (3, 1)),
                               rtol=1e-5)


def test_negative_x_probe_span_matches_mirrored_record():
    """The production -x ladder has a positive phase span and mirrored gain."""
    dx = 1e-4
    grid = Grid(1e9, (100 * dx, 8 * dx, 8 * dx), dx=dx, cpml_layers=0)
    port = MSLPort(70 * dx, dx, 2 * dx, 0.0, dx, "-x", 50.0)
    x = np.asarray(msl_probe_x_coords_n(
        grid, port, 5, n_offset_cells=29, n_spacing_cells=2))
    assert np.all(np.diff(x) < 0)
    x = x - x[0]
    length = 8 * dx
    beta = 0.3 / length
    alpha, gamma = 1.0 + 0.2j, 0.01j
    forward, backward = np.exp(-1j * beta * x), np.exp(1j * beta * x)
    v_fit = alpha * forward + gamma * backward
    waves = np.column_stack((forward, backward))
    column = -1j * x * alpha * forward + 1j * x * gamma * backward
    perpendicular = column - waves @ np.linalg.lstsq(waves, column, rcond=None)[0]
    expected = np.linalg.norm(v_fit) / (beta * np.linalg.norm(perpendicular))
    with enable_x64(False):
        span, amplification = decomp._msl_fit_conditioning(
            jnp.asarray(x), jnp.array([beta]), jnp.array([alpha]), jnp.array([gamma]))
        mirrored_span, mirrored_amplification = decomp._msl_fit_conditioning(
            jnp.asarray(-x), jnp.array([beta]),
            jnp.array([np.conj(alpha)]), jnp.array([np.conj(gamma)]))
    assert np.all(np.asarray(span) > 0)
    np.testing.assert_allclose(span, beta * length, rtol=1e-6)
    np.testing.assert_allclose(mirrored_span, beta * length, rtol=1e-6)
    np.testing.assert_allclose(amplification, expected, rtol=1e-5)
    np.testing.assert_allclose(amplification, mirrored_amplification, rtol=1e-5)


def test_extractor_fields_are_detached_under_jit_and_vmap():
    x = jnp.linspace(0, 1e-3, 5)
    beta = jnp.array([30.0, 300.0])
    v = jnp.exp(-1j * beta[:, None] * x) + 0.01j * jnp.exp(1j * beta[:, None] * x)

    def fields(scale):
        result = decomp.extract_msl_nprobe(v * scale, x, jnp.ones(2), beta)
        return jnp.stack((result["fit_span_phase"], result["fit_amplification"]))

    values, tangent = jax.jvp(jax.jit(fields), (jnp.float32(1),), (jnp.float32(1),))
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(tangent, np.zeros((2, 2)))
    batched = jax.jit(jax.vmap(fields))(jnp.array([1.0, 2.0]))
    assert batched.shape == (2, 2, 2)
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched[0], batched[1], rtol=1e-5)


def test_s_and_gradient_are_bit_identical_with_zero_fields(monkeypatch, tmp_path, capsys):
    """Reuse the one-port, one-step manufactured spatial-driver scene with eps_override."""
    start = perf_counter()
    sim, grid, profiles = _build_case("+x", False)
    calls, _ = _install_manufactured_backends(monkeypatch, sim, grid, profiles, "+x", False)
    scale = jnp.float32(1.25)

    def compute(value, **kwargs):
        return sim.compute_msl_s_matrix(
            freqs=FREQS, n_steps=1, num_periods=1, enforce_passivity=False,
            eps_override=_knob(grid, value), **kwargs)

    seen = []

    def objective(value):
        result = compute(value)
        seen.append((result.beta_railed, result.fit_span_phase, result.fit_amplification))
        return jnp.mean(jnp.real(result.S[0, 0]))

    dump = tmp_path / "conditioning.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        baseline = compute(scale, raw_3probe_dump_path=str(dump))
        gradient = jax.grad(objective)(scale)
        assert baseline.fit_span_phase.shape == baseline.beta_railed.shape == (1, len(FREQS))
        assert baseline.fit_amplification.shape == baseline.beta_railed.shape
        assert np.all(baseline.fit_span_phase > 0)
        assert np.all(baseline.fit_amplification > 0)
        with np.load(dump) as saved:
            for field in ("fit_span_phase", "fit_amplification"):
                np.testing.assert_array_equal(saved[field], getattr(baseline, field))

        def zeros(x, beta, alpha, gamma):
            return jnp.zeros_like(jnp.real(beta)), jnp.zeros_like(jnp.real(beta))

        monkeypatch.setattr(decomp, "_msl_fit_conditioning", zeros)
        zeroed = compute(scale)
        zeroed_gradient = jax.grad(objective)(scale)

    assert calls == ["forward"] * 4
    assert np.isfinite(gradient) and abs(float(gradient)) > 1e-4
    for a, b in ((baseline.S, zeroed.S), (gradient, zeroed_gradient)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(zeroed.fit_span_phase, np.zeros((1, len(FREQS))))
    np.testing.assert_array_equal(zeroed.fit_amplification, np.zeros((1, len(FREQS))))
    for railed, phase, amplification in seen:
        assert (railed is None) == (phase is None) == (amplification is None)
    with capsys.disabled():
        print(f"\nfixture=test_msl_spatial_driver._build_case('+x', False), "
              f"manufactured forward, n_steps=1: four calls wall={perf_counter() - start:.3f}s; "
              f"S bytes identical; grad={float(gradient):.9g}, bytes identical")


@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_mixed_result_carries_own_drive_fields(monkeypatch, direction):
    from tests.unit.sparams.test_msl_mixed_spatial_collocation import (
        FREQS as mixed_freqs, _case, _backend,
    )

    sim, target = _case(direction)
    monkeypatch.setattr(Simulation, "_forward_from_materials",
                        _backend(target, direction, {"affine": False}, []))
    original = decomp.extract_msl_nprobe
    fitted = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        fitted.append(result)
        return result

    monkeypatch.setattr(decomp, "extract_msl_nprobe", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result, diagnostics = sim.compute_mixed_s_matrix(
            freqs=mixed_freqs, n_steps=1, num_periods=1, skip_preflight=True,
            magnitude_channel="wave", enforce_passivity=False,
            return_diagnostics=True)
    assert len(fitted) == 1
    for field in ("beta_railed", "fit_span_phase", "fit_amplification"):
        assert getattr(result, field).shape == (1, len(mixed_freqs))
        np.testing.assert_array_equal(getattr(result, field)[0], fitted[0][field])
    for field in ("fit_span_phase", "fit_amplification"):
        np.testing.assert_array_equal(diagnostics[field], getattr(result, field))


def test_clearance_satisfied_z0_warning_points_to_conditioning():
    with pytest.warns(AccuracyWarning) as caught:
        _check_msl_extraction(
            RfxRefusal, jnp.zeros((1, 1, 1)), (), lambda error, _: error,
            [SimpleNamespace(name="p1")], np.array([0.5e9]), jax, 1,
            [SimpleNamespace(status="satisfied")], jnp.array([[[70.0]]]),
            None, False, [50.0])
    message = str(caught[0].message)
    assert "probe clearance is satisfied" in message
    assert "refine the mesh" not in message
    assert ("read fit_span_phase and fit_amplification at that bin. "
            "A finer mesh at fixed cell counts shortens the probe span "
            "and does not make the fitted Z0 more accurate. Otherwise "
            "reconcile the rasterized board with the declared one "
            "(sim.fidelity_report()) before quoting Z0.") in message
