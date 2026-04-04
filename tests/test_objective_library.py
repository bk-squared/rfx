"""Tests for the pre-built objective function library.

Validates:
1. Each objective returns a scalar.
2. Objectives are JAX-differentiable.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.optimize_objectives import (
    minimize_s11,
    maximize_s21,
    target_impedance,
    maximize_bandwidth,
)


# ---------------------------------------------------------------------------
# Lightweight mock Result
# ---------------------------------------------------------------------------

class _MockResult(NamedTuple):
    """Minimal stand-in for rfx.api.Result with s_params and freqs."""
    state: object
    time_series: jnp.ndarray
    s_params: jnp.ndarray
    freqs: jnp.ndarray
    ntff_data: object = None
    ntff_box: object = None


def _make_mock_result(
    n_ports: int = 2,
    n_freqs: int = 20,
    s11_mag: float = 0.5,
    s21_mag: float = 0.3,
) -> _MockResult:
    """Build a mock Result with controllable S-parameter magnitudes.

    S-parameters are real-valued for simplicity (phase = 0).
    """
    freqs = jnp.linspace(1e9, 10e9, n_freqs)
    s_params = jnp.zeros((n_ports, n_ports, n_freqs), dtype=jnp.complex64)
    # S11 = s11_mag (constant across freq)
    s_params = s_params.at[0, 0, :].set(s11_mag + 0j)
    # S21 = s21_mag
    if n_ports > 1:
        s_params = s_params.at[1, 0, :].set(s21_mag + 0j)
    return _MockResult(
        state=None,
        time_series=jnp.zeros((100, 1)),
        s_params=s_params,
        freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMinimizeS11:
    """Tests for minimize_s11 objective."""

    def test_minimize_s11_returns_scalar(self):
        """minimize_s11 objective must return a 0-d JAX array."""
        result = _make_mock_result(s11_mag=0.5)
        target_freqs = jnp.linspace(2e9, 8e9, 10)
        obj = minimize_s11(target_freqs, target_db=-10)
        loss = obj(result)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert loss.dtype in (jnp.float32, jnp.float64)
        # |S11|^2 = 0.25, threshold at -10dB = 0.1, so loss = 0.25 - 0.1 = 0.15
        assert float(loss) > 0.0, "S11 above threshold should give positive loss"
        print(f"\n  minimize_s11 loss = {float(loss):.6f}")

    def test_minimize_s11_below_threshold(self):
        """When S11 is already below target, loss should be 0."""
        result = _make_mock_result(s11_mag=0.1)  # |S11|^2 = 0.01, well below -10dB
        target_freqs = jnp.linspace(2e9, 8e9, 10)
        obj = minimize_s11(target_freqs, target_db=-10)
        loss = obj(result)
        assert float(loss) == 0.0, f"Expected 0.0, got {float(loss)}"


class TestMaximizeS21:
    """Tests for maximize_s21 objective."""

    def test_maximize_s21_returns_scalar(self):
        """maximize_s21 objective must return a 0-d JAX array."""
        result = _make_mock_result(s21_mag=0.8)
        target_freqs = jnp.linspace(2e9, 8e9, 10)
        obj = maximize_s21(target_freqs)
        loss = obj(result)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        # Negative of |S21|^2 = -0.64
        assert float(loss) < 0.0, "maximize_s21 should return negative loss"
        print(f"\n  maximize_s21 loss = {float(loss):.6f}")


class TestTargetImpedance:
    """Tests for target_impedance objective."""

    def test_target_impedance_returns_scalar(self):
        """target_impedance objective must return a 0-d JAX array."""
        result = _make_mock_result(s11_mag=0.2)
        obj = target_impedance(freq=5e9, z_target=50.0)
        loss = obj(result)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert float(loss) >= 0.0
        print(f"\n  target_impedance loss = {float(loss):.6f}")

    def test_target_impedance_matched(self):
        """When S11=0 (perfectly matched to 50 Ohm), Z_in=50 -> loss=0."""
        result = _make_mock_result(s11_mag=0.0)
        obj = target_impedance(freq=5e9, z_target=50.0)
        loss = obj(result)
        assert float(loss) < 1e-6, f"Expected ~0, got {float(loss)}"


class TestMaximizeBandwidth:
    """Tests for maximize_bandwidth objective."""

    def test_maximize_bandwidth_returns_scalar(self):
        """maximize_bandwidth objective must return a 0-d JAX array."""
        result = _make_mock_result(s11_mag=0.5)
        obj = maximize_bandwidth(f_center=5e9, f_bw=4e9, s11_threshold=-10)
        loss = obj(result)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert float(loss) > 0.0, "S11 above threshold should give positive bandwidth loss"
        print(f"\n  maximize_bandwidth loss = {float(loss):.6f}")

    def test_maximize_bandwidth_all_below(self):
        """When all S11 bins are below threshold, loss should be 0."""
        result = _make_mock_result(s11_mag=0.05)  # -26 dB, well below -10 dB
        obj = maximize_bandwidth(f_center=5e9, f_bw=4e9, s11_threshold=-10)
        loss = obj(result)
        assert float(loss) == 0.0, f"Expected 0.0, got {float(loss)}"


class TestDifferentiability:
    """Verify objectives are JAX-differentiable through S-parameters."""

    def test_objectives_are_differentiable(self):
        """jax.grad through each S-param-based objective should not error."""
        target_freqs = jnp.linspace(2e9, 8e9, 5)
        freqs = jnp.linspace(1e9, 10e9, 20)

        objectives = {
            "minimize_s11": minimize_s11(target_freqs, target_db=-10),
            "maximize_s21": maximize_s21(target_freqs),
            "target_impedance": target_impedance(freq=5e9, z_target=50.0),
            "maximize_bandwidth": maximize_bandwidth(
                f_center=5e9, f_bw=4e9, s11_threshold=-10,
            ),
        }

        for name, obj_fn in objectives.items():
            # Differentiate w.r.t. a scalar that scales S-params
            def loss_fn(scale):
                n_freqs = 20
                s_params = jnp.zeros((2, 2, n_freqs), dtype=jnp.complex64)
                s_params = s_params.at[0, 0, :].set(scale * 0.5 + 0j)
                s_params = s_params.at[1, 0, :].set(scale * 0.3 + 0j)
                mock = _MockResult(
                    state=None,
                    time_series=jnp.zeros((100, 1)),
                    s_params=s_params,
                    freqs=freqs,
                )
                return obj_fn(mock)

            grad_fn = jax.grad(loss_fn)
            g = grad_fn(jnp.float32(1.0))
            assert jnp.isfinite(g), f"{name}: gradient is not finite ({g})"
            print(f"\n  {name}: grad = {float(g):.6e}")
