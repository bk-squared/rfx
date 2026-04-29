"""Phase XVII differentiable Strategy B native S-matrix objective tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.phase17_strategy_b_smatrix_objective_gradient_validation import run_validation
from rfx import (
    GaussianPulse,
    Phase1SMatrixObjectiveRequest,
    Phase1SMatrixObjectiveTerm,
    Simulation,
    native_maximize_s21_request,
    native_minimize_s11_request,
)
from rfx.hybrid_adjoint import _phase17_weighted_smatrix_objective


def _make_one_port_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.009, 0.009, 0.009),
        dx=0.001,
        boundary=boundary,
        cpml_layers=0 if boundary == "pec" else 2,
    )
    sim.add_port(
        (0.004, 0.004, 0.004),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim


def _make_two_port_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.010, 0.009, 0.009),
        dx=0.001,
        boundary=boundary,
        cpml_layers=0 if boundary == "pec" else 2,
    )
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim.add_port((0.003, 0.004, 0.004), "ez", impedance=50.0, waveform=pulse)
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim


def _two_port_objective(freqs) -> Phase1SMatrixObjectiveRequest:
    return Phase1SMatrixObjectiveRequest(
        freqs=jnp.asarray(freqs, dtype=jnp.float32),
        terms=(
            Phase1SMatrixObjectiveTerm(row=0, col=0, target=0.0 + 0.0j, weight=1.0, mode="mse"),
            Phase1SMatrixObjectiveTerm(row=1, col=0, target=0.0 + 0.0j, weight=0.5, mode="negative_power"),
        ),
    )


def test_phase17_pure_weighted_objective_matches_hand_calculation():
    smatrix = jnp.asarray([[[1.0 + 2.0j, 2.0 + 0.0j]]], dtype=jnp.complex64)
    request = Phase1SMatrixObjectiveRequest(
        freqs=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        terms=(
            Phase1SMatrixObjectiveTerm(
                row=0,
                col=0,
                target=jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex64),
                weight=jnp.asarray([1.0, 3.0], dtype=jnp.float32),
                mode="mse",
            ),
            Phase1SMatrixObjectiveTerm(row=0, col=0, target=0.0, weight=2.0, mode="negative_power"),
        ),
    )

    expected_mse = 1.0 * 5.0 + 3.0 * 1.0
    expected_negative_power = 2.0 * (-(5.0 + 4.0))
    expected = (expected_mse + expected_negative_power) / (1.0 + 3.0 + 2.0 + 2.0)

    np.testing.assert_allclose(
        np.asarray(_phase17_weighted_smatrix_objective(smatrix, request)),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "objective_request, message",
    [
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([2.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0),),
            ),
            "freqs must match",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=2, col=0),),
            ),
            "indices exceed",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0, weight=-1.0),),
            ),
            "nonnegative",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0, weight=0.0),),
            ),
            "positive total weight",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0, mode="phase"),),
            ),
            "unsupported mode",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0, target=[0.0, 1.0]),),
            ),
            "targets must be scalar",
        ),
        (
            Phase1SMatrixObjectiveRequest(
                freqs=jnp.asarray([3.0e9], dtype=jnp.float32),
                terms=(Phase1SMatrixObjectiveTerm(row=0, col=0),),
                observable_source="standard_extractor_objective",
            ),
            "observable_source='strategy_b_native_smatrix_objective'",
        ),
    ],
)
def test_phase17_objective_spec_validation_fails_closed(objective_request, message):
    sim = _make_one_port_sim()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=jnp.asarray([3.0e9], dtype=jnp.float32))

    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        objective_request,
        checkpoint_every=8,
    )

    assert not report.supported
    assert message in report.reason_text
    with pytest.raises(ValueError, match=message):
        sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
            inputs,
            objective_request,
            strategy="b",
            checkpoint_every=8,
        )


def test_phase17_checkpoint_every_is_required_for_objective_mode():
    sim = _make_one_port_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=16, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)

    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        request,
        checkpoint_every=None,
    )

    assert not report.supported
    assert "checkpoint_every is required" in report.reason_text


def test_phase17_one_port_objective_value_matches_phase16_native_smatrix():
    sim = _make_one_port_sim()
    freqs = jnp.asarray([2.5e9, 3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=24, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)

    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=8)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=8,
    )

    expected = _phase17_weighted_smatrix_objective(native.s_params, request)
    np.testing.assert_allclose(np.asarray(objective), np.asarray(expected), rtol=0.0, atol=1e-7)


def test_phase17_two_port_objective_value_matches_phase16_native_smatrix():
    sim = _make_two_port_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=32, s_param_freqs=freqs)
    request = _two_port_objective(freqs)

    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=8)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=8,
    )

    expected = _phase17_weighted_smatrix_objective(native.s_params, request)
    np.testing.assert_allclose(np.asarray(objective), np.asarray(expected), rtol=0.0, atol=1e-7)


def test_phase17_gradient_is_finite_nonzero_and_matches_finite_difference():
    sim = _make_two_port_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=48, s_param_freqs=freqs)
    request = _two_port_objective(freqs)
    eps0 = inputs.materials.eps_r
    assert eps0 is not None
    port_cells = {tuple(port.cell) for port in inputs.s_param_request.ports}

    def loss_eps(eps):
        return sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
            inputs,
            request,
            eps_override=eps,
            strategy="b",
            checkpoint_every=12,
        )

    grad = jax.grad(loss_eps)(eps0)
    grad_np = np.asarray(grad)
    assert np.isfinite(grad_np).all()
    assert float(np.linalg.norm(grad_np)) > 1e-8

    mask = np.ones(grad_np.shape, dtype=bool)
    for cell in port_cells:
        mask[cell] = False
    trusted_flat = int(np.argmax(np.abs(np.where(mask, grad_np, 0.0)).ravel()))
    trusted_cell = np.unravel_index(trusted_flat, grad_np.shape)
    assert abs(float(grad_np[trusted_cell])) > 1e-8

    h = jnp.asarray(1e-2, dtype=jnp.float32)
    fd = (loss_eps(eps0.at[trusted_cell].add(h)) - loss_eps(eps0.at[trusted_cell].add(-h))) / (2.0 * h)
    analytic = grad[trusted_cell]
    abs_err = float(jnp.abs(fd - analytic))
    if abs(float(fd)) > 1e-6:
        rel_err = abs_err / (abs(float(fd)) + 1e-30)
        assert rel_err <= 0.15
    else:
        assert abs_err <= 5e-5


def test_phase17_cpml_objective_value_matches_phase16_after_phase18_promotion():
    sim = _make_one_port_sim(boundary="cpml")
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)

    phase16 = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=6)
    assert phase16.s_params is not None
    assert np.isfinite(np.asarray(phase16.s_params)).all()

    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        request,
        checkpoint_every=6,
    )
    assert report.supported, report.reason_text
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=6,
    )
    expected = _phase17_weighted_smatrix_objective(phase16.s_params, request)
    np.testing.assert_allclose(np.asarray(objective), np.asarray(expected), rtol=0.0, atol=1e-7)


def test_phase17_default_sidecar_remains_stopped_while_objective_is_differentiable():
    sim = _make_one_port_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=24, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)
    eps0 = inputs.materials.eps_r
    assert eps0 is not None
    port_cell = tuple(inputs.s_param_request.ports[0].cell)

    def eps_with_alpha(alpha):
        return eps0.at[port_cell].add(alpha)

    def sidecar_loss(alpha):
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps_with_alpha(alpha),
            strategy="b",
            checkpoint_every=8,
        )
        return jnp.real(jnp.sum(jnp.abs(result.s_params) ** 2))

    def objective_loss(alpha):
        return sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
            inputs,
            request,
            eps_override=eps_with_alpha(alpha),
            strategy="b",
            checkpoint_every=8,
        )

    np.testing.assert_allclose(np.asarray(jax.grad(sidecar_loss)(0.0)), np.asarray(0.0), rtol=0.0, atol=0.0)
    assert abs(float(jax.grad(objective_loss)(0.0))) > 1e-8


def test_phase17_native_objective_does_not_use_standard_extractors(monkeypatch):
    sim = _make_two_port_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    request = _two_port_objective(freqs)

    def _fail_standard(*_args, **_kwargs):
        raise AssertionError("native Strategy B S-matrix objective must not use standard extraction")

    import rfx.probes.probes as probes_mod

    monkeypatch.setattr(probes_mod, "extract_s_matrix", _fail_standard)
    monkeypatch.setattr(probes_mod, "extract_s_matrix_wire", _fail_standard)
    monkeypatch.setattr(Simulation, "run", _fail_standard)

    inputs = sim.build_hybrid_phase1_inputs(n_steps=24, s_param_freqs=freqs)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=8,
    )

    assert np.isfinite(np.asarray(objective))


def test_phase17_public_request_helpers_are_explicit_specs():
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)

    s11_request = native_minimize_s11_request(freqs)
    s21_request = native_maximize_s21_request(freqs)

    assert s11_request.observable_source == "strategy_b_native_smatrix_objective"
    assert s11_request.terms[0].mode == "mse"
    assert s21_request.terms[0].mode == "negative_power"
    assert (s21_request.terms[0].row, s21_request.terms[0].col) == (1, 0)


def test_phase17_validation_artifact_gates_pass():
    artifact = run_validation(n_steps=48)

    assert artifact["overall_status"] == "phase17_strategy_b_smatrix_objective_validated"
    assert all(artifact["gates"].values())
    assert artifact["schema_version"] == "phase17_strategy_b_smatrix_objective_gradient_v1"
