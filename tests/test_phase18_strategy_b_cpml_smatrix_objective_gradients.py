"""Phase XVIII CPML native Strategy B S-matrix objective gradient tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from scripts.phase18_strategy_b_cpml_smatrix_objective_gradient_gate import run_validation
from rfx import (
    GaussianPulse,
    Phase1SMatrixObjectiveRequest,
    Phase1SMatrixObjectiveTerm,
    Simulation,
    native_minimize_s11_request,
)
from rfx.hybrid_adjoint import _phase17_weighted_smatrix_objective


def _make_one_port_cpml_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.009, 0.009, 0.009),
        dx=0.001,
        boundary="cpml",
        cpml_layers=2,
    )
    sim.add_port(
        (0.004, 0.004, 0.004),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim


def _make_two_port_cpml_sim(*, explicit_direction: bool = False) -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.010, 0.009, 0.009),
        dx=0.001,
        boundary="cpml",
        cpml_layers=2,
    )
    direction = "+x" if explicit_direction else None
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim.add_port((0.003, 0.004, 0.004), "ez", impedance=50.0, waveform=pulse, direction=direction)
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, excite=False, direction=direction)
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


def _assert_fd_valid(sim, inputs, request, *, checkpoint_every: int):
    assert inputs.materials is not None
    assert inputs.s_param_request is not None
    eps0 = inputs.materials.eps_r
    port_cells = {tuple(port.cell) for port in inputs.s_param_request.ports}

    def loss_eps(eps):
        return sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
            inputs,
            request,
            eps_override=eps,
            strategy="b",
            checkpoint_every=checkpoint_every,
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
    assert trusted_cell not in port_cells
    assert abs(float(grad_np[trusted_cell])) > 1e-8

    h = jnp.asarray(1e-2, dtype=jnp.float32)
    fd = (loss_eps(eps0.at[trusted_cell].add(h)) - loss_eps(eps0.at[trusted_cell].add(-h))) / (2.0 * h)
    analytic = grad[trusted_cell]
    abs_err = float(jnp.abs(fd - analytic))
    if abs(float(fd)) > 1e-6:
        rel_err = abs_err / (abs(float(fd)) + 1e-30)
        assert rel_err <= 0.20
    else:
        assert abs_err <= 7.5e-5


def test_phase18_one_port_cpml_objective_value_matches_native_smatrix():
    sim = _make_one_port_cpml_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)

    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        request,
        checkpoint_every=6,
    )
    assert report.supported, report.reason_text
    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=6)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=6,
    )

    expected = _phase17_weighted_smatrix_objective(native.s_params, request)
    np.testing.assert_allclose(np.asarray(objective), np.asarray(expected), rtol=0.0, atol=1e-7)


def test_phase18_one_port_cpml_gradient_matches_finite_difference():
    sim = _make_one_port_cpml_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)
    _assert_fd_valid(
        sim,
        inputs,
        native_minimize_s11_request(freqs),
        checkpoint_every=6,
    )


def test_phase18_two_port_cpml_objective_value_and_gradient_match_finite_difference():
    sim = _make_two_port_cpml_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=36, s_param_freqs=freqs)
    request = _two_port_objective(freqs)

    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=12)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=12,
    )
    expected = _phase17_weighted_smatrix_objective(native.s_params, request)
    np.testing.assert_allclose(np.asarray(objective), np.asarray(expected), rtol=0.0, atol=1e-7)
    _assert_fd_valid(sim, inputs, request, checkpoint_every=12)


def test_phase18_cpml_default_sidecar_remains_stopped_gradient():
    sim = _make_one_port_cpml_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)
    assert inputs.materials is not None
    assert inputs.s_param_request is not None
    eps0 = inputs.materials.eps_r
    port_cell = tuple(inputs.s_param_request.ports[0].cell)

    def sidecar_loss(alpha):
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps0.at[port_cell].add(alpha),
            strategy="b",
            checkpoint_every=6,
        )
        return jnp.real(jnp.sum(jnp.abs(result.s_params) ** 2))

    np.testing.assert_allclose(np.asarray(jax.grad(sidecar_loss)(0.0)), np.asarray(0.0), rtol=0.0, atol=0.0)


def test_phase18_cpml_native_objective_does_not_use_standard_extractors(monkeypatch):
    sim = _make_two_port_cpml_sim()
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)

    def _fail_standard(*_args, **_kwargs):
        raise AssertionError("native Strategy B CPML S-matrix objective must not use standard extraction")

    import rfx.probes.probes as probes_mod

    monkeypatch.setattr(probes_mod, "extract_s_matrix", _fail_standard)
    monkeypatch.setattr(probes_mod, "extract_s_matrix_wire", _fail_standard)
    monkeypatch.setattr(Simulation, "run", _fail_standard)

    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        _two_port_objective(freqs),
        strategy="b",
        checkpoint_every=6,
    )

    assert np.isfinite(np.asarray(objective))


def test_phase18_direction_aware_cpml_ports_remain_fail_closed():
    sim = _make_two_port_cpml_sim(explicit_direction=True)
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=18, s_param_freqs=freqs)
    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        _two_port_objective(freqs),
        checkpoint_every=6,
    )

    assert not report.supported
    assert "explicit port direction" in report.reason_text


def test_phase18_validation_artifact_gates_pass():
    artifact = run_validation(n_steps=36)

    assert artifact["overall_status"] == "phase18_strategy_b_cpml_smatrix_objective_validated"
    assert all(artifact["gates"].values())
    assert artifact["schema_version"] == "phase18_strategy_b_cpml_smatrix_objective_gradient_v1"
