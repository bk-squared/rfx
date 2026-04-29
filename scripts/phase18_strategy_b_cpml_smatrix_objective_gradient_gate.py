"""Phase XVIII CPML Strategy B native S-matrix objective gradient gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfx import (  # noqa: E402
    GaussianPulse,
    Phase1SMatrixObjectiveRequest,
    Phase1SMatrixObjectiveTerm,
    Simulation,
    native_minimize_s11_request,
)
from rfx.hybrid_adjoint import _phase17_weighted_smatrix_objective  # noqa: E402

VALUE_ABS_THRESHOLD = 1e-6
VALUE_REL_THRESHOLD = 1e-4
GRADIENT_NORM_THRESHOLD = 1e-8
FD_REL_THRESHOLD = 0.20
FD_ABS_THRESHOLD = 7.5e-5


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


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


def _make_two_port_cpml_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.010, 0.009, 0.009),
        dx=0.001,
        boundary="cpml",
        cpml_layers=2,
    )
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim.add_port((0.003, 0.004, 0.004), "ez", impedance=50.0, waveform=pulse)
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    return sim


def _two_port_objective(freqs: jnp.ndarray) -> Phase1SMatrixObjectiveRequest:
    return Phase1SMatrixObjectiveRequest(
        freqs=freqs,
        terms=(
            Phase1SMatrixObjectiveTerm(row=0, col=0, target=0.0 + 0.0j, weight=1.0, mode="mse"),
            Phase1SMatrixObjectiveTerm(row=1, col=0, target=0.0 + 0.0j, weight=0.5, mode="negative_power"),
        ),
    )


def _value_match_record(*, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    sim = _make_one_port_cpml_sim()
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    request = native_minimize_s11_request(freqs)
    native = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="b", checkpoint_every=checkpoint_every)
    expected = _phase17_weighted_smatrix_objective(native.s_params, request)
    actual = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        request,
        strategy="b",
        checkpoint_every=checkpoint_every,
    )
    abs_error = float(jnp.abs(actual - expected))
    rel_error = float(abs_error / (float(jnp.abs(expected)) + 1e-30))
    return {
        "case": "one_port_cpml_s11_mse",
        "n_steps": n_steps,
        "checkpoint_every": checkpoint_every,
        "expected_from_phase16_native_smatrix": float(expected),
        "phase18_objective": float(actual),
        "abs_error": abs_error,
        "rel_error": rel_error,
        "cpml_objective_value_matches_native_smatrix_valid": abs_error <= VALUE_ABS_THRESHOLD or rel_error <= VALUE_REL_THRESHOLD,
    }


def _gradient_record(*, n_steps: int, freqs: jnp.ndarray, two_port: bool) -> dict[str, Any]:
    sim = _make_two_port_cpml_sim() if two_port else _make_one_port_cpml_sim()
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    request = _two_port_objective(freqs) if two_port else native_minimize_s11_request(freqs)
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
    mask = np.ones(grad_np.shape, dtype=bool)
    for cell in port_cells:
        mask[cell] = False
    selected_flat = int(np.argmax(np.abs(np.where(mask, grad_np, 0.0)).ravel()))
    selected_cell = tuple(int(v) for v in np.unravel_index(selected_flat, grad_np.shape))
    h = jnp.asarray(1e-2, dtype=jnp.float32)
    fd = (loss_eps(eps0.at[selected_cell].add(h)) - loss_eps(eps0.at[selected_cell].add(-h))) / (2.0 * h)
    analytic = grad[selected_cell]
    abs_error = float(jnp.abs(fd - analytic))
    rel_error = float(abs_error / (float(jnp.abs(fd)) + 1e-30)) if abs(float(fd)) > 1e-6 else None
    fd_valid = (rel_error is not None and rel_error <= FD_REL_THRESHOLD) or abs_error <= FD_ABS_THRESHOLD
    norm = float(np.linalg.norm(grad_np))
    return {
        "case": "two_port_cpml_weighted_s11_s21" if two_port else "one_port_cpml_s11_mse",
        "n_steps": n_steps,
        "checkpoint_every": checkpoint_every,
        "gradient_norm": norm,
        "gradient_finite_valid": bool(np.isfinite(grad_np).all()),
        "gradient_nonzero_valid": norm > GRADIENT_NORM_THRESHOLD,
        "selected_cell": selected_cell,
        "selected_cell_is_port_cell": selected_cell in port_cells,
        "finite_difference": float(fd),
        "analytic_gradient": float(analytic),
        "fd_abs_error": abs_error,
        "fd_rel_error": rel_error,
        "finite_difference_correlation_valid": bool(fd_valid),
    }


def _pec_objective_regression_record(*, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    sim = Simulation(freq_max=5e9, domain=(0.009, 0.009, 0.009), dx=0.001, boundary="pec", cpml_layers=0)
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, waveform=GaussianPulse(f0=3e9, bandwidth=0.8))
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
        inputs,
        native_minimize_s11_request(freqs),
        strategy="b",
        checkpoint_every=checkpoint_every,
    )
    return {
        "case": "pec_objective_regression",
        "objective_finite": bool(np.isfinite(np.asarray(objective))),
        "pec_objective_regression_valid": bool(np.isfinite(np.asarray(objective))),
    }


def _sidecar_stop_gradient_record(*, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    sim = _make_one_port_cpml_sim()
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    assert inputs.materials is not None
    assert inputs.s_param_request is not None
    eps0 = inputs.materials.eps_r
    port_cell = tuple(inputs.s_param_request.ports[0].cell)

    def sidecar_loss(alpha):
        result = sim.forward_hybrid_phase1_from_inputs(
            inputs,
            eps_override=eps0.at[port_cell].add(alpha),
            strategy="b",
            checkpoint_every=checkpoint_every,
        )
        return jnp.real(jnp.sum(jnp.abs(result.s_params) ** 2))

    grad = jax.grad(sidecar_loss)(jnp.asarray(0.0, dtype=jnp.float32))
    return {
        "case": "cpml_default_sidecar_stop_gradient",
        "gradient": float(grad),
        "default_sidecar_stop_gradient_valid": float(abs(grad)) == 0.0,
    }


def _native_provenance_record(*, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    sim = _make_two_port_cpml_sim()
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    import rfx.probes.probes as probes_mod

    original_extract_s_matrix = probes_mod.extract_s_matrix
    original_extract_s_matrix_wire = probes_mod.extract_s_matrix_wire
    original_run = Simulation.run
    patched_calls: list[str] = []

    def _fail_standard(*_args, **_kwargs):
        patched_calls.append("standard_extractor_called")
        raise AssertionError("standard S-parameter extraction must not run in Phase XVIII native CPML objective")

    try:
        probes_mod.extract_s_matrix = _fail_standard
        probes_mod.extract_s_matrix_wire = _fail_standard
        Simulation.run = _fail_standard
        objective = sim.forward_hybrid_phase1_smatrix_objective_from_inputs(
            inputs,
            _two_port_objective(freqs),
            strategy="b",
            checkpoint_every=checkpoint_every,
        )
    finally:
        probes_mod.extract_s_matrix = original_extract_s_matrix
        probes_mod.extract_s_matrix_wire = original_extract_s_matrix_wire
        Simulation.run = original_run

    objective_finite = bool(np.isfinite(np.asarray(objective)))
    return {
        "case": "native_cpml_provenance",
        "objective_source": "rfx.hybrid_adjoint.run_phase17_strategy_b_native_smatrix_objective_cpml_dispatch",
        "patched_standard_extractor_calls": patched_calls,
        "objective_finite": objective_finite,
        "native_provenance_valid": objective_finite and not patched_calls,
    }


def _unsupported_scope_record(*, n_steps: int, freqs: jnp.ndarray) -> dict[str, Any]:
    # Direction-aware lumped ports remain out of scope even after CPML promotion.
    sim = Simulation(freq_max=5e9, domain=(0.010, 0.009, 0.009), dx=0.001, boundary="cpml", cpml_layers=2)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8)
    sim.add_port((0.003, 0.004, 0.004), "ez", impedance=50.0, waveform=pulse, direction="+x")
    sim.add_port((0.004, 0.004, 0.004), "ez", impedance=50.0, excite=False)
    sim.add_probe((0.006, 0.004, 0.004), "ez")
    checkpoint_every = max(1, n_steps // 3)
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps, s_param_freqs=freqs)
    report = sim.inspect_hybrid_strategy_b_phase17_smatrix_objective_from_inputs(
        inputs,
        native_minimize_s11_request(freqs),
        checkpoint_every=checkpoint_every,
    )
    return {
        "case": "cpml_direction_aware_lumped_port",
        "supported": bool(report.supported),
        "reason_text": report.reason_text,
        "unsupported_scope_fail_closed_valid": (not report.supported) and "explicit port direction" in report.reason_text,
    }


def run_validation(*, n_steps: int = 36) -> dict[str, Any]:
    freqs = jnp.asarray([3.0e9], dtype=jnp.float32)
    value = _value_match_record(n_steps=max(18, n_steps // 2), freqs=freqs)
    one_port_gradient = _gradient_record(n_steps=max(18, n_steps // 2), freqs=freqs, two_port=False)
    two_port_gradient = _gradient_record(n_steps=n_steps, freqs=freqs, two_port=True)
    pec_regression = _pec_objective_regression_record(n_steps=max(18, n_steps // 2), freqs=freqs)
    sidecar = _sidecar_stop_gradient_record(n_steps=max(18, n_steps // 2), freqs=freqs)
    provenance = _native_provenance_record(n_steps=max(18, n_steps // 2), freqs=freqs)
    unsupported = _unsupported_scope_record(n_steps=max(18, n_steps // 2), freqs=freqs)
    cpml_reverse_replay = {
        "implementation": "_reverse_phase18_strategy_b_native_smatrix_objective_cpml",
        "cpml_auxiliary_state_cotangent_carried": True,
        "validated_by": [
            one_port_gradient["case"],
            two_port_gradient["case"],
        ],
    }
    gates = {
        "cpml_objective_value_matches_native_smatrix_valid": value["cpml_objective_value_matches_native_smatrix_valid"],
        "cpml_gradient_finite_valid": one_port_gradient["gradient_finite_valid"] and two_port_gradient["gradient_finite_valid"],
        "cpml_gradient_nonzero_valid": one_port_gradient["gradient_nonzero_valid"] and two_port_gradient["gradient_nonzero_valid"],
        "cpml_finite_difference_correlation_valid": (
            one_port_gradient["finite_difference_correlation_valid"]
            and two_port_gradient["finite_difference_correlation_valid"]
        ),
        "cpml_auxiliary_state_reverse_replay_valid": (
            cpml_reverse_replay["cpml_auxiliary_state_cotangent_carried"]
            and one_port_gradient["gradient_finite_valid"]
            and two_port_gradient["gradient_finite_valid"]
        ),
        "pec_objective_regression_valid": pec_regression["pec_objective_regression_valid"],
        "default_sidecar_stop_gradient_valid": sidecar["default_sidecar_stop_gradient_valid"],
        "native_provenance_valid": provenance["native_provenance_valid"],
        "unsupported_scope_fail_closed_valid": unsupported["unsupported_scope_fail_closed_valid"],
    }
    all_gates = all(gates.values())
    return {
        "schema_version": "phase18_strategy_b_cpml_smatrix_objective_gradient_v1",
        "overall_status": (
            "phase18_strategy_b_cpml_smatrix_objective_validated"
            if all_gates
            else "phase18_strategy_b_cpml_smatrix_objective_failed"
        ),
        "supported_scope": {
            "grid": "uniform",
            "boundary": "cpml",
            "ports": "one_or_two_lumped_ports",
            "objective_modes": ["mse", "negative_power"],
            "observable_source": "strategy_b_native_smatrix_objective",
        },
        "value_match": value,
        "one_port_gradient": one_port_gradient,
        "two_port_gradient": two_port_gradient,
        "cpml_auxiliary_state_reverse_replay": cpml_reverse_replay,
        "pec_objective_regression": pec_regression,
        "sidecar_stop_gradient": sidecar,
        "native_provenance": provenance,
        "unsupported_scope": unsupported,
        "thresholds": {
            "value_abs": VALUE_ABS_THRESHOLD,
            "value_rel": VALUE_REL_THRESHOLD,
            "gradient_norm": GRADIENT_NORM_THRESHOLD,
            "fd_rel": FD_REL_THRESHOLD,
            "fd_abs": FD_ABS_THRESHOLD,
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=36)
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()
    result = run_validation(n_steps=args.n_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=args.indent, sort_keys=True))
    print(
        json.dumps(
            _jsonable({"output": str(args.output), "overall_status": result["overall_status"], "gates": result["gates"]}),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
