"""Phase 1 hybrid-adjoint regression tests."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from rfx import Box, DebyePole, GaussianPulse
from rfx.api import Simulation
from rfx.hybrid_adjoint import phase1_forward_result


def _make_phase1_sim(
    *,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
) -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary=boundary,
        pec_faces=pec_faces,
    )
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _inspect_supported_phase1_snapshot(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=n_steps)
    assert prepared_runner is not None
    return sim, grid, prepared_runner, report


def _inspect_supported_phase1_snapshot_for_periods(
    *,
    num_periods: float = 8.0,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(num_periods=num_periods)
    assert prepared_runner is not None
    return sim, grid, prepared_runner, report


def _make_supported_phase1_snapshot_case(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim, grid, prepared_runner, report = _inspect_supported_phase1_snapshot(
        n_steps=n_steps,
        boundary=boundary,
        pec_faces=pec_faces,
    )
    return sim, grid, prepared_runner, report, n_steps


def _make_supported_phase1_snapshot_inputs_case(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case(
        n_steps=n_steps,
        boundary=boundary,
        pec_faces=pec_faces,
    )
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps)
    return sim, grid, prepared_runner, report, n_steps, inputs


def _make_supported_phase1_inputs(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    return sim, sim.build_hybrid_phase1_inputs(n_steps=n_steps)


def _make_supported_phase1_prepared_bundle(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    return sim, sim.prepare_hybrid_phase1(n_steps=n_steps)



def _make_supported_phase1_context(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    return sim, sim.build_hybrid_phase1_context(n_steps=n_steps)


def _make_supported_phase1_report_context(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    return sim, sim.inspect_hybrid_phase1(n_steps=n_steps), sim.build_hybrid_phase1_context(n_steps=n_steps)


def _make_supported_phase1_grid_materials(
    *,
    n_steps: int = 18,
    boundary: str = "pec",
    pec_faces: set[str] | None = None,
):
    sim = _make_phase1_sim(boundary=boundary, pec_faces=pec_faces)
    grid = sim._build_grid()
    base_materials, _, _, _, _, _ = sim._assemble_materials(grid)
    return sim, grid, base_materials, n_steps


def _make_cpml_supported_phase1_sim() -> Simulation:
    return _make_phase1_sim(boundary="cpml")


def _make_cpml_supported_phase1_sim_with_pec_face() -> Simulation:
    return _make_phase1_sim(boundary="cpml", pec_faces={"z_lo"})


def _make_cpml_lossy_unsupported_phase1_sim() -> Simulation:
    sim = _make_phase1_sim(boundary="cpml")
    sim.add_material("lossy", eps_r=2.0, sigma=5.0)
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="lossy")
    return sim


def _make_debye_unsupported_phase1_sim() -> Simulation:
    sim = _make_phase1_sim()
    sim.add_material(
        "disp",
        eps_r=2.0,
        debye_poles=[DebyePole(delta_eps=1.0, tau=8e-12)],
    )
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="disp")
    return sim



def _make_lumped_port_unsupported_phase1_sim() -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _make_lumped_port_unsupported_prepared_bundle(*, n_steps: int = 12):
    sim = _make_lumped_port_unsupported_phase1_sim()
    return sim, sim.prepare_hybrid_phase1(n_steps=n_steps)


def _assert_unsupported_prepare_bundle_basics(prepared, *, expected_reason: str):
    assert not prepared.supported
    assert prepared.context is None
    assert expected_reason in prepared.reason_text


def _assert_unsupported_inputs_basics(inputs, *, expected_reason: str):
    assert not inputs.supported
    assert expected_reason in inputs.reason_text


def _make_lumped_port_unsupported_inputs(*, n_steps: int = 12):
    sim = _make_lumped_port_unsupported_phase1_sim()
    return sim, sim.build_hybrid_phase1_inputs(n_steps=n_steps)



def _make_nonuniform_unsupported_inputs(*, n_steps: int = 12):
    sim = _make_nonuniform_unsupported_phase1_sim()
    return sim, sim.build_hybrid_phase1_inputs(n_steps=n_steps)


def _make_nonuniform_unsupported_prepared_bundle(*, n_steps: int = 12):
    sim = _make_nonuniform_unsupported_phase1_sim()
    return sim, sim.prepare_hybrid_phase1(n_steps=n_steps)



def _make_nonuniform_unsupported_phase1_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary="pec",
        dx=0.0025,
        dz_profile=np.array([0.002, 0.002, 0.002, 0.002, 0.002], dtype=float),
    )
    sim.add_source((0.005, 0.0075, 0.0075), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _make_nonuniform_inspected_runner_unsupported_phase1_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary="pec",
        dz_profile=np.full(5, 0.003),
    )
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _inspect_nonuniform_inspected_runner_unsupported_phase1(*, n_steps: int = 18):
    sim = _make_nonuniform_inspected_runner_unsupported_phase1_sim()
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=n_steps)
    return sim, grid, prepared_runner, report



def _resolved_n_steps(sim: Simulation, *, num_periods: float = 8.0) -> int:
    if sim._dz_profile is not None or sim._dx_profile is not None or sim._dy_profile is not None:
        grid = sim._build_nonuniform_grid()
        return int(np.ceil(num_periods * (1.0 / float(sim._freq_max)) / float(grid.dt)))
    return sim._build_grid().num_timesteps(num_periods=num_periods)


def _unsupported_phase1_cases() -> list[tuple[Simulation, str]]:
    return [
        (_make_debye_unsupported_phase1_sim(), "Debye"),
        (_make_lumped_port_unsupported_phase1_sim(), "add_source"),
        (_make_cpml_lossy_unsupported_phase1_sim(), "lossy materials"),
        (_make_nonuniform_unsupported_phase1_sim(), "non-uniform grids are unsupported"),
    ]


def _top_level_unsupported_report(
    sim: Simulation,
    *,
    n_steps: int | None = None,
    num_periods: float = 8.0,
):
    if n_steps is None:
        return sim.inspect_hybrid_phase1(num_periods=num_periods)
    return sim.inspect_hybrid_phase1(n_steps=n_steps)


def _top_level_unsupported_prepared(
    sim: Simulation,
    *,
    n_steps: int | None = None,
    num_periods: float = 8.0,
):
    if n_steps is None:
        return sim.prepare_hybrid_phase1(num_periods=num_periods)
    return sim.prepare_hybrid_phase1(n_steps=n_steps)


def _assert_top_level_raise_matches_report_reason_text(
    sim: Simulation,
    action,
    *,
    n_steps: int | None = None,
    num_periods: float = 8.0,
):
    report = _top_level_unsupported_report(sim, n_steps=n_steps, num_periods=num_periods)
    with pytest.raises(ValueError) as err:
        action(sim, n_steps=n_steps, num_periods=num_periods)
    assert str(err.value) == report.reason_text


def _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
    sim: Simulation,
    *,
    expected_error: str,
    fallback: str,
    num_periods: float = 8.0,
):
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)

    if fallback == "raise":
        with pytest.raises(ValueError, match=expected_error):
            sim.forward_hybrid_phase1(num_periods=num_periods, fallback="raise")

        with pytest.raises(ValueError, match=expected_error):
            sim.forward_hybrid_phase1(n_steps=explicit_n_steps, fallback="raise")
        return

    implicit = sim.forward_hybrid_phase1(num_periods=num_periods, fallback=fallback)
    explicit = sim.forward_hybrid_phase1(n_steps=explicit_n_steps, fallback=fallback)
    baseline = sim.forward(n_steps=explicit_n_steps, checkpoint=True)

    np.testing.assert_allclose(
        np.asarray(implicit.time_series),
        np.asarray(explicit.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(implicit.time_series),
        np.asarray(baseline.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(
    sim: Simulation,
    *,
    num_periods: float = 8.0,
):
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)

    implicit_report = sim.inspect_hybrid_phase1(num_periods=num_periods)
    explicit_report = sim.inspect_hybrid_phase1(n_steps=explicit_n_steps)
    implicit_inputs = sim.build_hybrid_phase1_inputs(num_periods=num_periods)
    explicit_inputs = sim.build_hybrid_phase1_inputs(n_steps=explicit_n_steps)
    implicit_prepared = sim.prepare_hybrid_phase1(num_periods=num_periods)
    explicit_prepared = sim.prepare_hybrid_phase1(n_steps=explicit_n_steps)
    implicit_context = sim.build_hybrid_phase1_context(num_periods=num_periods)
    explicit_context = sim.build_hybrid_phase1_context(n_steps=explicit_n_steps)
    implicit_forward = sim.forward_hybrid_phase1(num_periods=num_periods, fallback="raise")
    explicit_forward = sim.forward_hybrid_phase1(n_steps=explicit_n_steps, fallback="raise")

    assert implicit_report == explicit_report
    assert implicit_inputs.report == explicit_inputs.report == implicit_report
    assert implicit_inputs.reason_text == explicit_inputs.reason_text == ""
    assert implicit_inputs.supported and explicit_inputs.supported
    assert implicit_prepared.report == explicit_prepared.report == implicit_report
    assert implicit_prepared.reason_text == explicit_prepared.reason_text == ""
    assert implicit_prepared.supported and explicit_prepared.supported
    assert implicit_prepared.context is not None and explicit_prepared.context is not None
    np.testing.assert_allclose(np.asarray(implicit_context.eps_r), np.asarray(explicit_context.eps_r))
    np.testing.assert_allclose(np.asarray(implicit_inputs.require_context().eps_r), np.asarray(explicit_inputs.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(implicit_prepared.require_context().eps_r), np.asarray(explicit_prepared.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(implicit_context.run_time_series()), np.asarray(explicit_context.run_time_series()), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(implicit_inputs.require_context().run_time_series()), np.asarray(explicit_inputs.require_context().run_time_series()), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(implicit_prepared.run_time_series()), np.asarray(explicit_prepared.run_time_series()), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(implicit_forward.time_series), np.asarray(explicit_forward.time_series), rtol=1e-6, atol=1e-12)


def _assert_top_level_unsupported_public_family_matches_explicit_n_steps_when_omitted(
    sim: Simulation,
    *,
    expected_error: str,
    num_periods: float = 8.0,
):
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)

    implicit_report = _top_level_unsupported_report(sim, num_periods=num_periods)
    explicit_report = _top_level_unsupported_report(sim, n_steps=explicit_n_steps)
    implicit_prepared = _top_level_unsupported_prepared(sim, num_periods=num_periods)
    explicit_prepared = _top_level_unsupported_prepared(sim, n_steps=explicit_n_steps)

    assert implicit_report == explicit_report
    assert implicit_prepared.report == explicit_prepared.report

    with pytest.raises(ValueError, match=expected_error):
        sim.build_hybrid_phase1_context(num_periods=num_periods)

    with pytest.raises(ValueError, match=expected_error):
        sim.build_hybrid_phase1_context(n_steps=explicit_n_steps)


def _assert_top_level_unsupported_input_builder_contract(
    sim: Simulation,
    *,
    n_steps: int | None = None,
    num_periods: float = 8.0,
):
    report = _top_level_unsupported_report(sim, n_steps=n_steps, num_periods=num_periods)
    if n_steps is None:
        inputs = sim.build_hybrid_phase1_inputs(num_periods=num_periods)
    else:
        inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps)
    assert inputs.report == report
    assert inputs.reason_text == report.reason_text
    assert not inputs.supported
    with pytest.raises(ValueError):
        inputs.require_context()
    return inputs, report



def _assert_top_level_unsupported_prepare_bundle_contract(
    sim: Simulation,
    *,
    n_steps: int | None = None,
    num_periods: float = 8.0,
    expected_reason: str | None = None,
):
    report = _top_level_unsupported_report(sim, n_steps=n_steps, num_periods=num_periods)
    prepared = _top_level_unsupported_prepared(sim, n_steps=n_steps, num_periods=num_periods)
    assert prepared.report == report
    assert prepared.reason_text == report.reason_text
    assert prepared.context is None
    if expected_reason is not None:
        assert expected_reason in prepared.reason_text
    return prepared, report


def _single_cell_eps(grid, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    i, j, k = grid.position_to_index((0.0075, 0.0075, 0.0075))
    return base_eps.at[i, j, k].add(alpha)


def _assert_hybrid_forward_matches_pure_forward(sim: Simulation, *, n_steps: int = 18):
    baseline = sim.forward(n_steps=n_steps, checkpoint=True)
    hybrid = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")

    np.testing.assert_allclose(
        np.asarray(hybrid.time_series),
        np.asarray(baseline.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def _assert_single_cell_hybrid_gradient_matches_pure_ad(
    sim: Simulation,
    grid,
    base_materials,
    *,
    n_steps: int,
):
    def pure_loss(alpha):
        eps = _single_cell_eps(grid, base_materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=n_steps, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def hybrid_loss(alpha):
        eps = _single_cell_eps(grid, base_materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1(eps_override=eps, n_steps=n_steps, fallback="raise")
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_hybrid = jax.grad(hybrid_loss)(alpha0)
    rel_err = float(jnp.abs(grad_hybrid - grad_pure) / jnp.maximum(jnp.abs(grad_pure), 1e-12))

    assert np.isfinite(float(grad_pure))
    assert np.isfinite(float(grad_hybrid))
    assert rel_err <= 1e-4, (
        f"hybrid gradient drifted from pure AD: pure={float(grad_pure):.6e}, "
        f"hybrid={float(grad_hybrid):.6e}, rel_err={rel_err:.6e}"
    )


def _make_supported_phase1_eps_override(
    sim: Simulation,
    *,
    alpha: jnp.ndarray | None = None,
) -> tuple[Grid, jnp.ndarray]:
    if alpha is None:
        alpha = jnp.float32(0.2)
    grid = sim._build_grid()
    base_materials, _, _, _, _, _ = sim._assemble_materials(grid)
    return grid, _single_cell_eps(grid, base_materials.eps_r, alpha)


def _make_supported_phase1_eps_override_case(*, num_periods: float = 8.0):
    sim = _make_phase1_sim()
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    _, eps_override = _make_supported_phase1_eps_override(sim)
    return sim, explicit_n_steps, eps_override


def _make_supported_phase1_context_eps_override_case(*, num_periods: float = 8.0):
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )
    implicit_context = sim.build_hybrid_phase1_context(num_periods=num_periods)
    explicit_context = sim.build_hybrid_phase1_context(n_steps=explicit_n_steps)
    return sim, implicit_context, explicit_context, explicit_n_steps, eps_override


def _make_supported_phase1_prepared_eps_override_case(*, num_periods: float = 8.0):
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )
    implicit_prepared = sim.prepare_hybrid_phase1(num_periods=num_periods)
    explicit_prepared = sim.prepare_hybrid_phase1(n_steps=explicit_n_steps)
    return sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override


def _make_supported_phase1_inputs_eps_override_case(*, num_periods: float = 8.0):
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )
    implicit_inputs = sim.build_hybrid_phase1_inputs(num_periods=num_periods)
    explicit_inputs = sim.build_hybrid_phase1_inputs(n_steps=explicit_n_steps)
    return sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override


def _make_supported_phase1_prepare_context_case(*, num_periods: float = 8.0):
    sim = _make_phase1_sim()
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    implicit_prepared = sim.prepare_hybrid_phase1(num_periods=num_periods)
    implicit_context = sim.build_hybrid_phase1_context(num_periods=num_periods)
    explicit_prepared = sim.prepare_hybrid_phase1(n_steps=explicit_n_steps)
    explicit_context = sim.build_hybrid_phase1_context(n_steps=explicit_n_steps)
    return (
        sim,
        implicit_prepared,
        implicit_context,
        explicit_prepared,
        explicit_context,
        explicit_n_steps,
    )


def _make_supported_phase1_inspected_runner_eps_override_case(*, num_periods: float = 8.0):
    sim = _make_phase1_sim()
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(num_periods=num_periods)
    assert prepared_runner is not None
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    _, eps_override = _make_supported_phase1_eps_override(sim)
    return sim, grid, prepared_runner, report, explicit_n_steps, eps_override


def _make_supported_phase1_prepared_runner_eps_override_case(*, num_periods: float = 8.0):
    sim, grid, prepared_runner, _ = _inspect_supported_phase1_snapshot_for_periods(
        num_periods=num_periods
    )
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    _, eps_override = _make_supported_phase1_eps_override(sim)
    return sim, grid, prepared_runner, explicit_n_steps, eps_override


def test_phase1_root_hybrid_export_block_covers_all_public_seam_symbols():
    import ast
    from pathlib import Path

    hybrid_path = Path("rfx/hybrid_adjoint.py")
    root_path = Path("rfx/__init__.py")

    module = ast.parse(hybrid_path.read_text())
    public_symbols = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            public_symbols.append(node.name)

    root_lines = root_path.read_text().splitlines()
    start = root_lines.index("from rfx.hybrid_adjoint import (") + 1
    end = start
    while root_lines[end].strip() != ")":
        end += 1
    exported_symbols = {
        line.strip().rstrip(",")
        for line in root_lines[start:end]
        if line.strip()
    }

    assert set(public_symbols) <= exported_symbols



def test_phase1_hybrid_adjoint_public_functions_and_methods_keep_explicit_type_annotations():
    import ast
    from pathlib import Path

    module = ast.parse(Path("rfx/hybrid_adjoint.py").read_text())

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            for arg in node.args.args:
                if arg.arg in {"self", "cls"}:
                    continue
                assert arg.annotation is not None, f"{node.name}.{arg.arg} missing annotation"
            for arg in node.args.kwonlyargs:
                assert arg.annotation is not None, f"{node.name}.{arg.arg} missing annotation"
            assert node.returns is not None, f"{node.name} missing return annotation"
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Phase1"):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("__"):
                    for arg in child.args.args:
                        if arg.arg in {"self", "cls"}:
                            continue
                        assert arg.annotation is not None, f"{node.name}.{child.name}.{arg.arg} missing annotation"
                    for arg in child.args.kwonlyargs:
                        assert arg.annotation is not None, f"{node.name}.{child.name}.{arg.arg} missing annotation"
                    assert child.returns is not None, f"{node.name}.{child.name} missing return annotation"



def test_phase1_simulation_phase1_methods_keep_explicit_type_annotations():
    import inspect

    from rfx.api import Simulation

    method_names = [
        "inspect_hybrid_phase1",
        "build_hybrid_phase1_inputs",
        "prepare_hybrid_phase1",
        "build_hybrid_phase1_context",
        "inspect_hybrid_phase1_from_inputs",
        "build_hybrid_phase1_inputs_from_prepared_runner_state",
        "build_hybrid_phase1_inputs_from_inspected_runner_state",
        "inspect_hybrid_phase1_from_prepared_runner_state",
        "inspect_hybrid_phase1_from_inspected_runner_state",
        "prepare_hybrid_phase1_from_inputs",
        "prepare_hybrid_phase1_from_prepared_runner_state",
        "prepare_hybrid_phase1_from_inspected_runner_state",
        "build_hybrid_phase1_context_from_inputs",
        "build_hybrid_phase1_context_from_prepared_runner_state",
        "build_hybrid_phase1_context_from_inspected_runner_state",
        "forward_hybrid_phase1_from_inputs",
        "forward_hybrid_phase1_from_context",
        "forward_hybrid_phase1_from_prepared",
        "forward_hybrid_phase1_from_prepared_runner_state",
        "forward_hybrid_phase1_from_inspected_runner_state",
        "_resolve_phase1_hybrid_runner_state_n_steps",
        "_inspect_hybrid_phase1_prepared",
        "forward_hybrid_phase1",
    ]

    for name in method_names:
        signature = inspect.signature(getattr(Simulation, name))
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation is not inspect._empty, f"{name}.{param_name} missing annotation"
        assert signature.return_annotation is not inspect._empty, f"{name} missing return annotation"



def test_phase1_runner_state_n_steps_resolver_matches_expected_uniform_and_nonuniform_behavior():
    sim = _make_phase1_sim()
    uniform_grid = sim._build_grid()
    assert sim._resolve_phase1_hybrid_runner_state_n_steps(uniform_grid, n_steps=7, num_periods=8.0) == 7
    assert sim._resolve_phase1_hybrid_runner_state_n_steps(uniform_grid, n_steps=None, num_periods=8.0) == uniform_grid.num_timesteps(num_periods=8.0)
    assert sim._resolve_phase1_hybrid_runner_state_n_steps(None, n_steps=None, num_periods=8.0) is None

    nonuniform_sim = _make_nonuniform_unsupported_phase1_sim()
    nonuniform_grid = nonuniform_sim._build_nonuniform_grid()
    expected_nonuniform = int(np.ceil(8.0 * (1.0 / float(nonuniform_sim._freq_max)) / float(nonuniform_grid.dt)))
    assert nonuniform_sim._resolve_phase1_hybrid_runner_state_n_steps(nonuniform_grid, n_steps=None, num_periods=8.0) == expected_nonuniform


def test_phase1_private_prepared_inspection_matches_explicit_n_steps_when_omitted_on_supported_path():
    sim = _make_phase1_sim()
    num_periods = 8.0
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)

    implicit_grid, implicit_prepared, implicit_report = sim._inspect_hybrid_phase1_prepared(
        num_periods=num_periods
    )
    explicit_grid, explicit_prepared, explicit_report = sim._inspect_hybrid_phase1_prepared(
        n_steps=explicit_n_steps
    )

    assert implicit_prepared is not None
    assert explicit_prepared is not None
    assert implicit_report == explicit_report

    implicit_inputs = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        implicit_grid,
        implicit_prepared,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_inputs = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        explicit_grid,
        explicit_prepared,
        n_steps=explicit_n_steps,
    )

    assert implicit_inputs.report == explicit_inputs.report
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(explicit_inputs.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().run_time_series()),
        np.asarray(explicit_inputs.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_private_prepared_inspection_preserves_nonuniform_contract_when_n_steps_is_omitted():
    sim = _make_nonuniform_unsupported_phase1_sim()
    num_periods = 8.0
    explicit_n_steps = _resolved_n_steps(sim, num_periods=num_periods)

    implicit_grid, implicit_prepared, implicit_report = sim._inspect_hybrid_phase1_prepared(
        num_periods=num_periods
    )
    explicit_grid, explicit_prepared, explicit_report = sim._inspect_hybrid_phase1_prepared(
        n_steps=explicit_n_steps
    )

    assert implicit_prepared is None
    assert explicit_prepared is None
    assert implicit_report == explicit_report

    implicit_inputs = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        implicit_grid,
        implicit_prepared,
        implicit_report,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_inputs = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        explicit_grid,
        explicit_prepared,
        explicit_report,
        n_steps=explicit_n_steps,
    )

    assert not implicit_inputs.supported
    assert not explicit_inputs.supported
    assert implicit_inputs.report == explicit_inputs.report
    assert implicit_inputs.reason_text == explicit_inputs.reason_text == "non-uniform grids are unsupported"


def test_phase1_private_prepared_inspection_preserves_eps_override_on_supported_path():
    n_steps = 18
    sim, _explicit_n_steps_unused, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=8.0
    )

    prepared_grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(
        eps_override=eps_override,
        n_steps=n_steps,
    )

    assert prepared_runner is not None

    via_private = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_public = sim.build_hybrid_phase1_inputs(
        eps_override=eps_override,
        n_steps=n_steps,
    )

    assert report == via_public.report
    assert via_private.report == via_public.report
    np.testing.assert_allclose(
        np.asarray(via_private.require_context().eps_r),
        np.asarray(eps_override),
    )
    np.testing.assert_allclose(
        np.asarray(via_private.require_context().eps_r),
        np.asarray(via_public.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(via_private.require_context().run_time_series()),
        np.asarray(via_public.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_private_prepared_inspection_preserves_eps_override_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )

    implicit_grid, implicit_prepared, implicit_report = sim._inspect_hybrid_phase1_prepared(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    explicit_grid, explicit_prepared, explicit_report = sim._inspect_hybrid_phase1_prepared(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )

    assert implicit_prepared is not None
    assert explicit_prepared is not None
    assert implicit_report == explicit_report

    implicit_inputs = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        implicit_grid,
        implicit_prepared,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_inputs = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        explicit_grid,
        explicit_prepared,
        n_steps=explicit_n_steps,
    )
    public_inputs = sim.build_hybrid_phase1_inputs(
        eps_override=eps_override,
        num_periods=num_periods,
    )

    assert implicit_inputs.report == explicit_inputs.report == public_inputs.report
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(eps_override),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(explicit_inputs.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(public_inputs.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().run_time_series()),
        np.asarray(explicit_inputs.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().run_time_series()),
        np.asarray(public_inputs.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_simulation_top_level_public_entrypoints_keep_expected_defaults():
    import inspect

    from rfx.api import Simulation

    top_level_methods = {
        "inspect_hybrid_phase1": {"n_steps": None, "num_periods": 20.0, "eps_override": None},
        "build_hybrid_phase1_inputs": {"n_steps": None, "num_periods": 20.0, "eps_override": None},
        "prepare_hybrid_phase1": {"n_steps": None, "num_periods": 20.0, "eps_override": None},
        "build_hybrid_phase1_context": {"n_steps": None, "num_periods": 20.0, "eps_override": None},
        "forward_hybrid_phase1": {"n_steps": None, "num_periods": 20.0, "eps_override": None, "fallback": "pure_ad"},
    }

    assert set(top_level_methods) <= set(dir(Simulation))

    for name, expected_defaults in top_level_methods.items():
        signature = inspect.signature(getattr(Simulation, name))
        for param, expected in expected_defaults.items():
            assert signature.parameters[param].default == expected



def test_phase1_simulation_public_wrapper_family_is_present_and_runner_state_defaults_stay_symmetric():
    import inspect

    from rfx.api import Simulation

    required_methods = {
        "inspect_hybrid_phase1_from_inputs",
        "prepare_hybrid_phase1_from_inputs",
        "build_hybrid_phase1_context_from_inputs",
        "forward_hybrid_phase1_from_inputs",
        "build_hybrid_phase1_inputs_from_prepared_runner_state",
        "inspect_hybrid_phase1_from_prepared_runner_state",
        "prepare_hybrid_phase1_from_prepared_runner_state",
        "build_hybrid_phase1_context_from_prepared_runner_state",
        "forward_hybrid_phase1_from_prepared_runner_state",
        "build_hybrid_phase1_inputs_from_inspected_runner_state",
        "inspect_hybrid_phase1_from_inspected_runner_state",
        "prepare_hybrid_phase1_from_inspected_runner_state",
        "build_hybrid_phase1_context_from_inspected_runner_state",
        "forward_hybrid_phase1_from_inspected_runner_state",
    }
    assert required_methods <= set(dir(Simulation))

    symmetric_runner_methods = [
        "build_hybrid_phase1_inputs_from_prepared_runner_state",
        "inspect_hybrid_phase1_from_prepared_runner_state",
        "prepare_hybrid_phase1_from_prepared_runner_state",
        "build_hybrid_phase1_context_from_prepared_runner_state",
        "forward_hybrid_phase1_from_prepared_runner_state",
        "build_hybrid_phase1_inputs_from_inspected_runner_state",
        "inspect_hybrid_phase1_from_inspected_runner_state",
        "prepare_hybrid_phase1_from_inspected_runner_state",
        "build_hybrid_phase1_context_from_inspected_runner_state",
        "forward_hybrid_phase1_from_inspected_runner_state",
    ]
    for name in symmetric_runner_methods:
        signature = inspect.signature(getattr(Simulation, name))
        assert signature.parameters["n_steps"].default is None
        assert signature.parameters["num_periods"].default == 20.0


def test_phase1_public_forward_wrapper_family_keeps_eps_override_default_none():
    import inspect

    from rfx.api import Simulation

    forward_wrapper_methods = [
        "forward_hybrid_phase1_from_inputs",
        "forward_hybrid_phase1_from_context",
        "forward_hybrid_phase1_from_prepared",
        "forward_hybrid_phase1_from_prepared_runner_state",
        "forward_hybrid_phase1_from_inspected_runner_state",
    ]

    for name in forward_wrapper_methods:
        signature = inspect.signature(getattr(Simulation, name))
        assert signature.parameters["eps_override"].default is None



def test_phase1_seam_forward_helper_family_keeps_eps_override_default_none():
    import inspect

    from rfx.hybrid_adjoint import (
        forward_phase1_hybrid_from_context,
        forward_phase1_hybrid_from_prepared,
        forward_phase1_hybrid_from_inputs,
        forward_phase1_hybrid_from_prepared_runner_state,
        forward_phase1_hybrid_from_inspected_runner_state,
    )

    for fn in (
        forward_phase1_hybrid_from_context,
        forward_phase1_hybrid_from_prepared,
        forward_phase1_hybrid_from_inputs,
        forward_phase1_hybrid_from_prepared_runner_state,
        forward_phase1_hybrid_from_inspected_runner_state,
    ):
        signature = inspect.signature(fn)
        assert signature.parameters["eps_override"].default is None


def test_phase1_seam_surface_methods_keep_eps_override_default_none():
    import inspect

    from rfx.hybrid_adjoint import (
        Phase1HybridContext,
        Phase1HybridPrepared,
        Phase1HybridInputs,
    )

    for cls in (Phase1HybridContext, Phase1HybridPrepared, Phase1HybridInputs):
        for name in ("run_time_series", "forward_result"):
            signature = inspect.signature(getattr(cls, name))
            assert signature.parameters["eps_override"].default is None


def test_phase1_seam_runner_state_forward_helpers_keep_expected_n_steps_contract():
    import inspect

    from rfx.hybrid_adjoint import (
        forward_phase1_hybrid_from_prepared_runner_state,
        forward_phase1_hybrid_from_inspected_runner_state,
    )

    prepared_sig = inspect.signature(forward_phase1_hybrid_from_prepared_runner_state)
    inspected_sig = inspect.signature(forward_phase1_hybrid_from_inspected_runner_state)

    assert prepared_sig.parameters["n_steps"].default is inspect._empty
    assert inspected_sig.parameters["n_steps"].default is inspect._empty


def test_phase1_seam_runner_state_helper_family_keeps_expected_n_steps_signatures():
    import inspect

    from rfx.hybrid_adjoint import (
        build_phase1_hybrid_inputs_from_prepared_runner_state,
        build_phase1_hybrid_inputs_from_inspected_runner_state,
        inspect_phase1_hybrid_from_prepared_runner_state,
        inspect_phase1_hybrid_from_inspected_runner_state,
        prepare_phase1_hybrid_from_prepared_runner_state,
        prepare_phase1_hybrid_from_inspected_runner_state,
        build_phase1_hybrid_context_from_prepared_runner_state,
        build_phase1_hybrid_context_from_inspected_runner_state,
    )

    prepared_runner_helpers = (
        build_phase1_hybrid_inputs_from_prepared_runner_state,
        inspect_phase1_hybrid_from_prepared_runner_state,
        prepare_phase1_hybrid_from_prepared_runner_state,
        build_phase1_hybrid_context_from_prepared_runner_state,
    )
    inspected_runner_helpers = (
        build_phase1_hybrid_inputs_from_inspected_runner_state,
        inspect_phase1_hybrid_from_inspected_runner_state,
        prepare_phase1_hybrid_from_inspected_runner_state,
        build_phase1_hybrid_context_from_inspected_runner_state,
    )

    for fn in prepared_runner_helpers:
        signature = inspect.signature(fn)
        assert str(signature.parameters["n_steps"].annotation) == "int"
        assert signature.parameters["n_steps"].default is inspect._empty

    for fn in inspected_runner_helpers:
        signature = inspect.signature(fn)
        assert str(signature.parameters["n_steps"].annotation) == "int | None"
        assert signature.parameters["n_steps"].default is inspect._empty


def test_phase1_root_hybrid_export_block_has_no_duplicate_names():
    from pathlib import Path

    text = Path("rfx/__init__.py").read_text().splitlines()
    start = text.index("from rfx.hybrid_adjoint import (") + 1
    end = start
    while not text[end].strip() == ")":
        end += 1
    names = [line.strip().rstrip(",") for line in text[start:end] if line.strip()]
    assert len(names) == len(set(names)), f"duplicate seam exports found: {names}"




def test_phase1_hybrid_types_exported_from_package_root():
    from rfx import (
        Phase1FieldState,
        Phase1HybridInventory,
        Phase1HybridPreparedRunnerState,
        Phase1HybridContext,
        Phase1HybridInspection,
        Phase1HybridPrepared,
        Phase1HybridInputs,
        phase1_forward_result as exported_phase1_forward_result,
        run_phase1_forward_time_series as exported_run_time_series,
        make_phase1_hybrid_forward as exported_make_forward,
        phase1_hybrid_support_reasons as exported_support_reasons,
        build_phase1_hybrid_context as exported_build_context,
        build_phase1_hybrid_context_from_inputs as exported_build_context_from_inputs,
        build_phase1_hybrid_context_from_prepared_runner_state as exported_build_context_from_prepared,
        build_phase1_hybrid_context_from_inspected_runner_state as exported_build_context_from_inspected,
        build_phase1_hybrid_inputs_from_prepared_runner_state as exported_build_inputs_from_prepared,
        build_phase1_hybrid_inputs_from_inspected_runner_state as exported_build_inputs_from_inspected,
        inspect_phase1_hybrid as exported_inspect_phase1_hybrid,
        inspect_phase1_hybrid_from_inputs as exported_inspect_from_inputs,
        inspect_phase1_hybrid_from_prepared_runner_state as exported_inspect_from_prepared,
        inspect_phase1_hybrid_from_inspected_runner_state as exported_inspect_from_inspected,
        prepare_phase1_hybrid as exported_prepare_phase1_hybrid,
        prepare_phase1_hybrid_from_inputs as exported_prepare_from_inputs,
        prepare_phase1_hybrid_from_prepared_runner_state as exported_prepare_from_prepared,
        prepare_phase1_hybrid_from_inspected_runner_state as exported_prepare_from_inspected,
        forward_phase1_hybrid_from_context as exported_forward_from_context,
        forward_phase1_hybrid_from_prepared as exported_forward_from_prepared,
        forward_phase1_hybrid_from_inputs as exported_forward_from_inputs,
        forward_phase1_hybrid_from_prepared_runner_state as exported_forward_from_prepared_runner,
        forward_phase1_hybrid_from_inspected_runner_state as exported_forward_from_inspected_runner,
        unsupported_phase1_hybrid_nonuniform as exported_unsupported_prepared,
        unsupported_phase1_hybrid_nonuniform_report as exported_unsupported_report,
        unsupported_phase1_hybrid_nonuniform_inputs as exported_unsupported_inputs,
    )
    from rfx.hybrid_adjoint import (
        Phase1FieldState as MPhase1FieldState,
        Phase1HybridInventory as MPhase1HybridInventory,
        Phase1HybridPreparedRunnerState as MPhase1HybridPreparedRunnerState,
        Phase1HybridContext as MPhase1HybridContext,
        Phase1HybridInspection as MPhase1HybridInspection,
        Phase1HybridPrepared as MPhase1HybridPrepared,
        Phase1HybridInputs as MPhase1HybridInputs,
        phase1_forward_result as m_phase1_forward_result,
        run_phase1_forward_time_series as m_run_time_series,
        make_phase1_hybrid_forward as m_make_forward,
        phase1_hybrid_support_reasons as m_support_reasons,
        build_phase1_hybrid_context as m_build_context,
        build_phase1_hybrid_context_from_inputs as m_build_context_from_inputs,
        build_phase1_hybrid_context_from_prepared_runner_state as m_build_context_from_prepared,
        build_phase1_hybrid_context_from_inspected_runner_state as m_build_context_from_inspected,
        build_phase1_hybrid_inputs_from_prepared_runner_state as m_build_inputs_from_prepared,
        build_phase1_hybrid_inputs_from_inspected_runner_state as m_build_inputs_from_inspected,
        inspect_phase1_hybrid as m_inspect_phase1_hybrid,
        inspect_phase1_hybrid_from_inputs as m_inspect_from_inputs,
        inspect_phase1_hybrid_from_prepared_runner_state as m_inspect_from_prepared,
        inspect_phase1_hybrid_from_inspected_runner_state as m_inspect_from_inspected,
        prepare_phase1_hybrid as m_prepare_phase1_hybrid,
        prepare_phase1_hybrid_from_inputs as m_prepare_from_inputs,
        prepare_phase1_hybrid_from_prepared_runner_state as m_prepare_from_prepared,
        prepare_phase1_hybrid_from_inspected_runner_state as m_prepare_from_inspected,
        forward_phase1_hybrid_from_context as m_forward_from_context,
        forward_phase1_hybrid_from_prepared as m_forward_from_prepared,
        forward_phase1_hybrid_from_inputs as m_forward_from_inputs,
        forward_phase1_hybrid_from_prepared_runner_state as m_forward_from_prepared_runner,
        forward_phase1_hybrid_from_inspected_runner_state as m_forward_from_inspected_runner,
        unsupported_phase1_hybrid_nonuniform as m_unsupported_prepared,
        unsupported_phase1_hybrid_nonuniform_report as m_unsupported_report,
        unsupported_phase1_hybrid_nonuniform_inputs as m_unsupported_inputs,
    )

    assert Phase1FieldState is MPhase1FieldState
    assert Phase1HybridInventory is MPhase1HybridInventory
    assert Phase1HybridPreparedRunnerState is MPhase1HybridPreparedRunnerState
    assert Phase1HybridContext is MPhase1HybridContext
    assert Phase1HybridInspection is MPhase1HybridInspection
    assert Phase1HybridPrepared is MPhase1HybridPrepared
    assert Phase1HybridInputs is MPhase1HybridInputs
    assert exported_phase1_forward_result is m_phase1_forward_result
    assert exported_run_time_series is m_run_time_series
    assert exported_make_forward is m_make_forward
    assert exported_support_reasons is m_support_reasons
    assert exported_build_context is m_build_context
    assert exported_build_context_from_inputs is m_build_context_from_inputs
    assert exported_build_context_from_prepared is m_build_context_from_prepared
    assert exported_build_context_from_inspected is m_build_context_from_inspected
    assert exported_build_inputs_from_prepared is m_build_inputs_from_prepared
    assert exported_build_inputs_from_inspected is m_build_inputs_from_inspected
    assert exported_inspect_phase1_hybrid is m_inspect_phase1_hybrid
    assert exported_inspect_from_inputs is m_inspect_from_inputs
    assert exported_inspect_from_prepared is m_inspect_from_prepared
    assert exported_inspect_from_inspected is m_inspect_from_inspected
    assert exported_prepare_phase1_hybrid is m_prepare_phase1_hybrid
    assert exported_prepare_from_inputs is m_prepare_from_inputs
    assert exported_prepare_from_prepared is m_prepare_from_prepared
    assert exported_prepare_from_inspected is m_prepare_from_inspected
    assert exported_forward_from_context is m_forward_from_context
    assert exported_forward_from_prepared is m_forward_from_prepared
    assert exported_forward_from_inputs is m_forward_from_inputs
    assert exported_forward_from_prepared_runner is m_forward_from_prepared_runner
    assert exported_forward_from_inspected_runner is m_forward_from_inspected_runner
    assert exported_unsupported_prepared is m_unsupported_prepared
    assert exported_unsupported_report is m_unsupported_report
    assert exported_unsupported_inputs is m_unsupported_inputs




def test_phase1_seam_wrapper_matches_canonical_forward_objective():
    n_steps = 18
    sim, report, context = _make_supported_phase1_report_context()

    assert report.supported
    assert report.inventory is not None

    from rfx import Phase1HybridInventory

    assert isinstance(report.inventory, Phase1HybridInventory)
    assert isinstance(report.inventory.carry_fields, tuple)

    from rfx import Phase1FieldState

    assert isinstance(context.initial_state, Phase1FieldState)
    assert report.inventory.carry_fields == (
        "fdtd.ex",
        "fdtd.ey",
        "fdtd.ez",
        "fdtd.hx",
        "fdtd.hy",
        "fdtd.hz",
    )
    assert report.inventory.total_carry_bytes > 0
    assert report.inventory.replay_inputs == ("eps_r", "mu_r", "source_waveforms_raw")

    baseline = sim.forward(n_steps=n_steps, checkpoint=True)
    hybrid = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    baseline_obj = float(jnp.sum(baseline.time_series ** 2))
    hybrid_obj = float(jnp.sum(hybrid.time_series ** 2))
    np.testing.assert_allclose(hybrid_obj, baseline_obj, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(hybrid.time_series),
        np.asarray(baseline.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_input_builder_matches_prepare_bundle_surface():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs()
    prepared = inputs.prepare()
    public_prepared = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert inputs.source_count == 1
    assert inputs.probe_count == 1
    assert prepared.report == public_prepared.report
    assert prepared.context is not None
    assert public_prepared.context is not None
    np.testing.assert_allclose(np.asarray(prepared.context.eps_r), np.asarray(public_prepared.context.eps_r))




def test_phase1_prepared_runner_state_type_is_seam_owned():
    _sim, _grid, prepared_runner, _report, _n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import Phase1HybridPreparedRunnerState

    assert isinstance(prepared_runner, Phase1HybridPreparedRunnerState)




def test_phase1_report_classmethod_from_prepared_runner_state_matches_helper_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridInspection,
        inspect_phase1_hybrid_from_prepared_runner_state,
    )

    via_class = Phase1HybridInspection.from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    via_helper = inspect_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )

    assert via_class == via_helper




def test_phase1_report_classmethod_from_inspected_runner_state_matches_helper_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridInspection,
        inspect_phase1_hybrid_from_inspected_runner_state,
    )

    via_class = Phase1HybridInspection.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    via_helper = inspect_phase1_hybrid_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )

    assert via_class == via_helper




def test_phase1_report_from_prepared_runner_state_matches_public_report_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import inspect_phase1_hybrid_from_prepared_runner_state

    rebuilt = inspect_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    public_report = sim.inspect_hybrid_phase1(n_steps=n_steps)

    assert rebuilt == public_report




def test_phase1_prepared_from_inputs_matches_prepare_function_surface():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs(n_steps=n_steps)

    from rfx.hybrid_adjoint import Phase1HybridPrepared, prepare_phase1_hybrid

    via_class = Phase1HybridPrepared.from_inputs(inputs)
    via_func = prepare_phase1_hybrid(inputs)

    assert via_class.report == via_func.report
    assert via_class.context is not None
    assert via_func.context is not None
    np.testing.assert_allclose(np.asarray(via_class.context.eps_r), np.asarray(via_func.context.eps_r))




def test_phase1_prepare_from_prepared_runner_state_matches_public_prepare_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import prepare_phase1_hybrid_from_prepared_runner_state

    rebuilt = prepare_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    public_prepared = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert rebuilt.report == public_prepared.report
    assert rebuilt.context is not None
    assert public_prepared.context is not None
    np.testing.assert_allclose(np.asarray(rebuilt.context.eps_r), np.asarray(public_prepared.context.eps_r))




def test_phase1_prepare_helper_from_inspected_runner_state_matches_classmethod_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridPrepared,
        prepare_phase1_hybrid_from_inspected_runner_state,
    )

    via_class = Phase1HybridPrepared.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    via_helper = prepare_phase1_hybrid_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )

    assert via_class.report == via_helper.report
    assert via_class.context is not None
    assert via_helper.context is not None
    np.testing.assert_allclose(np.asarray(via_class.context.eps_r), np.asarray(via_helper.context.eps_r))




def test_phase1_prepared_from_inspected_runner_state_matches_public_prepare_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import Phase1HybridPrepared

    rebuilt = Phase1HybridPrepared.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    public_prepared = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert rebuilt.report == public_prepared.report
    assert rebuilt.context is not None
    assert public_prepared.context is not None
    np.testing.assert_allclose(np.asarray(rebuilt.context.eps_r), np.asarray(public_prepared.context.eps_r))




def test_phase1_prepared_from_prepared_runner_state_matches_prepare_helper_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridPrepared,
        prepare_phase1_hybrid_from_prepared_runner_state,
    )

    via_class = Phase1HybridPrepared.from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    via_func = prepare_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )

    assert via_class.report == via_func.report
    assert via_class.context is not None
    assert via_func.context is not None
    np.testing.assert_allclose(np.asarray(via_class.context.eps_r), np.asarray(via_func.context.eps_r))




def test_phase1_input_from_inspected_runner_state_matches_builder_surface():
    sim, grid, prepared_runner, report, n_steps, inputs = (
        _make_supported_phase1_snapshot_inputs_case()
    )

    from rfx.hybrid_adjoint import Phase1HybridInputs
    rebuilt = Phase1HybridInputs.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )

    assert rebuilt.source_count == inputs.source_count
    assert rebuilt.probe_count == inputs.probe_count
    assert rebuilt.reason_text == inputs.reason_text




def test_phase1_input_helper_from_prepared_runner_state_matches_classmethod_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridInputs,
        build_phase1_hybrid_inputs_from_prepared_runner_state,
    )

    via_class = Phase1HybridInputs.from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    via_helper = build_phase1_hybrid_inputs_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )

    assert via_class.source_count == via_helper.source_count
    assert via_class.probe_count == via_helper.probe_count
    assert via_class.reason_text == via_helper.reason_text



def test_phase1_input_helper_from_inspected_runner_state_matches_classmethod_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridInputs,
        build_phase1_hybrid_inputs_from_inspected_runner_state,
    )

    via_class = Phase1HybridInputs.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    via_helper = build_phase1_hybrid_inputs_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )

    assert via_class.source_count == via_helper.source_count
    assert via_class.probe_count == via_helper.probe_count
    assert via_class.reason_text == via_helper.reason_text




def test_phase1_input_from_prepared_runner_state_matches_builder_surface():
    sim, grid, prepared_runner, report, n_steps, inputs = (
        _make_supported_phase1_snapshot_inputs_case()
    )

    from rfx.hybrid_adjoint import Phase1HybridInputs
    rebuilt = Phase1HybridInputs.from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )

    assert rebuilt.source_count == inputs.source_count
    assert rebuilt.probe_count == inputs.probe_count
    assert rebuilt.pec_axes == inputs.pec_axes
    np.testing.assert_allclose(np.asarray(rebuilt.materials.eps_r), np.asarray(inputs.materials.eps_r))
    assert rebuilt.reason_text == inputs.reason_text




def test_phase1_input_builder_cached_report_and_bundle_surface():
    sim, inputs = _make_supported_phase1_inputs()
    assert inputs.inspect() is inputs.report
    assert inputs.prepare() is inputs.prepared_bundle




def test_phase1_inspect_from_inputs_matches_direct_inspect():
    sim, inputs = _make_supported_phase1_inputs()

    via_inputs = sim.inspect_hybrid_phase1_from_inputs(inputs)
    direct = sim.inspect_hybrid_phase1(n_steps=18)

    assert via_inputs == direct



def test_phase1_inspect_from_inputs_matches_direct_inspect_for_unsupported_nonuniform():
    sim = _make_nonuniform_unsupported_phase1_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    via_inputs = sim.inspect_hybrid_phase1_from_inputs(inputs)
    direct = sim.inspect_hybrid_phase1(n_steps=12)

    assert via_inputs == direct




def test_phase1_prepare_from_inputs_matches_direct_prepare_surface():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs()
    via_inputs = sim.prepare_hybrid_phase1_from_inputs(inputs)
    direct = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert via_inputs.report == direct.report
    assert via_inputs.context is not None
    assert direct.context is not None
    np.testing.assert_allclose(np.asarray(via_inputs.context.eps_r), np.asarray(direct.context.eps_r))



def test_phase1_context_from_inputs_matches_direct_context_surface():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs()
    via_inputs = sim.build_hybrid_phase1_context_from_inputs(inputs)
    direct = sim.build_hybrid_phase1_context(n_steps=n_steps)

    assert via_inputs.inventory == direct.inventory
    np.testing.assert_allclose(np.asarray(via_inputs.eps_r), np.asarray(direct.eps_r))



def test_phase1_prepare_from_inputs_preserves_unsupported_nonuniform_case():
    sim = _make_nonuniform_unsupported_phase1_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    via_inputs = sim.prepare_hybrid_phase1_from_inputs(inputs)
    direct = sim.prepare_hybrid_phase1(n_steps=12)

    assert via_inputs.report == direct.report
    assert via_inputs.context is None
    assert direct.context is None



def test_phase1_context_from_inputs_rejects_unsupported_nonuniform_case():
    sim = _make_nonuniform_unsupported_phase1_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        sim.build_hybrid_phase1_context_from_inputs(inputs)




def test_phase1_input_surface_helper_aliases_match_methods():
    sim, inputs = _make_supported_phase1_inputs()

    from rfx.hybrid_adjoint import (
        inspect_phase1_hybrid_from_inputs,
        prepare_phase1_hybrid_from_inputs,
        build_phase1_hybrid_context_from_inputs,
        forward_phase1_hybrid_from_inputs,
    )

    assert inspect_phase1_hybrid_from_inputs(inputs) == inputs.inspect()
    assert prepare_phase1_hybrid_from_inputs(inputs) is inputs.prepare()
    assert build_phase1_hybrid_context_from_inputs(inputs) is inputs.require_context()
    np.testing.assert_allclose(
        np.asarray(forward_phase1_hybrid_from_inputs(inputs).time_series),
        np.asarray(inputs.forward_result().time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_input_forward_helper_with_eps_override_matches_method_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override = (
        _make_supported_phase1_inputs_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inputs

    via_helper = forward_phase1_hybrid_from_inputs(implicit_inputs, eps_override=eps_override)
    via_method = implicit_inputs.forward_result(eps_override=eps_override)
    via_explicit_method = explicit_inputs.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_explicit_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_input_surface_helper_aliases_match_methods_for_unsupported_nonuniform():
    sim = _make_nonuniform_unsupported_phase1_sim()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)

    from rfx.hybrid_adjoint import (
        inspect_phase1_hybrid_from_inputs,
        prepare_phase1_hybrid_from_inputs,
        build_phase1_hybrid_context_from_inputs,
        forward_phase1_hybrid_from_inputs,
    )

    assert inspect_phase1_hybrid_from_inputs(inputs) == inputs.inspect()
    assert prepare_phase1_hybrid_from_inputs(inputs).report == inputs.prepare().report
    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        build_phase1_hybrid_context_from_inputs(inputs)
    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        forward_phase1_hybrid_from_inputs(inputs)




def test_phase1_forward_from_inputs_matches_direct_forward():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs(n_steps=n_steps)

    direct = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    via_inputs = sim.forward_hybrid_phase1_from_inputs(inputs)

    np.testing.assert_allclose(
        np.asarray(via_inputs.time_series),
        np.asarray(direct.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_input_forward_api_with_eps_override_matches_method_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override = (
        _make_supported_phase1_inputs_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_inputs(
        implicit_inputs,
        eps_override=eps_override,
    )
    via_method = implicit_inputs.forward_result(eps_override=eps_override)
    via_explicit_method = explicit_inputs.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_input_forward_api_with_eps_override_matches_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override = (
        _make_supported_phase1_inputs_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_inputs(
        implicit_inputs,
        eps_override=eps_override,
    )
    via_implicit_inputs = implicit_inputs.run_time_series(eps_override=eps_override)
    via_explicit_inputs = explicit_inputs.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_implicit_inputs),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_inputs),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_input_forward_api_with_eps_override_matches_helper_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override = (
        _make_supported_phase1_inputs_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inputs

    via_api = sim.forward_hybrid_phase1_from_inputs(
        implicit_inputs,
        eps_override=eps_override,
    )
    via_helper = forward_phase1_hybrid_from_inputs(
        implicit_inputs,
        eps_override=eps_override,
    )
    via_explicit_helper = forward_phase1_hybrid_from_inputs(
        explicit_inputs,
        eps_override=eps_override,
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_matches_input_builder_path():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs(n_steps=n_steps)

    direct = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    via_inputs = inputs.forward_result()

    np.testing.assert_allclose(
        np.asarray(via_inputs.time_series),
        np.asarray(direct.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_input_builder_cached_context_surface():
    sim, inputs = _make_supported_phase1_inputs()
    context = inputs.require_context()

    assert context is inputs.context
    np.testing.assert_allclose(np.asarray(context.eps_r), np.asarray(inputs.context.eps_r))




def test_phase1_inspection_from_inputs_matches_input_report_surface():
    sim, inputs = _make_supported_phase1_inputs()

    from rfx.hybrid_adjoint import Phase1HybridInspection

    assert Phase1HybridInspection.from_inputs(inputs) == inputs.report



def test_phase1_inspection_from_inputs_matches_input_report_surface_for_unsupported_nonuniform():
    sim = _make_nonuniform_unsupported_phase1_sim()
    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)

    from rfx.hybrid_adjoint import Phase1HybridInspection

    assert Phase1HybridInspection.from_inputs(inputs) == inputs.report




def test_phase1_input_builder_metadata_and_reason_surface():
    sim, inputs = _make_supported_phase1_inputs()
    report = inputs.inspect()

    assert inputs.supported
    assert inputs.reason_text == ""
    assert inputs.inventory == report.inventory
    assert inputs.source_count == report.source_count == 1
    assert inputs.probe_count == report.probe_count == 1
    inputs.require_supported()




def test_phase1_input_builder_report_matches_public_inspect_surface():
    sim, inputs = _make_supported_phase1_inputs()
    public_report = sim.inspect_hybrid_phase1(n_steps=18)

    assert inputs.report == public_report



def test_phase1_input_builder_report_matches_public_inspect_surface_for_unsupported_nonuniform():
    sim = _make_nonuniform_unsupported_phase1_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    public_report = sim.inspect_hybrid_phase1(n_steps=12)

    assert inputs.report == public_report




def test_phase1_prepare_bundle_matches_input_builder_surface_for_unsupported_nonuniform():
    sim, via_public = _make_nonuniform_unsupported_prepared_bundle(n_steps=12)

    via_inputs = sim.build_hybrid_phase1_inputs(n_steps=12).prepare()

    assert via_inputs.report == via_public.report
    assert via_inputs.context is None
    assert via_public.context is None




def test_phase1_prepare_bundle_matches_public_surfaces():
    n_steps = 18
    sim, report, context = _make_supported_phase1_report_context()

    prepared = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert prepared.report == report
    assert prepared.supported
    assert prepared.inventory == context.inventory
    np.testing.assert_allclose(np.asarray(prepared.require_context().eps_r), np.asarray(context.eps_r))




def test_phase1_support_reasons_helper_accepts_supported_cpml_fixture():
    sim = _make_cpml_supported_phase1_sim()
    n_steps = 18
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=n_steps)

    from rfx import phase1_hybrid_support_reasons

    assert prepared_runner is not None
    reasons = phase1_hybrid_support_reasons(
        boundary="cpml",
        periodic=(False, False, False),
        materials=prepared_runner.materials,
        sources=prepared_runner.raw_phase1_sources,
        probes=prepared_runner.probes,
        debye=None,
        lorentz=None,
        ntff_box=None,
        waveguide_ports=None,
        pec_mask=None,
        pec_occupancy=None,
    )

    assert reasons == report.reasons
    assert report.supported
    assert report.reason_text == ""



def test_phase1_run_time_series_and_make_forward_helpers_match_context_surface():
    sim, context = _make_supported_phase1_context()

    from rfx import make_phase1_hybrid_forward, run_phase1_forward_time_series

    via_runner = run_phase1_forward_time_series(context, context.eps_r)
    via_factory = make_phase1_hybrid_forward(context)(context.eps_r)
    via_context = context.run_time_series()

    np.testing.assert_allclose(
        np.asarray(via_runner),
        np.asarray(via_context),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_factory),
        np.asarray(via_context),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_run_time_series_and_make_forward_helpers_with_eps_override_match_context_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    from rfx import make_phase1_hybrid_forward, run_phase1_forward_time_series

    via_implicit_runner = run_phase1_forward_time_series(implicit_context, eps_override)
    via_explicit_runner = run_phase1_forward_time_series(explicit_context, eps_override)
    via_factory = make_phase1_hybrid_forward(implicit_context)(eps_override)
    via_context = implicit_context.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_runner),
        np.asarray(via_explicit_runner),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_runner),
        np.asarray(via_factory),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_runner),
        np.asarray(via_context),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_runner),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_result_helper_matches_context_method():
    sim, context = _make_supported_phase1_context()

    via_helper = phase1_forward_result(context.grid, context.run_time_series())
    via_method = context.forward_result()

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_forward_result_helper_with_eps_override_matches_context_method_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    implicit_time_series = implicit_context.run_time_series(eps_override=eps_override)
    explicit_time_series = explicit_context.run_time_series(eps_override=eps_override)
    via_helper = phase1_forward_result(implicit_context.grid, implicit_time_series)
    via_method = implicit_context.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(implicit_time_series),
        np.asarray(explicit_time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_helper_from_context_matches_method_surface():
    sim, context = _make_supported_phase1_context()

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_context

    via_helper = forward_phase1_hybrid_from_context(context)
    via_method = context.forward_result()

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_context_forward_helper_with_eps_override_matches_method_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_context

    via_implicit_helper = forward_phase1_hybrid_from_context(
        implicit_context,
        eps_override=eps_override,
    )
    via_explicit_helper = forward_phase1_hybrid_from_context(
        explicit_context,
        eps_override=eps_override,
    )
    via_method = implicit_context.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_explicit_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_forward_result_matches_api_execution():
    sim, context = _make_supported_phase1_context()

    via_context = context.forward_result()
    via_api = sim.forward_hybrid_phase1_from_context(context)

    np.testing.assert_allclose(
        np.asarray(via_context.time_series),
        np.asarray(via_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_forward_helper_from_prepared_matches_method_surface():
    sim, prepared = _make_supported_phase1_prepared_bundle()

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared

    via_helper = forward_phase1_hybrid_from_prepared(prepared)
    via_method = prepared.forward_result()

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_prepared_forward_helper_with_eps_override_matches_method_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared

    via_implicit_helper = forward_phase1_hybrid_from_prepared(
        implicit_prepared,
        eps_override=eps_override,
    )
    via_explicit_helper = forward_phase1_hybrid_from_prepared(
        explicit_prepared,
        eps_override=eps_override,
    )
    via_method = implicit_prepared.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_explicit_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_forward_result_matches_api_execution():
    sim, prepared = _make_supported_phase1_prepared_bundle()

    via_prepared = prepared.forward_result()
    via_api = sim.forward_hybrid_phase1_from_prepared(prepared)

    np.testing.assert_allclose(
        np.asarray(via_prepared.time_series),
        np.asarray(via_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_forward_from_prepared_matches_context_execution():
    sim, prepared = _make_supported_phase1_prepared_bundle()
    _, context = _make_supported_phase1_context()

    via_prepared = sim.forward_hybrid_phase1_from_prepared(prepared)
    via_context = sim.forward_hybrid_phase1_from_context(context)
    ts_direct = prepared.run_time_series()

    np.testing.assert_allclose(
        np.asarray(via_prepared.time_series),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(ts_direct),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_input_builder_from_prepared_runner_state_matches_public_surface_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, _ = _inspect_supported_phase1_snapshot_for_periods(num_periods=num_periods)

    via_api = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_public = sim.build_hybrid_phase1_inputs(num_periods=num_periods)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(via_api.require_context().run_time_series()), np.asarray(via_public.require_context().run_time_series()))



def test_phase1_api_input_builder_from_prepared_runner_state_matches_public_surface():
    sim = _make_phase1_sim()
    n_steps = 18
    grid, prepared_runner, _ = sim._inspect_hybrid_phase1_prepared(n_steps=n_steps)
    assert prepared_runner is not None

    via_api = sim.build_hybrid_phase1_inputs_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_public = sim.build_hybrid_phase1_inputs(n_steps=n_steps)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))



def test_phase1_top_level_prepare_and_context_match_when_n_steps_is_omitted_on_supported_path():
    num_periods = 8.0
    sim, prepared, context, _explicit_prepared, _explicit_context, _explicit_n_steps = (
        _make_supported_phase1_prepare_context_case(num_periods=num_periods)
    )

    assert prepared.supported
    assert prepared.report.inventory == context.inventory
    np.testing.assert_allclose(np.asarray(prepared.require_context().eps_r), np.asarray(context.eps_r))
    np.testing.assert_allclose(np.asarray(prepared.require_context().run_time_series()), np.asarray(context.run_time_series()), rtol=1e-6, atol=1e-12)



def test_phase1_prepared_and_context_execution_match_explicit_n_steps_when_omitted():
    num_periods = 8.0
    (
        sim,
        implicit_prepared,
        implicit_context,
        explicit_prepared,
        explicit_context,
        explicit_n_steps,
    ) = _make_supported_phase1_prepare_context_case(num_periods=num_periods)
    implicit_via_prepared = sim.forward_hybrid_phase1_from_prepared(implicit_prepared)
    implicit_via_context = sim.forward_hybrid_phase1_from_context(implicit_context)
    explicit_via_prepared = sim.forward_hybrid_phase1_from_prepared(explicit_prepared)
    explicit_via_context = sim.forward_hybrid_phase1_from_context(explicit_context)

    assert implicit_prepared.report == explicit_prepared.report
    np.testing.assert_allclose(np.asarray(implicit_context.eps_r), np.asarray(explicit_context.eps_r))
    np.testing.assert_allclose(np.asarray(implicit_via_prepared.time_series), np.asarray(explicit_via_prepared.time_series), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(implicit_via_context.time_series), np.asarray(explicit_via_context.time_series), rtol=1e-6, atol=1e-12)



def test_phase1_top_level_input_builder_matches_explicit_n_steps_when_omitted_on_supported_path():
    _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(_make_phase1_sim())



def test_phase1_top_level_forward_supported_semantics_match_explicit_n_steps_when_omitted():
    _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(_make_phase1_sim())



def test_phase1_top_level_prepare_bundle_supported_semantics_match_explicit_n_steps_when_omitted():
    _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(_make_phase1_sim())



def test_phase1_top_level_public_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(_make_phase1_sim())


def test_phase1_top_level_inspect_with_eps_override_matches_explicit_n_steps_when_omitted():
    num_periods = 8.0
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )

    implicit_report = sim.inspect_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    explicit_report = sim.inspect_hybrid_phase1(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )

    assert implicit_report == explicit_report
    assert implicit_report.supported
    assert implicit_report.reason_text == ""
    assert implicit_report.inventory is not None


def test_phase1_top_level_input_builder_with_eps_override_matches_explicit_n_steps_when_omitted():
    num_periods = 8.0
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )

    implicit_inputs = sim.build_hybrid_phase1_inputs(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    explicit_inputs = sim.build_hybrid_phase1_inputs(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )

    assert implicit_inputs.report == explicit_inputs.report
    assert implicit_inputs.supported
    assert implicit_inputs.reason_text == ""
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(eps_override),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().eps_r),
        np.asarray(explicit_inputs.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_inputs.require_context().run_time_series()),
        np.asarray(explicit_inputs.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_top_level_prepare_and_context_with_eps_override_match_explicit_n_steps_when_omitted():
    num_periods = 8.0
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )

    implicit_prepared = sim.prepare_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    implicit_context = sim.build_hybrid_phase1_context(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    explicit_prepared = sim.prepare_hybrid_phase1(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )
    explicit_context = sim.build_hybrid_phase1_context(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )

    assert implicit_prepared.report == explicit_prepared.report
    assert implicit_prepared.supported
    assert implicit_prepared.reason_text == ""
    np.testing.assert_allclose(
        np.asarray(implicit_prepared.require_context().eps_r),
        np.asarray(eps_override),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_prepared.require_context().eps_r),
        np.asarray(implicit_context.eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_prepared.require_context().eps_r),
        np.asarray(explicit_prepared.require_context().eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_context.eps_r),
        np.asarray(explicit_context.eps_r),
    )
    np.testing.assert_allclose(
        np.asarray(implicit_prepared.require_context().run_time_series()),
        np.asarray(explicit_prepared.require_context().run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(implicit_context.run_time_series()),
        np.asarray(explicit_context.run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_execution_with_eps_override_matches_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )

    via_implicit_prepared = sim.forward_hybrid_phase1_from_prepared(
        implicit_prepared,
        eps_override=eps_override,
    )
    via_explicit_prepared = sim.forward_hybrid_phase1_from_prepared(
        explicit_prepared,
        eps_override=eps_override,
    )
    via_prepared_method = implicit_prepared.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_explicit_prepared.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_prepared_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_forward_api_with_eps_override_matches_helper_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared

    via_api = sim.forward_hybrid_phase1_from_prepared(
        implicit_prepared,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_prepared(
        explicit_prepared,
        eps_override=eps_override,
    )
    via_helper = forward_phase1_hybrid_from_prepared(
        implicit_prepared,
        eps_override=eps_override,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_forward_api_with_eps_override_matches_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_prepared(
        implicit_prepared,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_prepared(
        explicit_prepared,
        eps_override=eps_override,
    )
    via_run_time_series = implicit_prepared.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_run_time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_run_time_series_with_eps_override_matches_context_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )
    implicit_context = sim.build_hybrid_phase1_context(num_periods=num_periods)

    via_implicit_prepared = implicit_prepared.run_time_series(eps_override=eps_override)
    via_explicit_prepared = explicit_prepared.run_time_series(eps_override=eps_override)
    via_context = implicit_context.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared),
        np.asarray(via_explicit_prepared),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared),
        np.asarray(via_context),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_top_level_forward_with_eps_override_matches_explicit_n_steps_when_omitted():
    num_periods = 8.0
    sim, explicit_n_steps, eps_override = _make_supported_phase1_eps_override_case(
        num_periods=num_periods
    )

    implicit_report = sim.inspect_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
    )
    explicit_report = sim.inspect_hybrid_phase1(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
    )
    implicit_forward = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )
    explicit_forward = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        n_steps=explicit_n_steps,
        fallback="raise",
    )

    assert implicit_report == explicit_report
    assert implicit_report.supported
    assert implicit_report.reason_text == ""
    np.testing.assert_allclose(
        np.asarray(implicit_forward.time_series),
        np.asarray(explicit_forward.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_forward_api_with_eps_override_matches_helper_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_context

    via_api = sim.forward_hybrid_phase1_from_context(
        implicit_context,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_context(
        explicit_context,
        eps_override=eps_override,
    )
    via_helper = forward_phase1_hybrid_from_context(
        implicit_context,
        eps_override=eps_override,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_forward_api_with_eps_override_matches_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_context(
        implicit_context,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_context(
        explicit_context,
        eps_override=eps_override,
    )
    via_run_time_series = implicit_context.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_run_time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_execution_with_eps_override_matches_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    via_implicit_context = sim.forward_hybrid_phase1_from_context(
        implicit_context,
        eps_override=eps_override,
    )
    via_explicit_context = sim.forward_hybrid_phase1_from_context(
        explicit_context,
        eps_override=eps_override,
    )
    via_context_method = implicit_context.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_context.time_series),
        np.asarray(via_explicit_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_context.time_series),
        np.asarray(via_context_method.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_context.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_forward_result_with_eps_override_matches_explicit_n_steps_and_top_level_when_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    via_implicit_context = implicit_context.forward_result(eps_override=eps_override)
    via_explicit_context = explicit_context.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_context.time_series),
        np.asarray(via_explicit_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_context.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_context_run_time_series_with_eps_override_matches_forward_result_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    via_implicit_context = implicit_context.run_time_series(eps_override=eps_override)
    via_explicit_context = explicit_context.run_time_series(eps_override=eps_override)
    via_forward_result = implicit_context.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_context),
        np.asarray(via_explicit_context),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_context),
        np.asarray(via_forward_result.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_context),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_prepared_forward_result_with_eps_override_matches_run_time_series_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_prepared, explicit_prepared, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_eps_override_case(num_periods=num_periods)
    )

    via_implicit_prepared = implicit_prepared.forward_result(eps_override=eps_override)
    via_explicit_prepared = explicit_prepared.forward_result(eps_override=eps_override)
    via_run_time_series = implicit_prepared.run_time_series(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_explicit_prepared.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_run_time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_prepared.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_api_runner_state_wrappers_match_public_surface_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report = _inspect_supported_phase1_snapshot_for_periods(num_periods=num_periods)

    n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    expected_report = sim.inspect_hybrid_phase1(n_steps=n_steps)
    expected_prepared = sim.prepare_hybrid_phase1(n_steps=n_steps)
    expected_context = sim.build_hybrid_phase1_context(n_steps=n_steps)
    expected_forward = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")

    via_prepared_report = sim.inspect_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_prepared_prepared = sim.prepare_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_prepared_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_prepared_forward = sim.forward_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )

    via_inspected_report = sim.inspect_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_inspected_prepared = sim.prepare_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_inspected_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_inspected_forward = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )

    assert via_prepared_report == expected_report
    assert via_inspected_report == expected_report
    assert via_prepared_prepared.report == expected_prepared.report
    assert via_inspected_prepared.report == expected_prepared.report
    np.testing.assert_allclose(np.asarray(via_prepared_context.eps_r), np.asarray(expected_context.eps_r))
    np.testing.assert_allclose(np.asarray(via_inspected_context.eps_r), np.asarray(expected_context.eps_r))
    np.testing.assert_allclose(np.asarray(via_prepared_forward.time_series), np.asarray(expected_forward.time_series), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(via_inspected_forward.time_series), np.asarray(expected_forward.time_series), rtol=1e-6, atol=1e-12)



def test_phase1_api_inspect_from_prepared_runner_state_matches_public_surface():
    sim, grid, prepared_runner, _report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.inspect_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_public = sim.inspect_hybrid_phase1(n_steps=n_steps)

    assert via_api == via_public



def test_phase1_api_prepare_from_prepared_runner_state_matches_public_surface():
    sim, grid, prepared_runner, _report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.prepare_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_public = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))



def test_phase1_api_context_from_prepared_runner_state_with_eps_override_matches_public_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    via_api = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_explicit_api = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=explicit_n_steps,
    )
    via_top_level = sim.build_hybrid_phase1_context(num_periods=num_periods)

    np.testing.assert_allclose(
        np.asarray(via_api.run_time_series(eps_override=eps_override)),
        np.asarray(via_explicit_api.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.run_time_series(eps_override=eps_override)),
        np.asarray(via_top_level.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_context_from_prepared_runner_state_matches_public_surface():
    sim, grid, prepared_runner, _report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_public = sim.build_hybrid_phase1_context(n_steps=n_steps)

    np.testing.assert_allclose(np.asarray(via_api.eps_r), np.asarray(via_public.eps_r))
    np.testing.assert_allclose(np.asarray(via_api.run_time_series()), np.asarray(via_public.run_time_series()))



def test_phase1_api_input_builder_from_inspected_runner_state_matches_public_surface_when_n_steps_is_inferred():
    num_periods = 8.0
    sim, grid, prepared_runner, report = _inspect_supported_phase1_snapshot_for_periods(
        num_periods=num_periods
    )

    resolved_n_steps = _resolved_n_steps(sim, num_periods=num_periods)
    via_api = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=resolved_n_steps,
    )
    via_public = sim.build_hybrid_phase1_inputs(num_periods=num_periods)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(via_api.require_context().run_time_series()), np.asarray(via_public.require_context().run_time_series()))



def test_phase1_api_input_builder_from_inspected_runner_state_matches_public_surface_when_n_steps_kwarg_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report = _inspect_supported_phase1_snapshot_for_periods(
        num_periods=num_periods
    )

    via_api = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        num_periods=num_periods,
    )
    via_public = sim.build_hybrid_phase1_inputs(num_periods=num_periods)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(via_api.require_context().run_time_series()), np.asarray(via_public.require_context().run_time_series()))



def test_phase1_api_input_builder_from_inspected_runner_state_matches_public_surface_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report = _inspect_supported_phase1_snapshot_for_periods(
        num_periods=num_periods
    )

    via_api = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_public = sim.build_hybrid_phase1_inputs(num_periods=num_periods)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))
    np.testing.assert_allclose(np.asarray(via_api.require_context().run_time_series()), np.asarray(via_public.require_context().run_time_series()))



def test_phase1_api_input_builder_from_inspected_runner_state_matches_public_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )
    via_public = sim.build_hybrid_phase1_inputs(n_steps=n_steps)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))



def test_phase1_api_inspect_from_inspected_runner_state_matches_public_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.inspect_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )
    via_public = sim.inspect_hybrid_phase1(n_steps=n_steps)

    assert via_api == via_public



def test_phase1_api_prepare_from_inspected_runner_state_matches_public_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.prepare_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )
    via_public = sim.prepare_hybrid_phase1(n_steps=n_steps)

    assert via_api.report == via_public.report
    np.testing.assert_allclose(np.asarray(via_api.require_context().eps_r), np.asarray(via_public.require_context().eps_r))



def test_phase1_api_context_from_inspected_runner_state_with_eps_override_matches_public_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report, explicit_n_steps, eps_override = (
        _make_supported_phase1_inspected_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    via_api = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_explicit_api = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=explicit_n_steps,
    )
    via_top_level = sim.build_hybrid_phase1_context(num_periods=num_periods)

    np.testing.assert_allclose(
        np.asarray(via_api.run_time_series(eps_override=eps_override)),
        np.asarray(via_explicit_api.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.run_time_series(eps_override=eps_override)),
        np.asarray(via_top_level.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_context_from_inspected_runner_state_matches_public_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )
    via_public = sim.build_hybrid_phase1_context(n_steps=n_steps)

    np.testing.assert_allclose(np.asarray(via_api.eps_r), np.asarray(via_public.eps_r))
    np.testing.assert_allclose(np.asarray(via_api.run_time_series()), np.asarray(via_public.run_time_series()))



def test_phase1_api_input_builder_from_inspected_runner_state_preserves_unsupported_nonuniform_case():
    n_steps = 18
    sim, grid, prepared_runner, report = _inspect_nonuniform_inspected_runner_unsupported_phase1(n_steps=n_steps)

    via_api = sim.build_hybrid_phase1_inputs_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )

    assert not via_api.supported
    assert via_api.reason_text == report.reason_text
    assert any("non-uniform grids are unsupported" in reason for reason in via_api.reasons)



def test_phase1_api_context_from_inspected_runner_state_rejects_unsupported_nonuniform_case():
    n_steps = 18
    sim, grid, prepared_runner, report = _inspect_nonuniform_inspected_runner_unsupported_phase1(n_steps=n_steps)

    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        sim.build_hybrid_phase1_context_from_inspected_runner_state(
            grid,
            prepared_runner,
            report,
            n_steps=n_steps,
        )



def test_phase1_api_forward_from_prepared_runner_state_matches_context_surface():
    sim, grid, prepared_runner, _report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.forward_hybrid_phase1_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=n_steps,
    )
    via_context = sim.forward_hybrid_phase1_from_context(
        sim.build_hybrid_phase1_context(n_steps=n_steps)
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_api_forward_from_prepared_runner_state_with_eps_override_matches_helper_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, prepared_grid, prepared_runner, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    via_api = sim.forward_hybrid_phase1_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared_runner_state

    via_helper = forward_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=prepared_grid,
        prepared=prepared_runner,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_forward_from_prepared_runner_state_with_eps_override_matches_context_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, prepared_grid, prepared_runner, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    via_api = sim.forward_hybrid_phase1_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=explicit_n_steps,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(implicit_context.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(explicit_context.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_forward_from_prepared_runner_state_with_eps_override_matches_context_forward_result_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, prepared_grid, prepared_runner, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    via_api = sim.forward_hybrid_phase1_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        prepared_grid,
        prepared_runner,
        n_steps=explicit_n_steps,
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(implicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(explicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_helper_from_prepared_runner_state_matches_context_surface():
    sim, grid, prepared_runner, _report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared_runner_state

    via_helper = forward_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    via_context = sim.forward_hybrid_phase1_from_context(
        sim.build_hybrid_phase1_context(n_steps=n_steps)
    )

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_forward_helper_from_prepared_runner_state_with_eps_override_matches_derived_context_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, explicit_n_steps, eps_override = (
        _make_supported_phase1_prepared_runner_eps_override_case(
            num_periods=num_periods
        )
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared_runner_state

    via_helper = forward_phase1_hybrid_from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_prepared_runner_state(
        grid,
        prepared_runner,
        n_steps=None,
        num_periods=num_periods,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(implicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_forward_from_inspected_runner_state_matches_context_surface_when_n_steps_is_omitted():
    sim = _make_phase1_sim()
    num_periods = 8.0
    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(num_periods=num_periods)
    assert prepared_runner is not None

    via_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
    )
    via_public = sim.forward_hybrid_phase1(n_steps=_resolved_n_steps(sim, num_periods=num_periods), fallback="raise")

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_public.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_api_forward_from_inspected_runner_state_matches_context_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    via_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=n_steps,
    )
    via_context = sim.forward_hybrid_phase1_from_context(
        sim.build_hybrid_phase1_context(n_steps=n_steps)
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_api_forward_from_inspected_runner_state_with_eps_override_matches_helper_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report, explicit_n_steps, eps_override = (
        _make_supported_phase1_inspected_runner_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        eps_override=eps_override,
    )
    via_explicit_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inspected_runner_state

    via_helper = forward_phase1_hybrid_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_explicit_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_helper.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_forward_from_inspected_runner_state_with_eps_override_matches_context_run_time_series_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report, explicit_n_steps, eps_override = (
        _make_supported_phase1_inspected_runner_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=explicit_n_steps,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(implicit_context.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(explicit_context.run_time_series(eps_override=eps_override)),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_api_forward_from_inspected_runner_state_with_eps_override_matches_context_forward_result_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report, explicit_n_steps, eps_override = (
        _make_supported_phase1_inspected_runner_eps_override_case(num_periods=num_periods)
    )

    via_api = sim.forward_hybrid_phase1_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    explicit_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=explicit_n_steps,
    )

    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(implicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_api.time_series),
        np.asarray(explicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_helper_from_inspected_runner_state_with_eps_override_matches_derived_context_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, grid, prepared_runner, report, explicit_n_steps, eps_override = (
        _make_supported_phase1_inspected_runner_eps_override_case(num_periods=num_periods)
    )

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inspected_runner_state

    via_helper = forward_phase1_hybrid_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=explicit_n_steps,
        eps_override=eps_override,
    )
    implicit_context = sim.build_hybrid_phase1_context_from_inspected_runner_state(
        grid,
        prepared_runner,
        report,
        n_steps=None,
        num_periods=num_periods,
    )
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(implicit_context.forward_result(eps_override=eps_override).time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_forward_helper_from_inspected_runner_state_matches_context_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inspected_runner_state

    via_helper = forward_phase1_hybrid_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    via_context = sim.forward_hybrid_phase1_from_context(
        sim.build_hybrid_phase1_context(n_steps=n_steps)
    )

    np.testing.assert_allclose(
        np.asarray(via_helper.time_series),
        np.asarray(via_context.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_prepare_bundle_reason_text_is_empty_when_supported():
    sim, prepared = _make_supported_phase1_prepared_bundle()

    assert prepared.reason_text == ""
    prepared.require_supported()




def test_phase1_prepare_bundle_metadata_passthroughs():
    sim, prepared = _make_supported_phase1_prepared_bundle()

    assert prepared.source_count == 1
    assert prepared.probe_count == 1
    assert prepared.boundary == "pec"
    assert prepared.periodic == (False, False, False)




def test_phase1_prepare_bundle_grid_and_eps_accessors():
    sim, prepared = _make_supported_phase1_prepared_bundle()

    assert prepared.grid is not None
    assert prepared.eps_r is not None
    assert prepared.grid.shape == prepared.eps_r.shape




def test_phase1_forward_matches_prepared_bundle_path():
    n_steps = 18
    sim, prepared = _make_supported_phase1_prepared_bundle(n_steps=n_steps)

    direct = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    via_prepared = sim.forward_hybrid_phase1_from_prepared(prepared)

    np.testing.assert_allclose(
        np.asarray(via_prepared.time_series),
        np.asarray(direct.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_context_helper_matches_context_classmethod_surface():
    sim, inputs = _make_supported_phase1_inputs()

    from rfx.hybrid_adjoint import Phase1HybridContext, build_phase1_hybrid_context

    via_class = Phase1HybridContext.from_inputs(inputs)
    via_func = build_phase1_hybrid_context(
        grid=inputs.grid,
        materials=inputs.materials,
        n_steps=inputs.n_steps,
        raw_sources=inputs.raw_sources,
        probes=inputs.probes,
        pec_axes=inputs.pec_axes,
    )

    assert via_class.inventory == via_func.inventory
    np.testing.assert_allclose(np.asarray(via_class.eps_r), np.asarray(via_func.eps_r))




def test_phase1_context_classmethod_from_inputs_matches_builder_surface():
    sim, inputs = _make_supported_phase1_inputs()
    _, public_context = _make_supported_phase1_context()

    from rfx.hybrid_adjoint import Phase1HybridContext

    rebuilt = Phase1HybridContext.from_inputs(inputs)

    assert rebuilt.inventory == public_context.inventory
    np.testing.assert_allclose(np.asarray(rebuilt.eps_r), np.asarray(public_context.eps_r))



def test_phase1_context_classmethod_from_inspected_runner_state_matches_helper_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import (
        Phase1HybridContext,
        build_phase1_hybrid_context_from_inspected_runner_state,
    )

    via_class = Phase1HybridContext.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )
    via_helper = build_phase1_hybrid_context_from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=n_steps,
    )

    assert via_class.inventory == via_helper.inventory
    np.testing.assert_allclose(np.asarray(via_class.eps_r), np.asarray(via_helper.eps_r))




def test_phase1_context_classmethod_from_prepared_runner_state_matches_helper_surface():
    sim, grid, prepared_runner, report, n_steps = _make_supported_phase1_snapshot_case()

    from rfx.hybrid_adjoint import Phase1HybridContext, build_phase1_hybrid_context

    via_class = Phase1HybridContext.from_prepared_runner_state(
        boundary="pec",
        grid=grid,
        prepared=prepared_runner,
        n_steps=n_steps,
    )
    via_func = build_phase1_hybrid_context(
        grid=grid,
        materials=prepared_runner.materials,
        n_steps=n_steps,
        raw_sources=prepared_runner.raw_phase1_sources,
        probes=prepared_runner.probes,
        pec_axes=prepared_runner.pec_axes_run,
    )

    assert via_class.inventory == via_func.inventory
    np.testing.assert_allclose(np.asarray(via_class.eps_r), np.asarray(via_func.eps_r))




def test_phase1_context_builder_matches_inspection_inventory():
    n_steps = 18
    sim, report, context = _make_supported_phase1_report_context()

    assert report.supported
    assert report.inventory is not None
    assert context.inventory == report.inventory
    hybrid = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    rebuilt = sim.forward_hybrid_phase1_from_context(
        context,
        eps_override=jnp.ones(context.grid.shape, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(rebuilt.time_series),
        np.asarray(hybrid.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_context_builder_preserves_lossless_dielectric_baseline():
    sim = _make_phase1_sim()
    sim.add_material("dielectric", eps_r=2.5)
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="dielectric")
    n_steps = 18

    context = sim.build_hybrid_phase1_context(n_steps=n_steps)
    direct = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")
    from_context = sim.forward_hybrid_phase1_from_context(context)

    np.testing.assert_allclose(
        np.asarray(from_context.time_series),
        np.asarray(direct.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    assert float(jnp.max(context.eps_r)) > 1.0




def test_phase1_context_run_time_series_matches_api_execution():
    sim, context = _make_supported_phase1_context()

    via_context = context.run_time_series()
    via_api = sim.forward_hybrid_phase1_from_context(context)

    np.testing.assert_allclose(
        np.asarray(via_context),
        np.asarray(via_api.time_series),
        rtol=1e-6,
        atol=1e-12,
    )




def test_phase1_forward_from_context_rejects_shape_mismatch():
    sim, context = _make_supported_phase1_context()

    with pytest.raises(ValueError, match="shape"):
        sim.forward_hybrid_phase1_from_context(
            context,
            eps_override=jnp.ones((2, 2, 2), dtype=jnp.float32),
        )




def test_phase1_hybrid_forward_matches_pure_forward():
    _assert_hybrid_forward_matches_pure_forward(_make_phase1_sim())


def test_phase1_hybrid_cpml_forward_matches_pure_forward():
    _assert_hybrid_forward_matches_pure_forward(_make_cpml_supported_phase1_sim())


def test_phase1_hybrid_gradient_matches_pure_ad_to_1e4():
    sim, grid, base_materials, n_steps = _make_supported_phase1_grid_materials()
    _assert_single_cell_hybrid_gradient_matches_pure_ad(
        sim,
        grid,
        base_materials,
        n_steps=n_steps,
    )


def test_phase1_hybrid_cpml_gradient_matches_pure_ad_to_1e4():
    sim, grid, base_materials, n_steps = _make_supported_phase1_grid_materials(boundary="cpml")
    _assert_single_cell_hybrid_gradient_matches_pure_ad(
        sim,
        grid,
        base_materials,
        n_steps=n_steps,
    )


def test_phase1_hybrid_replay_is_deterministic():
    sim, grid, base_materials, n_steps = _make_supported_phase1_grid_materials()
    eps = _single_cell_eps(grid, base_materials.eps_r, jnp.float32(0.05))

    first = sim.forward_hybrid_phase1(eps_override=eps, n_steps=n_steps, fallback="raise")
    second = sim.forward_hybrid_phase1(eps_override=eps, n_steps=n_steps, fallback="raise")
    np.testing.assert_allclose(
        np.asarray(first.time_series),
        np.asarray(second.time_series),
        rtol=1e-6,
        atol=1e-12,
    )

    def hybrid_loss(alpha):
        eps_local = _single_cell_eps(grid, base_materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1(eps_override=eps_local, n_steps=n_steps, fallback="raise")
        return jnp.sum(result.time_series ** 2)

    g1 = jax.grad(hybrid_loss)(jnp.float32(0.05))
    g2 = jax.grad(hybrid_loss)(jnp.float32(0.05))
    np.testing.assert_allclose(float(g1), float(g2), rtol=1e-6, atol=1e-12)


def test_phase1_hybrid_cpml_with_pec_face_matches_pure_forward():
    _assert_hybrid_forward_matches_pure_forward(_make_cpml_supported_phase1_sim_with_pec_face())


def test_phase1_hybrid_gradient_matches_pure_ad_at_source_cell():
    sim, grid, base_materials, n_steps = _make_supported_phase1_grid_materials()

    def source_cell_eps(alpha):
        i, j, k = grid.position_to_index((0.005, 0.0075, 0.0075))
        return base_materials.eps_r.at[i, j, k].add(alpha)

    def pure_loss(alpha):
        result = sim.forward(eps_override=source_cell_eps(alpha), n_steps=n_steps, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def hybrid_loss(alpha):
        result = sim.forward_hybrid_phase1(
            eps_override=source_cell_eps(alpha),
            n_steps=n_steps,
            fallback="raise",
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_hybrid = jax.grad(hybrid_loss)(alpha0)
    rel_err = float(jnp.abs(grad_hybrid - grad_pure) / jnp.maximum(jnp.abs(grad_pure), 1e-12))

    assert rel_err <= 1e-4, (
        f"source-cell hybrid gradient drifted from pure AD: pure={float(grad_pure):.6e}, "
        f"hybrid={float(grad_hybrid):.6e}, rel_err={rel_err:.6e}"
    )


def test_phase1_hybrid_rejects_invalid_fallback_value():
    sim = _make_phase1_sim()

    with pytest.raises(ValueError, match="fallback must be 'pure_ad' or 'raise'"):
        sim.forward_hybrid_phase1(n_steps=12, fallback="bogus")



def test_phase1_hybrid_raise_error_matches_public_report_reason_text_for_unsupported_cases():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_raise_matches_report_reason_text(
            sim,
            lambda s, *, n_steps=None, num_periods=8.0: s.forward_hybrid_phase1(
                n_steps=n_steps,
                num_periods=num_periods,
                fallback="raise",
            ),
            num_periods=8.0,
        )



def test_phase1_hybrid_raise_error_matches_public_report_reason_text_for_unsupported_cases_with_explicit_n_steps():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_raise_matches_report_reason_text(
            sim,
            lambda s, *, n_steps=None, num_periods=8.0: s.forward_hybrid_phase1(
                n_steps=n_steps,
                num_periods=num_periods,
                fallback="raise",
            ),
            n_steps=_resolved_n_steps(sim, num_periods=8.0),
        )



def test_phase1_top_level_cpml_supported_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_supported_family_matches_explicit_n_steps_when_omitted(
        _make_cpml_supported_phase1_sim(),
    )


def test_phase1_hybrid_debye_raise_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_debye_unsupported_phase1_sim(),
        expected_error="Debye",
        fallback="raise",
    )



def test_phase1_hybrid_debye_fallback_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_debye_unsupported_phase1_sim(),
        expected_error="Debye",
        fallback="pure_ad",
    )



def test_phase1_hybrid_rejects_or_falls_back_on_debye():
    sim = _make_debye_unsupported_phase1_sim()
    n_steps = 12

    with pytest.raises(ValueError, match="Debye"):
        sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")

    fallback = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="pure_ad")
    baseline = sim.forward(n_steps=n_steps, checkpoint=True)
    np.testing.assert_allclose(
        np.asarray(fallback.time_series),
        np.asarray(baseline.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_nonuniform_unsupported_classmethods_match_helper_functions():
    from rfx.hybrid_adjoint import (
        Phase1HybridInspection,
        Phase1HybridPrepared,
        Phase1HybridInputs,
        unsupported_phase1_hybrid_nonuniform,
        unsupported_phase1_hybrid_nonuniform_report,
        unsupported_phase1_hybrid_nonuniform_inputs,
    )

    report_cls = Phase1HybridInspection.nonuniform_unsupported(probe_count=1, boundary="pec")
    prepared_cls = Phase1HybridPrepared.nonuniform_unsupported(probe_count=1, boundary="pec")
    inputs_cls = Phase1HybridInputs.nonuniform_unsupported(probe_count=1, boundary="pec")

    assert report_cls == unsupported_phase1_hybrid_nonuniform_report(probe_count=1, boundary="pec")
    assert prepared_cls.report == unsupported_phase1_hybrid_nonuniform(probe_count=1, boundary="pec").report
    assert inputs_cls.reason_text == unsupported_phase1_hybrid_nonuniform_inputs(probe_count=1, boundary="pec").reason_text




def test_phase1_nonuniform_unsupported_helpers_match_public_surfaces():
    sim = _make_nonuniform_unsupported_phase1_sim()

    from rfx.hybrid_adjoint import (
        unsupported_phase1_hybrid_nonuniform,
        unsupported_phase1_hybrid_nonuniform_report,
    )

    report = unsupported_phase1_hybrid_nonuniform_report(
        probe_count=1,
        boundary="pec",
    )
    prepared = unsupported_phase1_hybrid_nonuniform(
        probe_count=1,
        boundary="pec",
    )
    public_report = sim.inspect_hybrid_phase1(n_steps=12)
    public_prepared = sim.prepare_hybrid_phase1(n_steps=12)

    assert report == public_report
    assert prepared.report == public_prepared.report
    assert prepared.context is None




def test_phase1_prepared_from_inspected_runner_state_preserves_nonuniform_unsupported_case():
    from rfx.hybrid_adjoint import Phase1HybridPrepared

    sim = _make_nonuniform_unsupported_phase1_sim()

    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=12)
    rebuilt = Phase1HybridPrepared.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=None,
    )

    assert not rebuilt.supported
    assert rebuilt.reason_text == "non-uniform grids are unsupported"




def test_phase1_private_nonuniform_prep_report_matches_canonical_helper_surface():
    sim = _make_nonuniform_unsupported_phase1_sim()

    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=12)

    from rfx.hybrid_adjoint import unsupported_phase1_hybrid_nonuniform_report

    assert prepared_runner is None
    assert report == unsupported_phase1_hybrid_nonuniform_report(probe_count=1, boundary="pec")
    assert grid is None



def test_phase1_hybrid_nonuniform_raise_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_nonuniform_unsupported_phase1_sim(),
        expected_error="non-uniform grids are unsupported",
        fallback="raise",
    )



def test_phase1_hybrid_nonuniform_fallback_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_nonuniform_unsupported_phase1_sim(),
        expected_error="non-uniform grids are unsupported",
        fallback="pure_ad",
    )



def test_phase1_top_level_nonuniform_public_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_public_family_matches_explicit_n_steps_when_omitted(
        _make_nonuniform_unsupported_phase1_sim(),
        expected_error="non-uniform grids are unsupported",
    )



def test_phase1_top_level_lumped_port_public_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_public_family_matches_explicit_n_steps_when_omitted(
        _make_lumped_port_unsupported_phase1_sim(),
        expected_error="add_source",
    )



def test_phase1_top_level_debye_public_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_public_family_matches_explicit_n_steps_when_omitted(
        _make_debye_unsupported_phase1_sim(),
        expected_error="Debye",
    )



def test_phase1_top_level_cpml_lossy_public_family_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_public_family_matches_explicit_n_steps_when_omitted(
        _make_cpml_lossy_unsupported_phase1_sim(),
        expected_error="lossy materials",
    )



def test_phase1_top_level_context_raise_error_matches_public_report_reason_text_for_unsupported_cases_with_explicit_n_steps():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_raise_matches_report_reason_text(
            sim,
            lambda s, *, n_steps=None, num_periods=8.0: s.build_hybrid_phase1_context(
                n_steps=n_steps,
                num_periods=num_periods,
            ),
            n_steps=_resolved_n_steps(sim, num_periods=8.0),
        )



def test_phase1_top_level_context_raise_error_matches_public_report_reason_text_for_unsupported_cases():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_raise_matches_report_reason_text(
            sim,
            lambda s, *, n_steps=None, num_periods=8.0: s.build_hybrid_phase1_context(
                n_steps=n_steps,
                num_periods=num_periods,
            ),
            num_periods=8.0,
        )



def test_phase1_top_level_input_builder_matches_public_report_and_no_context_for_unsupported_cases_with_explicit_n_steps():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_unsupported_input_builder_contract(
            sim,
            n_steps=_resolved_n_steps(sim, num_periods=8.0),
        )



def test_phase1_top_level_input_builder_matches_public_report_and_no_context_for_unsupported_cases_when_n_steps_is_omitted():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_unsupported_input_builder_contract(sim, num_periods=8.0)



def test_phase1_top_level_prepare_bundle_preserves_report_reason_and_no_context_for_unsupported_cases_with_explicit_n_steps():
    for sim, expected in _unsupported_phase1_cases():
        _assert_top_level_unsupported_prepare_bundle_contract(
            sim,
            n_steps=_resolved_n_steps(sim, num_periods=8.0),
            expected_reason=expected,
        )



def test_phase1_top_level_prepare_bundle_reason_text_matches_public_report_for_unsupported_cases_when_n_steps_is_omitted():
    for sim, _ in _unsupported_phase1_cases():
        _assert_top_level_unsupported_prepare_bundle_contract(sim, num_periods=8.0)



def test_phase1_top_level_prepare_bundle_preserves_report_and_no_context_for_unsupported_cases_when_n_steps_is_omitted():
    for sim, expected in _unsupported_phase1_cases():
        _assert_top_level_unsupported_prepare_bundle_contract(
            sim,
            num_periods=8.0,
            expected_reason=expected,
        )



def test_phase1_prepare_bundle_reports_nonuniform_unsupported():
    sim, prepared = _make_nonuniform_unsupported_prepared_bundle(n_steps=12)
    assert not prepared.supported
    assert prepared.context is None
    assert prepared.reason_text == "non-uniform grids are unsupported"
    assert prepared.boundary == "pec"
    assert prepared.periodic == (False, False, False)




def test_phase1_hybrid_inspection_reports_lumped_port_unsupported():
    sim = _make_lumped_port_unsupported_phase1_sim()

    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert not report.supported
    assert report.inventory is None
    assert any("add_source()" in reason for reason in report.reasons)



def test_phase1_context_resolved_eps_r_rejects_shape_mismatch():
    _sim, context = _make_supported_phase1_context()

    with pytest.raises(ValueError, match="shape"):
        context.resolved_eps_r(jnp.ones((2, 2, 2), dtype=jnp.float32))




def test_phase1_context_resolved_eps_r_with_eps_override_matches_explicit_n_steps_when_omitted():
    num_periods = 8.0
    sim, implicit_context, explicit_context, explicit_n_steps, eps_override = (
        _make_supported_phase1_context_eps_override_case(num_periods=num_periods)
    )

    implicit_eps = implicit_context.resolved_eps_r(eps_override)
    explicit_eps = explicit_context.resolved_eps_r(eps_override)

    np.testing.assert_allclose(np.asarray(implicit_eps), np.asarray(eps_override))
    np.testing.assert_allclose(np.asarray(implicit_eps), np.asarray(explicit_eps))


def test_phase1_forward_helper_from_prepared_rejects_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_prepared
    with pytest.raises(ValueError, match="add_source"):
        forward_phase1_hybrid_from_prepared(prepared)




def test_phase1_prepare_bundle_forward_result_rejects_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    with pytest.raises(ValueError, match="add_source"):
        prepared.forward_result()




def test_phase1_prepare_bundle_run_time_series_rejects_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    with pytest.raises(ValueError, match="add_source"):
        prepared.run_time_series()




def test_phase1_prepare_bundle_metadata_passthroughs_on_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    _assert_unsupported_prepare_bundle_basics(prepared, expected_reason="add_source()")
    assert prepared.source_count == 0
    assert prepared.probe_count == 1
    assert prepared.boundary == "pec"
    assert prepared.periodic == (False, False, False)




def test_phase1_prepare_bundle_reason_text_is_canonical_for_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    _assert_unsupported_prepare_bundle_basics(prepared, expected_reason="add_source()")
    with pytest.raises(ValueError, match="add_source"):
        prepared.require_supported()




def test_phase1_prepare_bundle_require_context_rejects_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    _assert_unsupported_prepare_bundle_basics(prepared, expected_reason="add_source()")
    with pytest.raises(ValueError, match="add_source"):
        prepared.require_context()




def test_phase1_api_forward_from_inspected_runner_state_rejects_unsupported_nonuniform_case():
    n_steps = 18
    sim, grid, prepared_runner, report = _inspect_nonuniform_inspected_runner_unsupported_phase1(n_steps=n_steps)

    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        sim.forward_hybrid_phase1_from_inspected_runner_state(
            grid,
            prepared_runner,
            report,
            n_steps=n_steps,
        )



def test_phase1_forward_from_inspected_runner_state_rejects_unsupported_nonuniform_case():
    n_steps = 18
    sim, grid, prepared_runner, report = _inspect_nonuniform_inspected_runner_unsupported_phase1(n_steps=n_steps)

    from rfx.hybrid_adjoint import forward_phase1_hybrid_from_inspected_runner_state

    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        forward_phase1_hybrid_from_inspected_runner_state(
            boundary="pec",
            probe_count=1,
            grid=grid,
            prepared=prepared_runner,
            report=report,
            n_steps=n_steps,
        )



def test_phase1_forward_from_inputs_rejects_unsupported_nonuniform_case():
    sim = _make_nonuniform_unsupported_phase1_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=12)
    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        sim.forward_hybrid_phase1_from_inputs(inputs)




def test_phase1_input_from_inspected_runner_state_preserves_nonuniform_unsupported_case():
    sim = _make_nonuniform_unsupported_phase1_sim()

    grid, prepared_runner, report = sim._inspect_hybrid_phase1_prepared(n_steps=12)
    assert prepared_runner is None

    from rfx.hybrid_adjoint import Phase1HybridInputs
    rebuilt = Phase1HybridInputs.from_inspected_runner_state(
        boundary="pec",
        probe_count=1,
        grid=grid,
        prepared=prepared_runner,
        report=report,
        n_steps=None,
    )

    assert not rebuilt.supported
    assert rebuilt.reason_text == "non-uniform grids are unsupported"




def test_phase1_input_builder_preserves_nonuniform_unsupported_case():
    sim, inputs = _make_nonuniform_unsupported_inputs(n_steps=12)

    _, prepared = _make_nonuniform_unsupported_prepared_bundle(n_steps=12)

    _assert_unsupported_inputs_basics(inputs, expected_reason="non-uniform grids are unsupported")
    assert prepared.report.reason_text == inputs.reason_text
    with pytest.raises(ValueError, match="non-uniform grids are unsupported"):
        inputs.require_supported()




def test_phase1_input_builder_preserves_unsupported_uniform_case():
    sim, inputs = _make_lumped_port_unsupported_inputs(n_steps=12)

    prepared = inputs.prepare()
    _assert_unsupported_inputs_basics(inputs, expected_reason="add_source()")
    assert inputs.source_count == 0
    assert inputs.probe_count == 1
    assert not prepared.supported
    assert "add_source()" in prepared.reason_text




def test_phase1_input_builder_run_surface_matches_prepare_bundle():
    n_steps = 18
    sim, inputs = _make_supported_phase1_inputs(n_steps=n_steps)
    prepared = inputs.prepare()

    np.testing.assert_allclose(
        np.asarray(inputs.run_time_series()),
        np.asarray(prepared.run_time_series()),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(inputs.forward_result().time_series),
        np.asarray(prepared.forward_result().time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase1_input_run_time_series_with_eps_override_matches_forward_result_and_top_level_when_n_steps_is_omitted():
    num_periods = 8.0
    sim, implicit_inputs, explicit_inputs, explicit_n_steps, eps_override = (
        _make_supported_phase1_inputs_eps_override_case(num_periods=num_periods)
    )

    via_implicit_inputs = implicit_inputs.run_time_series(eps_override=eps_override)
    via_explicit_inputs = explicit_inputs.run_time_series(eps_override=eps_override)
    via_forward_result = implicit_inputs.forward_result(eps_override=eps_override)
    via_top_level = sim.forward_hybrid_phase1(
        eps_override=eps_override,
        num_periods=num_periods,
        fallback="raise",
    )

    np.testing.assert_allclose(
        np.asarray(via_implicit_inputs),
        np.asarray(via_explicit_inputs),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_inputs),
        np.asarray(via_forward_result.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(via_implicit_inputs),
        np.asarray(via_top_level.time_series),
        rtol=1e-6,
        atol=1e-12,
    )



def test_phase1_input_builder_reason_surface_for_unsupported_uniform_case():
    sim, inputs = _make_lumped_port_unsupported_inputs(n_steps=12)
    _assert_unsupported_inputs_basics(inputs, expected_reason="add_source()")
    with pytest.raises(ValueError, match="add_source"):
        inputs.require_supported()




def test_phase1_prepare_bundle_reports_unsupported_without_context():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    _assert_unsupported_prepare_bundle_basics(prepared, expected_reason="add_source()")
    assert not prepared.report.supported
    assert any("add_source()" in reason for reason in prepared.report.reasons)




def test_phase1_forward_from_prepared_rejects_unsupported_bundle():
    sim, prepared = _make_lumped_port_unsupported_prepared_bundle(n_steps=12)
    with pytest.raises(ValueError, match="add_source"):
        sim.forward_hybrid_phase1_from_prepared(prepared)




def test_phase1_context_builder_rejects_lumped_port():
    sim = _make_lumped_port_unsupported_phase1_sim()

    with pytest.raises(ValueError, match="add_source"):
        sim.build_hybrid_phase1_context(n_steps=12)



def test_phase1_hybrid_lumped_port_raise_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_lumped_port_unsupported_phase1_sim(),
        expected_error="add_source",
        fallback="raise",
    )



def test_phase1_hybrid_lumped_port_fallback_matches_explicit_n_steps_when_omitted():
    _assert_top_level_unsupported_forward_matches_explicit_n_steps_when_omitted(
        _make_lumped_port_unsupported_phase1_sim(),
        expected_error="add_source",
        fallback="pure_ad",
    )



def test_phase1_hybrid_rejects_or_falls_back_on_lumped_port():
    sim = _make_lumped_port_unsupported_phase1_sim()
    n_steps = 12

    with pytest.raises(ValueError, match="add_source"):
        sim.forward_hybrid_phase1(n_steps=n_steps, fallback="raise")

    fallback = sim.forward_hybrid_phase1(n_steps=n_steps, fallback="pure_ad")
    baseline = sim.forward(n_steps=n_steps, checkpoint=True)
    np.testing.assert_allclose(
        np.asarray(fallback.time_series),
        np.asarray(baseline.time_series),
        rtol=1e-6,
        atol=1e-12,
    )
