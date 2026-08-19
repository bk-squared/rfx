"""CPU-safe contract tests for the experimental Apple Metal lane."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.core.yee import init_materials
from rfx.grid import Grid

import rfx.api._execute as execute_module
import rfx.api._preflight as preflight_module
import rfx.api._sparams as sparams_module
import rfx.adi as adi_module
import rfx.backends as backend_module
import rfx.nonuniform as nonuniform_module
import rfx.probes.probes as probes_module
import rfx.runners.disjoint as disjoint_runner_module
import rfx.runners.distributed as distributed_runner_module
import rfx.runners.distributed_nu as distributed_nu_runner_module
import rfx.runners.distributed_v2 as distributed_v2_runner_module
import rfx.runners.nonuniform as nonuniform_runner_module
import rfx.runners.subgridded as subgridded_runner_module
import rfx.runners.uniform as uniform_runner_module
import rfx.simulation as simulation_module
import rfx.subgridding.jit_runner as subgridding_jit_module
import rfx.subgridding.runner as subgridding_runner_module
import rfx.topology as topology_module
import rfx.vmap_sweep as vmap_module


material_fit_module = importlib.import_module("rfx.differentiable_material_fit")


def _basic_simulation() -> Simulation:
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
    return sim


def test_metal_preflight_identifies_eligible_research_configuration(monkeypatch):
    monkeypatch.setattr(preflight_module, "is_metal_backend", lambda: True)

    report = _basic_simulation().preflight(check_resolution=False)

    issues = report.by_code("metal_research_backend")
    assert report.ok
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "research-only" in str(issues[0])
    assert "JAX_PLATFORMS=cpu" in str(issues[0])


def test_metal_preflight_rejects_complex_dft_configuration(monkeypatch):
    monkeypatch.setattr(preflight_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()
    sim.add_dft_plane_probe(
        axis="x",
        coordinate=0.01,
        component="ez",
        freqs=jnp.asarray([2e9], dtype=jnp.float32),
    )

    report = sim.preflight(check_resolution=False)

    issues = report.by_code("metal_backend_unsupported")
    assert not report.ok
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "DFT plane probes (complex64)" in str(issues[0])


def test_metal_run_guard_cannot_be_bypassed_with_skip_preflight(monkeypatch):
    monkeypatch.setattr(execute_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()
    sim.add_port((0.01, 0.01, 0.01), "ez")

    with pytest.raises(NotImplementedError, match="compute_s_params=False"):
        sim.run(n_steps=2, compute_s_params=True, skip_preflight=True)


def test_metal_forward_guard_blocks_before_reverse_mode_scan(monkeypatch):
    monkeypatch.setattr(execute_module, "is_metal_backend", lambda: True)

    with pytest.raises(NotImplementedError, match="reverse-mode AD"):
        _basic_simulation().forward(n_steps=2, skip_preflight=True)


def test_specialized_sparameter_calculators_fail_before_setup(monkeypatch):
    monkeypatch.setattr(sparams_module, "is_metal_backend", lambda: True)

    with pytest.raises(NotImplementedError, match="complex64/complex128"):
        _basic_simulation().compute_waveguide_s_matrix(n_steps=2)


def test_sparameter_preflight_never_reports_metal_as_ready(monkeypatch):
    monkeypatch.setattr(preflight_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()
    sim.add_port((0.01, 0.01, 0.01), "ez")

    report = sim.preflight_sparameters(calculator="run")

    assert not report.ok
    issues = report.by_code("metal_backend_unsupported")
    assert len(issues) == 1
    assert "complex64/complex128" in str(issues[0])


def test_low_level_metal_runner_rejects_spectral_arrays_before_setup(monkeypatch):
    monkeypatch.setattr(simulation_module, "is_metal_backend", lambda: True)
    grid = Grid(
        freq_max=1e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.01,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)

    with pytest.raises(NotImplementedError, match="DFT plane probes"):
        simulation_module.run(
            grid,
            materials,
            1,
            dft_planes=[object()],
        )


def test_low_level_decay_runner_reuses_metal_capability_fence(monkeypatch):
    monkeypatch.setattr(simulation_module, "is_metal_backend", lambda: True)
    grid = Grid(
        freq_max=1e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.01,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)

    with pytest.raises(NotImplementedError, match="DFT plane probes"):
        simulation_module.run_until_decay(
            grid,
            materials,
            max_steps=1,
            min_steps=1,
            check_interval=1,
            dft_planes=[object()],
        )


def test_low_level_metal_runner_rejects_transformed_inputs(monkeypatch):
    monkeypatch.setattr(simulation_module, "is_metal_backend", lambda: True)
    grid = Grid(
        freq_max=1e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.01,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)

    def loss(scale):
        traced_materials = materials._replace(eps_r=materials.eps_r * scale)
        return simulation_module.run(grid, traced_materials, 1).time_series.sum()

    with pytest.raises(NotImplementedError, match="automatic-differentiation"):
        jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float32))


@pytest.mark.parametrize("dtype", [np.float64, np.complex64])
def test_low_level_metal_runner_rejects_non_float32_array_leaves(
    monkeypatch, dtype,
):
    monkeypatch.setattr(simulation_module, "is_metal_backend", lambda: True)
    grid = Grid(
        freq_max=1e9,
        domain=(0.02, 0.02, 0.02),
        dx=0.01,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)._replace(
        eps_r=np.ones(grid.shape, dtype=dtype),
    )

    with pytest.raises(NotImplementedError, match="array dtype"):
        simulation_module.run(grid, materials, 1)


def test_public_optimization_bypasses_are_fenced_on_metal(monkeypatch):
    monkeypatch.setattr(topology_module, "is_metal_backend", lambda: True)
    monkeypatch.setattr(material_fit_module, "is_metal_backend", lambda: True)

    with pytest.raises(NotImplementedError, match="topology_optimize"):
        topology_module.topology_optimize(None, None, None)
    with pytest.raises(NotImplementedError, match="differentiable_material_fit"):
        material_fit_module.differentiable_material_fit(None, None, None)


def test_metal_vmap_dft_sweep_fails_before_complex_accumulator(monkeypatch):
    monkeypatch.setattr(vmap_module, "is_metal_backend", lambda: True)
    dummy_sim = type("DummySimulation", (), {"_dft_planes": [object()]})()

    with pytest.raises(NotImplementedError, match="complex64"):
        vmap_module.vmap_material_sweep(dummy_sim, "eps_r", [1.0])


def test_metal_vmap_is_outside_single_run_research_lane(monkeypatch):
    monkeypatch.setattr(vmap_module, "is_metal_backend", lambda: True)
    monkeypatch.setattr(execute_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()

    with pytest.raises(NotImplementedError, match="vmap/batched"):
        vmap_module.vmap_material_sweep(sim, "eps_r", [1.0])


def test_public_nonuniform_runners_are_fenced_on_metal(monkeypatch):
    monkeypatch.setattr(backend_module, "current_backend", lambda: "metal")

    with pytest.raises(NotImplementedError, match="run_nonuniform"):
        nonuniform_module.run_nonuniform(None, None, 1)
    with pytest.raises(NotImplementedError, match="run_nonuniform_until_decay"):
        nonuniform_module.run_nonuniform_until_decay(None, None)


def test_public_adi_runners_are_fenced_on_metal(monkeypatch):
    monkeypatch.setattr(backend_module, "current_backend", lambda: "metal")

    with pytest.raises(NotImplementedError, match="run_adi_2d"):
        adi_module.run_adi_2d(
            None, None, None, None, None, 0.0, 0.0, 0.0, 1,
        )
    with pytest.raises(NotImplementedError, match="run_adi_3d"):
        adi_module.run_adi_3d(
            None, None, None, None, None, None,
            None, None, 0.0, 0.0, 0.0, 0.0, 1,
        )


def test_direct_uniform_runner_applies_configuration_fence(monkeypatch):
    monkeypatch.setattr(uniform_runner_module, "is_metal_backend", lambda: True)
    monkeypatch.setattr(execute_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()
    sim.add_dft_plane_probe(
        axis="x",
        coordinate=0.01,
        component="ez",
        freqs=jnp.asarray([2e9], dtype=jnp.float32),
    )

    with pytest.raises(NotImplementedError, match="DFT plane probes"):
        uniform_runner_module.run_uniform(sim, n_steps=1)


def test_direct_uniform_runner_rejects_caller_supplied_float64_materials(
    monkeypatch,
):
    monkeypatch.setattr(uniform_runner_module, "is_metal_backend", lambda: True)
    monkeypatch.setattr(execute_module, "is_metal_backend", lambda: True)
    monkeypatch.setattr(simulation_module, "is_metal_backend", lambda: True)
    sim = _basic_simulation()
    grid = sim._build_grid()
    materials = init_materials(grid.shape)._replace(
        eps_r=np.ones(grid.shape, dtype=np.float64),
    )

    with pytest.raises(NotImplementedError, match="array dtype.*float64"):
        uniform_runner_module.run_uniform(
            sim,
            n_steps=1,
            grid=grid,
            base_materials=materials,
        )


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: probes_module.extract_s_matrix_wire(None, None, [], None),
        lambda: nonuniform_runner_module.run_nonuniform_path(None, n_steps=1),
        lambda: subgridded_runner_module.run_subgridded_path(
            None, None, None, None, 1,
        ),
        lambda: distributed_runner_module.run_distributed(None, n_steps=1),
        lambda: distributed_v2_runner_module.run_distributed(None, n_steps=1),
        lambda: distributed_nu_runner_module.run_nonuniform_distributed_pec(
            None, None, None, 1, n_devices=1,
        ),
        lambda: disjoint_runner_module.run_disjoint_stage2_path(None, None, 1),
        lambda: subgridding_runner_module.run_subgridded(
            None, None, None, None, None, 1,
        ),
        lambda: subgridding_jit_module.run_subgridded_jit(
            None, None, None, None, 1,
        ),
    ],
)
def test_advanced_runner_front_doors_fail_before_setup(monkeypatch, invoke):
    monkeypatch.setattr(backend_module, "current_backend", lambda: "metal")

    with pytest.raises(NotImplementedError, match="experimental Apple Metal"):
        invoke()
