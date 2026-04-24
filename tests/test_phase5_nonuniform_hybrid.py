"""Phase V nonuniform hybrid foundation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, DebyePole, GaussianPulse, Simulation


def _make_nonuniform_supported_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary=boundary,
        dx=0.0025,
        dz_profile=np.array([0.0020, 0.0015, 0.0010, 0.0015, 0.0020], dtype=float),
    )
    sim.add_source((0.005, 0.0075, 0.0075), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _make_nonuniform_xy_unsupported_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary="pec",
        dx=0.0025,
        dx_profile=np.array([0.0025, 0.0015, 0.0015, 0.0015, 0.0015, 0.0025], dtype=float),
        dz_profile=np.array([0.0020, 0.0015, 0.0010, 0.0015, 0.0020], dtype=float),
    )
    sim.add_source((0.005, 0.0075, 0.0075), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _make_periodic_supported_sim(*, boundary: str = "pec") -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary=boundary,
    )
    sim.add_source((0.005, 0.0075, 0.0075), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    sim.set_periodic_axes("x")
    return sim


def _single_cell_eps(sim: Simulation, base_eps: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    grid = sim._build_nonuniform_grid()
    i, j, k = sim._pos_to_nu_index(grid, (0.0075, 0.0075, 0.0075))
    return base_eps.at[i, j, k].add(alpha)


def test_phase5_nonuniform_support_inspection_accepts_z_graded_source_probe():
    sim = _make_nonuniform_supported_sim()

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert report.supported
    assert report.source_count == 1
    assert report.probe_count == 1
    assert report.boundary == "pec"
    assert report.inventory is not None


def test_phase5_nonuniform_support_inspection_accepts_cpml_z_graded_source_probe():
    sim = _make_nonuniform_supported_sim(boundary="cpml")

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert report.supported
    assert report.boundary == "cpml"


def test_phase5_nonuniform_prepare_bundle_supports_z_graded_source_probe():
    sim = _make_nonuniform_supported_sim()

    inputs = sim.build_hybrid_phase1_inputs(n_steps=8)
    prepared = sim.prepare_hybrid_phase1(n_steps=8)

    assert inputs.supported
    assert prepared.supported
    assert prepared.context is not None
    assert prepared.context.grid.shape == inputs.grid.shape


def test_phase5_nonuniform_forward_matches_pure_ad_pec():
    sim = _make_nonuniform_supported_sim()

    pure = sim.forward(n_steps=8, checkpoint=True)
    hybrid = sim.forward_hybrid_phase1(n_steps=8, fallback="raise")

    np.testing.assert_allclose(
        np.asarray(hybrid.time_series),
        np.asarray(pure.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase5_nonuniform_forward_matches_pure_ad_cpml():
    sim = _make_nonuniform_supported_sim(boundary="cpml")

    pure = sim.forward(n_steps=8, checkpoint=True)
    hybrid = sim.forward_hybrid_phase1(n_steps=8, fallback="raise")

    np.testing.assert_allclose(
        np.asarray(hybrid.time_series),
        np.asarray(pure.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase5_nonuniform_gradient_matches_pure_ad():
    sim = _make_nonuniform_supported_sim()
    grid = sim._build_nonuniform_grid()
    materials, *_ = sim._assemble_materials_nu(grid)

    def pure_loss(alpha):
        eps = _single_cell_eps(sim, materials.eps_r, alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def hybrid_loss(alpha):
        eps = _single_cell_eps(sim, materials.eps_r, alpha)
        result = sim.forward_hybrid_phase1(
            eps_override=eps,
            n_steps=8,
            fallback="raise",
        )
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_hybrid = jax.grad(hybrid_loss)(alpha0)
    rel_err = float(
        jnp.abs(grad_hybrid - grad_pure)
        / jnp.maximum(jnp.abs(grad_pure), 1e-12)
    )

    assert np.isfinite(float(grad_pure))
    assert np.isfinite(float(grad_hybrid))
    assert rel_err <= 1e-4


def test_phase5_nonuniform_xy_profile_remains_rejected():
    sim = _make_nonuniform_xy_unsupported_sim()

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "non-uniform grids are unsupported" in report.reason_text


def test_phase5_nonuniform_periodic_axes_remain_rejected():
    sim = _make_nonuniform_supported_sim()
    sim.set_periodic_axes("x")

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "combined non-uniform + periodic" in report.reason_text


def test_phase5_periodic_support_inspection_accepts_bounded_source_probe():
    sim = _make_periodic_supported_sim()

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert report.supported
    assert report.boundary == "pec"
    assert report.periodic == (True, False, False)
    assert report.inventory is not None


def test_phase5_periodic_prepare_bundle_supports_bounded_source_probe():
    sim = _make_periodic_supported_sim()

    prepared = sim.prepare_hybrid_phase1(n_steps=8)

    assert prepared.supported
    assert prepared.context is not None
    assert prepared.periodic == (True, False, False)


def test_phase5_periodic_forward_matches_pure_ad():
    sim = _make_periodic_supported_sim()

    pure = sim.forward(n_steps=8, checkpoint=True)
    hybrid = sim.forward_hybrid_phase1(n_steps=8, fallback="raise")

    np.testing.assert_allclose(
        np.asarray(hybrid.time_series),
        np.asarray(pure.time_series),
        rtol=1e-6,
        atol=1e-12,
    )


def test_phase5_periodic_gradient_matches_pure_ad():
    sim = _make_periodic_supported_sim()
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)

    def pure_loss(alpha):
        eps = materials.eps_r.at[grid.position_to_index((0.0075, 0.0075, 0.0075))].add(alpha)
        result = sim.forward(eps_override=eps, n_steps=8, checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    def hybrid_loss(alpha):
        eps = materials.eps_r.at[grid.position_to_index((0.0075, 0.0075, 0.0075))].add(alpha)
        result = sim.forward_hybrid_phase1(eps_override=eps, n_steps=8, fallback="raise")
        return jnp.sum(result.time_series ** 2)

    alpha0 = jnp.float32(0.1)
    grad_pure = jax.grad(pure_loss)(alpha0)
    grad_hybrid = jax.grad(hybrid_loss)(alpha0)
    rel_err = float(
        jnp.abs(grad_hybrid - grad_pure)
        / jnp.maximum(jnp.abs(grad_pure), 1e-12)
    )

    assert np.isfinite(float(grad_pure))
    assert np.isfinite(float(grad_hybrid))
    assert rel_err <= 1e-4


def test_phase5_periodic_floquet_workflows_remain_rejected():
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")
    sim.add_floquet_port(0.005, axis="z", f0=3e9)

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "floquet periodic workflows are unsupported" in report.reason_text


def test_phase5_nonuniform_ntff_remains_rejected():
    sim = _make_nonuniform_supported_sim()
    sim.add_ntff_box(
        corner_lo=(0.003, 0.003, 0.003),
        corner_hi=(0.012, 0.012, 0.012),
        n_freqs=4,
    )

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "does not support NTFF" in report.reason_text


def test_phase5_nonuniform_strategy_b_remains_rejected():
    sim = _make_nonuniform_supported_sim()

    report = sim.inspect_hybrid_strategy_b_phase3(n_steps=8, checkpoint_every=4)

    assert not report.supported
    assert "supports only uniform grids" in report.reason_text

    try:
        sim.forward_hybrid_phase1(
            n_steps=8,
            strategy="b",
            checkpoint_every=4,
            fallback="raise",
        )
    except ValueError as exc:
        assert "supports only uniform grids" in str(exc)
    else:
        raise AssertionError("nonuniform Strategy B unexpectedly ran")


def test_phase5_nonuniform_port_workflow_remains_rejected():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary="pec",
        dx=0.0025,
        dz_profile=np.array([0.0020, 0.0015, 0.0010, 0.0015, 0.0020], dtype=float),
    )
    sim.add_port(
        (0.005, 0.0075, 0.0075),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.010, 0.0075, 0.0075), "ez")

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "supports only add_source()/probe workflows" in report.reason_text


def test_phase5_nonuniform_lossy_materials_remain_rejected():
    sim = _make_nonuniform_supported_sim()
    sim.add_material("lossy", eps_r=2.0, sigma=5.0)
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="lossy")

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "supports only zero sigma materials" in report.reason_text


def test_phase5_nonuniform_dispersive_materials_remain_rejected():
    sim = _make_nonuniform_supported_sim()
    sim.add_material(
        "disp",
        eps_r=2.0,
        debye_poles=[DebyePole(delta_eps=1.0, tau=8e-12)],
    )
    sim.add(Box((0.006, 0.006, 0.006), (0.009, 0.009, 0.009)), material="disp")

    report = sim.inspect_hybrid_phase1(n_steps=8)

    assert not report.supported
    assert "supports only lossless nondispersive materials" in report.reason_text
