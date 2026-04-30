"""Phase-1 3D SBP-SAT z-slab tests."""

import inspect
import numpy as np
import jax.numpy as jnp

from rfx.subgridding.face_ops import build_face_ops, prolong_face, restrict_face
from rfx.subgridding import jit_runner
from rfx.subgridding.sbp_sat_3d import (
    _apply_sat_pair_face,
    _apply_time_centered_paired_face_helper,
    _active_corners,
    _active_edges,
    _active_faces,
    _face_interior_masks,
    apply_sat_e_interfaces,
    apply_sat_h_interfaces,
    compute_energy_3d,
    extract_tangential_h_face,
    init_subgrid_3d,
    sat_penalty_coefficients,
    step_subgrid_3d,
    step_subgrid_3d_with_cpml,
)


def _face_mismatch_energy(coarse, fine, ops, coarse_mask, fine_mask) -> float:
    coarse_mismatch = restrict_face(fine, ops) - coarse
    fine_mismatch = prolong_face(coarse, ops) - fine
    return float(
        jnp.sum((coarse_mismatch**2) * ops.coarse_norm * coarse_mask)
        + jnp.sum((fine_mismatch**2) * ops.fine_norm * fine_mask)
    )


def test_sat_face_pair_reduces_weighted_mismatch_energy():
    ops = build_face_ops((4, 3), ratio=2, dx_c=0.004)
    coarse = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) * 1.0e-3
    fine = prolong_face(coarse, ops) + jnp.linspace(
        -2.0e-3,
        2.0e-3,
        48,
        dtype=jnp.float32,
    ).reshape(8, 6)
    coarse_mask, fine_mask = _face_interior_masks(coarse.shape, ops.ratio)
    alpha_c, alpha_f = sat_penalty_coefficients(ops.ratio, 0.5)

    before = _face_mismatch_energy(coarse, fine, ops, coarse_mask, fine_mask)
    coarse_after, fine_after = _apply_sat_pair_face(
        coarse,
        fine,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
    )
    after = _face_mismatch_energy(
        coarse_after,
        fine_after,
        ops,
        coarse_mask,
        fine_mask,
    )

    assert after < before


def test_sat_face_pair_preserves_matched_constant_trace():
    ops = build_face_ops((3, 2), ratio=2, dx_c=0.004)
    coarse = jnp.ones((3, 2), dtype=jnp.float32) * 1.25
    fine = prolong_face(coarse, ops)
    coarse_mask, fine_mask = _face_interior_masks(coarse.shape, ops.ratio)
    alpha_c, alpha_f = sat_penalty_coefficients(ops.ratio, 0.5)

    coarse_after, fine_after = _apply_sat_pair_face(
        coarse,
        fine,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
    )

    np.testing.assert_allclose(np.asarray(coarse_after), np.asarray(coarse), atol=1e-7)
    np.testing.assert_allclose(np.asarray(fine_after), np.asarray(fine), atol=1e-7)


def test_zslab_energy_pec_1000_steps():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
        tau=0.5,
    )
    state = state._replace(ez_c=state.ez_c.at[3, 3, 2].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            energy = compute_energy_3d(state, config)
            assert energy <= initial_energy * 1.02, (
                f"Energy exceeded transient bound at step {i + 1}: "
                f"{energy / initial_energy:.4f}"
            )
            max_energy = max(max_energy, energy)

    final_energy = compute_energy_3d(state, config)
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy
    assert max_energy <= initial_energy * 1.02


def test_init_subgrid_3d_accepts_arbitrary_box_region():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(1, 7, 1, 6, 3, 7),
        ratio=2,
    )
    assert (config.fi_lo, config.fi_hi) == (1, 7)
    assert (config.fj_lo, config.fj_hi) == (1, 6)
    assert (config.fk_lo, config.fk_hi) == (3, 7)


def test_init_subgrid_3d_default_region_is_full_span_xy():
    config, _ = init_subgrid_3d(shape_c=(8, 8, 10), dx_c=0.004, ratio=2)
    assert (config.fi_lo, config.fi_hi) == (0, 8)
    assert (config.fj_lo, config.fj_hi) == (0, 8)


def test_zslab_fine_grid_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[4, 4, 2].set(1.0))
    for _ in range(200):
        state = step_subgrid_3d(state, config)

    max_f = float(jnp.max(jnp.abs(state.ez_f)))
    assert np.isfinite(max_f)
    assert max_f > 1e-8


def test_box_fine_grid_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 4, 5].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    max_f = float(jnp.max(jnp.abs(state.ez_f)))
    assert np.isfinite(max_f)
    assert max_f > 1e-8


def test_box_energy_pec_1000_steps():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
        tau=0.5,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 4, 5].set(1.0))
    initial_energy = compute_energy_3d(state, config)

    max_energy = initial_energy
    for i in range(1000):
        state = step_subgrid_3d(state, config)
        if (i + 1) % 100 == 0:
            energy = compute_energy_3d(state, config)
            assert energy <= initial_energy * 1.05, (
                f"Box energy exceeded transient bound at step {i + 1}: "
                f"{energy / initial_energy:.4f}"
            )
            max_energy = max(max_energy, energy)

    final_energy = compute_energy_3d(state, config)
    assert np.isfinite(final_energy)
    assert final_energy <= initial_energy * 1.05
    assert max_energy <= initial_energy * 1.05


def test_box_edge_region_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 1, 5].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    edge_mag = float(jnp.max(jnp.abs(state.ez_f[0:2, 0:2, :])))
    assert np.isfinite(edge_mag)
    assert edge_mag > 1e-8


def test_box_corner_region_receives_signal():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = state._replace(ez_c=state.ez_c.at[1, 1, 1].set(1.0))
    for _ in range(250):
        state = step_subgrid_3d(state, config)

    corner_mag = float(jnp.max(jnp.abs(state.ez_f[0:2, 0:2, 0:2])))
    assert np.isfinite(corner_mag)
    assert corner_mag > 1e-8


def test_phase1_uses_single_canonical_stepper():
    source = inspect.getsource(jit_runner.run_subgridded_jit)
    assert "step_subgrid_3d" in source
    assert "_shared_node_coupling_3d" not in source
    assert "_shared_node_coupling_h_3d" not in source


def test_time_centered_paired_face_helper_changes_private_h_trace():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    z_face = "z_lo"
    coarse_perturbation = (
        jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4) * 1.0e-3
    )
    fine_perturbation = (
        jnp.linspace(-1.0, 1.0, 64, dtype=jnp.float32).reshape(8, 8) * 2.0e-3
    )
    coarse_z = config.fk_lo
    state = state._replace(
        ex_c=state.ex_c.at[
            config.fi_lo : config.fi_hi, config.fj_lo : config.fj_hi, coarse_z
        ].set(coarse_perturbation),
        ey_c=state.ey_c.at[
            config.fi_lo : config.fi_hi, config.fj_lo : config.fj_hi, coarse_z
        ].set(-0.5 * coarse_perturbation),
        hx_c=state.hx_c.at[
            config.fi_lo : config.fi_hi, config.fj_lo : config.fj_hi, coarse_z
        ].set(1.0e-6 + coarse_perturbation * 1.0e-3),
        hy_c=state.hy_c.at[
            config.fi_lo : config.fi_hi, config.fj_lo : config.fj_hi, coarse_z
        ].set(2.0e-6 - coarse_perturbation * 1.5e-3),
        ex_f=state.ex_f.at[:, :, 0].set(fine_perturbation),
        ey_f=state.ey_f.at[:, :, 0].set(-0.5 * fine_perturbation),
        hx_f=state.hx_f.at[:, :, 0].set(1.0e-6 + fine_perturbation * 1.0e-3),
        hy_f=state.hy_f.at[:, :, 0].set(2.0e-6 - fine_perturbation * 1.5e-3),
    )
    h_pre_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_pre_fine = (state.hx_f, state.hy_f, state.hz_f)
    e_pre_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_pre_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_post_coarse, h_post_fine = apply_sat_h_interfaces(
        h_pre_coarse,
        h_pre_fine,
        config,
    )
    e_post_coarse, e_post_fine = apply_sat_e_interfaces(
        e_pre_coarse,
        e_pre_fine,
        config,
    )

    corrected_coarse, corrected_fine = _apply_time_centered_paired_face_helper(
        h_post_coarse,
        h_post_fine,
        h_pre_sat_coarse=h_pre_coarse,
        h_pre_sat_fine=h_pre_fine,
        h_post_sat_coarse=h_post_coarse,
        h_post_sat_fine=h_post_fine,
        e_pre_sat_coarse=e_pre_coarse,
        e_pre_sat_fine=e_pre_fine,
        e_post_sat_coarse=e_post_coarse,
        e_post_sat_fine=e_post_fine,
        config=config,
    )

    _, h2_before = extract_tangential_h_face(
        h_post_coarse, config, z_face, grid="coarse"
    )
    _, h2_after = extract_tangential_h_face(
        corrected_coarse, config, z_face, grid="coarse"
    )
    assert float(jnp.max(jnp.abs(h2_after - h2_before))) > 0.0
    for before, after in zip(h_post_coarse, corrected_coarse):
        assert before.shape == after.shape
        assert bool(jnp.all(jnp.isfinite(after)))
    for before, after in zip(h_post_fine, corrected_fine):
        assert before.shape == after.shape
        assert bool(jnp.all(jnp.isfinite(after)))


def test_time_centered_paired_face_helper_is_wired_after_e_sat():
    for function in (step_subgrid_3d, step_subgrid_3d_with_cpml):
        source = inspect.getsource(function)
        helper_index = source.index("_apply_time_centered_paired_face_helper")
        h_sat_index = source.index("apply_sat_h_interfaces")
        e_sat_index = source.index("apply_sat_e_interfaces")
        assert h_sat_index < e_sat_index < helper_index
        assert "private_post_h_hook" in source
        assert "private_post_e_hook" in source
        helper_source = source[helper_index:]
        assert "private_post_h_hook" not in helper_source
        assert "private_post_e_hook" not in helper_source


def test_active_interfaces_for_full_span_zslab():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(0, 8, 0, 8, 3, 7),
        ratio=2,
    )
    assert _active_faces(config) == ("z_lo", "z_hi")
    assert _active_edges(config) == ()
    assert _active_corners(config) == ()


def test_active_interfaces_for_interior_box():
    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    assert set(_active_faces(config)) == {
        "x_lo",
        "x_hi",
        "y_lo",
        "y_hi",
        "z_lo",
        "z_hi",
    }
    assert len(_active_edges(config)) == 12
    assert len(_active_corners(config)) == 8
