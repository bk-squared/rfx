"""Phase-1 3D SBP-SAT z-slab tests."""

import inspect
import numpy as np
import jax.numpy as jnp

from rfx.boundaries.cpml import init_cpml
from rfx.grid import Grid
from rfx.subgridding.face_ops import build_face_ops, prolong_face, restrict_face
from rfx.subgridding import jit_runner, sbp_sat_3d
from rfx.subgridding.sbp_sat_3d import (
    FACE_ORIENTATIONS,
    _apply_observable_proxy_modal_retry_face_helper,
    _apply_operator_projected_skew_eh_face_helper,
    _apply_propagation_aware_modal_retry_face_helper,
    _apply_sat_pair_face,
    _apply_time_centered_paired_face_helper,
    _active_corners,
    _active_edges,
    _active_faces,
    _face_interior_masks,
    _get_face_ops,
    _init_private_interface_owner_state,
    _private_interface_owner_joint_score,
    _project_private_modal_basis_packets,
    _stage_private_time_aligned_owner_packets,
    _update_private_interface_owner_state_from_scan,
    _update_private_source_owner_state_from_scan,
    apply_sat_e_interfaces,
    apply_sat_h_interfaces,
    compute_energy_3d,
    extract_tangential_e_face,
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


def _seed_all_components_for_box(config, state):
    coarse_values = (
        jnp.linspace(-1.0, 1.0, int(np.prod(state.ex_c.shape)), dtype=jnp.float32)
        .reshape(state.ex_c.shape)
        * 1.0e-3
    )
    fine_values = (
        jnp.linspace(-1.0, 1.0, int(np.prod(state.ex_f.shape)), dtype=jnp.float32)
        .reshape(state.ex_f.shape)
        * 1.5e-3
    )
    return state._replace(
        ex_c=coarse_values,
        ey_c=-0.5 * coarse_values,
        ez_c=0.25 * coarse_values,
        hx_c=1.0e-6 + 1.0e-3 * coarse_values,
        hy_c=2.0e-6 - 1.5e-3 * coarse_values,
        hz_c=-1.0e-6 + 0.75e-3 * coarse_values,
        ex_f=fine_values,
        ey_f=-0.5 * fine_values,
        ez_f=0.25 * fine_values,
        hx_f=1.0e-6 + 1.0e-3 * fine_values,
        hy_f=2.0e-6 - 1.5e-3 * fine_values,
        hz_f=-1.0e-6 + 0.75e-3 * fine_values,
    )


def test_operator_projected_skew_eh_helper_has_same_call_slot_map():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = _seed_all_components_for_box(config, state)
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = _apply_operator_projected_skew_eh_face_helper(
        e_coarse,
        e_fine,
        h_coarse,
        h_fine,
        config,
    )

    assert len(out_e_c) == len(out_e_f) == len(out_h_c) == len(out_h_f) == 3
    assert set(_active_faces(config)) == set(FACE_ORIENTATIONS)
    changed = False
    for before, after in zip(e_coarse + e_fine + h_coarse + h_fine, out_e_c + out_e_f + out_h_c + out_h_f):
        assert before.shape == after.shape
        assert bool(jnp.all(jnp.isfinite(after)))
        changed = changed or bool(jnp.any(jnp.abs(after - before) > 0.0))
    assert changed


def test_operator_projected_skew_eh_helper_declares_exact_face_local_slot_map():
    expected = {
        "x_lo": {"normal_sign": -1, "e": ("ey", "ez"), "h": ("hy", "hz")},
        "x_hi": {"normal_sign": 1, "e": ("ey", "ez"), "h": ("hy", "hz")},
        "y_lo": {"normal_sign": -1, "e": ("ex", "ez"), "h": ("hx", "hz")},
        "y_hi": {"normal_sign": 1, "e": ("ex", "ez"), "h": ("hx", "hz")},
        "z_lo": {"normal_sign": -1, "e": ("ex", "ey"), "h": ("hx", "hy")},
        "z_hi": {"normal_sign": 1, "e": ("ex", "ey"), "h": ("hx", "hy")},
    }

    assert set(FACE_ORIENTATIONS) == set(expected)
    for face, expected_map in expected.items():
        orientation = FACE_ORIENTATIONS[face]
        assert orientation.normal_sign == expected_map["normal_sign"]
        assert orientation.tangential_e_components == expected_map["e"]
        assert orientation.tangential_h_components == expected_map["h"]

    source = inspect.getsource(_apply_operator_projected_skew_eh_face_helper)
    assert "ex_c=e_c_face[0]" in source
    assert "ey_c=e_c_face[1]" in source
    assert "hx_c=h_c_face[0]" in source
    assert "hy_c=h_c_face[1]" in source
    assert "ex_f=e_f_face[0]" in source
    assert "ey_f=e_f_face[1]" in source
    assert "hx_f=h_f_face[0]" in source
    assert "hy_f=h_f_face[1]" in source
    assert "normal_sign=orientation.normal_sign" in source
    assert "include_scalar_projection=False" in source


def test_operator_projected_skew_eh_helper_orients_opposite_faces_once():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )

    def _seed_fine_x_face_hz(face: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        hz_f = jnp.zeros_like(state.hz_f)
        x_index = 0 if face == "x_lo" else config.nx_f - 1
        hz_f = hz_f.at[x_index, :, :].set(1.0e-6)
        return jnp.zeros_like(state.hx_f), jnp.zeros_like(state.hy_f), hz_f

    deltas = {}
    for face in ("x_lo", "x_hi"):
        e_coarse = (
            jnp.zeros_like(state.ex_c),
            jnp.zeros_like(state.ey_c),
            jnp.zeros_like(state.ez_c),
        )
        e_fine = (
            jnp.zeros_like(state.ex_f),
            jnp.zeros_like(state.ey_f),
            jnp.zeros_like(state.ez_f),
        )
        h_coarse = (
            jnp.zeros_like(state.hx_c),
            jnp.zeros_like(state.hy_c),
            jnp.zeros_like(state.hz_c),
        )
        h_fine = _seed_fine_x_face_hz(face)

        out_e_c, _, _, _ = _apply_operator_projected_skew_eh_face_helper(
            e_coarse,
            e_fine,
            h_coarse,
            h_fine,
            config,
        )
        before_t1 = extract_tangential_e_face(
            e_coarse,
            config,
            face,
            grid="coarse",
        )[0]
        after_t1 = extract_tangential_e_face(
            out_e_c,
            config,
            face,
            grid="coarse",
        )[0]
        ops = _get_face_ops(config, face)
        coarse_mask, _ = _face_interior_masks(ops.coarse_shape, config.ratio)
        interior = np.asarray(coarse_mask) == 1.0
        deltas[face] = float(jnp.mean((after_t1 - before_t1)[interior]))

    assert deltas["x_lo"] * deltas["x_hi"] < 0.0
    np.testing.assert_allclose(
        abs(deltas["x_lo"]),
        abs(deltas["x_hi"]),
        rtol=1.0e-6,
    )


def test_operator_projected_skew_eh_helper_keeps_edges_and_corners_unchanged():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 10),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 3, 7),
        ratio=2,
    )
    state = _seed_all_components_for_box(config, state)
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = _apply_operator_projected_skew_eh_face_helper(
        e_coarse,
        e_fine,
        h_coarse,
        h_fine,
        config,
    )

    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        for before_fields, after_fields, extractor, mask in (
            (e_coarse, out_e_c, extract_tangential_e_face, coarse_mask),
            (h_coarse, out_h_c, extract_tangential_h_face, coarse_mask),
            (e_fine, out_e_f, extract_tangential_e_face, fine_mask),
            (h_fine, out_h_f, extract_tangential_h_face, fine_mask),
        ):
            grid = "coarse" if mask.shape == ops.coarse_shape else "fine"
            before_t = extractor(before_fields, config, face, grid=grid)
            after_t = extractor(after_fields, config, face, grid=grid)
            edge_mask = np.asarray(mask) == 0.0
            for before_face, after_face in zip(before_t, after_t):
                np.testing.assert_allclose(
                    np.asarray(after_face)[edge_mask],
                    np.asarray(before_face)[edge_mask],
                    atol=1.0e-12,
                )


def test_time_centered_paired_face_helper_is_wired_after_e_sat():
    for function in (step_subgrid_3d, step_subgrid_3d_with_cpml):
        source = inspect.getsource(function)
        skew_index = source.index("_apply_operator_projected_skew_eh_face_helper")
        helper_index = source.index("_apply_time_centered_paired_face_helper")
        proxy_index = source.index("_apply_observable_proxy_modal_retry_face_helper")
        stage_index = source.index("_stage_private_time_aligned_owner_packets")
        source_index = source.index("_update_private_source_owner_state_from_scan")
        propagation_index = source.index(
            "_apply_propagation_aware_modal_retry_face_helper"
        )
        h_sat_index = source.index("apply_sat_h_interfaces")
        e_sat_index = source.index("apply_sat_e_interfaces")
        assert (
            h_sat_index
            < e_sat_index
            < skew_index
            < helper_index
            < proxy_index
            < stage_index
            < source_index
            < propagation_index
        )
        assert "private_post_h_hook" in source
        assert "private_post_e_hook" in source
        helper_source = source[helper_index:]
        assert "private_post_h_hook" not in helper_source
        assert "private_post_e_hook" not in helper_source


def _expected_private_owner_packet_lengths(config):
    return np.asarray(
        [
            int(np.prod(_get_face_ops(config, face).coarse_shape))
            for face in _active_faces(config)
        ],
        dtype=np.int32,
    )


def _assert_private_owner_packet_shape(owner_state, config):
    lengths = _expected_private_owner_packet_lengths(config)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int32)
    packet_size = int(np.sum(lengths))
    active_faces = _active_faces(config)

    np.testing.assert_array_equal(
        np.asarray(owner_state.face_packet_lengths),
        lengths,
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.face_packet_offsets),
        offsets,
    )
    assert owner_state.face_proxy_reference_real.shape == (packet_size,)
    assert owner_state.face_proxy_reference_imag.shape == (packet_size,)
    assert owner_state.face_proxy_reference_prev_real.shape == (packet_size,)
    assert owner_state.face_proxy_reference_prev_imag.shape == (packet_size,)
    assert owner_state.face_proxy_weight.shape == (packet_size,)
    assert owner_state.face_proxy_mask.shape == (packet_size,)
    np.testing.assert_array_equal(
        np.asarray(owner_state.face_normal_axis),
        [FACE_ORIENTATIONS[face].normal_axis for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.face_normal_sign),
        [FACE_ORIENTATIONS[face].normal_sign for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.face_tangential_axis_0),
        [FACE_ORIENTATIONS[face].tangential_axes[0] for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.face_tangential_axis_1),
        [FACE_ORIENTATIONS[face].tangential_axes[1] for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_packet_lengths),
        lengths,
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_packet_offsets),
        offsets,
    )
    assert owner_state.source_owner_reference_real.shape == (packet_size,)
    assert owner_state.source_owner_reference_imag.shape == (packet_size,)
    assert owner_state.source_owner_reference_prev_real.shape == (packet_size,)
    assert owner_state.source_owner_reference_prev_imag.shape == (packet_size,)
    assert owner_state.source_owner_weight.shape == (packet_size,)
    assert owner_state.source_owner_mask.shape == (packet_size,)
    assert owner_state.source_incident_normalizer_real.shape == (packet_size,)
    assert owner_state.source_incident_normalizer_imag.shape == (packet_size,)
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_normal_axis),
        [FACE_ORIENTATIONS[face].normal_axis for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_normal_sign),
        [FACE_ORIENTATIONS[face].normal_sign for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_tangential_axis_0),
        [FACE_ORIENTATIONS[face].tangential_axes[0] for face in active_faces],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_state.source_tangential_axis_1),
        [FACE_ORIENTATIONS[face].tangential_axes[1] for face in active_faces],
    )
    assert np.all(np.asarray(owner_state.face_proxy_mask) >= 0.0)
    assert np.all(np.asarray(owner_state.face_proxy_weight) >= 0.0)
    assert np.any(np.asarray(owner_state.face_proxy_mask) > 0.0)
    assert np.all(np.asarray(owner_state.source_owner_mask) >= 0.0)
    assert np.all(np.asarray(owner_state.source_owner_weight) >= 0.0)
    np.testing.assert_allclose(
        np.asarray(owner_state.source_owner_mask),
        np.asarray(owner_state.face_proxy_mask),
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_incident_normalizer_real),
        np.asarray(owner_state.source_owner_mask),
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_incident_normalizer_imag),
        0.0,
    )


def test_private_interface_owner_state_initializes_for_active_faces():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )

    owner_state = state.private_interface_owner_state

    assert owner_state is not None
    assert owner_state.face_phase_reference.shape == (len(_active_faces(config)),)
    assert owner_state.face_magnitude_reference.shape == (len(_active_faces(config)),)
    assert owner_state.face_update_count.shape == (len(_active_faces(config)),)
    _assert_private_owner_packet_shape(owner_state, config)
    np.testing.assert_allclose(np.asarray(owner_state.face_phase_reference), 0.0)
    np.testing.assert_allclose(np.asarray(owner_state.face_magnitude_reference), 0.0)
    np.testing.assert_array_equal(np.asarray(owner_state.face_update_count), 0)
    np.testing.assert_allclose(np.asarray(owner_state.face_proxy_reference_real), 0.0)
    np.testing.assert_allclose(np.asarray(owner_state.face_proxy_reference_imag), 0.0)
    np.testing.assert_allclose(
        np.asarray(owner_state.face_proxy_reference_prev_real),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.face_proxy_reference_prev_imag),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_owner_reference_real),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_owner_reference_imag),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_owner_reference_prev_real),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_owner_reference_prev_imag),
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_incident_normalizer_real),
        np.asarray(owner_state.source_owner_mask),
    )
    np.testing.assert_allclose(
        np.asarray(owner_state.source_incident_normalizer_imag),
        0.0,
    )


def test_private_source_owner_buffers_are_separate_from_interface_packet():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _init_private_interface_owner_state(config)
    source_real_before = np.asarray(owner_state.source_owner_reference_real)
    source_imag_before = np.asarray(owner_state.source_owner_reference_imag)
    source_normalizer_before = np.asarray(
        owner_state.source_incident_normalizer_real
    )

    updated = _update_private_interface_owner_state_from_scan(
        owner_state,
        e_post_sat_coarse=(state.ex_c + 1.0, state.ey_c + 2.0, state.ez_c + 3.0),
        e_post_sat_fine=(state.ex_f + 1.5, state.ey_f + 2.5, state.ez_f + 3.5),
        h_post_sat_coarse=(state.hx_c + 0.1, state.hy_c + 0.2, state.hz_c + 0.3),
        h_post_sat_fine=(state.hx_f + 0.15, state.hy_f + 0.25, state.hz_f + 0.35),
        config=config,
    )
    active_proxy = np.asarray(updated.face_proxy_mask) > 0.0

    assert np.any(
        np.abs(np.asarray(updated.face_proxy_reference_real)[active_proxy]) > 0.0
    )
    np.testing.assert_allclose(
        np.asarray(updated.source_owner_reference_real),
        source_real_before,
    )
    np.testing.assert_allclose(
        np.asarray(updated.source_owner_reference_imag),
        source_imag_before,
    )
    np.testing.assert_allclose(
        np.asarray(updated.source_incident_normalizer_real),
        source_normalizer_before,
    )


def test_private_source_owner_population_updates_source_packet_only():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _init_private_interface_owner_state(config)
    face_real_before = np.asarray(owner_state.face_proxy_reference_real)
    face_imag_before = np.asarray(owner_state.face_proxy_reference_imag)

    updated = _update_private_source_owner_state_from_scan(
        owner_state,
        e_source_coarse=(state.ex_c + 1.0, state.ey_c + 2.0, state.ez_c + 3.0),
        e_source_fine=(state.ex_f + 1.5, state.ey_f + 2.5, state.ez_f + 3.5),
        h_source_coarse=(state.hx_c + 0.1, state.hy_c + 0.2, state.hz_c + 0.3),
        h_source_fine=(state.hx_f + 0.15, state.hy_f + 0.25, state.hz_f + 0.35),
        config=config,
    )
    active_source = np.asarray(updated.source_owner_mask) > 0.0

    assert np.any(
        np.abs(np.asarray(updated.source_owner_reference_real)[active_source]) > 0.0
    )
    assert np.any(
        np.abs(np.asarray(updated.source_owner_reference_imag)[active_source]) > 0.0
    )
    np.testing.assert_allclose(
        np.asarray(updated.face_proxy_reference_real),
        face_real_before,
    )
    np.testing.assert_allclose(
        np.asarray(updated.face_proxy_reference_imag),
        face_imag_before,
    )
    np.testing.assert_allclose(
        np.asarray(updated.source_incident_normalizer_real),
        np.asarray(updated.source_owner_mask),
    )
    np.testing.assert_allclose(
        np.asarray(updated.source_incident_normalizer_imag),
        0.0,
    )


def test_private_time_aligned_owner_packet_staging_snapshots_previous_pair():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _init_private_interface_owner_state(config)
    owner_state = _update_private_interface_owner_state_from_scan(
        owner_state,
        e_post_sat_coarse=(state.ex_c + 1.0, state.ey_c + 2.0, state.ez_c + 3.0),
        e_post_sat_fine=(state.ex_f + 1.5, state.ey_f + 2.5, state.ez_f + 3.5),
        h_post_sat_coarse=(state.hx_c + 0.1, state.hy_c + 0.2, state.hz_c + 0.3),
        h_post_sat_fine=(state.hx_f + 0.15, state.hy_f + 0.25, state.hz_f + 0.35),
        config=config,
    )
    owner_state = _update_private_source_owner_state_from_scan(
        owner_state,
        e_source_coarse=(state.ex_c + 4.0, state.ey_c + 5.0, state.ez_c + 6.0),
        e_source_fine=(state.ex_f + 4.5, state.ey_f + 5.5, state.ez_f + 6.5),
        h_source_coarse=(state.hx_c + 0.4, state.hy_c + 0.5, state.hz_c + 0.6),
        h_source_fine=(state.hx_f + 0.45, state.hy_f + 0.55, state.hz_f + 0.65),
        config=config,
    )

    staged = _stage_private_time_aligned_owner_packets(owner_state)
    overwritten = _update_private_source_owner_state_from_scan(
        staged,
        e_source_coarse=(state.ex_c + 7.0, state.ey_c + 8.0, state.ez_c + 9.0),
        e_source_fine=(state.ex_f + 7.5, state.ey_f + 8.5, state.ez_f + 9.5),
        h_source_coarse=(state.hx_c + 0.7, state.hy_c + 0.8, state.hz_c + 0.9),
        h_source_fine=(state.hx_f + 0.75, state.hy_f + 0.85, state.hz_f + 0.95),
        config=config,
    )

    np.testing.assert_allclose(
        np.asarray(staged.face_proxy_reference_prev_real),
        np.asarray(owner_state.face_proxy_reference_real),
    )
    np.testing.assert_allclose(
        np.asarray(staged.face_proxy_reference_prev_imag),
        np.asarray(owner_state.face_proxy_reference_imag),
    )
    np.testing.assert_allclose(
        np.asarray(staged.source_owner_reference_prev_real),
        np.asarray(owner_state.source_owner_reference_real),
    )
    np.testing.assert_allclose(
        np.asarray(staged.source_owner_reference_prev_imag),
        np.asarray(owner_state.source_owner_reference_imag),
    )
    np.testing.assert_allclose(
        np.asarray(overwritten.source_owner_reference_prev_real),
        np.asarray(staged.source_owner_reference_prev_real),
    )
    np.testing.assert_allclose(
        np.asarray(overwritten.source_owner_reference_prev_imag),
        np.asarray(staged.source_owner_reference_prev_imag),
    )
    assert not np.allclose(
        np.asarray(overwritten.source_owner_reference_real),
        np.asarray(overwritten.source_owner_reference_prev_real),
    )


def test_private_interface_owner_state_propagates_through_non_cpml_step():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    state = state._replace(private_interface_owner_state=None)

    next_state = step_subgrid_3d(state, config)

    owner_state = next_state.private_interface_owner_state
    assert owner_state is not None
    assert owner_state.face_update_count.shape == (len(_active_faces(config)),)
    _assert_private_owner_packet_shape(owner_state, config)
    np.testing.assert_array_equal(np.asarray(owner_state.face_update_count), 1)


def test_private_interface_owner_state_propagates_through_cpml_step():
    grid = Grid(
        freq_max=5e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.004,
        cpml_layers=1,
    )
    config, state = init_subgrid_3d(
        shape_c=grid.shape,
        dx_c=grid.dx,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    cpml_params, cpml_state = init_cpml(grid)
    state = state._replace(private_interface_owner_state=None)

    next_state, _ = step_subgrid_3d_with_cpml(
        state,
        config,
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        grid_c=grid,
        cpml_axes="xyz",
    )

    owner_state = next_state.private_interface_owner_state
    assert owner_state is not None
    assert owner_state.face_update_count.shape == (len(_active_faces(config)),)
    _assert_private_owner_packet_shape(owner_state, config)
    np.testing.assert_array_equal(np.asarray(owner_state.face_update_count), 1)


def test_private_interface_owner_state_initializes_in_jit_runner():
    source = inspect.getsource(jit_runner.run_subgridded_jit)
    assert "_init_private_interface_owner_state(config)" in source

    config, _ = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _init_private_interface_owner_state(config)

    assert owner_state.face_update_count.shape == (len(_active_faces(config)),)
    _assert_private_owner_packet_shape(owner_state, config)


def _private_owner_with_packet_reference(config, *, active):
    owner_state = _init_private_interface_owner_state(config)
    mask = owner_state.face_proxy_mask
    update_count = jnp.ones_like(owner_state.face_update_count)
    if not active:
        update_count = jnp.zeros_like(owner_state.face_update_count)
    return owner_state._replace(
        face_update_count=update_count,
        face_proxy_reference_real=0.25 * mask,
        face_proxy_reference_imag=-0.125 * mask,
    )


def test_observable_proxy_modal_retry_helper_uses_lagged_packet_and_bounds_update():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _private_owner_with_packet_reference(config, active=True)
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = (
        _apply_observable_proxy_modal_retry_face_helper(
            e_coarse,
            e_fine,
            h_coarse,
            h_fine,
            owner_state,
            config,
        )
    )

    for before, after in zip(h_coarse, out_h_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_fine, out_h_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)

    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        before_c = extract_tangential_e_face(e_coarse, config, face, grid="coarse")
        after_c = extract_tangential_e_face(out_e_c, config, face, grid="coarse")
        before_f = extract_tangential_e_face(e_fine, config, face, grid="fine")
        after_f = extract_tangential_e_face(out_e_f, config, face, grid="fine")
        coarse_active = np.asarray(coarse_mask) > 0.0
        fine_active = np.asarray(fine_mask) > 0.0
        delta_c0 = np.asarray(after_c[0] - before_c[0])
        delta_c1 = np.asarray(after_c[1] - before_c[1])
        delta_f0 = np.asarray(after_f[0] - before_f[0])
        delta_f1 = np.asarray(after_f[1] - before_f[1])
        assert np.any(np.abs(delta_c0[coarse_active]) > 0.0)
        assert np.any(np.abs(delta_c1[coarse_active]) > 0.0)
        assert np.max(np.abs(delta_c0[coarse_active])) <= 0.25 * 0.02 + 1.0e-8
        assert np.max(np.abs(delta_c1[coarse_active])) <= 0.125 * 0.02 + 1.0e-8
        assert np.max(np.abs(delta_f0[fine_active])) <= 0.25 * 0.02 + 1.0e-8
        assert np.max(np.abs(delta_f1[fine_active])) <= 0.125 * 0.02 + 1.0e-8


def test_observable_proxy_modal_retry_helper_waits_for_lagged_packet():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _private_owner_with_packet_reference(config, active=False)
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = (
        _apply_observable_proxy_modal_retry_face_helper(
            e_coarse,
            e_fine,
            h_coarse,
            h_fine,
            owner_state,
            config,
        )
    )

    for before, after in zip(e_coarse, out_e_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(e_fine, out_e_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_coarse, out_h_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_fine, out_h_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)


def _private_owner_with_source_interface_reference(config, *, source_active):
    owner_state = _init_private_interface_owner_state(config)
    mask = owner_state.face_proxy_mask
    source_mask = owner_state.source_owner_mask
    source_real = 0.20 * source_mask
    source_imag = 0.10 * source_mask
    if not source_active:
        source_real = jnp.zeros_like(source_real)
        source_imag = jnp.zeros_like(source_imag)
    return owner_state._replace(
        face_update_count=jnp.ones_like(owner_state.face_update_count),
        face_proxy_reference_real=0.30 * mask,
        face_proxy_reference_imag=0.25 * mask,
        face_proxy_reference_prev_real=0.30 * mask,
        face_proxy_reference_prev_imag=0.25 * mask,
        source_owner_reference_real=source_real,
        source_owner_reference_imag=source_imag,
        source_owner_reference_prev_real=source_real,
        source_owner_reference_prev_imag=source_imag,
        source_incident_normalizer_real=2.0 * source_mask,
        source_incident_normalizer_imag=jnp.zeros_like(source_mask),
    )


def test_project_private_modal_basis_packets_projects_source_onto_shared_basis():
    source_real = jnp.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=jnp.float32)
    source_imag = jnp.zeros_like(source_real)
    interface_real = 5.0 * jnp.ones_like(source_real)
    interface_imag = jnp.zeros_like(source_real)
    normalizer_real = jnp.ones_like(source_real)
    normalizer_imag = jnp.zeros_like(source_real)
    packet_weight = jnp.ones_like(source_real)
    packet_mask = jnp.ones_like(source_real)

    target_real, target_imag, projection_gate = (
        _project_private_modal_basis_packets(
            source_real=source_real,
            source_imag=source_imag,
            interface_real=interface_real,
            interface_imag=interface_imag,
            normalizer_real=normalizer_real,
            normalizer_imag=normalizer_imag,
            source_weight=packet_weight,
            interface_weight=packet_weight,
            source_mask=packet_mask,
            interface_mask=packet_mask,
        )
    )

    np.testing.assert_allclose(np.asarray(projection_gate), 1.0)
    np.testing.assert_allclose(np.asarray(target_real), 3.0)
    np.testing.assert_allclose(np.asarray(target_imag), 0.0)


def test_project_private_modal_basis_packets_fails_closed_without_basis_energy():
    packet = jnp.ones((2, 2), dtype=jnp.float32)

    target_real, target_imag, projection_gate = (
        _project_private_modal_basis_packets(
            source_real=packet,
            source_imag=packet,
            interface_real=2.0 * packet,
            interface_imag=3.0 * packet,
            normalizer_real=jnp.zeros_like(packet),
            normalizer_imag=jnp.zeros_like(packet),
            source_weight=packet,
            interface_weight=packet,
            source_mask=packet,
            interface_mask=packet,
        )
    )

    np.testing.assert_allclose(np.asarray(projection_gate), 0.0)
    np.testing.assert_allclose(np.asarray(target_real), 2.0)
    np.testing.assert_allclose(np.asarray(target_imag), 3.0)


def test_propagation_aware_modal_retry_helper_uses_source_owner_packet():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _private_owner_with_source_interface_reference(
        config,
        source_active=True,
    )
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = (
        _apply_propagation_aware_modal_retry_face_helper(
            e_coarse,
            e_fine,
            h_coarse,
            h_fine,
            owner_state,
            config,
        )
    )

    for before, after in zip(h_coarse, out_h_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_fine, out_h_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)

    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        before_c = extract_tangential_e_face(e_coarse, config, face, grid="coarse")
        after_c = extract_tangential_e_face(out_e_c, config, face, grid="coarse")
        before_f = extract_tangential_e_face(e_fine, config, face, grid="fine")
        after_f = extract_tangential_e_face(out_e_f, config, face, grid="fine")
        coarse_active = np.asarray(coarse_mask) > 0.0
        fine_active = np.asarray(fine_mask) > 0.0
        delta_c0 = np.asarray(after_c[0] - before_c[0])
        delta_c1 = np.asarray(after_c[1] - before_c[1])
        delta_f0 = np.asarray(after_f[0] - before_f[0])
        delta_f1 = np.asarray(after_f[1] - before_f[1])
        assert np.any(np.abs(delta_c0[coarse_active]) > 0.0)
        assert np.any(np.abs(delta_c1[coarse_active]) > 0.0)
        assert np.max(np.abs(delta_c0[coarse_active])) <= 0.20 * 0.01 + 1.0e-8
        assert np.max(np.abs(delta_c1[coarse_active])) <= 0.20 * 0.01 + 1.0e-8
        assert np.max(np.abs(delta_f0[fine_active])) <= 0.20 * 0.01 + 1.0e-8
        assert np.max(np.abs(delta_f1[fine_active])) <= 0.20 * 0.01 + 1.0e-8


def test_propagation_aware_modal_retry_helper_requires_private_contract():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _private_owner_with_source_interface_reference(
        config,
        source_active=True,
    )
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)
    contract_breakers = (
        owner_state._replace(
            source_normal_sign=-owner_state.source_normal_sign,
        ),
        owner_state._replace(
            source_incident_normalizer_real=jnp.zeros_like(
                owner_state.source_incident_normalizer_real
            ),
        ),
        owner_state._replace(
            source_owner_weight=2.0 * owner_state.source_owner_weight,
        ),
    )

    for broken_owner_state in contract_breakers:
        out_e_c, out_e_f, out_h_c, out_h_f = (
            _apply_propagation_aware_modal_retry_face_helper(
                e_coarse,
                e_fine,
                h_coarse,
                h_fine,
                broken_owner_state,
                config,
            )
        )

        for before_fields, after_fields in (
            (e_coarse, out_e_c),
            (e_fine, out_e_f),
            (h_coarse, out_h_c),
            (h_fine, out_h_f),
        ):
            for before, after in zip(before_fields, after_fields):
                np.testing.assert_allclose(
                    np.asarray(after),
                    np.asarray(before),
                    atol=0.0,
                )


def test_propagation_aware_modal_retry_helper_waits_for_source_packet():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    owner_state = _private_owner_with_source_interface_reference(
        config,
        source_active=False,
    )
    e_coarse = (state.ex_c, state.ey_c, state.ez_c)
    e_fine = (state.ex_f, state.ey_f, state.ez_f)
    h_coarse = (state.hx_c, state.hy_c, state.hz_c)
    h_fine = (state.hx_f, state.hy_f, state.hz_f)

    out_e_c, out_e_f, out_h_c, out_h_f = (
        _apply_propagation_aware_modal_retry_face_helper(
            e_coarse,
            e_fine,
            h_coarse,
            h_fine,
            owner_state,
            config,
        )
    )

    for before, after in zip(e_coarse, out_e_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(e_fine, out_e_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_coarse, out_h_c):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)
    for before, after in zip(h_fine, out_h_f):
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=0.0)


def _seed_private_owner_scan_fields(state):
    return state._replace(
        ex_c=jnp.ones_like(state.ex_c),
        ey_c=2.0 * jnp.ones_like(state.ey_c),
        ez_c=3.0 * jnp.ones_like(state.ez_c),
        hx_c=0.10 * jnp.ones_like(state.hx_c),
        hy_c=0.20 * jnp.ones_like(state.hy_c),
        hz_c=0.30 * jnp.ones_like(state.hz_c),
        ex_f=1.5 * jnp.ones_like(state.ex_f),
        ey_f=2.5 * jnp.ones_like(state.ey_f),
        ez_f=3.5 * jnp.ones_like(state.ez_f),
        hx_f=0.15 * jnp.ones_like(state.hx_f),
        hy_f=0.25 * jnp.ones_like(state.hy_f),
        hz_f=0.35 * jnp.ones_like(state.hz_f),
    )


def _assert_private_owner_joint_score(owner_state, config):
    assert owner_state is not None
    assert owner_state.face_magnitude_reference.shape == (len(_active_faces(config)),)
    assert owner_state.face_phase_reference.shape == (len(_active_faces(config)),)
    _assert_private_owner_packet_shape(owner_state, config)
    assert np.all(np.isfinite(np.asarray(owner_state.face_magnitude_reference)))
    assert np.all(np.isfinite(np.asarray(owner_state.face_phase_reference)))
    assert np.any(np.asarray(owner_state.face_magnitude_reference) > 0.0)
    assert np.all(np.isfinite(np.asarray(owner_state.face_proxy_reference_real)))
    assert np.all(np.isfinite(np.asarray(owner_state.face_proxy_reference_imag)))
    active_proxy = np.asarray(owner_state.face_proxy_mask) > 0.0
    assert np.any(np.abs(np.asarray(owner_state.face_proxy_reference_real)[active_proxy]) > 0.0)
    active_source = np.asarray(owner_state.source_owner_mask) > 0.0
    assert np.all(np.isfinite(np.asarray(owner_state.source_owner_reference_real)))
    assert np.all(np.isfinite(np.asarray(owner_state.source_owner_reference_imag)))
    assert np.any(
        np.abs(np.asarray(owner_state.source_owner_reference_real)[active_source])
        > 0.0
    )
    score = _private_interface_owner_joint_score(owner_state)
    assert int(np.asarray(score.usable_face_count)) == len(_active_faces(config))
    assert np.isfinite(float(np.asarray(score.transverse_magnitude_cv)))
    assert np.isfinite(float(np.asarray(score.transverse_phase_spread_deg)))
    assert float(np.asarray(score.transverse_magnitude_cv)) >= 0.0
    assert float(np.asarray(score.transverse_phase_spread_deg)) >= 0.0


def test_private_interface_owner_scan_wiring_is_after_same_step_eh_sat():
    for step_func in (step_subgrid_3d, step_subgrid_3d_with_cpml):
        source = inspect.getsource(step_func)
        h_sat_index = source.index("apply_sat_h_interfaces")
        e_sat_index = source.index("apply_sat_e_interfaces")
        helper_index = source.index("_apply_time_centered_paired_face_helper")
        proxy_index = source.index("_apply_observable_proxy_modal_retry_face_helper")
        stage_index = source.index("_stage_private_time_aligned_owner_packets")
        source_index = source.index("_update_private_source_owner_state_from_scan")
        propagation_index = source.index(
            "_apply_propagation_aware_modal_retry_face_helper"
        )
        scan_index = source.index("_update_private_interface_owner_state_from_scan")
        assert (
            h_sat_index
            < e_sat_index
            < helper_index
            < proxy_index
            < stage_index
            < source_index
            < propagation_index
            < scan_index
        )


def test_private_interface_owner_scan_wiring_records_joint_score_non_cpml():
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    state = _seed_private_owner_scan_fields(state)

    next_state = step_subgrid_3d(state, config)

    owner_state = next_state.private_interface_owner_state
    np.testing.assert_array_equal(np.asarray(owner_state.face_update_count), 1)
    _assert_private_owner_joint_score(owner_state, config)


def test_private_interface_owner_scan_wiring_records_joint_score_cpml():
    grid = Grid(
        freq_max=5e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.004,
        cpml_layers=1,
    )
    config, state = init_subgrid_3d(
        shape_c=grid.shape,
        dx_c=grid.dx,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    cpml_params, cpml_state = init_cpml(grid)
    state = _seed_private_owner_scan_fields(state)

    next_state, _ = step_subgrid_3d_with_cpml(
        state,
        config,
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        grid_c=grid,
        cpml_axes="xyz",
    )

    owner_state = next_state.private_interface_owner_state
    np.testing.assert_array_equal(np.asarray(owner_state.face_update_count), 1)
    _assert_private_owner_joint_score(owner_state, config)


def _spy_operator_projected_helper(monkeypatch):
    calls = []
    original = sbp_sat_3d._apply_operator_projected_skew_eh_face_helper

    def spy(e_coarse, e_fine, h_coarse, h_fine, config):
        calls.append(
            {
                "coarse_shape": tuple(e_coarse[0].shape),
                "fine_shape": tuple(e_fine[0].shape),
                "active_faces": tuple(_active_faces(config)),
            }
        )
        return original(e_coarse, e_fine, h_coarse, h_fine, config)

    monkeypatch.setattr(
        sbp_sat_3d,
        "_apply_operator_projected_skew_eh_face_helper",
        spy,
    )
    return calls


def test_operator_projected_helper_executes_under_representative_non_cpml_boundaries(
    monkeypatch,
):
    config, state = init_subgrid_3d(
        shape_c=(8, 8, 8),
        dx_c=0.004,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    state = _seed_all_components_for_box(config, state)
    calls = _spy_operator_projected_helper(monkeypatch)
    non_cpml_cases = (
        (
            "all_pec",
            {},
        ),
        (
            "selected_pmc_reflector",
            {
                "outer_pec_faces": frozenset(
                    {"x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
                ),
                "outer_pmc_faces": frozenset({"x_lo"}),
            },
        ),
        (
            "periodic_axis_interior_box",
            {
                "outer_pec_faces": frozenset({"y_lo", "y_hi", "z_lo", "z_hi"}),
                "periodic": (True, False, False),
                "fine_periodic": (True, False, False),
            },
        ),
    )

    for case_name, kwargs in non_cpml_cases:
        calls.clear()
        next_state = step_subgrid_3d(state, config, **kwargs)

        assert next_state.step == state.step + 1, case_name
        assert len(calls) == 1, case_name
        assert calls[0]["coarse_shape"] == state.ex_c.shape
        assert calls[0]["fine_shape"] == state.ex_f.shape
        assert calls[0]["active_faces"] == tuple(_active_faces(config))


def test_operator_projected_helper_executes_under_representative_cpml_boundary(
    monkeypatch,
):
    grid = Grid(
        freq_max=5e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.004,
        cpml_layers=1,
    )
    assert grid.shape == (8, 8, 8)
    config, state = init_subgrid_3d(
        shape_c=grid.shape,
        dx_c=grid.dx,
        fine_region=(2, 6, 2, 6, 2, 6),
        ratio=2,
    )
    state = _seed_all_components_for_box(config, state)
    cpml_params, cpml_state = init_cpml(grid)
    calls = _spy_operator_projected_helper(monkeypatch)

    next_state, next_cpml_state = step_subgrid_3d_with_cpml(
        state,
        config,
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        grid_c=grid,
        cpml_axes="xyz",
    )

    assert next_state.step == state.step + 1
    assert next_cpml_state is not None
    assert len(calls) == 1
    assert calls[0]["coarse_shape"] == state.ex_c.shape
    assert calls[0]["fine_shape"] == state.ex_f.shape
    assert calls[0]["active_faces"] == tuple(_active_faces(config))


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
