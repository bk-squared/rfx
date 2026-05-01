"""Operator tests for the canonical Phase-1 z-slab face ops."""

import numpy as np
import jax.numpy as jnp

from rfx.subgridding.face_ops import (
    build_edge_ops,
    build_face_ops,
    build_zface_ops,
    check_edge_norm_compatibility,
    check_norm_compatibility,
    prolong_edge,
    prolong_face,
    prolong_zface,
    restrict_edge,
    restrict_face,
    restrict_zface,
)

from rfx.subgridding.sbp_operators import (
    FaceFluxSpec,
    all_face_weighted_flux_report,
    box_surface_partition_report,
    build_sbp_first_derivative_1d,
    build_tensor_face_mortar,
    build_yee_staggered_derivative_pair_1d,
    face_mortar_adjoint_report,
    face_mortar_reproduction_report,
    sbp_identity_residual,
    weighted_em_flux_residual,
    yee_staggered_identity_residual,
)
from rfx.subgridding.sbp_sat_3d import (
    extract_tangential_e_face,
    extract_tangential_h_face,
    init_subgrid_3d,
    scatter_tangential_e_face,
)


def test_zface_norm_compatibility():
    ops = build_zface_ops((4, 3), ratio=2, dx_c=0.006)
    report = check_norm_compatibility(ops, atol=1e-6)
    assert report["passes"], report


def test_zface_restriction_matches_norm_definition():
    ops = build_zface_ops((3, 2), ratio=3, dx_c=0.009)
    coarse = jnp.ones((3, 2), dtype=jnp.float32) * 2.5
    fine = prolong_zface(coarse, ops)
    restricted = restrict_zface(fine, ops)
    np.testing.assert_allclose(np.array(restricted), np.array(coarse), atol=1e-6)


def test_zface_linear_prolongation_weights_are_explicit():
    ops = build_zface_ops((2, 2), ratio=2, dx_c=0.006)
    expected = np.array(
        [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(ops.prolong_i), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(ops.prolong_j), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(ops.restrict_i), expected.T / 2.0, atol=1e-6)
    np.testing.assert_allclose(np.array(ops.restrict_j), expected.T / 2.0, atol=1e-6)


def test_zface_restriction_is_pt_over_ratio():
    ops = build_zface_ops((3, 2), ratio=3, dx_c=0.009)
    np.testing.assert_allclose(
        np.array(ops.restrict_i),
        np.array(ops.prolong_i).T / ops.ratio,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(ops.restrict_j),
        np.array(ops.prolong_j).T / ops.ratio,
        atol=1e-6,
    )


def test_generic_face_ops_match_zface_ops():
    z_ops = build_zface_ops((3, 4), ratio=2, dx_c=0.006)
    g_ops = build_face_ops((3, 4), ratio=2, dx_c=0.006)
    np.testing.assert_allclose(
        np.array(g_ops.prolong_i), np.array(z_ops.prolong_i), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(g_ops.prolong_j), np.array(z_ops.prolong_j), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(g_ops.restrict_i), np.array(z_ops.restrict_i), atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(g_ops.restrict_j), np.array(z_ops.restrict_j), atol=1e-6
    )


def test_face_ops_aliases_match_zface_helpers():
    ops = build_face_ops((3, 4), ratio=2, dx_c=0.006)
    coarse = jnp.ones((3, 4), dtype=jnp.float32) * 3.25
    fine = prolong_face(coarse, ops)
    restricted = restrict_face(fine, ops)
    np.testing.assert_allclose(np.array(restricted), np.array(coarse), atol=1e-6)


def test_edge_norm_compatibility():
    ops = build_edge_ops(5, ratio=2, dx_c=0.006)
    report = check_edge_norm_compatibility(ops, atol=1e-6)
    assert report["passes"], report


def test_edge_restriction_matches_norm_definition():
    ops = build_edge_ops(4, ratio=3, dx_c=0.009)
    coarse = jnp.ones((4,), dtype=jnp.float32) * 1.75
    fine = prolong_edge(coarse, ops)
    restricted = restrict_edge(fine, ops)
    np.testing.assert_allclose(np.array(restricted), np.array(coarse), atol=1e-6)


def test_extract_tangential_e_zface_shapes():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(0, 6, 0, 5, 2, 5),
        ratio=2,
    )
    ex_face_c, ey_face_c = extract_tangential_e_face(
        (state.ex_c, state.ey_c, state.ez_c), config, "z_lo", grid="coarse"
    )
    ex_face_f, ey_face_f = extract_tangential_e_face(
        (state.ex_f, state.ey_f, state.ez_f), config, "z_lo", grid="fine"
    )
    assert ex_face_c.shape == (6, 5)
    assert ey_face_c.shape == (6, 5)
    assert ex_face_f.shape == (12, 10)
    assert ey_face_f.shape == (12, 10)


def test_extract_tangential_h_zface_shapes():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(0, 6, 0, 5, 2, 5),
        ratio=2,
    )
    hx_face_c, hy_face_c = extract_tangential_h_face(
        (state.hx_c, state.hy_c, state.hz_c), config, "z_hi", grid="coarse"
    )
    hx_face_f, hy_face_f = extract_tangential_h_face(
        (state.hx_f, state.hy_f, state.hz_f), config, "z_hi", grid="fine"
    )
    assert hx_face_c.shape == (6, 5)
    assert hy_face_c.shape == (6, 5)
    assert hx_face_f.shape == (12, 10)
    assert hy_face_f.shape == (12, 10)


def test_extract_tangential_e_xface_shapes():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(1, 5, 1, 4, 2, 6),
        ratio=2,
    )
    ey_face_c, ez_face_c = extract_tangential_e_face(
        (state.ex_c, state.ey_c, state.ez_c), config, "x_lo", grid="coarse"
    )
    ey_face_f, ez_face_f = extract_tangential_e_face(
        (state.ex_f, state.ey_f, state.ez_f), config, "x_lo", grid="fine"
    )
    assert ey_face_c.shape == (3, 4)
    assert ez_face_c.shape == (3, 4)
    assert ey_face_f.shape == (6, 8)
    assert ez_face_f.shape == (6, 8)


def test_extract_tangential_h_yface_shapes():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(1, 5, 1, 4, 2, 6),
        ratio=2,
    )
    hx_face_c, hz_face_c = extract_tangential_h_face(
        (state.hx_c, state.hy_c, state.hz_c), config, "y_hi", grid="coarse"
    )
    hx_face_f, hz_face_f = extract_tangential_h_face(
        (state.hx_f, state.hy_f, state.hz_f), config, "y_hi", grid="fine"
    )
    assert hx_face_c.shape == (4, 4)
    assert hz_face_c.shape == (4, 4)
    assert hx_face_f.shape == (8, 8)
    assert hz_face_f.shape == (8, 8)


def test_scatter_roundtrip_zface_preserves_nonface_entries():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(0, 6, 0, 5, 2, 5),
        ratio=2,
    )
    ex = jnp.arange(np.prod(state.ex_c.shape), dtype=jnp.float32).reshape(
        state.ex_c.shape
    )
    ey = ex + 1000.0
    ez = ex + 2000.0

    ex_face, ey_face = extract_tangential_e_face(
        (ex, ey, ez), config, "z_lo", grid="coarse"
    )
    ex_new = ex_face + 1.0
    ey_new = ey_face + 2.0
    ex_out, ey_out, ez_out = scatter_tangential_e_face(
        (ex, ey, ez), (ex_new, ey_new), config, "z_lo", grid="coarse"
    )

    k = config.fk_lo
    ex_masked = np.array(ex_out)
    ey_masked = np.array(ey_out)
    ex_masked[:, :, k] = np.array(ex)[:, :, k]
    ey_masked[:, :, k] = np.array(ey)[:, :, k]
    np.testing.assert_allclose(ex_masked, np.array(ex))
    np.testing.assert_allclose(ey_masked, np.array(ey))
    np.testing.assert_allclose(np.array(ez_out), np.array(ez))



def test_private_sbp_derivative_norm_boundary_contract():
    operator = build_sbp_first_derivative_1d(6, 0.004, grid_role="electric_primal")

    assert operator.derivative.shape == (6, 6)
    assert operator.norm.shape == (6,)
    assert np.all(np.asarray(operator.norm) > 0.0)
    assert operator.left_boundary[0] == 1.0
    assert operator.right_boundary[-1] == 1.0
    np.testing.assert_allclose(
        np.asarray(sbp_identity_residual(operator)),
        np.zeros((6, 6), dtype=np.float32),
        atol=1.0e-6,
    )


def test_private_yee_staggered_dual_derivative_contract():
    pair = build_yee_staggered_derivative_pair_1d(7, 0.004)

    assert pair.primal_to_dual.shape == (6, 7)
    assert pair.dual_to_primal.shape == (7, 6)
    assert np.all(np.asarray(pair.primal_norm) > 0.0)
    assert np.all(np.asarray(pair.dual_norm) > 0.0)
    np.testing.assert_allclose(
        np.asarray(yee_staggered_identity_residual(pair)),
        np.zeros((7, 6), dtype=np.float32),
        atol=1.0e-6,
    )


def test_private_norm_compatible_mortar_contract_is_adjoint_and_noop():
    mortar = build_tensor_face_mortar((4, 3), ratio=2, dx_c=0.004)

    adjoint = face_mortar_adjoint_report(mortar)
    reproduction = face_mortar_reproduction_report(mortar)

    assert mortar.fine_shape == (8, 6)
    assert adjoint["passes"], adjoint
    assert reproduction["passes"], reproduction
    assert reproduction["constant_max_error"] <= 1.0e-6
    assert reproduction["linear_i_max_error"] <= 1.0e-6
    assert reproduction["linear_j_max_error"] <= 1.0e-6


def test_private_weighted_em_flux_contract_closes_for_all_face_signs():
    mortar = build_tensor_face_mortar((4, 3), ratio=2, dx_c=0.004)
    i = jnp.arange(4, dtype=jnp.float32)[:, None]
    j = jnp.arange(3, dtype=jnp.float32)[None, :]
    ex = 1.0 + 0.1 * i + 0.2 * j
    ey = -0.5 + 0.05 * i - 0.1 * j
    zeros = jnp.zeros((4, 3), dtype=jnp.float32)
    hx = 2.0e-6 + 0.1e-6 * i + zeros
    hy = -1.0e-6 + 0.2e-6 * j + zeros
    metric = 1.0 + 0.01 * i + 0.02 * j

    for normal_sign in (-1, 1):
        residual = weighted_em_flux_residual(
            mortar,
            ex_c=ex,
            ey_c=ey,
            hx_c=hx,
            hy_c=hy,
            normal_sign=normal_sign,
            coarse_metric_weight=metric,
        )
        assert abs(residual) <= 1.0e-12


def test_private_all_face_surface_partition_and_flux_contract():
    face_specs = (
        FaceFluxSpec("x_lo", (4, 4), -1),
        FaceFluxSpec("x_hi", (4, 4), 1),
        FaceFluxSpec("y_lo", (4, 4), -1),
        FaceFluxSpec("y_hi", (4, 4), 1),
        FaceFluxSpec("z_lo", (4, 4), -1),
        FaceFluxSpec("z_hi", (4, 4), 1),
    )

    partition = box_surface_partition_report((4, 4, 4))
    flux = all_face_weighted_flux_report(face_specs, ratio=2, dx_c=0.004)

    assert partition["status"] == "all_face_edge_corner_accounting_closed"
    assert partition["active_faces"] == 6
    assert partition["active_edges"] == 12
    assert partition["active_corners"] == 8
    assert partition["surface_cells"] == 56
    assert partition["counted_surface_cells"] == partition["surface_cells"]
    assert partition["partition_closes"] is True
    assert flux["faces_tested"] == ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    assert flux["face_count"] == 6
    assert flux["passes"] is True
    assert flux["max_abs_residual"] <= 1.0e-12
