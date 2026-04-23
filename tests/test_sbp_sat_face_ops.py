"""Operator tests for the canonical Phase-1 z-slab face ops."""

import numpy as np
import jax.numpy as jnp

from rfx.subgridding.face_ops import (
    build_zface_ops,
    check_norm_compatibility,
    prolong_zface,
    restrict_zface,
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


def test_scatter_roundtrip_zface_preserves_nonface_entries():
    config, state = init_subgrid_3d(
        shape_c=(6, 5, 8),
        dx_c=0.004,
        fine_region=(0, 6, 0, 5, 2, 5),
        ratio=2,
    )
    ex = jnp.arange(np.prod(state.ex_c.shape), dtype=jnp.float32).reshape(state.ex_c.shape)
    ey = ex + 1000.0
    ez = ex + 2000.0

    ex_face, ey_face = extract_tangential_e_face((ex, ey, ez), config, "z_lo", grid="coarse")
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
