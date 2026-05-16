from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.subgridding.jit_runner import _modal_filter_face_delta


def test_modal_xy_linear_trace_filter_extracts_diagonal_mode():
    axis = jnp.linspace(-1.0, 1.0, 5)
    face = axis[:, None] * axis[None, :]

    filtered = _modal_filter_face_delta(face, "modal_xy_linear_delta")

    np.testing.assert_allclose(np.asarray(filtered), np.asarray(face), atol=1e-6)


def test_modal_mixed_quadratic_trace_filter_extracts_mixed_modes():
    axis = jnp.linspace(-1.0, 1.0, 5)
    quadratic = axis * axis - jnp.mean(axis * axis)
    face = axis[:, None] * quadratic[None, :] + quadratic[:, None] * axis[None, :]

    filtered = _modal_filter_face_delta(face, "modal_mixed_quadratic_delta")

    np.testing.assert_allclose(np.asarray(filtered), np.asarray(face), atol=1e-6)


def test_modal_low2_trace_filter_removes_face_mean():
    face = jnp.ones((5, 5))

    filtered = _modal_filter_face_delta(face, "modal_low2_delta")

    np.testing.assert_allclose(np.asarray(filtered), np.zeros((5, 5)), atol=1e-6)
