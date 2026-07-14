"""AD regression coverage for the 3D ADI-FDTD scheme (issue #338 follow-up).

The Zheng-Chen-Zhang two-sub-step 3D ADI (``rfx.adi.adi_step_3d``) replaced
an LOD-with-artificial-diffusion scheme whose gradient behavior was known to
go NaN under some configurations; that specific regression was fixed during
development but never pinned as a permanent test at the level where the bug
actually was (``tests/test_adi.py::TestThomasSolve.test_differentiable``
only covers the low-level tridiagonal-solve primitive, not ``adi_step_3d``
or the full 3D-ADI ``Simulation`` forward path). This file closes that gap.

Deliberately NOT gpu-marked (unlike ``tests/test_adi.py`` and
``tests/test_gradient_coverage.py``, both module-level
``pytestmark = pytest.mark.gpu``) — this grid is tiny and CPU-fast, so it
runs in the default/fast pytest lane where a future regression would
actually be caught on every push.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.adi import adi_step_3d

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


def test_adi_step_3d_gradient_is_finite_and_nonzero():
    """jax.grad of a scalar field-energy loss w.r.t. the source amplitude,
    through several unrolled adi_step_3d calls, must be finite and nonzero.

    Small grid (8^3) and few steps (10) keep this CPU-fast; the point is
    regression coverage of AD-through-adi_step_3d, not accuracy.
    """
    nx = ny = nz = 8
    dx = dy = dz = 2e-3
    dt_yee = dx / (C0 * np.sqrt(3.0)) * 0.99
    dt = dt_yee * 2.0  # matches the tier1-committed 2x CFL rung
    n_steps = 10

    eps_r = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    sigma = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

    def loss(amplitude):
        zeros = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
        ex, ey, ez, hx, hy, hz = zeros, zeros, zeros, zeros, zeros, zeros
        ez = ez.at[nx // 2, ny // 2, nz // 2].set(amplitude)
        for _ in range(n_steps):
            ex, ey, ez, hx, hy, hz = adi_step_3d(
                ex, ey, ez, hx, hy, hz, eps_r, sigma, dt, dx, dy, dz,
            )
        return jnp.sum(ex ** 2) + jnp.sum(ey ** 2) + jnp.sum(ez ** 2)

    amplitude = jnp.float32(1.0)
    val, grad = jax.value_and_grad(loss)(amplitude)

    val = float(val)
    grad = float(grad)
    print(f"\nadi_step_3d AD: loss={val:.6e}, d(loss)/d(amplitude)={grad:.6e}")

    assert np.isfinite(val), f"loss is not finite: {val}"
    assert val > 0.0, "loss should be positive (nonzero field energy)"
    assert np.isfinite(grad), f"gradient through adi_step_3d is not finite: {grad}"
    assert abs(grad) > 0.0, "gradient through adi_step_3d is exactly zero"


def test_adi_step_3d_gradient_with_internal_pec_mask_is_finite():
    """The same AD path with an internal pec_mask applied (rfx/adi.py:748-750
    post-solve-projection path) must also stay finite — the jnp.where-based
    masking is the mechanism most likely to break reverse-mode AD if it were
    ever changed to a non-differentiable indexing form.
    """
    nx = ny = nz = 8
    dx = dy = dz = 2e-3
    dt_yee = dx / (C0 * np.sqrt(3.0)) * 0.99
    dt = dt_yee * 2.0
    n_steps = 10

    eps_r = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    sigma = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    pec_mask = jnp.zeros((nx, ny, nz), dtype=bool)
    pec_mask = pec_mask.at[nx // 2, ny // 2, :].set(True)  # a thin internal post

    def loss(amplitude):
        zeros = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
        ex, ey, ez, hx, hy, hz = zeros, zeros, zeros, zeros, zeros, zeros
        ez = ez.at[nx // 4, ny // 2, nz // 2].set(amplitude)
        for _ in range(n_steps):
            ex, ey, ez, hx, hy, hz = adi_step_3d(
                ex, ey, ez, hx, hy, hz, eps_r, sigma, dt, dx, dy, dz,
                pec_mask=pec_mask,
            )
        return jnp.sum(ex ** 2) + jnp.sum(ey ** 2) + jnp.sum(ez ** 2)

    amplitude = jnp.float32(1.0)
    val, grad = jax.value_and_grad(loss)(amplitude)

    val = float(val)
    grad = float(grad)
    print(f"\nadi_step_3d AD (with pec_mask): loss={val:.6e}, grad={grad:.6e}")

    assert np.isfinite(val), f"loss is not finite: {val}"
    assert np.isfinite(grad), (
        f"gradient through adi_step_3d with an internal pec_mask is not "
        f"finite: {grad}"
    )
