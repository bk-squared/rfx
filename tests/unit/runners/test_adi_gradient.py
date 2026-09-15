"""AD regression coverage for the 3D ADI-FDTD scheme (issue #338 follow-up).

The Zheng-Chen-Zhang two-sub-step 3D ADI (``rfx.adi.adi_step_3d``) replaced
an LOD-with-artificial-diffusion scheme whose gradient behavior was known to
go NaN under some configurations; that specific regression was fixed during
development but never pinned as a permanent test at the level where the bug
actually was (``tests/unit/runners/test_adi.py::TestThomasSolve.test_differentiable``
only covers the low-level tridiagonal-solve primitive, not ``adi_step_3d``
or the full 3D-ADI ``Simulation`` forward path). This file closes that gap.

Deliberately NOT gpu-marked (unlike ``tests/unit/runners/test_adi.py`` and
``tests/unit/autodiff/test_gradient_coverage.py``, both module-level
``pytestmark = pytest.mark.gpu``) — this grid is tiny and CPU-fast, so it
runs in the default/fast pytest lane where a future regression would
actually be caught on every push.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MU_0
from rfx.adi import adi_step_3d
from rfx.boundaries.pec import realized_pec_edge_masks

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
    # Compile the small AD witness as one program for the shared CPU pod.
    val, grad = jax.jit(jax.value_and_grad(loss))(amplitude)

    val = float(val)
    grad = float(grad)
    print(f"\nadi_step_3d AD: loss={val:.6e}, d(loss)/d(amplitude)={grad:.6e}")

    assert np.isfinite(val), f"loss is not finite: {val}"
    assert val > 0.0, "loss should be positive (nonzero field energy)"
    assert np.isfinite(grad), f"gradient through adi_step_3d is not finite: {grad}"
    assert abs(grad) > 0.0, "gradient through adi_step_3d is exactly zero"


@pytest.mark.parametrize("transform", ["jit", "grad", "jit_grad"])
def test_adi_step_3d_refuses_internal_pec_under_transformations(transform):
    """Ten finite steps never certified projected-PEC stability.

    Passing the masks as transformed-function arguments makes them tracers
    under JIT. Refusal must survive tracing without a boolean conversion
    error, an ignored mask, or an attempt to execute unstable numerics.
    """
    nx = ny = nz = 8
    dx = dy = dz = 2e-3
    dt_yee = dx / (C0 * np.sqrt(3.0)) * 0.99
    dt = dt_yee * 5.0

    eps_r = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    sigma = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    cell_mask = jnp.zeros((nx, ny, nz), dtype=bool)
    cell_mask = cell_mask.at[nx // 2, ny // 2, :].set(True)  # internal post
    pec_edge_masks = realized_pec_edge_masks(cell_mask)

    def loss(amplitude, masks):
        zeros = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
        ex, ey, ez, hx, hy, hz = zeros, zeros, zeros, zeros, zeros, zeros
        ez = ez.at[nx // 4, ny // 2, nz // 2].set(amplitude)
        ex, ey, ez, hx, hy, hz = adi_step_3d(
            ex, ey, ez, hx, hy, hz, eps_r, sigma, dt, dx, dy, dz,
            pec_edge_masks=masks,
        )
        return jnp.sum(ex ** 2) + jnp.sum(ey ** 2) + jnp.sum(ez ** 2)

    transformed = {
        "jit": jax.jit(loss),
        "grad": jax.grad(loss),
        "jit_grad": jax.jit(jax.grad(loss)),
    }[transform]
    with pytest.raises(ValueError, match="adi_interior_pec_unsupported"):
        transformed(jnp.float32(1.0), pec_edge_masks)
