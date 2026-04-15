"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

This module supplies the NU analogues of the uniform kernels used by
``rfx/runners/distributed_v2.py`` so the distributed path can accept
``NonUniformGrid`` without silently dropping the profile. The uniform
kernels in ``distributed.py`` remain the reference baseline and must
not be modified from this module.

Scope (Phase B minimal):
- x-axis shard only (1D slab decomposition).
- Global grading ratio <= 5:1 (shared single dt).
- x-axis CPML cells are uniform (guaranteed by make_nonuniform_grid
  boundary padding).
- TFSF single-device only (enforced upstream).
- Dispersive (Debye/Lorentz) E on NU distributed is NOT implemented
  here. The public entry point in ``distributed_v2`` falls back when
  dispersion is active.

Key helper: ``_build_sharded_inv_dx_arrays`` returns per-device
slabs of ``inv_dx`` / ``inv_dx_h`` whose slab boundary entry of
``inv_dx_h`` is derived from the global spacing straddling the slab
seam (NOT from the local slab alone) so H-field mean-spacing math
remains consistent across the shard boundary.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    MU_0,
    EPS_0,
    _shift_fwd,
    _shift_bwd,
)


# ---------------------------------------------------------------------------
# Sharded inv-spacing arrays
# ---------------------------------------------------------------------------

def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already padded the global x-extent by ``pad_x`` cells
    (to align ``nx`` on ``n_devices``).  We replicate that padding onto
    the cell-size profile using the boundary cell value (matching how
    ``make_nonuniform_grid`` pads CPML cells) and rebuild the global
    ``inv_dx`` and ``inv_dx_h`` from the padded profile, then reshape to
    per-device slabs.

    For ``inv_dx_h``, the last entry of each device's slab is the global
    mean-spacing straddling the slab seam with the next device (or 0 at
    the domain boundary), NOT derived from the local slab alone.

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of PEC-padded cells appended to the high-x end of the
        domain so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_global : (nx_padded,) np.ndarray
        Replicated — every device sees the whole thing when used with
        ``P("x")`` (see caller packing).
    inv_dx_h_global : (nx_padded,) np.ndarray
    dx_padded : (nx_padded,) np.ndarray
        The padded cell-size profile (float32) — useful for diagnostics
        and the unit test.
    """
    dx_arr = np.asarray(grid.dx_arr, dtype=np.float64)
    if pad_x > 0:
        # pad at the high-x end with boundary-cell-size value
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
    nx = dx_arr.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    inv_dx = 1.0 / dx_arr
    # inv_dx_h[i] = 2 / (dx[i] + dx[i+1]) for i<N-1 ; 0 at end.
    inv_dx_h_mean = 2.0 / (dx_arr[:-1] + dx_arr[1:])
    inv_dx_h = np.concatenate([inv_dx_h_mean, np.zeros(1, dtype=np.float64)])

    return (
        inv_dx.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------

def _update_h_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full,
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h_full[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h_full[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h_full[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h_slab[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h_slab[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h_full[None, :, None]
    )

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)
