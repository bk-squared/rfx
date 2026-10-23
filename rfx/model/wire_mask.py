"""Rasterize a polyline tube on explicit node coordinates."""

import numpy as np
import jax
import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer


def polyline_tube_mask(points, radius, x, y, z):
    r2_thresh = radius ** 2
    pts = np.array(points, dtype=np.float64)

    # Filter out degenerate (zero-length) segments
    A = pts[:-1]  # (n_seg, 3)
    B = pts[1:]   # (n_seg, 3)
    D = B - A     # direction vectors
    seg_len2 = np.sum(D ** 2, axis=1)  # (n_seg,)
    valid = seg_len2 > 1e-30
    A = A[valid]
    D = D[valid]
    seg_len2 = seg_len2[valid]

    if len(A) == 0:
        return jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)

    if not any(is_tracer(c) for c in (x, y, z)):
        # #802: concrete coordinates — host float64 per-segment loop
        # (same clip/dist2 algebra as the traced scan below), so
        # surface-touching cells do not depend on jax_enable_x64. The
        # loop keeps the scan's memory profile: one (Nx,Ny,Nz) array
        # per step, never (n_seg, Nx, Ny, Nz) at once.
        xn = np.asarray(x, dtype=np.float64)[:, None, None]
        yn = np.asarray(y, dtype=np.float64)[None, :, None]
        zn = np.asarray(z, dtype=np.float64)[None, None, :]
        mask_np = np.zeros((xn.size, yn.size, zn.size), dtype=bool)
        for (ax, ay, az), (dx_s, dy_s, dz_s), sl2 in zip(A, D, seg_len2):
            t = ((xn - ax) * dx_s + (yn - ay) * dy_s
                 + (zn - az) * dz_s) / sl2
            t = np.clip(t, 0.0, 1.0)
            dist2 = ((xn - (ax + t * dx_s)) ** 2
                     + (yn - (ay + t * dy_s)) ** 2
                     + (zn - (az + t * dz_s)) ** 2)
            mask_np |= dist2 <= r2_thresh
        return jnp.asarray(mask_np)

    # Traced coordinates (mesh-as-design-variable): jnp path unchanged.
    # Pre-compute segment data as JAX arrays: (n_valid, 3) and (n_valid,)
    A_j = jnp.array(A, dtype=jnp.float32)
    D_j = jnp.array(D, dtype=jnp.float32)
    seg_len2_j = jnp.array(seg_len2, dtype=jnp.float32)

    # 3D coordinate grids (Nx, Ny, Nz)
    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    # Use lax.scan to accumulate boolean mask without materializing
    # all n_seg intermediate distance arrays simultaneously.
    def _scan_step(mask_acc, seg_idx):
        ax, ay, az = A_j[seg_idx, 0], A_j[seg_idx, 1], A_j[seg_idx, 2]
        dx_s, dy_s, dz_s = D_j[seg_idx, 0], D_j[seg_idx, 1], D_j[seg_idx, 2]
        sl2 = seg_len2_j[seg_idx]

        t = ((X - ax) * dx_s + (Y - ay) * dy_s + (Z - az) * dz_s) / sl2
        t = jnp.clip(t, 0.0, 1.0)

        cx = ax + t * dx_s
        cy = ay + t * dy_s
        cz = az + t * dz_s

        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        return mask_acc | (dist2 <= r2_thresh), None

    init_mask = jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)
    mask, _ = jax.lax.scan(
        _scan_step, init_mask, jnp.arange(len(A_j)))

    return mask
