"""Probe-0 S recovered from a microstrip result's own feed-plane fields.

``compute_msl_s_matrix`` / ``compute_mixed_s_matrix`` report S at each port's
declared feed plane: ``S = D_i S_probe0 D_j`` with
``D_p = exp(-1j * shift_beta[p] * reference_plane_offsets_m[p])``. Tests that
pin the PROBE-0 extraction (V/I wave split, multi-drive solve) against
planted or manufactured waves compare against ``probe0_s(result)``: the shift
undone by ``exp(+1j * beta * L)`` per port. Works on traced S (the factor is
a constant).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def probe0_s(result, S=None):
    """``S * exp(+j beta_i L_i) * exp(+j beta_j L_j)``; ``S`` defaults to result.S."""
    S = result.S if S is None else S
    beta = np.asarray(result.shift_beta, dtype=np.float64)
    offsets = np.asarray(result.reference_plane_offsets_m, dtype=np.float64)
    u = np.exp(1j * beta * offsets[:, None])
    factor = u[:, None, :] * u[None, :, :]
    if isinstance(S, np.ndarray):
        return S * factor.astype(S.dtype)
    S = jnp.asarray(S)
    return S * jnp.asarray(factor).astype(S.dtype)
