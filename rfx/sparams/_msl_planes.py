"""One reference-plane resolver for microstrip (MSL) ports.

A microstrip port launches its quasi-TEM wave at the feed plane the user
declares and measures V and I at probe 0, ``n_probe_offset`` cells further
in. The S calculators form S at probe 0; this module moves it back to the
declared feed plane, so that S11 is the reflection seen at the port the user
placed. Both ``compute_msl_s_matrix`` and the MSL rows/columns of
``compute_mixed_s_matrix`` call it; neither keeps its own expression
(measurement plan MSL-PLANE.md, decision record 5, D1-D3).

Per MSL port ``p``:

* ``L_p = s * (X(i_p + s*offset_p) - X(i_p))`` from the REALIZED node
  coordinates (``X`` = ``_msl_coord_for_index`` on the calculator's grid,
  ``i_p`` the feed index exactly as the probe ladder resolves it, ``s`` the
  direction sign). On a graded mesh this is the sum of the realized cells,
  not ``offset * dx``.
* the shift beta per bin: the real part of the port's own-drive fitted beta,
  or the Hammerstad-Jensen beta where that fit railed at its scan window.
* ``D_p = exp(-j * beta_p * L_p)`` and ``S_feed[i, j] = D_i * S[i, j] * D_j``.

The sign follows the code's DFT convention as measured (the phase of S21
falls with line length), so moving the plane outward by ``L`` multiplies by
``exp(-j*beta*L)`` per port. beta is real, so ``|D_p| = 1``: magnitudes,
singular values (passivity) and cond(A) do not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class MSLPlaneShift:
    """Resolved feed-plane shift for the MSL ports of one calculator call.

    ``offsets_m`` is a NumPy float64 array (geometry, never traced);
    ``shift_beta`` (rad/m, real) and ``closed_form`` follow the fit's
    dataflow and may be JAX tracers on a differentiable channel.
    """
    offsets_m: np.ndarray        # (n_msl,)
    shift_beta: jnp.ndarray      # (n_msl, n_freqs) real
    closed_form: jnp.ndarray     # (n_msl, n_freqs) bool


def msl_feed_to_probe0_distance(grid, port, n_probe_offset: int,
                                probe0_coordinate: float) -> float:
    """Realized distance (m) from a port's feed node to its probe-0 node.

    ``port`` is the calculator's :class:`~rfx.sources.msl_port.MSLPort`;
    ``probe0_coordinate`` is the calculator's own ``probe_xs[p][0]``, which
    this asserts equals the node the formula lands on (one ladder, two
    readers).
    """
    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX,
        _msl_coord_for_index,
        _msl_position_to_index,
        msl_axis_roles,
        msl_physical_point,
    )
    prop, _w, _n, sign_f = msl_axis_roles(port.direction)
    sign = int(sign_f)
    idx = _msl_position_to_index(
        grid, msl_physical_point(port.direction, port.feed_x, port.y_lo,
                                 port.z_lo))
    i_feed = int(idx[_MSL_AXIS_INDEX[prop]])
    x_feed = float(_msl_coord_for_index(grid, prop, i_feed))
    x_probe0 = float(_msl_coord_for_index(
        grid, prop, i_feed + sign * int(n_probe_offset)))
    if x_probe0 != float(probe0_coordinate):
        raise AssertionError(
            "MSL feed-plane resolver: probe 0 of the resolver "
            f"({x_probe0!r} m) is not the calculator's probe 0 "
            f"({float(probe0_coordinate)!r} m); the plane shift would be "
            "taken over a different distance than the one S was measured "
            "across.")
    distance = sign * (x_probe0 - x_feed)
    if not distance > 0.0:
        raise AssertionError(
            f"MSL feed-plane resolver: realized feed-to-probe-0 distance "
            f"{distance!r} m is not positive (feed node {x_feed!r} m, "
            f"probe 0 {x_probe0!r} m, direction {port.direction!r}).")
    return float(distance)


def resolve_msl_planes(grid, ports, n_probe_offsets, probe0_coordinates,
                       beta_fit, beta_closed_form, railed) -> MSLPlaneShift:
    """Resolve L, the shift beta and the closed-form mask per MSL port.

    Parameters
    ----------
    grid : the calculator's realized grid.
    ports : sequence of :class:`~rfx.sources.msl_port.MSLPort`.
    n_probe_offsets : resolved ``n_probe_offset`` per port (cells).
    probe0_coordinates : the calculator's ``probe_xs[p][0]`` per port.
    beta_fit : per port, the own-drive fitted beta (n_freqs,), any dtype.
    beta_closed_form : per port, the Hammerstad-Jensen beta (n_freqs,) the
        calculator already formed (``beta0_per_port``).
    railed : per port, the own-drive ``beta_railed`` (n_freqs,) bool.
    """
    offsets = np.asarray([
        msl_feed_to_probe0_distance(grid, port, off, x0)
        for port, off, x0 in zip(ports, n_probe_offsets, probe0_coordinates)
    ], dtype=np.float64)
    rows = []
    masks = []
    for b_fit, b_hj, r in zip(beta_fit, beta_closed_form, railed):
        mask = jnp.asarray(r, dtype=bool)
        fitted = jnp.real(jnp.asarray(b_fit))
        closed = jnp.asarray(np.asarray(b_hj, dtype=np.float64),
                             dtype=fitted.dtype)
        rows.append(jnp.where(mask, closed, fitted))
        masks.append(mask)
    return MSLPlaneShift(offsets_m=offsets,
                         shift_beta=jnp.stack(rows),
                         closed_form=jnp.stack(masks))


def apply_feed_plane_shift(S, offsets_m, shift_beta):
    """``S_feed[i, j, f] = D_i(f) * S[i, j, f] * D_j(f)``, ``D = exp(-j beta L)``.

    ``offsets_m`` (n_ports,) and ``shift_beta`` (n_ports, n_freqs) cover
    every port of ``S``; a port with ``L = 0`` (a lumped/wire port in the
    mixed calculator) has ``D = 1`` exactly.
    """
    S = jnp.asarray(S)
    real_dtype = jnp.real(S).dtype
    L = jnp.asarray(np.asarray(offsets_m, dtype=np.float64),
                    dtype=real_dtype)
    beta = jnp.asarray(shift_beta).astype(real_dtype)
    d = jnp.exp(-1j * (beta * L[:, None])).astype(S.dtype)
    return d[:, None, :] * S * d[None, :, :]


def full_port_shift(n_ports, msl_port_indices, plane: MSLPlaneShift,
                    n_freqs):
    """Embed the MSL rows of ``plane`` into all-port arrays (others L = 0)."""
    offsets = np.zeros(n_ports, dtype=np.float64)
    offsets[list(msl_port_indices)] = plane.offsets_m
    beta = jnp.zeros((n_ports, n_freqs), dtype=plane.shift_beta.dtype)
    mask = jnp.zeros((n_ports, n_freqs), dtype=bool)
    idx = jnp.asarray(list(msl_port_indices), dtype=jnp.int32)
    beta = beta.at[idx].set(plane.shift_beta)
    mask = mask.at[idx].set(plane.closed_form)
    return offsets, beta, mask
