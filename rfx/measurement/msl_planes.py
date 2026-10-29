"""One reference-plane resolver for microstrip (MSL) ports.

A microstrip port launches its quasi-TEM wave at the feed plane the user
declares and measures V and I at probe 0, ``n_probe_offset`` cells further
in. The S calculators form S at probe 0; this module moves it back to the
declared feed plane, so that S11 is the reflection seen at the port the user
placed. Both ``compute_msl_s_matrix`` and the MSL rows/columns of
``compute_mixed_s_matrix`` call it; neither keeps its own expression
(measurement plan MSL-PLANE.md, decision records 5 and 7).

Per MSL port ``p``:

* ``L_p = s * (X(i_p + s*offset_p) - X(i_p))`` from the REALIZED node
  coordinates (``X`` = ``_msl_coord_for_index`` on the calculator's grid,
  ``i_p`` the feed index exactly as the probe ladder resolves it, ``s`` the
  direction sign). On a graded mesh this is the sum of the realized cells,
  not ``offset * dx``.
* the shift beta per bin, always the closed form:
  ``beta_p(f) = (2 pi f / c) * sqrt(eps_eff_KJ(f))`` -- the
  Kirschning-Jansen dispersion (:func:`kirschning_jansen_eps_eff`) on top of
  rfx's Hammerstad-Jensen ``eps_eff(0)``, from the port's width, height and
  the substrate eps_r the calculator resolved. The fitted beta never enters
  S; it stays a reported diagnostic.
* ``D_p = exp(-j * beta_p * L_p)`` and ``S_feed[i, j] = D_i * S[i, j] * D_j``.

The sign follows the code's DFT convention as measured (the phase of S21
falls with line length), so moving the plane outward by ``L`` multiplies by
``exp(-j*beta*L)`` per port. beta is real, so ``|D_p| = 1``: magnitudes,
singular values (passivity) and cond(A) do not change. beta depends only on
Python floats, so no gradient flows through it.

Accuracy of the closed-form beta against the simulated line (two-line beta,
two lines differing only in length, probes at the auto offset; MSL-PLANE.md
decision records 10-15 in bk-squared/rfx-archive): within 0.76 % on every
bin of the resolved boards measured (the microstrip chain battery thru at
100 um, 50 um and on a z-graded substrate, and an eps_r 10.2 / h 0.6 mm
board), where the shift's phase error 2 L |beta - beta_2L| is at most
5.5e-3 rad; within 1.23 % on a 4-cell substrate (0.8 mm of eps_r 2.2,
dx 0.2 mm, 2-18 GHz, L = 4 mm), where the phase error is at most 1.9e-2 rad
(at most 1.33e-2 rad up to 15 GHz).

Scope. The shift is a pure phase factor, which is exact for a line whose
characteristic impedance equals the power-wave reference R of the port (the
closed-form Hammerstad-Jensen Z0); where the simulated line's Zc differs from
R, moving the plane along it also mixes the a and b waves, which a phase
factor does not represent. On the ``eps_override`` channel the calculator
therefore forms beta AND the reference from the permittivity the solve used
under the port line (:func:`msl_line_beta_z0`, traced), applies the phase in
that line's own reference, and only then refers S to ``impedance=`` (#2199).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(kw_only=True)
class MSLFeedPlaneFields:
    """Where a microstrip port's S is referred (MSL-PLANE.md records 5, 7).

    Attributes
    ----------
    reference_plane_offsets_m : (n_ports,) float, optional
        Realized distance in metres from each port's declared feed node to
        its probe-0 node, read from the grid's node coordinates (on a graded
        mesh the sum of the realized cells). 0 for lumped/wire ports of the
        mixed calculator.
    shift_beta : (n_ports, n_freqs) float, optional
        rad/m. The closed-form beta, ``(2 pi f / c) sqrt(eps_eff(f))``:
        Hammerstad-Jensen ``eps_eff(0)`` with Kirschning-Jansen dispersion,
        from the port's width, height and resolved substrate eps_r. Never
        the fitted ``beta``. 0 for lumped/wire ports.

    ``S`` is the probe-0 S times ``D_i D_j`` with
    ``D_p = exp(-1j * shift_beta[p] * reference_plane_offsets_m[p])``, so
    the probe-0 matrix is ``S / (D_i D_j)``. Phase only: ``|D_p| = 1``. In
    the mixed result ``S_wave`` gets the same shift.
    """
    reference_plane_offsets_m: np.ndarray | None = None
    shift_beta: np.ndarray | None = None


def kirschning_jansen_eps_eff(eps_eff0: float, width: float, height: float,
                              eps_r: float, freqs) -> np.ndarray:
    """Dispersive microstrip effective permittivity eps_eff(f).

    M. Kirschning and R. H. Jansen, "Accurate model for effective dielectric
    constant of microstrip with validity up to millimetre-wave frequencies",
    Electron. Lett. 18(6):272-274, 1982; as given in B. C. Wadell,
    Transmission Line Design Handbook, Artech House 1991, section 3.5.1:

        fn = f[GHz] * h[mm],  u = w / h
        P1 = 0.27488 + (0.6315 + 0.525 / (1 + 0.0157 fn)^20) u
             - 0.065683 exp(-8.7513 u)
        P2 = 0.33622 (1 - exp(-0.03442 eps_r))
        P3 = 0.0363 exp(-4.6 u) (1 - exp(-(fn / 38.7)^4.97))
        P4 = 1 + 2.751 (1 - exp(-(eps_r / 15.916)^8))
        P  = P1 P2 ((0.1844 + P3 P4) fn)^1.5763
        eps_eff(f) = eps_r - (eps_r - eps_eff0) / (1 + P)

    ``eps_eff0`` is the quasi-static value (rfx uses its Hammerstad-Jensen
    form, :func:`rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff`).
    Lengths in metres, ``freqs`` in Hz.
    """
    f = np.asarray(freqs, dtype=np.float64)
    fn = (f * 1e-9) * (float(height) * 1e3)
    u = float(width) / float(height)
    er = float(eps_r)
    p1 = (0.27488 + (0.6315 + 0.525 / (1.0 + 0.0157 * fn) ** 20) * u
          - 0.065683 * np.exp(-8.7513 * u))
    p2 = 0.33622 * (1.0 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1.0 + 2.751 * (1.0 - np.exp(-(er / 15.916) ** 8))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return er - (er - float(eps_eff0)) / (1.0 + p)


def msl_line_beta_z0(width: float, height: float, eps_r, freqs):
    """Closed-form ``(beta(f), Z0)`` of a microstrip line, traceable in ``eps_r`` (#2199).

    The same formulas as :func:`msl_closed_form_beta` and
    :func:`rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff`, written
    in ``jnp`` so that ``eps_r`` may be a tracer (the permittivity an
    ``eps_override`` sets under the port line). The width/height branch is on
    Python floats. Returns ``beta`` (n_freqs,) rad/m and the scalar ``Z0``.
    """
    from rfx.grid import C0 as c0
    er = jnp.asarray(eps_r)
    u = float(width) / float(height)
    eps_eff0 = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5
    if u <= 1.0:
        z0 = (60.0 / jnp.sqrt(eps_eff0)) * np.log(8.0 / u + u / 4.0)
    else:
        z0 = 120.0 * np.pi / (
            jnp.sqrt(eps_eff0) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    f = np.asarray(freqs, dtype=np.float64)
    fn = (f * 1e-9) * (float(height) * 1e3)
    p1 = (0.27488 + (0.6315 + 0.525 / (1.0 + 0.0157 * fn) ** 20) * u
          - 0.065683 * np.exp(-8.7513 * u))
    p2 = 0.33622 * (1.0 - jnp.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1.0 + 2.751 * (1.0 - jnp.exp(-(er / 15.916) ** 8))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    eps_f = er - (er - eps_eff0) / (1.0 + p)
    return 2.0 * np.pi * f * jnp.sqrt(eps_f) / c0, z0


def readable_values(x):
    """The values of ``x`` as a NumPy array, or ``None`` when none exist.

    A concrete array is returned as is. Under an eager transform the values
    still exist inside the tracer: the primal of a ``grad``/``jacfwd`` tracer,
    the stacked batch of a ``vmap`` tracer (its batch axes then come first, in
    whatever order, so read only quantities whose every axis is a batch axis
    that way). Under ``jax.jit`` there are none.
    """
    try:
        return np.asarray(x)
    except Exception:
        pass
    to_concrete = getattr(x, "to_concrete_value", None)
    value = to_concrete() if to_concrete is not None else None
    if value is not None:
        return np.asarray(value)
    for inner in ("primal", "val"):
        value = getattr(x, inner, None)
        if isinstance(value, (jax.Array, np.ndarray)):
            return readable_values(value)
    return None


def msl_line_permittivity(eps_override, *, port_name: str, direction: str,
                          i_feed: int, sign: int, n_probe_offset: int,
                          j_centre: int, k_lo: int, k_hi: int):
    """The permittivity the solve used under a port line, for the plane shift (#2199).

    Reads ``eps_override`` on the trace-centre column, every substrate cell
    ``k_lo <= k < k_hi``, from the feed cell to the probe-0 cell inclusive.
    The shift applies ONE closed-form line over that length, so a
    permittivity that varies there is refused by name rather than read at
    one cell (per batch member under ``vmap``). Returns the value at
    ``(i_feed, j_centre, (k_lo + k_hi) // 2)`` -- the cell the calculator
    samples without an override -- traced.
    """
    from rfx._traced_refusal import TracedStructureRefusal
    from rfx.sources.msl_port import msl_cell
    i_end = int(i_feed) + int(sign) * int(n_probe_offset)
    i_range = range(min(i_feed, i_end), max(i_feed, i_end) + 1)
    k_range = range(int(k_lo), max(int(k_hi), int(k_lo) + 1))
    cells = [msl_cell(direction, i, j_centre, k) for i in i_range for k in k_range]
    idx = tuple(np.asarray(c) for c in zip(*cells))
    eps = jnp.asarray(eps_override)
    line = eps[idx]
    lo, hi = readable_values(jnp.min(line)), readable_values(jnp.max(line))
    if lo is None or hi is None:
        raise TracedStructureRefusal(
            f"compute_msl_s_matrix(eps_override=...): MSL port {port_name!r} "
            "refers S from probe 0 to its feed plane with one line formed from "
            "the permittivity under the line, and must check that it is "
            "uniform there; under jax.jit the override has no value to check. "
            "Call it outside jax.jit (jax.grad and jax.jacfwd work).")
    if np.any(hi - lo > 1e-6 * np.maximum(np.abs(hi), 1.0)):
        raise NotImplementedError(
            f"compute_msl_s_matrix(eps_override=...): MSL port {port_name!r} "
            f"sees a permittivity from {float(np.min(lo)):.6g} to "
            f"{float(np.max(hi)):.6g} under its line between the feed plane "
            f"and probe 0 ({len(i_range)} cells along the line, substrate "
            "cells under the trace centre). The plane shift moves S along "
            "that length with one closed-form line, which a varying "
            "permittivity does not have. Keep the override uniform under each "
            "port line from the feed to probe 0, or move the probes with "
            "n_probe_offset (#2199).")
    return eps[msl_cell(direction, i_feed, j_centre, (int(k_lo) + int(k_hi)) // 2)]


def msl_closed_form_beta(width: float, height: float, eps_r: float,
                         freqs) -> np.ndarray:
    """``beta(f) = (2 pi f / c) sqrt(eps_eff_KJ(f))`` in rad/m (float64)."""
    from rfx.grid import C0 as c0
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    _, eps_eff0 = hammerstad_jensen_z0_eps_eff(width, height, eps_r)
    f = np.asarray(freqs, dtype=np.float64)
    eps_f = kirschning_jansen_eps_eff(eps_eff0, width, height, eps_r, f)
    return 2.0 * np.pi * f * np.sqrt(eps_f) / c0


@dataclass(frozen=True)
class MSLPlaneShift:
    """Resolved feed-plane shift for the MSL ports of one calculator call."""
    offsets_m: np.ndarray        # (n_msl,) realized feed-to-probe-0 distance
    shift_beta: np.ndarray       # (n_msl, n_freqs) rad/m, closed form


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
                       widths, heights, eps_r, freqs) -> MSLPlaneShift:
    """Resolve L and the closed-form shift beta per MSL port.

    Parameters
    ----------
    grid : the calculator's realized grid.
    ports : sequence of :class:`~rfx.sources.msl_port.MSLPort`.
    n_probe_offsets : resolved ``n_probe_offset`` per port (cells).
    probe0_coordinates : the calculator's ``probe_xs[p][0]`` per port.
    widths, heights : the port's declared trace width and substrate height.
    eps_r : the substrate eps_r the calculator resolved per port
        (``eps_r_sub`` or the rasterized cell under the trace).
    freqs : (n_freqs,) Hz.
    """
    offsets = np.asarray([
        msl_feed_to_probe0_distance(grid, port, off, x0)
        for port, off, x0 in zip(ports, n_probe_offsets, probe0_coordinates)
    ], dtype=np.float64)
    beta = np.stack([
        msl_closed_form_beta(w, h, er, freqs)
        for w, h, er in zip(widths, heights, eps_r)
    ])
    return MSLPlaneShift(offsets_m=offsets, shift_beta=beta)


def feed_plane_factors(offsets_m, shift_beta):
    """``D_p(f) = exp(-j beta_p(f) L_p)`` as a complex128 (n_ports, n_freqs) array.

    The one place the feed-plane phase factor is formed: the calculators (through
    :func:`apply_feed_plane_shift`) and the dump replay both read it.
    """
    phase = (np.asarray(shift_beta, dtype=np.float64)
             * np.asarray(offsets_m, dtype=np.float64)[:, None])
    return np.exp(-1j * phase)


def apply_feed_plane_shift(S, offsets_m, shift_beta):
    """``S_feed[i, j, f] = D_i(f) * S[i, j, f] * D_j(f)``, ``D = exp(-j beta L)``.

    ``offsets_m`` (n_ports,) and ``shift_beta`` (n_ports, n_freqs) cover
    every port of ``S``; a port with ``L = 0`` (a lumped/wire port in the
    mixed calculator) has ``D = 1`` exactly. D is formed in float64 and cast
    to S's dtype.
    """
    S = jnp.asarray(S)
    if isinstance(shift_beta, jax.Array):
        # A traced or device beta (the eps_override channel, #2199): form D in jnp.
        phase = jnp.asarray(shift_beta) * jnp.asarray(offsets_m)[:, None]
        d = jnp.exp(-1j * phase).astype(S.dtype)
    else:
        d = jnp.asarray(feed_plane_factors(offsets_m, shift_beta)).astype(S.dtype)
    return d[:, None, :] * S * d[None, :, :]


def full_port_shift(n_ports, msl_port_indices, plane: MSLPlaneShift):
    """Embed the MSL rows of ``plane`` into all-port arrays (others L = 0)."""
    idx = list(msl_port_indices)
    offsets = np.zeros(n_ports, dtype=np.float64)
    offsets[idx] = plane.offsets_m
    beta = np.zeros((n_ports, plane.shift_beta.shape[1]), dtype=np.float64)
    beta[idx] = plane.shift_beta
    return offsets, beta
