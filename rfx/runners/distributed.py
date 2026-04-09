"""Multi-GPU distributed FDTD runner using jax.pmap.

Uses 1D slab decomposition along the x-axis with ghost cell
exchange via jax.lax.ppermute.  Supports PEC and CPML boundaries,
soft sources, point probes, lumped ports, and dispersive materials
(Debye / Lorentz / Drude).

Phase 1: PEC boundary (1D slab decomposition)
Phase 2: CPML absorbing boundary support — "apply everywhere, override
         with ghosts" strategy.  Each device runs CPML on all 6 faces of
         its local slab.  Ghost exchange after CPML overwrites x-face
         artifacts on interior devices; physical boundary devices (first /
         last) keep their CPML absorption intact.
Phase 3: Lumped ports + Debye/Lorentz dispersive materials.
         Port impedance is folded into materials before splitting.
         ADE (auxiliary differential equation) state is carried through
         the scan loop; updates are purely local (no cross-device
         exchange needed for polarization fields).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    EPS_0,
    MU_0,
    _shift_fwd,
    _shift_bwd,
)
from rfx.simulation import (
    make_source,
    make_j_source,
    make_probe,
    make_port_source,
)
from rfx.sources.sources import LumpedPort, setup_lumped_port
from rfx.materials.debye import DebyeCoeffs, DebyeState
from rfx.materials.lorentz import LorentzCoeffs, LorentzState


# ---------------------------------------------------------------------------
# Domain splitting / gathering
# ---------------------------------------------------------------------------

def split_array_x(arr, n_devices, ghost=1, pad_value=0.0):
    """Split a 3D array into N slabs along x with ghost cells.

    Parameters
    ----------
    arr : ndarray, shape (nx, ny, nz)
    n_devices : int
    ghost : int
        Number of ghost cells on each side.
    pad_value : float
        Value used for ghost cells at the physical boundary (device 0
        left ghost and device N-1 right ghost).  Default 0.0 is correct
        for field arrays; use 1.0 for eps_r and mu_r to avoid division
        by zero in the Yee update.

    Returns
    -------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    """
    nx = arr.shape[0]
    nx_per = nx // n_devices
    slabs = []
    for i in range(n_devices):
        x_start = i * nx_per
        x_end = x_start + nx_per

        # Desired range including ghosts
        want_lo = x_start - ghost
        want_hi = x_end + ghost

        # Clamp to valid array range
        g_lo = max(0, want_lo)
        g_hi = min(nx, want_hi)

        slab_data = arr[g_lo:g_hi]

        # Pad where the desired range exceeds array bounds
        pad_lo = g_lo - want_lo   # > 0 when want_lo < 0
        pad_hi = want_hi - g_hi   # > 0 when want_hi > nx

        if pad_lo > 0 or pad_hi > 0:
            pad_widths = [(pad_lo, pad_hi)] + [(0, 0)] * (arr.ndim - 1)
            slab_data = jnp.pad(slab_data, pad_widths, mode='constant',
                                constant_values=pad_value)

        slabs.append(slab_data)
    return jnp.stack(slabs)


def gather_array_x(slabs, ghost=1):
    """Gather slabs back into a single array, stripping ghost cells.

    Parameters
    ----------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    ghost : int

    Returns
    -------
    arr : ndarray, shape (nx, ny, nz)
    """
    # Strip ghost cells from each slab and concatenate
    inner = slabs[:, ghost:-ghost, :, :]  # (n_devices, nx_per, ny, nz)
    n_devices = inner.shape[0]
    # Reshape: merge device and x dims
    nx_per = inner.shape[1]
    ny = inner.shape[2]
    nz = inner.shape[3]
    return inner.reshape(n_devices * nx_per, ny, nz)


def _split_state(state, n_devices, ghost=1):
    """Split an FDTDState into per-device slabs with ghost cells."""
    return FDTDState(
        ex=split_array_x(state.ex, n_devices, ghost),
        ey=split_array_x(state.ey, n_devices, ghost),
        ez=split_array_x(state.ez, n_devices, ghost),
        hx=split_array_x(state.hx, n_devices, ghost),
        hy=split_array_x(state.hy, n_devices, ghost),
        hz=split_array_x(state.hz, n_devices, ghost),
        step=jnp.broadcast_to(state.step, (n_devices,)),
    )


def _gather_state(state, ghost=1):
    """Gather per-device FDTDState slabs back into a single state."""
    return FDTDState(
        ex=gather_array_x(state.ex, ghost),
        ey=gather_array_x(state.ey, ghost),
        ez=gather_array_x(state.ez, ghost),
        hx=gather_array_x(state.hx, ghost),
        hy=gather_array_x(state.hy, ghost),
        hz=gather_array_x(state.hz, ghost),
        step=state.step[0],
    )


def _split_materials(materials, n_devices, ghost=1):
    """Split MaterialArrays into per-device slabs with ghost cells.

    Uses pad_value=1.0 for eps_r and mu_r (vacuum) to prevent
    division-by-zero in Yee updates at boundary ghost cells, and
    pad_value=0.0 for sigma (lossless).
    """
    return MaterialArrays(
        eps_r=split_array_x(materials.eps_r, n_devices, ghost, pad_value=1.0),
        sigma=split_array_x(materials.sigma, n_devices, ghost, pad_value=0.0),
        mu_r=split_array_x(materials.mu_r, n_devices, ghost, pad_value=1.0),
    )


# ---------------------------------------------------------------------------
# Dispersive material (Debye / Lorentz) splitting
# ---------------------------------------------------------------------------

def _split_debye_coeffs(coeffs: DebyeCoeffs, n_devices, ghost=1):
    """Split DebyeCoeffs arrays along x into per-device slabs.

    3-D arrays (nx, ny, nz) are split with ghost cells.
    4-D arrays (n_poles, nx, ny, nz) are split per-pole along x.
    """
    # ca, cb are (nx, ny, nz)
    ca = split_array_x(coeffs.ca, n_devices, ghost, pad_value=0.0)
    cb = split_array_x(coeffs.cb, n_devices, ghost, pad_value=0.0)

    # cc, alpha, beta are (n_poles, nx, ny, nz) — split each pole along x
    n_poles = coeffs.alpha.shape[0]
    cc_slabs = jnp.stack([
        split_array_x(coeffs.cc[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)  # (n_devices, n_poles, nx_local, ny, nz)
    alpha_slabs = jnp.stack([
        split_array_x(coeffs.alpha[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)
    beta_slabs = jnp.stack([
        split_array_x(coeffs.beta[p], n_devices, ghost, pad_value=0.0)
        for p in range(n_poles)
    ], axis=1)

    return DebyeCoeffs(ca=ca, cb=cb, cc=cc_slabs, alpha=alpha_slabs, beta=beta_slabs)


def _split_debye_state(state: DebyeState, n_devices, ghost=1):
    """Split DebyeState arrays along x into per-device slabs.

    Each field is (n_poles, nx, ny, nz) — split each pole along x.
    """
    n_poles = state.px.shape[0]

    def _split_poles(arr):
        return jnp.stack([
            split_array_x(arr[p], n_devices, ghost, pad_value=0.0)
            for p in range(n_poles)
        ], axis=1)  # (n_devices, n_poles, nx_local, ny, nz)

    return DebyeState(
        px=_split_poles(state.px),
        py=_split_poles(state.py),
        pz=_split_poles(state.pz),
    )


def _split_lorentz_coeffs(coeffs: LorentzCoeffs, n_devices, ghost=1):
    """Split LorentzCoeffs arrays along x into per-device slabs."""
    ca = split_array_x(coeffs.ca, n_devices, ghost, pad_value=0.0)
    cb = split_array_x(coeffs.cb, n_devices, ghost, pad_value=0.0)
    cc = split_array_x(coeffs.cc, n_devices, ghost, pad_value=0.0)

    n_poles = coeffs.a.shape[0]

    def _split_poles(arr):
        return jnp.stack([
            split_array_x(arr[p], n_devices, ghost, pad_value=0.0)
            for p in range(n_poles)
        ], axis=1)

    return LorentzCoeffs(
        ca=ca, cb=cb,
        a=_split_poles(coeffs.a),
        b=_split_poles(coeffs.b),
        c=_split_poles(coeffs.c),
        cc=cc,
    )


def _split_lorentz_state(state: LorentzState, n_devices, ghost=1):
    """Split LorentzState arrays along x into per-device slabs."""
    n_poles = state.px.shape[0]

    def _split_poles(arr):
        return jnp.stack([
            split_array_x(arr[p], n_devices, ghost, pad_value=0.0)
            for p in range(n_poles)
        ], axis=1)

    return LorentzState(
        px=_split_poles(state.px),
        py=_split_poles(state.py),
        pz=_split_poles(state.pz),
        px_prev=_split_poles(state.px_prev),
        py_prev=_split_poles(state.py_prev),
        pz_prev=_split_poles(state.pz_prev),
    )


# ---------------------------------------------------------------------------
# Ghost cell exchange
# ---------------------------------------------------------------------------

def _exchange_component(field, n_devices, axis_name="devices"):
    """Exchange ghost cells for a single field component.

    field : shape (nx_local + 2*ghost, ny, nz)  -- per-device via pmap
    The first and last x-planes are ghost cells.

    After exchange:
    - field[0, :, :] = right-neighbor's field[1, :, :]  (NO -- left neighbor's boundary)

    Convention:
    - ghost[0]  = left ghost  <- should contain left neighbor's rightmost real cell
    - ghost[-1] = right ghost <- should contain right neighbor's leftmost real cell
    - real cells: field[1:-1]
    """
    # The rightmost real cell of each device -> left ghost of right neighbor
    right_boundary = field[-2:-1, :, :]  # last real cell: index -2, shape (1, ny, nz)

    # The leftmost real cell of each device -> right ghost of left neighbor
    left_boundary = field[1:2, :, :]   # first real cell: index 1, shape (1, ny, nz)

    # ppermute: send from device i to device (i+1) % n  (right shift)
    # This sends each device's right_boundary to its right neighbor's left ghost
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    left_ghost_recv = lax.ppermute(right_boundary, axis_name, perm=perm_right)

    # ppermute: send from device i to device (i-1) % n  (left shift)
    # This sends each device's left_boundary to its left neighbor's right ghost
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    right_ghost_recv = lax.ppermute(left_boundary, axis_name, perm=perm_left)

    # Determine device index to mask boundary devices
    device_idx = lax.axis_index(axis_name)

    # For device 0: don't overwrite left ghost (physical boundary, keep zero)
    # For device n-1: don't overwrite right ghost (physical boundary, keep zero)
    left_ghost_val = jnp.where(device_idx > 0,
                               left_ghost_recv,
                               field[0:1, :, :])
    right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                right_ghost_recv,
                                field[-1:, :, :])

    field = field.at[0:1, :, :].set(left_ghost_val)
    field = field.at[-1:, :, :].set(right_ghost_val)
    return field


def _exchange_h_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all H components."""
    return state._replace(
        hx=_exchange_component(state.hx, n_devices, axis_name),
        hy=_exchange_component(state.hy, n_devices, axis_name),
        hz=_exchange_component(state.hz, n_devices, axis_name),
    )


def _exchange_e_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all E components."""
    return state._replace(
        ex=_exchange_component(state.ex, n_devices, axis_name),
        ey=_exchange_component(state.ey, n_devices, axis_name),
        ez=_exchange_component(state.ez, n_devices, axis_name),
    )


# ---------------------------------------------------------------------------
# Local update functions (operate on per-device slab with ghosts)
# ---------------------------------------------------------------------------

def _update_h_local(state, materials, dt, dx):
    """H update on a local slab (including ghost cells).

    Identical to yee.update_h but without jit decorator and always
    non-periodic (ghost cells handle inter-device coupling).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

    curl_x = (
        (_shift_fwd(ez, 1) - ez) / dx
        - (_shift_fwd(ey, 2) - ey) / dx
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) / dx
        - (_shift_fwd(ez, 0) - ez) / dx
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) / dx
        - (_shift_fwd(ex, 1) - ex) / dx
    )

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local(state, materials, dt, dx):
    """E update on a local slab (including ghost cells).

    Identical to yee.update_e but without jit decorator and always
    non-periodic.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) / dx
        - (hy - _shift_bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) / dx
        - (hz - _shift_bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) / dx
        - (hx - _shift_bwd(hx, 1)) / dx
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Local dispersive E-update functions (operate on per-device slab)
# ---------------------------------------------------------------------------

def _update_e_debye_local(state, debye_coeffs, debye_state, dt, dx):
    """E update with Debye ADE on a local slab (always non-periodic)."""
    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
    alpha, beta = debye_coeffs.alpha, debye_coeffs.beta

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx

    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    ex_new = ca * ex_old + cb * curl_x + jnp.sum(cc * debye_state.px, axis=0)
    ey_new = ca * ey_old + cb * curl_y + jnp.sum(cc * debye_state.py, axis=0)
    ez_new = ca * ez_old + cb * curl_z + jnp.sum(cc * debye_state.pz, axis=0)

    px_new = alpha * debye_state.px + beta * (ex_new[None] + ex_old[None])
    py_new = alpha * debye_state.py + beta * (ey_new[None] + ey_old[None])
    pz_new = alpha * debye_state.pz + beta * (ez_new[None] + ez_old[None])

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
    return new_fdtd, new_debye


def _update_e_lorentz_local(state, lorentz_coeffs, lor_state, dt, dx):
    """E update with Lorentz/Drude ADE on a local slab (always non-periodic)."""
    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc
    a, b, c = lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx

    px_new = a * lor_state.px + b * lor_state.px_prev + c * state.ex[None]
    py_new = a * lor_state.py + b * lor_state.py_prev + c * state.ey[None]
    pz_new = a * lor_state.pz + b * lor_state.pz_prev + c * state.ez[None]

    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    ex_new = ca * state.ex + cb * curl_x - cc * dpx
    ey_new = ca * state.ey + cb * curl_y - cc * dpy
    ez_new = ca * state.ez + cb * curl_z - cc * dpz

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_lor


def _update_e_local_with_dispersion(state, materials, dt, dx,
                                     debye=None, lorentz=None):
    """E update on a local slab with optional Debye/Lorentz dispersion.

    Returns (new_state, new_debye_state_or_None, new_lorentz_state_or_None).
    """
    if debye is None and lorentz is None:
        return _update_e_local(state, materials, dt, dx), None, None

    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        new_state, new_debye = _update_e_debye_local(
            state, debye_coeffs, debye_state, dt, dx)
        return new_state, new_debye, None

    if lorentz is not None and debye is None:
        lorentz_coeffs, lorentz_state = lorentz
        new_state, new_lorentz = _update_e_lorentz_local(
            state, lorentz_coeffs, lorentz_state, dt, dx)
        return new_state, None, new_lorentz

    # Mixed Debye + Lorentz
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lorentz_state = lorentz
    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2))) / dx
    curl_y = ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0))) / dx
    curl_z = ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1))) / dx
    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    px_l_new = (lorentz_coeffs.a * lorentz_state.px
                + lorentz_coeffs.b * lorentz_state.px_prev
                + lorentz_coeffs.c * ex_old[None])
    py_l_new = (lorentz_coeffs.a * lorentz_state.py
                + lorentz_coeffs.b * lorentz_state.py_prev
                + lorentz_coeffs.c * ey_old[None])
    pz_l_new = (lorentz_coeffs.a * lorentz_state.pz
                + lorentz_coeffs.b * lorentz_state.pz_prev
                + lorentz_coeffs.c * ez_old[None])
    dpx_l = jnp.sum(px_l_new - lorentz_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lorentz_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lorentz_state.pz, axis=0)

    ca_d, cb_d, cc_d = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
    alpha_d, beta_d = debye_coeffs.alpha, debye_coeffs.beta
    cc_l = lorentz_coeffs.cc

    ex_new = ca_d * ex_old + cb_d * curl_x + jnp.sum(cc_d * debye_state.px, axis=0) - cc_l * dpx_l
    ey_new = ca_d * ey_old + cb_d * curl_y + jnp.sum(cc_d * debye_state.py, axis=0) - cc_l * dpy_l
    ez_new = ca_d * ez_old + cb_d * curl_z + jnp.sum(cc_d * debye_state.pz, axis=0) - cc_l * dpz_l

    px_d_new = alpha_d * debye_state.px + beta_d * (ex_new[None] + ex_old[None])
    py_d_new = alpha_d * debye_state.py + beta_d * (ey_new[None] + ey_old[None])
    pz_d_new = alpha_d * debye_state.pz + beta_d * (ez_new[None] + ez_old[None])

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new, step=state.step + 1)
    new_debye_st = DebyeState(px=px_d_new, py=py_d_new, pz=pz_d_new)
    new_lor_st = LorentzState(
        px=px_l_new, py=py_l_new, pz=pz_l_new,
        px_prev=lorentz_state.px, py_prev=lorentz_state.py, pz_prev=lorentz_state.pz,
    )
    return new_fdtd, new_debye_st, new_lor_st


def _apply_pec_local(state, n_devices, nx_local_with_ghost, axis_name="devices"):
    """Apply PEC boundary on a local slab.

    - y and z PEC: always applied (all devices own the full y/z extent).
    - x PEC: only device 0 applies x-lo, only device N-1 applies x-hi.
      These operate on the first/last REAL cell (index ghost and
      nx_local+ghost-1), not the ghost cell itself.
    """
    device_idx = lax.axis_index(axis_name)
    ghost = 1

    ex, ey, ez = state.ex, state.ey, state.ez

    # Y-axis PEC (all devices)
    ex = ex.at[:, 0, :].set(0.0)
    ex = ex.at[:, -1, :].set(0.0)
    ez = ez.at[:, 0, :].set(0.0)
    ez = ez.at[:, -1, :].set(0.0)

    # Z-axis PEC (all devices)
    ex = ex.at[:, :, 0].set(0.0)
    ex = ex.at[:, :, -1].set(0.0)
    ey = ey.at[:, :, 0].set(0.0)
    ey = ey.at[:, :, -1].set(0.0)

    # X-axis PEC: only at physical boundaries
    # Device 0: x-lo PEC at real cell index ghost (=1)
    # We zero tangential components (ey, ez) at x=0 of the global domain,
    # which is index `ghost` (first real cell) of device 0.
    is_first = (device_idx == 0)
    ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
    ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
    ey = ey.at[ghost, :, :].set(ey_xlo)
    ez = ez.at[ghost, :, :].set(ez_xlo)

    # Device N-1: x-hi PEC at last real cell
    is_last = (device_idx == n_devices - 1)
    last_real = nx_local_with_ghost - 1 - ghost  # last real cell index
    ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
    ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
    ey = ey.at[last_real, :, :].set(ey_xhi)
    ez = ez.at[last_real, :, :].set(ez_xhi)

    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Distributed CPML support (Phase 2)
# ---------------------------------------------------------------------------

def _init_cpml_distributed(grid, nx_local, n_devices):
    """Initialize CPML params and per-device state for distributed slabs.

    In 1D slab decomposition along x:
    - Y/Z CPML faces: applied on ALL devices (each owns full y/z extent).
      Psi arrays that index along x use ``nx_local`` (slab width with ghosts).
    - X CPML faces: applied only on boundary devices (device 0 for x-lo,
      device N-1 for x-hi). Psi arrays allocated on all devices but
      masked to zero on non-boundary devices via ``jnp.where``.

    Parameters
    ----------
    grid : Grid
        Full-domain grid with CPML configuration.
    nx_local : int
        Per-device slab width including ghost cells.
    n_devices : int
        Number of devices.

    Returns
    -------
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state_stacked : dict
        Per-device CPML state arrays stacked along axis 0,
        shape ``(n_devices, ...)``.
    """
    from rfx.boundaries.cpml import _cpml_profile, CPMLState

    kappa_max = getattr(grid, "kappa_max", None) or 1.0
    n = grid.cpml_layers
    params = _cpml_profile(n, grid.dt, grid.dx, kappa_max=kappa_max)

    ny, nz = grid.ny, grid.nz

    def _zeros(dim1, dim2):
        """Zero psi array: (n_devices, n_cpml, dim1, dim2)."""
        return jnp.zeros((n_devices, n, dim1, dim2), dtype=jnp.float32)

    # X-face psi: perpendicular dims are (ny, nz) or transposed
    # Y/Z-face psi: perpendicular dims include nx_local (slab-local x)
    state_stacked = CPMLState(
        # E-field psi (12 faces)
        psi_ex_ylo=_zeros(nx_local, nz),
        psi_ex_yhi=_zeros(nx_local, nz),
        psi_ex_zlo=_zeros(nx_local, ny),
        psi_ex_zhi=_zeros(nx_local, ny),
        psi_ey_xlo=_zeros(ny, nz),
        psi_ey_xhi=_zeros(ny, nz),
        psi_ey_zlo=_zeros(ny, nx_local),
        psi_ey_zhi=_zeros(ny, nx_local),
        psi_ez_xlo=_zeros(nz, ny),
        psi_ez_xhi=_zeros(nz, ny),
        psi_ez_ylo=_zeros(nz, nx_local),
        psi_ez_yhi=_zeros(nz, nx_local),
        # H-field psi (12 faces)
        psi_hx_ylo=_zeros(nx_local, nz),
        psi_hx_yhi=_zeros(nx_local, nz),
        psi_hx_zlo=_zeros(nx_local, ny),
        psi_hx_zhi=_zeros(nx_local, ny),
        psi_hy_xlo=_zeros(ny, nz),
        psi_hy_xhi=_zeros(ny, nz),
        psi_hy_zlo=_zeros(ny, nx_local),
        psi_hy_zhi=_zeros(ny, nx_local),
        psi_hz_xlo=_zeros(nz, ny),
        psi_hz_xhi=_zeros(nz, ny),
        psi_hz_ylo=_zeros(nz, nx_local),
        psi_hz_yhi=_zeros(nz, nx_local),
    )

    return params, state_stacked


def _apply_cpml_e_distributed(
    state, cpml_params, cpml_state, n_cpml, dt, dx,
    n_devices, ghost=1, axis_name="devices",
):
    """Apply CPML E-field correction on a distributed slab.

    Y/Z faces: applied on all devices (each owns full y/z extent).
    X faces: x-lo on device 0 only, x-hi on device N-1 only.
    Non-active faces are masked to zero via jnp.where.

    X-axis indexing accounts for ghost cells: x-lo uses
    ``[ghost:ghost+n]``, x-hi uses ``[-(ghost+n):-ghost]``.

    Parameters
    ----------
    state : FDTDState
        Per-device slab state (nx_local, ny, nz).
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state : CPMLState
        Per-device CPML auxiliary state.
    n_cpml : int
        Number of CPML layers.
    dt, dx : float
        Timestep and cell size.
    n_devices : int
    ghost : int
        Number of ghost cells on each side of x (default 1).
    axis_name : str
    """
    n = n_cpml
    g = ghost
    coeff_e = dt / (8.854187817e-12 * 1.0)  # dt / eps_0

    b = cpml_params.b
    c = cpml_params.c
    kappa = cpml_params.kappa
    b_r = jnp.flip(b)
    c_r = jnp.flip(c)
    kappa_r = jnp.flip(kappa)

    ex = state.ex
    ey = state.ey
    ez = state.ez

    device_idx = lax.axis_index(axis_name)
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    # X-axis slice helpers (account for ghost offset)
    xlo = slice(g, g + n)          # first n real cells
    xhi = slice(-(g + n), -g)      # last n real cells

    # =======================================================================
    # X-axis CPML (device-conditional)
    # =======================================================================
    b_x = b[:, None, None]
    c_x = c[:, None, None]
    b_xr = b_r[:, None, None]
    c_xr = c_r[:, None, None]
    k_x = kappa[:, None, None]
    k_xr = kappa_r[:, None, None]

    # --- X-lo: Ey correction from dHz/dx (device 0 only) ---
    hz_xlo = state.hz[xlo, :, :]
    hz_shifted_xlo = _shift_bwd(state.hz, 0)[xlo, :, :]
    curl_hz_dx_xlo = (hz_xlo - hz_shifted_xlo) / dx

    new_psi_ey_xlo = b_x * cpml_state.psi_ey_xlo + c_x * curl_hz_dx_xlo
    ey_corr_xlo = -coeff_e * new_psi_ey_xlo - coeff_e * (1.0 / k_x - 1.0) * curl_hz_dx_xlo
    # Mask: only device 0
    ey_corr_xlo = jnp.where(is_first, ey_corr_xlo, 0.0)
    ey = ey.at[xlo, :, :].add(ey_corr_xlo)
    new_psi_ey_xlo = jnp.where(is_first, new_psi_ey_xlo, cpml_state.psi_ey_xlo)

    # --- X-hi: Ey correction from dHz/dx (device N-1 only) ---
    hz_xhi = state.hz[xhi, :, :]
    hz_shifted_xhi = _shift_bwd(state.hz, 0)[xhi, :, :]
    curl_hz_dx_xhi = (hz_xhi - hz_shifted_xhi) / dx

    new_psi_ey_xhi = b_xr * cpml_state.psi_ey_xhi + c_xr * curl_hz_dx_xhi
    ey_corr_xhi = -coeff_e * new_psi_ey_xhi - coeff_e * (1.0 / k_xr - 1.0) * curl_hz_dx_xhi
    ey_corr_xhi = jnp.where(is_last, ey_corr_xhi, 0.0)
    ey = ey.at[xhi, :, :].add(ey_corr_xhi)
    new_psi_ey_xhi = jnp.where(is_last, new_psi_ey_xhi, cpml_state.psi_ey_xhi)

    # --- X-lo: Ez correction from dHy/dx (device 0 only) ---
    hy_xlo = state.hy[xlo, :, :]
    hy_shifted_xlo = _shift_bwd(state.hy, 0)[xlo, :, :]
    curl_hy_dx_xlo = (hy_xlo - hy_shifted_xlo) / dx
    curl_hy_dx_xlo_t = jnp.transpose(curl_hy_dx_xlo, (0, 2, 1))

    new_psi_ez_xlo = b_x * cpml_state.psi_ez_xlo + c_x * curl_hy_dx_xlo_t
    correction_ez_xlo = jnp.transpose(new_psi_ez_xlo, (0, 2, 1))
    ez_corr_xlo = coeff_e * correction_ez_xlo + coeff_e * (1.0 / k_x - 1.0) * curl_hy_dx_xlo
    ez_corr_xlo = jnp.where(is_first, ez_corr_xlo, 0.0)
    ez = ez.at[xlo, :, :].add(ez_corr_xlo)
    new_psi_ez_xlo = jnp.where(is_first, new_psi_ez_xlo, cpml_state.psi_ez_xlo)

    # --- X-hi: Ez correction from dHy/dx (device N-1 only) ---
    hy_xhi = state.hy[xhi, :, :]
    hy_shifted_xhi = _shift_bwd(state.hy, 0)[xhi, :, :]
    curl_hy_dx_xhi = (hy_xhi - hy_shifted_xhi) / dx
    curl_hy_dx_xhi_t = jnp.transpose(curl_hy_dx_xhi, (0, 2, 1))

    new_psi_ez_xhi = b_xr * cpml_state.psi_ez_xhi + c_xr * curl_hy_dx_xhi_t
    correction_ez_xhi = jnp.transpose(new_psi_ez_xhi, (0, 2, 1))
    ez_corr_xhi = coeff_e * correction_ez_xhi + coeff_e * (1.0 / k_xr - 1.0) * curl_hy_dx_xhi
    ez_corr_xhi = jnp.where(is_last, ez_corr_xhi, 0.0)
    ez = ez.at[xhi, :, :].add(ez_corr_xhi)
    new_psi_ez_xhi = jnp.where(is_last, new_psi_ez_xhi, cpml_state.psi_ez_xhi)

    # =======================================================================
    # Y-axis CPML (all devices)
    # =======================================================================
    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    # --- Y-lo: Ex correction from dHz/dy ---
    hz_ylo = state.hz[:, :n, :]
    hz_shifted_ylo = _shift_bwd(state.hz, 1)[:, :n, :]
    curl_hz_dy_ylo = (hz_ylo - hz_shifted_ylo) / dx
    curl_hz_dy_ylo_t = jnp.transpose(curl_hz_dy_ylo, (1, 0, 2))

    new_psi_ex_ylo = b_yn * cpml_state.psi_ex_ylo + c_yn * curl_hz_dy_ylo_t
    correction_ex_ylo = jnp.transpose(new_psi_ex_ylo, (1, 0, 2))
    ex = ex.at[:, :n, :].add(coeff_e * correction_ex_ylo)
    kappa_corr_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hz_dy_ylo_t, (1, 0, 2))
    ex = ex.at[:, :n, :].add(coeff_e * kappa_corr_ylo)

    # --- Y-hi: Ex correction from dHz/dy ---
    hz_yhi = state.hz[:, -n:, :]
    hz_shifted_yhi = _shift_bwd(state.hz, 1)[:, -n:, :]
    curl_hz_dy_yhi = (hz_yhi - hz_shifted_yhi) / dx
    curl_hz_dy_yhi_t = jnp.transpose(curl_hz_dy_yhi, (1, 0, 2))

    new_psi_ex_yhi = b_yrn * cpml_state.psi_ex_yhi + c_yrn * curl_hz_dy_yhi_t
    correction_ex_yhi = jnp.transpose(new_psi_ex_yhi, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(coeff_e * correction_ex_yhi)
    kappa_corr_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hz_dy_yhi_t, (1, 0, 2))
    ex = ex.at[:, -n:, :].add(coeff_e * kappa_corr_yhi)

    # --- Y-lo: Ez correction from dHx/dy ---
    hx_ylo = state.hx[:, :n, :]
    hx_shifted_ylo = _shift_bwd(state.hx, 1)[:, :n, :]
    curl_hx_dy_ylo = (hx_ylo - hx_shifted_ylo) / dx
    curl_hx_dy_ylo_t = jnp.transpose(curl_hx_dy_ylo, (1, 2, 0))

    new_psi_ez_ylo = b_yn * cpml_state.psi_ez_ylo + c_yn * curl_hx_dy_ylo_t
    correction_ez_ylo = jnp.transpose(new_psi_ez_ylo, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-coeff_e * correction_ez_ylo)
    kappa_corr_ez_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dy_ylo_t, (2, 0, 1))
    ez = ez.at[:, :n, :].add(-coeff_e * kappa_corr_ez_ylo)

    # --- Y-hi: Ez correction from dHx/dy ---
    hx_yhi = state.hx[:, -n:, :]
    hx_shifted_yhi = _shift_bwd(state.hx, 1)[:, -n:, :]
    curl_hx_dy_yhi = (hx_yhi - hx_shifted_yhi) / dx
    curl_hx_dy_yhi_t = jnp.transpose(curl_hx_dy_yhi, (1, 2, 0))

    new_psi_ez_yhi = b_yrn * cpml_state.psi_ez_yhi + c_yrn * curl_hx_dy_yhi_t
    correction_ez_yhi = jnp.transpose(new_psi_ez_yhi, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-coeff_e * correction_ez_yhi)
    kappa_corr_ez_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dy_yhi_t, (2, 0, 1))
    ez = ez.at[:, -n:, :].add(-coeff_e * kappa_corr_ez_yhi)

    # =======================================================================
    # Z-axis CPML (all devices)
    # =======================================================================

    # --- Z-lo: Ex correction from dHy/dz ---
    hy_zlo = state.hy[:, :, :n]
    hy_shifted_zlo = _shift_bwd(state.hy, 2)[:, :, :n]
    curl_hy_dz_zlo = (hy_zlo - hy_shifted_zlo) / dx
    curl_hy_dz_zlo_t = jnp.transpose(curl_hy_dz_zlo, (2, 0, 1))

    new_psi_ex_zlo = b_yn * cpml_state.psi_ex_zlo + c_yn * curl_hy_dz_zlo_t
    correction_ex_zlo = jnp.transpose(new_psi_ex_zlo, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-coeff_e * correction_ex_zlo)
    kappa_corr_ex_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hy_dz_zlo_t, (1, 2, 0))
    ex = ex.at[:, :, :n].add(-coeff_e * kappa_corr_ex_zlo)

    # --- Z-hi: Ex correction from dHy/dz ---
    hy_zhi = state.hy[:, :, -n:]
    hy_shifted_zhi = _shift_bwd(state.hy, 2)[:, :, -n:]
    curl_hy_dz_zhi = (hy_zhi - hy_shifted_zhi) / dx
    curl_hy_dz_zhi_t = jnp.transpose(curl_hy_dz_zhi, (2, 0, 1))

    new_psi_ex_zhi = b_yrn * cpml_state.psi_ex_zhi + c_yrn * curl_hy_dz_zhi_t
    correction_ex_zhi = jnp.transpose(new_psi_ex_zhi, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-coeff_e * correction_ex_zhi)
    kappa_corr_ex_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hy_dz_zhi_t, (1, 2, 0))
    ex = ex.at[:, :, -n:].add(-coeff_e * kappa_corr_ex_zhi)

    # --- Z-lo: Ey correction from dHx/dz ---
    hx_zlo = state.hx[:, :, :n]
    hx_shifted_zlo = _shift_bwd(state.hx, 2)[:, :, :n]
    curl_hx_dz_zlo = (hx_zlo - hx_shifted_zlo) / dx
    curl_hx_dz_zlo_t = jnp.transpose(curl_hx_dz_zlo, (2, 1, 0))

    new_psi_ey_zlo = b_yn * cpml_state.psi_ey_zlo + c_yn * curl_hx_dz_zlo_t
    correction_ey_zlo = jnp.transpose(new_psi_ey_zlo, (2, 1, 0))
    ey = ey.at[:, :, :n].add(coeff_e * correction_ey_zlo)
    kappa_corr_ey_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dz_zlo_t, (2, 1, 0))
    ey = ey.at[:, :, :n].add(coeff_e * kappa_corr_ey_zlo)

    # --- Z-hi: Ey correction from dHx/dz ---
    hx_zhi = state.hx[:, :, -n:]
    hx_shifted_zhi = _shift_bwd(state.hx, 2)[:, :, -n:]
    curl_hx_dz_zhi = (hx_zhi - hx_shifted_zhi) / dx
    curl_hx_dz_zhi_t = jnp.transpose(curl_hx_dz_zhi, (2, 1, 0))

    new_psi_ey_zhi = b_yrn * cpml_state.psi_ey_zhi + c_yrn * curl_hx_dz_zhi_t
    correction_ey_zhi = jnp.transpose(new_psi_ey_zhi, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(coeff_e * correction_ey_zhi)
    kappa_corr_ey_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dz_zhi_t, (2, 1, 0))
    ey = ey.at[:, :, -n:].add(coeff_e * kappa_corr_ey_zhi)

    state = state._replace(ex=ex, ey=ey, ez=ez)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_ey_xlo=new_psi_ey_xlo,
        psi_ey_xhi=new_psi_ey_xhi,
        psi_ez_xlo=new_psi_ez_xlo,
        psi_ez_xhi=new_psi_ez_xhi,
        # y-axis
        psi_ex_ylo=new_psi_ex_ylo,
        psi_ex_yhi=new_psi_ex_yhi,
        psi_ez_ylo=new_psi_ez_ylo,
        psi_ez_yhi=new_psi_ez_yhi,
        # z-axis
        psi_ex_zlo=new_psi_ex_zlo,
        psi_ex_zhi=new_psi_ex_zhi,
        psi_ey_zlo=new_psi_ey_zlo,
        psi_ey_zhi=new_psi_ey_zhi,
    )

    return state, cpml_state


def _apply_cpml_h_distributed(
    state, cpml_params, cpml_state, n_cpml, dt, dx,
    n_devices, ghost=1, axis_name="devices",
):
    """Apply CPML H-field correction on a distributed slab.

    Y/Z faces: applied on all devices.
    X faces: x-lo on device 0 only, x-hi on device N-1 only.

    X-axis indexing accounts for ghost cells: x-lo uses
    ``[ghost:ghost+n]``, x-hi uses ``[-(ghost+n):-ghost]``.

    Parameters
    ----------
    state : FDTDState
        Per-device slab state (nx_local, ny, nz).
    cpml_params : CPMLParams
        Shared CPML profile coefficients.
    cpml_state : CPMLState
        Per-device CPML auxiliary state.
    n_cpml : int
        Number of CPML layers.
    dt, dx : float
    n_devices : int
    ghost : int
        Number of ghost cells on each side of x (default 1).
    axis_name : str
    """
    n = n_cpml
    g = ghost
    coeff_h = dt / 1.2566370614e-6  # dt / mu_0

    b = cpml_params.b
    c = cpml_params.c
    kappa = cpml_params.kappa
    b_r = jnp.flip(b)
    c_r = jnp.flip(c)
    kappa_r = jnp.flip(kappa)

    hx = state.hx
    hy = state.hy
    hz = state.hz

    device_idx = lax.axis_index(axis_name)
    is_first = (device_idx == 0)
    is_last = (device_idx == n_devices - 1)

    # X-axis slice helpers (account for ghost offset)
    xlo = slice(g, g + n)          # first n real cells
    xhi = slice(-(g + n), -g)      # last n real cells

    # =======================================================================
    # X-axis CPML (device-conditional)
    # =======================================================================
    b_x = b[:, None, None]
    c_x = c[:, None, None]
    b_xr = b_r[:, None, None]
    c_xr = c_r[:, None, None]
    k_x = kappa[:, None, None]
    k_xr = kappa_r[:, None, None]

    # --- X-lo: Hy correction from dEz/dx (device 0 only) ---
    ez_xlo = state.ez[xlo, :, :]
    ez_shifted_xlo = _shift_fwd(state.ez, 0)[xlo, :, :]
    curl_ez_dx_xlo = (ez_shifted_xlo - ez_xlo) / dx

    new_psi_hy_xlo = b_x * cpml_state.psi_hy_xlo + c_x * curl_ez_dx_xlo
    hy_corr_xlo = coeff_h * new_psi_hy_xlo + coeff_h * (1.0 / k_x - 1.0) * curl_ez_dx_xlo
    hy_corr_xlo = jnp.where(is_first, hy_corr_xlo, 0.0)
    hy = hy.at[xlo, :, :].add(hy_corr_xlo)
    new_psi_hy_xlo = jnp.where(is_first, new_psi_hy_xlo, cpml_state.psi_hy_xlo)

    # --- X-hi: Hy correction from dEz/dx (device N-1 only) ---
    ez_xhi = state.ez[xhi, :, :]
    ez_shifted_xhi = _shift_fwd(state.ez, 0)[xhi, :, :]
    curl_ez_dx_xhi = (ez_shifted_xhi - ez_xhi) / dx

    new_psi_hy_xhi = b_xr * cpml_state.psi_hy_xhi + c_xr * curl_ez_dx_xhi
    hy_corr_xhi = coeff_h * new_psi_hy_xhi + coeff_h * (1.0 / k_xr - 1.0) * curl_ez_dx_xhi
    hy_corr_xhi = jnp.where(is_last, hy_corr_xhi, 0.0)
    hy = hy.at[xhi, :, :].add(hy_corr_xhi)
    new_psi_hy_xhi = jnp.where(is_last, new_psi_hy_xhi, cpml_state.psi_hy_xhi)

    # --- X-lo: Hz correction from dEy/dx (device 0 only) ---
    ey_xlo = state.ey[xlo, :, :]
    ey_shifted_xlo = _shift_fwd(state.ey, 0)[xlo, :, :]
    curl_ey_dx_xlo = (ey_shifted_xlo - ey_xlo) / dx
    curl_ey_dx_xlo_t = jnp.transpose(curl_ey_dx_xlo, (0, 2, 1))

    new_psi_hz_xlo = b_x * cpml_state.psi_hz_xlo + c_x * curl_ey_dx_xlo_t
    correction_hz_xlo = jnp.transpose(new_psi_hz_xlo, (0, 2, 1))
    hz_corr_xlo = -coeff_h * correction_hz_xlo - coeff_h * (1.0 / k_x - 1.0) * curl_ey_dx_xlo
    hz_corr_xlo = jnp.where(is_first, hz_corr_xlo, 0.0)
    hz = hz.at[xlo, :, :].add(hz_corr_xlo)
    new_psi_hz_xlo = jnp.where(is_first, new_psi_hz_xlo, cpml_state.psi_hz_xlo)

    # --- X-hi: Hz correction from dEy/dx (device N-1 only) ---
    ey_xhi = state.ey[xhi, :, :]
    ey_shifted_xhi = _shift_fwd(state.ey, 0)[xhi, :, :]
    curl_ey_dx_xhi = (ey_shifted_xhi - ey_xhi) / dx
    curl_ey_dx_xhi_t = jnp.transpose(curl_ey_dx_xhi, (0, 2, 1))

    new_psi_hz_xhi = b_xr * cpml_state.psi_hz_xhi + c_xr * curl_ey_dx_xhi_t
    correction_hz_xhi = jnp.transpose(new_psi_hz_xhi, (0, 2, 1))
    hz_corr_xhi = -coeff_h * correction_hz_xhi - coeff_h * (1.0 / k_xr - 1.0) * curl_ey_dx_xhi
    hz_corr_xhi = jnp.where(is_last, hz_corr_xhi, 0.0)
    hz = hz.at[xhi, :, :].add(hz_corr_xhi)
    new_psi_hz_xhi = jnp.where(is_last, new_psi_hz_xhi, cpml_state.psi_hz_xhi)

    # =======================================================================
    # Y-axis CPML (all devices)
    # =======================================================================
    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    # --- Y-lo: Hx correction from dEz/dy ---
    ez_ylo = state.ez[:, :n, :]
    ez_shifted_ylo = _shift_fwd(state.ez, 1)[:, :n, :]
    curl_ez_dy_ylo = (ez_shifted_ylo - ez_ylo) / dx
    curl_ez_dy_ylo_t = jnp.transpose(curl_ez_dy_ylo, (1, 0, 2))

    new_psi_hx_ylo = b_yn * cpml_state.psi_hx_ylo + c_yn * curl_ez_dy_ylo_t
    correction_hx_ylo = jnp.transpose(new_psi_hx_ylo, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-coeff_h * correction_hx_ylo)
    kappa_corr_hx_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ez_dy_ylo_t, (1, 0, 2))
    hx = hx.at[:, :n, :].add(-coeff_h * kappa_corr_hx_ylo)

    # --- Y-hi: Hx correction from dEz/dy ---
    ez_yhi = state.ez[:, -n:, :]
    ez_shifted_yhi = _shift_fwd(state.ez, 1)[:, -n:, :]
    curl_ez_dy_yhi = (ez_shifted_yhi - ez_yhi) / dx
    curl_ez_dy_yhi_t = jnp.transpose(curl_ez_dy_yhi, (1, 0, 2))

    new_psi_hx_yhi = b_yrn * cpml_state.psi_hx_yhi + c_yrn * curl_ez_dy_yhi_t
    correction_hx_yhi = jnp.transpose(new_psi_hx_yhi, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-coeff_h * correction_hx_yhi)
    kappa_corr_hx_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ez_dy_yhi_t, (1, 0, 2))
    hx = hx.at[:, -n:, :].add(-coeff_h * kappa_corr_hx_yhi)

    # --- Y-lo: Hz correction from dEx/dy ---
    ex_ylo = state.ex[:, :n, :]
    ex_shifted_ylo = _shift_fwd(state.ex, 1)[:, :n, :]
    curl_ex_dy_ylo = (ex_shifted_ylo - ex_ylo) / dx
    curl_ex_dy_ylo_t = jnp.transpose(curl_ex_dy_ylo, (1, 2, 0))

    new_psi_hz_ylo = b_yn * cpml_state.psi_hz_ylo + c_yn * curl_ex_dy_ylo_t
    correction_hz_ylo = jnp.transpose(new_psi_hz_ylo, (2, 0, 1))
    hz = hz.at[:, :n, :].add(coeff_h * correction_hz_ylo)
    kappa_corr_hz_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dy_ylo_t, (2, 0, 1))
    hz = hz.at[:, :n, :].add(coeff_h * kappa_corr_hz_ylo)

    # --- Y-hi: Hz correction from dEx/dy ---
    ex_yhi = state.ex[:, -n:, :]
    ex_shifted_yhi = _shift_fwd(state.ex, 1)[:, -n:, :]
    curl_ex_dy_yhi = (ex_shifted_yhi - ex_yhi) / dx
    curl_ex_dy_yhi_t = jnp.transpose(curl_ex_dy_yhi, (1, 2, 0))

    new_psi_hz_yhi = b_yrn * cpml_state.psi_hz_yhi + c_yrn * curl_ex_dy_yhi_t
    correction_hz_yhi = jnp.transpose(new_psi_hz_yhi, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(coeff_h * correction_hz_yhi)
    kappa_corr_hz_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dy_yhi_t, (2, 0, 1))
    hz = hz.at[:, -n:, :].add(coeff_h * kappa_corr_hz_yhi)

    # =======================================================================
    # Z-axis CPML (all devices)
    # =======================================================================

    # --- Z-lo: Hx correction from dEy/dz ---
    ey_zlo = state.ey[:, :, :n]
    ey_shifted_zlo = _shift_fwd(state.ey, 2)[:, :, :n]
    curl_ey_dz_zlo = (ey_shifted_zlo - ey_zlo) / dx
    curl_ey_dz_zlo_t = jnp.transpose(curl_ey_dz_zlo, (2, 0, 1))

    new_psi_hx_zlo = b_yn * cpml_state.psi_hx_zlo + c_yn * curl_ey_dz_zlo_t
    correction_hx_zlo = jnp.transpose(new_psi_hx_zlo, (1, 2, 0))
    hx = hx.at[:, :, :n].add(coeff_h * correction_hx_zlo)
    kappa_corr_hx_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ey_dz_zlo_t, (1, 2, 0))
    hx = hx.at[:, :, :n].add(coeff_h * kappa_corr_hx_zlo)

    # --- Z-hi: Hx correction from dEy/dz ---
    ey_zhi = state.ey[:, :, -n:]
    ey_shifted_zhi = _shift_fwd(state.ey, 2)[:, :, -n:]
    curl_ey_dz_zhi = (ey_shifted_zhi - ey_zhi) / dx
    curl_ey_dz_zhi_t = jnp.transpose(curl_ey_dz_zhi, (2, 0, 1))

    new_psi_hx_zhi = b_yrn * cpml_state.psi_hx_zhi + c_yrn * curl_ey_dz_zhi_t
    correction_hx_zhi = jnp.transpose(new_psi_hx_zhi, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(coeff_h * correction_hx_zhi)
    kappa_corr_hx_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ey_dz_zhi_t, (1, 2, 0))
    hx = hx.at[:, :, -n:].add(coeff_h * kappa_corr_hx_zhi)

    # --- Z-lo: Hy correction from dEx/dz ---
    ex_zlo = state.ex[:, :, :n]
    ex_shifted_zlo = _shift_fwd(state.ex, 2)[:, :, :n]
    curl_ex_dz_zlo = (ex_shifted_zlo - ex_zlo) / dx
    curl_ex_dz_zlo_t = jnp.transpose(curl_ex_dz_zlo, (2, 1, 0))

    new_psi_hy_zlo = b_yn * cpml_state.psi_hy_zlo + c_yn * curl_ex_dz_zlo_t
    correction_hy_zlo = jnp.transpose(new_psi_hy_zlo, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-coeff_h * correction_hy_zlo)
    kappa_corr_hy_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dz_zlo_t, (2, 1, 0))
    hy = hy.at[:, :, :n].add(-coeff_h * kappa_corr_hy_zlo)

    # --- Z-hi: Hy correction from dEx/dz ---
    ex_zhi = state.ex[:, :, -n:]
    ex_shifted_zhi = _shift_fwd(state.ex, 2)[:, :, -n:]
    curl_ex_dz_zhi = (ex_shifted_zhi - ex_zhi) / dx
    curl_ex_dz_zhi_t = jnp.transpose(curl_ex_dz_zhi, (2, 1, 0))

    new_psi_hy_zhi = b_yrn * cpml_state.psi_hy_zhi + c_yrn * curl_ex_dz_zhi_t
    correction_hy_zhi = jnp.transpose(new_psi_hy_zhi, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-coeff_h * correction_hy_zhi)
    kappa_corr_hy_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dz_zhi_t, (2, 1, 0))
    hy = hy.at[:, :, -n:].add(-coeff_h * kappa_corr_hy_zhi)

    state = state._replace(hx=hx, hy=hy, hz=hz)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_hy_xlo=new_psi_hy_xlo,
        psi_hy_xhi=new_psi_hy_xhi,
        psi_hz_xlo=new_psi_hz_xlo,
        psi_hz_xhi=new_psi_hz_xhi,
        # y-axis
        psi_hx_ylo=new_psi_hx_ylo,
        psi_hx_yhi=new_psi_hx_yhi,
        psi_hz_ylo=new_psi_hz_ylo,
        psi_hz_yhi=new_psi_hz_yhi,
        # z-axis
        psi_hx_zlo=new_psi_hx_zlo,
        psi_hx_zhi=new_psi_hx_zhi,
        psi_hy_zlo=new_psi_hy_zlo,
        psi_hy_zhi=new_psi_hy_zhi,
    )

    return state, cpml_state


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_distributed(sim, *, n_steps, devices=None, exchange_interval=1,
                    **kwargs):
    """Run FDTD simulation distributed across multiple devices.

    Uses 1D slab decomposition along the x-axis.  Supports PEC and
    CPML boundaries, soft sources, point probes, lumped ports, and
    Debye/Lorentz dispersive materials.

    TFSF plane-wave sources and waveguide ports are not yet supported
    in the distributed runner.  When detected, the runner issues a
    warning and transparently falls back to the single-device path.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance.
    n_steps : int
        Number of timesteps.
    devices : list of jax.Device or None
        If None, use all available devices.
    exchange_interval : int, optional
        How often (in timesteps) to perform ghost cell exchange via
        ``lax.ppermute``.  Default is 1 (every step, original behavior).
        Setting to 2 or 4 reduces synchronization overhead at the cost
        of O(interval * dt) boundary error from stale ghost data.
        Typical FDTD simulations tolerate ``exchange_interval <= 4``
        with negligible accuracy loss.

    Returns
    -------
    Result
    """
    import warnings

    if sim._boundary == "upml":
        raise ValueError("boundary='upml' does not support distributed execution")

    # ------------------------------------------------------------------
    # Graceful fallback for features that require the full domain on a
    # single device (TFSF auxiliary grid, waveguide eigenmode solver).
    # ------------------------------------------------------------------
    if sim._tfsf is not None:
        warnings.warn(
            "Distributed runner does not yet support TFSF plane-wave "
            "sources. Falling back to single-device execution.",
            stacklevel=2,
        )
        return sim.run(n_steps=n_steps)

    if sim._waveguide_ports:
        warnings.warn(
            "Distributed runner does not yet support waveguide ports. "
            "Falling back to single-device execution.",
            stacklevel=2,
        )
        return sim.run(n_steps=n_steps)

    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    # Build grid and materials (full domain)
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, _ = (
        sim._assemble_materials(grid)
    )
    materials = base_materials

    nx, ny, nz = grid.shape
    if nx % n_devices != 0:
        raise ValueError(
            f"Grid nx={nx} is not evenly divisible by {n_devices} devices. "
            f"Adjust domain size or dx so that nx is a multiple of n_devices."
        )

    nx_per = nx // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    # Determine boundary type
    use_cpml = sim._boundary == "cpml" and grid.cpml_layers > 0
    n_cpml = grid.cpml_layers if use_cpml else 0

    # Build sources and probes on the full grid
    # First pass: fold lumped port impedances into materials (must happen
    # before splitting so the owning device's slab gets correct sigma).
    sources = []
    probes = []
    for pe in sim._ports:
        if pe.impedance > 0.0 and pe.extent is None:
            # Single-cell lumped port: fold impedance into materials
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))
        elif pe.impedance == 0.0:
            if sim._boundary == "cpml":
                sources.append(make_j_source(grid, pe.position, pe.component,
                                             pe.waveform, n_steps, materials))
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))
    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    # Map source/probe global indices to (device_id, local_index)
    # Source mapping
    src_device_ids = []
    src_local_specs = []  # (local_i, j, k, component, waveform)
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost  # offset by ghost
        src_device_ids.append(dev_id)
        src_local_specs.append((local_i, s.j, s.k, s.component))

    # Probe mapping
    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(dev_id)
        prb_local_specs.append((local_i, p.j, p.k, p.component))

    # Precompute source waveform matrix: (n_steps, n_sources)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Replicate waveforms to all devices: (n_devices, n_steps, n_sources)
    src_waveforms_rep = jnp.broadcast_to(
        src_waveforms[None, :, :],
        (n_devices, n_steps, src_waveforms.shape[-1]),
    )

    # Build per-device source mask: (n_devices, n_sources) bool
    # device d only injects source s if src_device_ids[s] == d
    src_device_mask = jnp.array(
        [[1.0 if src_device_ids[s] == d else 0.0
          for s in range(len(sources))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if sources else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Build per-device probe mask similarly
    prb_device_mask = jnp.array(
        [[1.0 if prb_device_ids[p] == d else 0.0
          for p in range(len(probes))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if probes else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Split domain into per-device slabs
    from rfx.core.yee import init_state
    full_state = init_state(grid.shape)
    state_slabs = _split_state(full_state, n_devices, ghost)
    materials_slabs = _split_materials(materials, n_devices, ghost)

    # Initialize dispersion (Debye / Lorentz) on the full domain,
    # then split coefficients and state into per-device slabs.
    _, debye_full, lorentz_full = sim._init_dispersion(
        materials, grid.dt, debye_spec, lorentz_spec)

    has_debye = debye_full is not None
    has_lorentz = lorentz_full is not None

    if has_debye:
        debye_coeffs_full, debye_state_full = debye_full
        debye_coeffs_slabs = _split_debye_coeffs(debye_coeffs_full, n_devices, ghost)
        debye_state_slabs = _split_debye_state(debye_state_full, n_devices, ghost)
    else:
        _dz3 = jnp.zeros((n_devices, 1, nx_local, ny, nz), dtype=jnp.float32)
        _dz = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        debye_coeffs_slabs = DebyeCoeffs(ca=_dz, cb=_dz, cc=_dz3, alpha=_dz3, beta=_dz3)
        debye_state_slabs = DebyeState(px=_dz3, py=_dz3.copy(), pz=_dz3.copy())

    if has_lorentz:
        lorentz_coeffs_full, lorentz_state_full = lorentz_full
        lorentz_coeffs_slabs = _split_lorentz_coeffs(lorentz_coeffs_full, n_devices, ghost)
        lorentz_state_slabs = _split_lorentz_state(lorentz_state_full, n_devices, ghost)
    else:
        _lz3 = jnp.zeros((n_devices, 1, nx_local, ny, nz), dtype=jnp.float32)
        _lz = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.float32)
        lorentz_coeffs_slabs = LorentzCoeffs(ca=_lz, cb=_lz, a=_lz3, b=_lz3, c=_lz3, cc=_lz)
        lorentz_state_slabs = LorentzState(
            px=_lz3, py=_lz3.copy(), pz=_lz3.copy(),
            px_prev=_lz3.copy(), py_prev=_lz3.copy(), pz_prev=_lz3.copy(),
        )

    # Initialize CPML if needed
    if use_cpml:
        cpml_params, cpml_state_slabs = _init_cpml_distributed(
            grid, nx_local, n_devices)

    # Static metadata captured by closure
    n_src = len(sources)
    n_prb = len(probes)

    dt = grid.dt
    dx = grid.dx

    # Capture exchange_interval for use inside pmap closures
    _exchange_interval = int(exchange_interval)

    if use_cpml:
        # CPML path: uses dict carry for state + cpml_state
        @partial(jax.pmap, axis_name="devices", devices=devices)
        def distributed_scan(state_slab, materials_slab, cpml_state_dev,
                             step_indices, src_waveforms_dev, src_mask,
                             prb_mask, debye_coeffs_dev, debye_state_dev,
                             lorentz_coeffs_dev, lorentz_state_dev):
            """Scan body over timesteps on one device (CPML path)."""

            def step_fn(carry, xs):
                _step_idx, src_vals = xs
                st = carry["fdtd"]
                cpml_st = carry["cpml"]
                db_st = carry["debye"]
                lr_st = carry["lorentz"]

                # 1. H update (local)
                st = _update_h_local(st, materials_slab, dt, dx)

                # 2. CPML H correction (before ghost exchange)
                st, cpml_st = _apply_cpml_h_distributed(
                    st, cpml_params, cpml_st, n_cpml, dt, dx,
                    n_devices, ghost=ghost, axis_name="devices")

                # 3. Exchange H ghost cells (conditionally skip)
                do_exchange = (_step_idx % _exchange_interval == 0)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_h_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 4. E update (local)
                _debye_arg = (debye_coeffs_dev, db_st) if has_debye else None
                _lorentz_arg = (lorentz_coeffs_dev, lr_st) if has_lorentz else None
                st, new_db, new_lr = _update_e_local_with_dispersion(
                    st, materials_slab, dt, dx,
                    debye=_debye_arg, lorentz=_lorentz_arg)
                if has_debye:
                    db_st = new_db
                if has_lorentz:
                    lr_st = new_lr

                # 5. CPML E correction (before ghost exchange)
                st, cpml_st = _apply_cpml_e_distributed(
                    st, cpml_params, cpml_st, n_cpml, dt, dx,
                    n_devices, ghost=ghost, axis_name="devices")

                # 6. Exchange E ghost cells (conditionally skip)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_e_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 7. Source injection (only on owning device)
                for idx_s in range(n_src):
                    li, lj, lk, lc = src_local_specs[idx_s]
                    val = src_vals[idx_s] * src_mask[idx_s]
                    field = getattr(st, lc)
                    field = field.at[li, lj, lk].add(val)
                    st = st._replace(**{lc: field})

                # 8. Probe sampling (only on owning device)
                samples = []
                for idx_p in range(n_prb):
                    li, lj, lk, lc = prb_local_specs[idx_p]
                    val = getattr(st, lc)[li, lj, lk] * prb_mask[idx_p]
                    samples.append(val)

                if samples:
                    probe_out = jnp.stack(samples)
                else:
                    probe_out = jnp.zeros(0, dtype=jnp.float32)

                return {"fdtd": st, "cpml": cpml_st,
                        "debye": db_st, "lorentz": lr_st}, probe_out

            xs = (step_indices, src_waveforms_dev)
            carry_init = {"fdtd": state_slab, "cpml": cpml_state_dev,
                          "debye": debye_state_dev, "lorentz": lorentz_state_dev}
            final_carry, probe_ts = lax.scan(step_fn, carry_init, xs)
            return final_carry["fdtd"], probe_ts

    else:
        # PEC path: original simple carry
        @partial(jax.pmap, axis_name="devices", devices=devices)
        def distributed_scan(state_slab, materials_slab, cpml_state_dev,
                             step_indices, src_waveforms_dev, src_mask,
                             prb_mask, debye_coeffs_dev, debye_state_dev,
                             lorentz_coeffs_dev, lorentz_state_dev):
            """Scan body over timesteps on one device (PEC path)."""

            def step_fn(carry, xs):
                _step_idx, src_vals = xs
                st = carry["fdtd"]
                db_st = carry["debye"]
                lr_st = carry["lorentz"]

                # 1. H update (local)
                st = _update_h_local(st, materials_slab, dt, dx)

                # 2. Exchange H ghost cells (conditionally skip)
                do_exchange = (_step_idx % _exchange_interval == 0)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_h_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 3. E update (local)
                _debye_arg = (debye_coeffs_dev, db_st) if has_debye else None
                _lorentz_arg = (lorentz_coeffs_dev, lr_st) if has_lorentz else None
                st, new_db, new_lr = _update_e_local_with_dispersion(
                    st, materials_slab, dt, dx,
                    debye=_debye_arg, lorentz=_lorentz_arg)
                if has_debye:
                    db_st = new_db
                if has_lorentz:
                    lr_st = new_lr

                # 4. Exchange E ghost cells (conditionally skip)
                st = lax.cond(
                    do_exchange,
                    lambda s: _exchange_e_ghosts(s, n_devices, "devices"),
                    lambda s: s,
                    st,
                )

                # 5. PEC boundaries
                st = _apply_pec_local(st, n_devices, nx_local, "devices")

                # 6. Source injection (only on owning device)
                for idx_s in range(n_src):
                    li, lj, lk, lc = src_local_specs[idx_s]
                    val = src_vals[idx_s] * src_mask[idx_s]
                    field = getattr(st, lc)
                    field = field.at[li, lj, lk].add(val)
                    st = st._replace(**{lc: field})

                # 7. Probe sampling (only on owning device)
                samples = []
                for idx_p in range(n_prb):
                    li, lj, lk, lc = prb_local_specs[idx_p]
                    val = getattr(st, lc)[li, lj, lk] * prb_mask[idx_p]
                    samples.append(val)

                if samples:
                    probe_out = jnp.stack(samples)
                else:
                    probe_out = jnp.zeros(0, dtype=jnp.float32)

                return {"fdtd": st, "debye": db_st, "lorentz": lr_st}, probe_out

            xs = (step_indices, src_waveforms_dev)
            carry_init = {"fdtd": state_slab, "debye": debye_state_dev,
                          "lorentz": lorentz_state_dev}
            final_carry, probe_ts = lax.scan(step_fn, carry_init, xs)
            return final_carry["fdtd"], probe_ts

    # Prepare scan inputs: replicate step indices across devices
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    step_indices_rep = jnp.broadcast_to(
        step_indices[None, :], (n_devices, n_steps)
    )

    # Build dummy CPML state for PEC path (keeps pmap signature uniform)
    if not use_cpml:
        from rfx.boundaries.cpml import CPMLState
        # Minimal dummy: all zeros, shape (n_devices, 0, ...) won't work
        # for NamedTuple, so use shape (n_devices, 1, 1, 1)
        _z = jnp.zeros((n_devices, 1, 1, 1), dtype=jnp.float32)
        cpml_state_slabs = CPMLState(
            psi_ex_ylo=_z, psi_ex_yhi=_z,
            psi_ex_zlo=_z, psi_ex_zhi=_z,
            psi_ey_xlo=_z, psi_ey_xhi=_z,
            psi_ey_zlo=_z, psi_ey_zhi=_z,
            psi_ez_xlo=_z, psi_ez_xhi=_z,
            psi_ez_ylo=_z, psi_ez_yhi=_z,
            psi_hx_ylo=_z, psi_hx_yhi=_z,
            psi_hx_zlo=_z, psi_hx_zhi=_z,
            psi_hy_xlo=_z, psi_hy_xhi=_z,
            psi_hy_zlo=_z, psi_hy_zhi=_z,
            psi_hz_xlo=_z, psi_hz_xhi=_z,
            psi_hz_ylo=_z, psi_hz_yhi=_z,
        )

    # Run the distributed simulation
    final_state_slabs, probe_ts_all = distributed_scan(
        state_slabs,
        materials_slabs,
        cpml_state_slabs,
        step_indices_rep,
        src_waveforms_rep,
        src_device_mask,
        prb_device_mask,
        debye_coeffs_slabs,
        debye_state_slabs,
        lorentz_coeffs_slabs,
        lorentz_state_slabs,
    )

    # Gather final state
    final_state = _gather_state(final_state_slabs, ghost)

    # Aggregate probe time series: sum across devices
    # probe_ts_all: (n_devices, n_steps, n_probes)
    # Each probe is non-zero only on its owning device, so sum works
    if n_prb > 0:
        time_series = jnp.sum(probe_ts_all, axis=0)  # (n_steps, n_probes)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    return Result(
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=grid,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
