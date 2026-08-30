"""Remis-class dual-cell discrete energy for the rfx NU leapfrog (W1).

Definition (full derivation in
docs/design_notes/20260829_spec01_multiband_predeclaration.md):

    E_n = 1/2 sum_c eps0*eps_r * w(Ec) * (Ec^{n})^2
        + 1/2 sum_c mu0*mu_r  * w(Hc) * Hc^{n-1/2} * Hc^{n+1/2}

with per-component tensor-product weights built from the PRIMAL cell
width  pd[k] = 1/inv_d_h[k]  (0 where inv_d_h == 0: the #562 trailing
bounding-node duplicate) and the DUAL spacing  dd[k] = 1/inv_d_e[k]
(the same dual metric as ``rfx.nonuniform.e_node_dual_spacings``):

    w(Ex) = pd_x (x) dd_y (x) dd_z        w(Hx) = dd_x (x) pd_y (x) pd_z
    w(Ey) = dd_x (x) pd_y (x) dd_z        w(Hy) = pd_x (x) dd_y (x) pd_z
    w(Ez) = dd_x (x) dd_y (x) pd_z        w(Hz) = pd_x (x) pd_y (x) dd_z

This is the weighted inner product under which the rfx NU curl pair
(curl_bwd with inv_d_e in the E update / curl_fwd with inv_d_h in the H
update) satisfies exact summation-by-parts on a PEC-pinned lossless
grid, so E_n is conserved to rounding by the leapfrog. The weights are
computed as float64 reciprocals of the grid's OWN float32 inv arrays so
the SBP identity cancels against the update coefficients exactly.

The mixed-time H product uses the state H (H^{n+1/2} after a full rfx
step) and ONE extra float64 H-update applied on the host (H^{n+3/2}),
paired with the state E (E^{n+1}).
"""

from __future__ import annotations

import numpy as np

from rfx.core.yee import EPS_0, MU_0


def _pd_dd(inv_e: np.ndarray, inv_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_e = np.asarray(inv_e, dtype=np.float64)
    inv_h = np.asarray(inv_h, dtype=np.float64)
    pd = np.where(inv_h > 0, 1.0 / np.where(inv_h > 0, inv_h, 1.0), 0.0)
    dd = 1.0 / inv_e
    return pd, dd


def energy_weights(grid) -> dict[str, np.ndarray]:
    """Per-component (nx,ny,nz) float64 weight arrays."""
    pdx, ddx = _pd_dd(grid.inv_dx, grid.inv_dx_h)
    pdy, ddy = _pd_dd(grid.inv_dy, grid.inv_dy_h)
    pdz, ddz = _pd_dd(grid.inv_dz, grid.inv_dz_h)
    o = lambda a, b, c: a[:, None, None] * b[None, :, None] * c[None, None, :]
    return {
        "ex": o(pdx, ddy, ddz), "ey": o(ddx, pdy, ddz), "ez": o(ddx, ddy, pdz),
        "hx": o(ddx, pdy, pdz), "hy": o(pdx, ddy, pdz), "hz": o(pdx, pdy, ddz),
    }


def _curl_fwd_f64(ex, ey, ez, inv_dx_h, inv_dy_h, inv_dz_h):
    """float64 mirror of rfx.core.yee.update_h_nu's forward-difference
    curl (zero-padded shifts), using the grid's exact inv values."""
    def sf(a, ax):
        out = np.zeros_like(a)
        if ax == 0:
            out[:-1] = a[1:]
        elif ax == 1:
            out[:, :-1] = a[:, 1:]
        else:
            out[:, :, :-1] = a[:, :, 1:]
        return out
    cx = (sf(ez, 1) - ez) * inv_dy_h[None, :, None] \
        - (sf(ey, 2) - ey) * inv_dz_h[None, None, :]
    cy = (sf(ex, 2) - ex) * inv_dz_h[None, None, :] \
        - (sf(ez, 0) - ez) * inv_dx_h[:, None, None]
    cz = (sf(ey, 0) - ey) * inv_dx_h[:, None, None] \
        - (sf(ex, 1) - ex) * inv_dy_h[None, :, None]
    return cx, cy, cz


def remis_energy(grid, materials, state, weights=None) -> float:
    """E_n from a post-step state (E^{n+1}, H^{n+1/2}); the H^{n+3/2}
    half is reconstructed here in float64 with the same operator."""
    if weights is None:
        weights = energy_weights(grid)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    ex, ey, ez = f64(state.ex), f64(state.ey), f64(state.ez)
    hx, hy, hz = f64(state.hx), f64(state.hy), f64(state.hz)
    dt = float(grid.dt)
    # KERNEL-REALIZED coefficients: update_e_nu/update_h_nu compute
    # eps = eps_r*EPS_0, cb = dt/eps and dt/mu in float32 (the material
    # arrays are float32, the weak-typed Python scalars round to f32).
    # The conserved functional is the one written in the REALIZED
    # coefficients, so the witness must use eps_eff = dt/cb32 and
    # mu_eff = dt/(dt/mu)32 — using exact EPS_0/MU_0 instead leaves a
    # bounded ~1e-8 coefficient-rounding residual in the witness itself
    # (measured 2.4e-8 on the float64 validity run before this fix).
    er32 = np.asarray(materials.eps_r, dtype=np.float32)
    mr32 = np.asarray(materials.mu_r, dtype=np.float32)
    cb32 = (np.float32(dt) / (er32 * np.float32(EPS_0))).astype(np.float32)
    dtmu32 = (np.float32(dt) / (mr32 * np.float32(MU_0))).astype(np.float32)
    eps = dt / f64(cb32)
    mu = dt / f64(dtmu32)
    ivxh = f64(grid.inv_dx_h)
    ivyh = f64(grid.inv_dy_h)
    ivzh = f64(grid.inv_dz_h)
    cx, cy, cz = _curl_fwd_f64(ex, ey, ez, ivxh, ivyh, ivzh)
    hx2 = hx - (dt / mu) * cx
    hy2 = hy - (dt / mu) * cy
    hz2 = hz - (dt / mu) * cz
    e_term = 0.5 * (np.sum(eps * weights["ex"] * ex * ex)
                    + np.sum(eps * weights["ey"] * ey * ey)
                    + np.sum(eps * weights["ez"] * ez * ez))
    h_term = 0.5 * (np.sum(mu * weights["hx"] * hx * hx2)
                    + np.sum(mu * weights["hy"] * hy * hy2)
                    + np.sum(mu * weights["hz"] * hz * hz2))
    return float(e_term + h_term)


def adjointness_residual(grid, seed: int = 0) -> float:
    """Witness-validity check: the SBP identity

        sum_c w(Ec) e_c (curl_bwd h)_c  ==  sum_c w(Hc) h_c (curl_fwd e)_c

    for random fields with tangential-E PEC-pinned planes zeroed and the
    weight-0 planes zeroed. Returns the relative residual (float64);
    must be at rounding level (< 1e-12) for the energy to telescope."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    w = energy_weights(grid)
    f = {k: rng.standard_normal((nx, ny, nz)) for k in
         ("ex", "ey", "ez", "hx", "hy", "hz")}
    # PEC pinning exactly as rfx.boundaries.pec.apply_pec does
    for arr, planes in (
        (f["ey"], [(0, 0), (0, -1)]), (f["ez"], [(0, 0), (0, -1)]),
        (f["ex"], [(1, 0), (1, -1)]), (f["ez"], [(1, 0), (1, -1)]),
        (f["ex"], [(2, 0), (2, -1)]), (f["ey"], [(2, 0), (2, -1)]),
        (f["ez"], [(2, -1)]),
    ):
        for ax, idx in planes:
            sl = [slice(None)] * 3
            sl[ax] = idx
            arr[tuple(sl)] = 0.0
    # zero the weight-0 planes so their (undefined-metric) values don't
    # enter either side
    for k in f:
        f[k] = np.where(w[k] > 0, f[k], 0.0)

    f64 = lambda a: np.asarray(a, dtype=np.float64)
    ivx, ivy, ivz = f64(grid.inv_dx), f64(grid.inv_dy), f64(grid.inv_dz)
    ivxh, ivyh, ivzh = f64(grid.inv_dx_h), f64(grid.inv_dy_h), f64(grid.inv_dz_h)

    def sb(a, ax):
        out = np.zeros_like(a)
        if ax == 0:
            out[1:] = a[:-1]
        elif ax == 1:
            out[:, 1:] = a[:, :-1]
        else:
            out[:, :, 1:] = a[:, :, :-1]
        return out

    # backward curl of H (E-update stencil, rfx._curl_h_nu)
    bx = (f["hz"] - sb(f["hz"], 1)) * ivy[None, :, None] \
        - (f["hy"] - sb(f["hy"], 2)) * ivz[None, None, :]
    by = (f["hx"] - sb(f["hx"], 2)) * ivz[None, None, :] \
        - (f["hz"] - sb(f["hz"], 0)) * ivx[:, None, None]
    bz = (f["hy"] - sb(f["hy"], 0)) * ivx[:, None, None] \
        - (f["hx"] - sb(f["hx"], 1)) * ivy[None, :, None]
    # forward curl of E (H-update stencil)
    cx, cy, cz = _curl_fwd_f64(f["ex"], f["ey"], f["ez"], ivxh, ivyh, ivzh)

    lhs = (np.sum(w["ex"] * f["ex"] * bx) + np.sum(w["ey"] * f["ey"] * by)
           + np.sum(w["ez"] * f["ez"] * bz))
    rhs = (np.sum(w["hx"] * f["hx"] * cx) + np.sum(w["hy"] * f["hy"] * cy)
           + np.sum(w["hz"] * f["hz"] * cz))
    scale = max(abs(lhs), abs(rhs), 1e-300)
    return abs(lhs - rhs) / scale
