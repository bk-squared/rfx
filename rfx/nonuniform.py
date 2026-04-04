"""Non-uniform Yee grid FDTD runner.

Supports spatially-varying dz (z-graded mesh) with uniform dx, dy.
This is the standard approach used by CST/OpenEMS for thin-substrate
structures where z-resolution must be fine near the substrate but
coarse in the air region.

Uses update_h_nu / update_e_nu from core/yee.py with pre-computed
inverse spacing arrays. Fully JIT-compiled via jax.lax.scan.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    update_h_nu, update_e_nu, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec, apply_pec_mask

C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))


class NonUniformGrid(NamedTuple):
    """Non-uniform grid specification (uniform dx/dy, graded dz)."""
    nx: int
    ny: int
    nz: int
    dx: float              # uniform x cell size
    dy: float              # uniform y cell size
    dz: jnp.ndarray        # (nz,) z cell sizes
    dt: float              # timestep (from min cell CFL)
    cpml_layers: int
    # Pre-computed inverse spacing arrays (length N, padded)
    inv_dx: jnp.ndarray    # (nx,) — 1/dx (uniform, all same)
    inv_dy: jnp.ndarray    # (ny,)
    inv_dz: jnp.ndarray    # (nz,) — 1/dz[k] per cell
    inv_dx_h: jnp.ndarray  # (nx,) — 2/(dx+dx) = 1/dx (uniform)
    inv_dy_h: jnp.ndarray  # (ny,)
    inv_dz_h: jnp.ndarray  # (nz,) — 2/(dz[k]+dz[k+1]), padded


def make_nonuniform_grid(
    domain_xy: tuple[float, float],
    dz_profile: np.ndarray,
    dx: float,
    cpml_layers: int = 12,
) -> NonUniformGrid:
    """Create a non-uniform grid with graded z-spacing.

    Parameters
    ----------
    domain_xy : (Lx, Ly) in metres
    dz_profile : 1D array of z cell sizes in metres (physical domain only)
    dx : uniform x/y cell size
    cpml_layers : number of CPML cells (added to all 6 faces)
    """
    dy = dx
    nx = int(round(domain_xy[0] / dx)) + 2 * cpml_layers
    ny = int(round(domain_xy[1] / dy)) + 2 * cpml_layers

    # Add CPML cells in z with the boundary cell size
    dz_lo_pad = np.full(cpml_layers, float(dz_profile[0]))
    dz_hi_pad = np.full(cpml_layers, float(dz_profile[-1]))
    dz_full = np.concatenate([dz_lo_pad, dz_profile, dz_hi_pad])
    nz = len(dz_full)

    # CFL from minimum cell size
    dz_min = float(np.min(dz_full))
    dt = 0.99 / (C0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz_min**2))

    dz_arr = jnp.array(dz_full, dtype=jnp.float32)

    # Inverse spacing arrays
    inv_dx = jnp.ones(nx, dtype=jnp.float32) / dx
    inv_dy = jnp.ones(ny, dtype=jnp.float32) / dy
    inv_dz = 1.0 / dz_arr

    # H-curl mean spacing (padded with 0 at end)
    inv_dx_h = jnp.ones(nx, dtype=jnp.float32) / dx  # uniform → same
    inv_dy_h = jnp.ones(ny, dtype=jnp.float32) / dy
    inv_dz_mean = 2.0 / (dz_arr[:-1] + dz_arr[1:])
    inv_dz_h = jnp.concatenate([inv_dz_mean, jnp.zeros(1)])

    return NonUniformGrid(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz_arr, dt=float(dt),
        cpml_layers=cpml_layers,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
    )


def z_position_to_index(grid: NonUniformGrid, z_phys: float) -> int:
    """Convert physical z-coordinate to grid index."""
    cpml = grid.cpml_layers
    dz_np = np.array(grid.dz)
    z_cumsum = np.cumsum(dz_np[cpml:])  # physical z positions
    z_cumsum = np.insert(z_cumsum, 0, 0)  # start at 0
    idx = int(np.argmin(np.abs(z_cumsum - z_phys)))
    return idx + cpml


def make_z_profile(
    features: list[float],
    domain_z: float,
    dx_fine: float,
    dx_coarse: float | None = None,
    grading: float = 1.4,
) -> np.ndarray:
    """Generate z-profile that snaps to feature boundaries.

    Fine cells are used near feature boundaries; coarse cells fill the
    remaining space.  Adjacent cells differ by at most ``grading``.

    Parameters
    ----------
    features : list of z-positions that must align to cell boundaries
    domain_z : total z domain height
    dx_fine : fine cell size (near features)
    dx_coarse : coarse cell size (away from features). If None, uses dx_fine
        everywhere (no grading).
    grading : max ratio between adjacent cells (default 1.4)
    """
    if dx_coarse is None:
        dx_coarse = dx_fine

    features = sorted(set(features + [0, domain_z]))

    cells = []
    for i in range(len(features) - 1):
        span = features[i + 1] - features[i]
        if span <= 0:
            continue

        if dx_coarse <= dx_fine * 1.01 or span <= 4 * dx_fine:
            # Uniform fine cells for thin segments or when no grading needed
            n = max(1, int(round(span / dx_fine)))
            dz = span / n
            cells.extend([dz] * n)
        else:
            # Graded transition: fine → coarse → fine
            # Build from both ends toward the middle
            left = []
            dz = dx_fine
            remaining = span
            while remaining > 0 and dz < dx_coarse:
                dz_use = min(dz, remaining)
                left.append(dz_use)
                remaining -= dz_use
                dz = min(dz * grading, dx_coarse)

            # Fill middle with coarse cells
            if remaining > dx_coarse * 0.5:
                n_mid = max(1, int(round(remaining / dx_coarse)))
                mid = [remaining / n_mid] * n_mid
            else:
                mid = [remaining] if remaining > 1e-15 else []

            cells.extend(left + mid)

    return np.array(cells)


def make_current_source(grid: NonUniformGrid, position_ijk, component,
                        waveform_fn, n_steps, materials):
    """Create a properly normalized current source for non-uniform grid.

    The waveform specifies CURRENT (Amperes). The E-field addition is:
    E += (dt/ε) × I_source / dV
    where dV = dx × dy × dz_local (actual cell volume).

    This gives resolution-independent injected POWER regardless of cell size.
    Same approach as Meep's internal source normalization.
    """
    import jax
    i, j, k = position_ijk
    eps = float(materials.eps_r[i, j, k]) * EPS_0
    sigma = float(materials.sigma[i, j, k])
    loss = sigma * grid.dt / (2.0 * eps)

    # Cb = dt / (eps * (1 + loss))
    cb = (grid.dt / eps) / (1.0 + loss)

    # Cell volume: dx * dy * dz_local
    dz_local = float(grid.dz[k])
    dV = grid.dx * grid.dy * dz_local

    # Normalized waveform: Cb * I(t) / dV
    # This ensures power = ∫(J·E)dV is independent of cell size
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / dV) * jax.vmap(waveform_fn)(times)

    return (i, j, k, component, np.array(waveform))


def _curl_h_nu(state, inv_dx, inv_dy, inv_dz):
    """Compute curl(H) using non-uniform backward differences.

    Shared by both plain and dispersive E updates on non-uniform grids.
    """
    from rfx.core.yee import _shift_bwd
    hx, hy, hz = state.hx, state.hy, state.hz

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )
    return curl_x, curl_y, curl_z


def _update_e_nu_dispersive(
    state: FDTDState,
    materials: MaterialArrays,
    dt: float,
    inv_dx: jnp.ndarray,
    inv_dy: jnp.ndarray,
    inv_dz: jnp.ndarray,
    *,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
) -> tuple[FDTDState, object | None, object | None]:
    """E-field update with ADE dispersion on non-uniform grid.

    Uses per-axis inverse spacing arrays for curl(H), then applies the
    same ADE coefficient math as the uniform path. The ADE coefficients
    (ca, cb, cc, alpha, beta, etc.) are pre-baked spatial arrays that do
    not depend on dx, so they work unchanged on non-uniform grids.

    Mirrors the structure of ``_update_e_with_optional_dispersion`` in
    ``rfx/simulation.py`` but replaces the uniform ``curl / dx`` with
    non-uniform ``curl * inv_d[axis]``.
    """
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    curl_x, curl_y, curl_z = _curl_h_nu(state, inv_dx, inv_dy, inv_dz)
    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # --- Debye only ---
    if debye is not None and lorentz is None:
        debye_coeffs, debye_state = debye
        ca, cb, cc = debye_coeffs.ca, debye_coeffs.cb, debye_coeffs.cc
        alpha, beta = debye_coeffs.alpha, debye_coeffs.beta

        ex_new = ca * ex_old + cb * curl_x + jnp.sum(cc * debye_state.px, axis=0)
        ey_new = ca * ey_old + cb * curl_y + jnp.sum(cc * debye_state.py, axis=0)
        ez_new = ca * ez_old + cb * curl_z + jnp.sum(cc * debye_state.pz, axis=0)

        px_new = alpha * debye_state.px + beta * (ex_new[None] + ex_old[None])
        py_new = alpha * debye_state.py + beta * (ey_new[None] + ey_old[None])
        pz_new = alpha * debye_state.pz + beta * (ez_new[None] + ez_old[None])

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)
        return new_fdtd, new_debye, None

    # --- Lorentz only ---
    if lorentz is not None and debye is None:
        lorentz_coeffs, lor_state = lorentz
        ca, cb, cc = lorentz_coeffs.ca, lorentz_coeffs.cb, lorentz_coeffs.cc
        a, b, c = lorentz_coeffs.a, lorentz_coeffs.b, lorentz_coeffs.c

        px_new = a * lor_state.px + b * lor_state.px_prev + c * ex_old[None]
        py_new = a * lor_state.py + b * lor_state.py_prev + c * ey_old[None]
        pz_new = a * lor_state.pz + b * lor_state.pz_prev + c * ez_old[None]

        dpx = jnp.sum(px_new - lor_state.px, axis=0)
        dpy = jnp.sum(py_new - lor_state.py, axis=0)
        dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

        ex_new = ca * ex_old + cb * curl_x - cc * dpx
        ey_new = ca * ey_old + cb * curl_y - cc * dpy
        ez_new = ca * ez_old + cb * curl_z - cc * dpz

        new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                                  step=state.step + 1)
        new_lor = LorentzState(
            px=px_new, py=py_new, pz=pz_new,
            px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
        )
        return new_fdtd, None, new_lor

    # --- Mixed Debye + Lorentz ---
    debye_coeffs, debye_state = debye
    lorentz_coeffs, lor_state = lorentz

    # Explicit Lorentz polarization update first
    px_l_new = (lorentz_coeffs.a * lor_state.px
                + lorentz_coeffs.b * lor_state.px_prev
                + lorentz_coeffs.c * ex_old[None])
    py_l_new = (lorentz_coeffs.a * lor_state.py
                + lorentz_coeffs.b * lor_state.py_prev
                + lorentz_coeffs.c * ey_old[None])
    pz_l_new = (lorentz_coeffs.a * lor_state.pz
                + lorentz_coeffs.b * lor_state.pz_prev
                + lorentz_coeffs.c * ez_old[None])

    dpx_l = jnp.sum(px_l_new - lor_state.px, axis=0)
    dpy_l = jnp.sum(py_l_new - lor_state.py, axis=0)
    dpz_l = jnp.sum(pz_l_new - lor_state.pz, axis=0)

    beta_sum = jnp.sum(debye_coeffs.beta, axis=0)
    gamma_base = 1.0 / lorentz_coeffs.cc
    gamma_total = jnp.maximum(gamma_base + beta_sum, EPS_0 * 1e-10)
    numer_base = lorentz_coeffs.ca * gamma_base

    ca = (numer_base - beta_sum) / gamma_total
    cb = dt / gamma_total
    cc_debye = (1.0 - debye_coeffs.alpha) / gamma_total
    cc_lorentz = 1.0 / gamma_total

    ex_new = (ca * ex_old + cb * curl_x
              + jnp.sum(cc_debye * debye_state.px, axis=0)
              - cc_lorentz * dpx_l)
    ey_new = (ca * ey_old + cb * curl_y
              + jnp.sum(cc_debye * debye_state.py, axis=0)
              - cc_lorentz * dpy_l)
    ez_new = (ca * ez_old + cb * curl_z
              + jnp.sum(cc_debye * debye_state.pz, axis=0)
              - cc_lorentz * dpz_l)

    new_fdtd = state._replace(ex=ex_new, ey=ey_new, ez=ez_new,
                              step=state.step + 1)
    new_debye = DebyeState(
        px=debye_coeffs.alpha * debye_state.px + debye_coeffs.beta * (ex_new[None] + ex_old[None]),
        py=debye_coeffs.alpha * debye_state.py + debye_coeffs.beta * (ey_new[None] + ey_old[None]),
        pz=debye_coeffs.alpha * debye_state.pz + debye_coeffs.beta * (ez_new[None] + ez_old[None]),
    )
    new_lor = LorentzState(
        px=px_l_new, py=py_l_new, pz=pz_l_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )
    return new_fdtd, new_debye, new_lor


def run_nonuniform(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    pec_mask=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
) -> dict:
    """Run non-uniform FDTD via jax.lax.scan.

    Parameters
    ----------
    sources : list of (i, j, k, component, waveform_array)
    probes : list of (i, j, k, component)
    wire_ports : list of dict with keys:
        mid_i, mid_j, mid_k, component, impedance, waveform_array
    s_param_freqs : (n_freqs,) array for S-param DFT
    debye : (DebyeCoeffs, DebyeState) or None
    lorentz : (LorentzCoeffs, LorentzState) or None
    """
    sources = sources or []
    probes = probes or []
    wire_ports = wire_ports or []
    dt = grid.dt
    use_wire_ports = len(wire_ports) > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None

    # CPML: create a uniform-dx proxy grid for CPML coefficient computation.
    # CPML operates on the outer shell (cpml_layers cells on each face).
    # Using dx as the reference cell size is conservative and correct for
    # the x/y faces. Z-faces use the boundary dz (first/last cells).
    from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e

    # CPML proxy: lightweight object mimicking Grid with exact shape
    class _CpmlProxy:
        def __init__(self, g):
            self.nx = g.nx; self.ny = g.ny; self.nz = g.nz
            self.dx = g.dx; self.dt = dt
            self.cpml_layers = g.cpml_layers
            self.shape = (g.nx, g.ny, g.nz)
            self.is_2d = False
    cpml_proxy = _CpmlProxy(grid)
    cpml_params, cpml_state_init = init_cpml(cpml_proxy)

    use_pec_mask = pec_mask is not None

    if sources:
        src_waveforms = jnp.stack([jnp.array(s[4]) for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_meta = [(s[0], s[1], s[2], s[3]) for s in sources]
    prb_meta = [(p[0], p[1], p[2], p[3]) for p in probes]

    state = init_state((grid.nx, grid.ny, grid.nz))

    inv_dx_h = grid.inv_dx_h
    inv_dy_h = grid.inv_dy_h
    inv_dz_h = grid.inv_dz_h
    inv_dx = grid.inv_dx
    inv_dy = grid.inv_dy
    inv_dz = grid.inv_dz

    carry_init = {"fdtd": state, "cpml": cpml_state_init}

    # Debye/Lorentz ADE state
    if use_debye:
        debye_coeffs, debye_state = debye
        carry_init["debye"] = debye_state

    if use_lorentz:
        lorentz_coeffs, lorentz_state = lorentz
        carry_init["lorentz"] = lorentz_state

    # Wire port S-param DFT accumulators
    if use_wire_ports and s_param_freqs is not None:
        sp_freqs = jnp.asarray(s_param_freqs, dtype=jnp.float32)
        nf = len(sp_freqs)
        carry_init["wire_sparams"] = tuple(
            (jnp.zeros(nf, dtype=jnp.complex64),  # v_dft
             jnp.zeros(nf, dtype=jnp.complex64),  # i_dft
             jnp.zeros(nf, dtype=jnp.complex64))   # v_inc_dft
            for _ in wire_ports
        )
        wp_meta = [(wp['mid_i'], wp['mid_j'], wp['mid_k'],
                     wp['component'], wp['impedance']) for wp in wire_ports]
    else:
        use_wire_ports = False

    def step_fn(carry, xs):
        step_idx, src_vals = xs
        st = carry["fdtd"]

        # H update (non-uniform) + CPML
        st = update_h_nu(st, materials, dt, inv_dx_h, inv_dy_h, inv_dz_h)
        st, cpml_new = apply_cpml_h(st, cpml_params, carry["cpml"],
                                     cpml_proxy, "xyz")

        # E update: use ADE-aware path when dispersive materials are present
        debye_new = None
        lorentz_new = None
        if use_debye or use_lorentz:
            st, debye_new, lorentz_new = _update_e_nu_dispersive(
                st, materials, dt, inv_dx, inv_dy, inv_dz,
                debye=(debye_coeffs, carry["debye"]) if use_debye else None,
                lorentz=(lorentz_coeffs, carry["lorentz"]) if use_lorentz else None,
            )
        else:
            st = update_e_nu(st, materials, dt, inv_dx, inv_dy, inv_dz)

        st, cpml_new = apply_cpml_e(st, cpml_params, cpml_new,
                                     cpml_proxy, "xyz")

        # PEC
        st = apply_pec(st)
        if use_pec_mask:
            st = apply_pec_mask(st, pec_mask)

        # Sources (point sources + wire port excitation)
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        # Wire port V/I DFT accumulation
        t = step_idx.astype(jnp.float32) * dt
        new_wire_sp = None
        if use_wire_ports:
            new_wire_sp = []
            for (v_dft, i_dft, vinc_dft), (mi, mj, mk, comp, z0) in \
                    zip(carry.get("wire_sparams", ()), wp_meta):
                # V = -E_comp * d_parallel, I = H-loop * d_transverse
                # Non-uniform z: Ez uses dz[k], Ex/Ey use dx/dy (uniform)
                dz_local = grid.dz[mk]
                if comp == "ez":
                    v = -st.ez[mi, mj, mk] * dz_local
                    i_val = (st.hy[mi,mj,mk] - st.hy[mi-1,mj,mk]
                             - st.hx[mi,mj,mk] + st.hx[mi,mj-1,mk]) * grid.dx
                elif comp == "ex":
                    v = -st.ex[mi, mj, mk] * grid.dx
                    i_val = (st.hz[mi,mj,mk] - st.hz[mi,mj-1,mk]
                             - st.hy[mi,mj,mk] + st.hy[mi,mj,mk-1]) * dz_local
                else:
                    v = -st.ey[mi, mj, mk] * grid.dy
                    i_val = (st.hx[mi,mj,mk] - st.hx[mi,mj,mk-1]
                             - st.hz[mi,mj,mk] + st.hz[mi-1,mj,mk]) * dz_local
                t_f64 = t.astype(jnp.float64) if hasattr(t, 'astype') else jnp.float64(t)
                phase = jnp.exp(-1j * 2.0 * jnp.pi * sp_freqs.astype(jnp.float64) * t_f64).astype(jnp.complex64) * dt
                new_wire_sp.append((
                    v_dft + v * phase,
                    i_dft + i_val * phase,
                    vinc_dft,
                ))

        # Probes
        samples = [getattr(st, pc)[pi, pj, pk] for pi, pj, pk, pc in prb_meta]
        probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

        new_carry = {"fdtd": st, "cpml": cpml_new}
        if use_debye and debye_new is not None:
            new_carry["debye"] = debye_new
        if use_lorentz and lorentz_new is not None:
            new_carry["lorentz"] = lorentz_new
        if use_wire_ports and new_wire_sp is not None:
            new_carry["wire_sparams"] = tuple(new_wire_sp)
        return new_carry, probe_out

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
    final, time_series = jax.lax.scan(step_fn, carry_init, xs)

    result = {
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Extract S-params from wire port DFTs
    if use_wire_ports and "wire_sparams" in final:
        import numpy as _np
        n_wp = len(wire_ports)
        nf = len(sp_freqs)
        S = _np.zeros((n_wp, n_wp, nf), dtype=_np.complex64)
        for j, ((v_dft, i_dft, _), (_, _, _, _, z0)) in enumerate(
                zip(final["wire_sparams"], wp_meta)):
            # Wave decomposition: S11 = (V - Z0*I) / (V + Z0*I)
            a = (v_dft + z0 * i_dft) / (2.0 * _np.sqrt(z0))
            safe_a = jnp.where(jnp.abs(a) > 0, a, jnp.ones_like(a))
            b = (v_dft - z0 * i_dft) / (2.0 * _np.sqrt(z0))
            S[j, j, :] = _np.array(b / safe_a)
        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)

    return result
