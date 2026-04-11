"""Non-uniform Yee grid FDTD runner.

Supports spatially-varying dx, dy, dz profiles. The most common use
is dz-graded meshes for thin-substrate structures where z-resolution
must be fine near the substrate but coarse in the air region; dx/dy
graded meshes additionally allow fine cells near metal edges
(fringing-field physics in patch antennas, microstrip filters, etc.)
without paying the cost of a uniform fine mesh everywhere.

Back-compat: `make_nonuniform_grid(domain_xy, dz_profile, dx, ...)`
with a scalar `dx` still produces a uniform-xy grid; only `dz` is
graded. Pass `dx_profile=` / `dy_profile=` to enable per-cell x/y
spacing.

Uses update_h_nu / update_e_nu from core/yee.py with pre-computed
per-axis inverse spacing arrays. Fully JIT-compiled via jax.lax.scan.
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
    """Non-uniform grid with per-axis cell-size arrays.

    ``dx`` / ``dy`` are the BOUNDARY cell sizes (used by CPML and any
    legacy code that reads a scalar spacing); ``dx_arr`` / ``dy_arr`` /
    ``dz`` hold the per-cell spacings. In the uniform-xy case,
    ``dx_arr`` is ``jnp.full(nx, dx)`` and ``dy_arr`` is analogous.
    """
    nx: int
    ny: int
    nz: int
    dx: float              # BOUNDARY x cell size (CPML + legacy scalars)
    dy: float              # BOUNDARY y cell size
    dx_arr: jnp.ndarray    # (nx,) — per-cell dx (includes CPML padding)
    dy_arr: jnp.ndarray    # (ny,) — per-cell dy
    dz: jnp.ndarray        # (nz,) z cell sizes (already per-cell)
    dt: float              # timestep (from min cell CFL)
    cpml_layers: int
    # Pre-computed inverse spacing arrays (length N, padded)
    inv_dx: jnp.ndarray    # (nx,) — 1/dx_arr[i]
    inv_dy: jnp.ndarray    # (ny,) — 1/dy_arr[j]
    inv_dz: jnp.ndarray    # (nz,) — 1/dz[k] per cell
    inv_dx_h: jnp.ndarray  # (nx,) — 2/(dx_arr[i]+dx_arr[i+1]) padded
    inv_dy_h: jnp.ndarray  # (ny,) — 2/(dy_arr[j]+dy_arr[j+1]) padded
    inv_dz_h: jnp.ndarray  # (nz,) — 2/(dz[k]+dz[k+1]), padded

    @property
    def shape(self):
        """Grid shape (nx, ny, nz) — duck-typing compatible with Grid."""
        return (self.nx, self.ny, self.nz)


def _pad_profile(profile: np.ndarray, cpml_layers: int) -> np.ndarray:
    """Pad a 1-D cell-size profile with CPML cells on both ends.

    CPML uses constant spacing matching the boundary cell size, so the
    padding on each end is ``cpml_layers`` copies of ``profile[0]`` and
    ``profile[-1]``, respectively.
    """
    lo_pad = np.full(cpml_layers, float(profile[0]))
    hi_pad = np.full(cpml_layers, float(profile[-1]))
    return np.concatenate([lo_pad, np.asarray(profile, dtype=np.float64), hi_pad])


def _profile_to_inv_arrays(profile_full: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (inv_d, inv_d_h) from a padded 1-D cell-size profile.

    ``inv_d[i] = 1/profile[i]`` and
    ``inv_d_h[i] = 2/(profile[i]+profile[i+1])`` (padded with 0 at end).
    """
    arr = jnp.asarray(profile_full, dtype=jnp.float32)
    inv_d = 1.0 / arr
    inv_d_mean = 2.0 / (arr[:-1] + arr[1:])
    inv_d_h = jnp.concatenate([inv_d_mean, jnp.zeros(1, dtype=jnp.float32)])
    return inv_d, inv_d_h


def make_nonuniform_grid(
    domain_xy: tuple[float, float],
    dz_profile: np.ndarray,
    dx: float,
    cpml_layers: int = 8,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
) -> NonUniformGrid:
    """Create a non-uniform Yee grid.

    Parameters
    ----------
    domain_xy : (Lx, Ly) in metres
        Only used when ``dx_profile`` / ``dy_profile`` are None — in
        that case the xy mesh is uniform with spacing ``dx``.
    dz_profile : 1D array of z cell sizes in metres (physical domain only)
    dx : float
        Boundary cell size (also used for CPML padding and as the
        uniform-xy spacing when no xy profile is provided).
    cpml_layers : int
        Number of CPML cells added to each face.
    dx_profile, dy_profile : 1D arrays or None
        Optional per-cell x / y spacings for the physical (non-CPML)
        interior. When provided, the first and last values must match
        ``dx`` (they set the boundary cell size used by the CPML).
    """
    # --- x profile (uniform or provided) ---
    if dx_profile is None:
        nx_interior = int(round(domain_xy[0] / dx))
        dx_prof_phys = np.full(nx_interior, float(dx))
    else:
        dx_prof_phys = np.asarray(dx_profile, dtype=np.float64)
        # Guard: CPML cells must have the same size as the boundary interior
        # cells, otherwise the CPML σ/κ profile is miscalibrated.
        if abs(float(dx_prof_phys[0]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[0]={float(dx_prof_phys[0])} must equal boundary "
                f"dx={float(dx)} (CPML cells use the boundary spacing)."
            )
        if abs(float(dx_prof_phys[-1]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[-1]={float(dx_prof_phys[-1])} must equal boundary "
                f"dx={float(dx)}."
            )
    dx_full = _pad_profile(dx_prof_phys, cpml_layers)
    nx = len(dx_full)

    # --- y profile ---
    if dy_profile is None:
        # Uniform y uses `dx` as the cell size (legacy behaviour)
        ny_interior = int(round(domain_xy[1] / dx))
        dy_prof_phys = np.full(ny_interior, float(dx))
        dy_boundary = float(dx)
    else:
        dy_prof_phys = np.asarray(dy_profile, dtype=np.float64)
        dy_boundary = float(dy_prof_phys[0])
        if abs(float(dy_prof_phys[-1]) - dy_boundary) > 1e-12:
            raise ValueError(
                f"dy_profile boundary cells must match each other "
                f"(got lo={dy_boundary}, hi={float(dy_prof_phys[-1])})."
            )
    dy_full = _pad_profile(dy_prof_phys, cpml_layers)
    ny = len(dy_full)

    # --- z profile ---
    dz_full = _pad_profile(dz_profile, cpml_layers)
    nz = len(dz_full)

    # --- CFL from minimum cell size on every axis ---
    dx_min = float(np.min(dx_full))
    dy_min = float(np.min(dy_full))
    dz_min = float(np.min(dz_full))
    dt = 0.99 / (C0 * np.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))

    # --- Per-cell arrays + inverse spacings ---
    dx_arr = jnp.asarray(dx_full, dtype=jnp.float32)
    dy_arr = jnp.asarray(dy_full, dtype=jnp.float32)
    dz_arr = jnp.asarray(dz_full, dtype=jnp.float32)

    inv_dx, inv_dx_h = _profile_to_inv_arrays(dx_full)
    inv_dy, inv_dy_h = _profile_to_inv_arrays(dy_full)
    inv_dz, inv_dz_h = _profile_to_inv_arrays(dz_full)

    return NonUniformGrid(
        nx=nx, ny=ny, nz=nz,
        dx=float(dx), dy=dy_boundary,
        dx_arr=dx_arr, dy_arr=dy_arr, dz=dz_arr,
        dt=float(dt), cpml_layers=cpml_layers,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
    )


def _interior_line_positions(d_arr_np: np.ndarray, cpml: int) -> np.ndarray:
    """Return cell-edge positions (0 at first interior face) for a padded
    cell-size array. Length = n_interior + 1.
    """
    interior = d_arr_np[cpml : len(d_arr_np) - cpml]
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    return edges


def z_position_to_index(grid: NonUniformGrid, z_phys: float) -> int:
    """Convert physical z-coordinate to (cpml-offset) grid index."""
    cpml = grid.cpml_layers
    edges = _interior_line_positions(np.asarray(grid.dz), cpml)
    idx = int(np.argmin(np.abs(edges - float(z_phys))))
    return idx + cpml


def _axis_position_to_index(d_arr: jnp.ndarray, cpml: int, pos: float) -> int:
    """Generic non-uniform axis lookup.

    Uses cell-edge positions (same convention as z_position_to_index):
    position 0 is the first interior face, position ``sum(interior)`` is
    the last interior face.
    """
    edges = _interior_line_positions(np.asarray(d_arr), cpml)
    idx = int(np.argmin(np.abs(edges - float(pos))))
    return idx + cpml


def position_to_index(grid: NonUniformGrid, pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to grid indices for NonUniformGrid.

    All three axes use cumulative cell-size lookup. In the uniform-xy
    case (``dx_arr`` constant) this reduces to the legacy
    ``round(pos[0]/dx) + cpml`` behaviour within one cell.
    """
    cpml = grid.cpml_layers
    i = _axis_position_to_index(grid.dx_arr, cpml, pos[0])
    j = _axis_position_to_index(grid.dy_arr, cpml, pos[1])
    k = z_position_to_index(grid, pos[2])
    return (i, j, k)


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

    # Cell volume: dx_i * dy_j * dz_k (per-cell on each axis)
    dx_local = float(np.asarray(grid.dx_arr)[i])
    dy_local = float(np.asarray(grid.dy_arr)[j])
    dz_local = float(grid.dz[k])
    dV = dx_local * dy_local * dz_local

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
    pec_faces: set[str] | None = None,
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

    # CPML: only initialize when cpml_layers > 0 (skip for PEC boundary)
    use_cpml = grid.cpml_layers > 0

    cpml_params = None
    cpml_state_init = None
    cpml_grid = None

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e

        # Pass NonUniformGrid directly — init_cpml duck-types dx/dy/dz
        cpml_params, cpml_state_init = init_cpml(grid, pec_faces=pec_faces)
        cpml_grid = grid

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

    carry_init = {"fdtd": state}
    if use_cpml:
        carry_init["cpml"] = cpml_state_init

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
        # Pre-compute per-cell dx/dy at each wire-port cell so the
        # JIT'd scan below uses the correct line-integral lengths even
        # on non-uniform xy meshes.
        _dx_arr_np = np.asarray(grid.dx_arr)
        _dy_arr_np = np.asarray(grid.dy_arr)
        wp_meta = [(
            wp['mid_i'], wp['mid_j'], wp['mid_k'],
            wp['component'], wp['impedance'],
            float(_dx_arr_np[wp['mid_i']]),
            float(_dy_arr_np[wp['mid_j']]),
        ) for wp in wire_ports]
    else:
        use_wire_ports = False

    def step_fn(carry, xs):
        step_idx, src_vals = xs
        st = carry["fdtd"]

        # H update (non-uniform)
        st = update_h_nu(st, materials, dt, inv_dx_h, inv_dy_h, inv_dz_h)
        if use_cpml:
            st, cpml_new = apply_cpml_h(st, cpml_params, carry["cpml"],
                                         cpml_grid, "xyz")
        else:
            cpml_new = None

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

        if use_cpml:
            st, cpml_new = apply_cpml_e(st, cpml_params, cpml_new,
                                         cpml_grid, "xyz")

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
            for (v_dft, i_dft, vinc_dft), (mi, mj, mk, comp, z0, dxi, dyj) in \
                    zip(carry.get("wire_sparams", ()), wp_meta):
                # V = -E_comp * d_parallel, I = H-loop * d_transverse
                # dxi / dyj are the per-cell xy spacings at this port's
                # (mi, mj); dz_local is the per-cell z spacing.
                dz_local = grid.dz[mk]
                if comp == "ez":
                    v = -st.ez[mi, mj, mk] * dz_local
                    i_val = (st.hy[mi,mj,mk] - st.hy[mi-1,mj,mk]
                             - st.hx[mi,mj,mk] + st.hx[mi,mj-1,mk]) * dxi
                elif comp == "ex":
                    v = -st.ex[mi, mj, mk] * dxi
                    i_val = (st.hz[mi,mj,mk] - st.hz[mi,mj-1,mk]
                             - st.hy[mi,mj,mk] + st.hy[mi,mj,mk-1]) * dz_local
                else:
                    v = -st.ey[mi, mj, mk] * dyj
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

        new_carry = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
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
        for j, ((v_dft, i_dft, _), (_, _, _, _, z0, _, _)) in enumerate(
                zip(final["wire_sparams"], wp_meta)):
            # Wave decomposition: S11 = (V - Z0*I) / (V + Z0*I)
            a = (v_dft + z0 * i_dft) / (2.0 * _np.sqrt(z0))
            safe_a = jnp.where(jnp.abs(a) > 0, a, jnp.ones_like(a))
            b = (v_dft - z0 * i_dft) / (2.0 * _np.sqrt(z0))
            S[j, j, :] = _np.array(b / safe_a)
        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)

    return result
