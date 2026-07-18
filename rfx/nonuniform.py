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
from rfx.boundaries.pec import apply_pec, apply_pec_mask, apply_pec_occupancy
from rfx.core.jax_utils import is_tracer

C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))


class NonUniformGrid(NamedTuple):
    """Non-uniform grid with per-axis cell-size arrays.

    ``dx`` / ``dy`` are the BOUNDARY cell sizes (used by CPML and any
    legacy code that reads a scalar spacing); ``dx_arr`` / ``dy_arr`` /
    ``dz`` hold the per-cell spacings. In the uniform-xy case,
    ``dx_arr`` is ``jnp.full(nx, dx)`` and ``dy_arr`` is analogous.

    ``pad_{axis}_{lo|hi}`` mirror ``rfx.grid.Grid`` per-face padding —
    a face whose token is ``pmc``/``pec``/``periodic`` gets 0 cells on
    that side even when the axis as a whole uses CPML. For back-compat
    the six fields default to ``cpml_layers`` on every face (the
    pre-per-face-allocation symmetric layout).
    """
    nx: int
    ny: int
    nz: int
    dx: float              # BOUNDARY x cell size (CPML + legacy scalars)
    dy: float              # BOUNDARY y cell size
    dx_arr: jnp.ndarray    # (nx,) — per-cell dx (includes CPML padding)
    dy_arr: jnp.ndarray    # (ny,) — per-cell dy
    dz: jnp.ndarray        # (nz,) z cell sizes (already per-cell)
    dt: float | jax.Array  # timestep; eager float in normal use, traced
                           # jax.Array on the run_nonuniform traced path.
                           # Readers must not apply float() / .item() without
                           # an is_tracer(dt) guard or a host-boundary context.
    cpml_layers: int
    # Pre-computed inverse spacing arrays (length N, padded).
    # CORE-C2: inv_d* feed the E update (mean spacing), inv_d*_h feed
    # the H update (local cell width). See _profile_to_inv_arrays.
    inv_dx: jnp.ndarray    # (nx,) — E update: 2/(dx[i-1]+dx[i]); [0]=1/dx[0]
    inv_dy: jnp.ndarray    # (ny,) — E update: 2/(dy[j-1]+dy[j]); [0]=1/dy[0]
    inv_dz: jnp.ndarray    # (nz,) — E update: 2/(dz[k-1]+dz[k]); [0]=1/dz[0]
    inv_dx_h: jnp.ndarray  # (nx,) — H update: 1/dx[i]; [nx-1]=0
    inv_dy_h: jnp.ndarray  # (ny,) — H update: 1/dy[j]; [ny-1]=0
    inv_dz_h: jnp.ndarray  # (nz,) — H update: 1/dz[k]; [nz-1]=0
    # Per-face CPML padding (PMC+CPML composition fix, 2026-04)
    pad_x_lo: int = 0
    pad_x_hi: int = 0
    pad_y_lo: int = 0
    pad_y_hi: int = 0
    pad_z_lo: int = 0
    pad_z_hi: int = 0

    @property
    def shape(self):
        """Grid shape (nx, ny, nz) — duck-typing compatible with Grid."""
        return (self.nx, self.ny, self.nz)

    @property
    def axis_pads(self):
        """Leading (``lo``) per-axis pad — the number coordinate-offset
        callers subtract from array indices to recover user coords."""
        return (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)




def _pad_profile(profile, pad_lo: int, pad_hi: int | None = None):
    """Pad a 1-D cell-size profile with CPML cells on each end.

    ``pad_lo`` and ``pad_hi`` may differ (per-face allocation, 2026-04 —
    a PMC/PEC face gets 0 cells on that side while the opposing CPML
    face keeps its allocation). If ``pad_hi`` is omitted the symmetric
    ``pad_lo`` count is used on both ends (pre-per-face-allocation behaviour).

    CPML uses constant spacing matching the boundary cell size, so the
    ``pad_lo`` cells on the leading side carry ``profile[0]`` and the
    ``pad_hi`` cells on the trailing side carry ``profile[-1]``.

    When ``profile`` is a JAX tracer the padding stays in-trace (needed
    for ``jax.grad`` w.r.t. ``dz_profile`` — mesh-as-design-variable).
    Otherwise the numpy path is used, preserving the Python-float ``dt``
    that Simulation-level callers depend on.
    """
    if pad_hi is None:
        pad_hi = pad_lo
    if is_tracer(profile):
        prof = jnp.asarray(profile, dtype=jnp.float32)
        lo_pad = jnp.full(pad_lo, prof[0])
        hi_pad = jnp.full(pad_hi, prof[-1])
        return jnp.concatenate([lo_pad, prof, hi_pad])
    lo_pad = np.full(pad_lo, float(profile[0]))
    hi_pad = np.full(pad_hi, float(profile[-1]))
    return np.concatenate([lo_pad, np.asarray(profile, dtype=np.float64), hi_pad])


def _profile_to_inv_arrays(profile_full: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(inv_d_e, inv_d_h)`` — the E-update and H-update inverse
    cell-spacing arrays for a padded 1-D cell-size profile ``d``.

    CORE-C2 fix (2026-05-16). The non-uniform Yee stencil needs a
    *different* metric for each update:

    * **H update** — the H-curl differences two E nodes that straddle
      one cell, so it divides by the **local cell width**::

          inv_d_h[k] = 1/d[k]       (k < N-1);   inv_d_h[N-1] = 0

    * **E update** — the E-curl differences two H cell-centres, whose
      separation is the **mean** of the two adjacent cell widths::

          inv_d_e[k] = 2/(d[k-1]+d[k])   (k >= 1);   inv_d_e[0] = 1/d[0]

    On a uniform mesh both collapse to ``1/d`` so the uniform path is
    bit-identical. The pre-fix code had the two swapped (H got the mean,
    E got the local width) — a stencil inconsistency that scaled the
    curl by ``2 d[k]/(d[k]+d[k±1])`` on every graded cell. The boundary
    entries (``inv_d_h[N-1]=0``, ``inv_d_e[0]=1/d[0]``) reproduce the
    prior boundary behaviour exactly, so PEC/CPML faces are untouched.

    The return order ``(inv_d_e, inv_d_h)`` matches the caller's
    ``inv_dx, inv_dx_h = _profile_to_inv_arrays(...)`` unpacking: the
    first slot feeds the E update, the second feeds ``update_h_nu``.
    """
    arr = jnp.asarray(profile_full, dtype=jnp.float32)
    inv_local = 1.0 / arr                          # 1/d[k]
    inv_mean = 2.0 / (arr[:-1] + arr[1:])          # 2/(d[k]+d[k+1])
    # H update: local cell width; trailing 0 (forward-diff has no d[N]).
    inv_d_h = jnp.concatenate([inv_local[:-1], jnp.zeros(1, dtype=jnp.float32)])
    # E update: mean of (d[k-1], d[k]); leading 1/d[0] (backward-diff
    # boundary — reproduces the pre-fix value so the face cell is
    # unchanged).
    inv_d_e = jnp.concatenate([inv_local[:1], inv_mean])
    return inv_d_e, inv_d_h


def make_nonuniform_grid(
    domain_xy: tuple[float, float],
    dz_profile: np.ndarray,
    dx: float,
    cpml_layers: int = 8,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    cpml_axes: str = "xyz",
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
        Number of CPML cells added to each face (when that face is
        absorbing — see ``pec_faces`` / ``pmc_faces``).
    dx_profile, dy_profile : 1D arrays or None
        Optional per-cell x / y spacings for the physical (non-CPML)
        interior. When provided, the first and last values must match
        ``dx`` (they set the boundary cell size used by the CPML).
    pec_faces, pmc_faces : set of str or None
        Face labels (``x_lo``, ``x_hi``, ``y_lo``, ``y_hi``, ``z_lo``,
        ``z_hi``) where the boundary is PEC / PMC. Per-face pad count
        is forced to 0 on these faces so the reflector plane aligns
        with the user domain edge. Added 2026-04 to close the
        PMC+CPML composition gap on the non-uniform mesh path.
    cpml_axes : str
        Axes that participate in CPML allocation (default ``"xyz"``).
        Per-face allocation still applies; an axis absent from
        ``cpml_axes`` gets pad=0 on both faces.
    """
    _pec = pec_faces or set()
    _pmc = pmc_faces or set()

    def _face_pad(axis: str, side: str) -> int:
        face = f"{axis}_{side}"
        if face in _pec or face in _pmc:
            return 0
        if axis not in cpml_axes:
            return 0
        return int(cpml_layers)

    pad_x_lo = _face_pad("x", "lo")
    pad_x_hi = _face_pad("x", "hi")
    pad_y_lo = _face_pad("y", "lo")
    pad_y_hi = _face_pad("y", "hi")
    pad_z_lo = _face_pad("z", "lo")
    pad_z_hi = _face_pad("z", "hi")
    # --- x profile (uniform or provided) ---
    if dx_profile is None:
        nx_interior = int(round(domain_xy[0] / dx))
        dx_prof_phys = np.full(nx_interior, float(dx))
    elif is_tracer(dx_profile):
        # Tracer path (mesh-as-design-variable): stay in jnp and skip the
        # concrete boundary validation. Caller is responsible for keeping
        # `dx_profile[0] == dx_profile[-1] == dx` (the CPML uses the
        # boundary scalar). Mirrors the 2026-04-17 dz tracer refactor.
        dx_prof_phys = jnp.asarray(dx_profile, dtype=jnp.float32)
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
    dx_full = _pad_profile(dx_prof_phys, pad_x_lo, pad_x_hi)
    nx = int(dx_full.shape[0])

    # --- y profile ---
    if dy_profile is None:
        # Uniform y uses `dx` as the cell size (legacy behaviour)
        ny_interior = int(round(domain_xy[1] / dx))
        dy_prof_phys = np.full(ny_interior, float(dx))
        dy_boundary = float(dx)
    elif is_tracer(dy_profile):
        # Tracer path: stay in jnp. Use the concrete scalar `dx` as the
        # boundary cell size — the caller must align `dy_profile[0]` and
        # `dy_profile[-1]` with `dx` (same contract as the concrete path).
        dy_prof_phys = jnp.asarray(dy_profile, dtype=jnp.float32)
        dy_boundary = float(dx)
    else:
        dy_prof_phys = np.asarray(dy_profile, dtype=np.float64)
        dy_boundary = float(dy_prof_phys[0])
        if abs(float(dy_prof_phys[-1]) - dy_boundary) > 1e-12:
            raise ValueError(
                f"dy_profile boundary cells must match each other "
                f"(got lo={dy_boundary}, hi={float(dy_prof_phys[-1])})."
            )
    dy_full = _pad_profile(dy_prof_phys, pad_y_lo, pad_y_hi)
    ny = int(dy_full.shape[0])

    # --- z profile ---
    dz_full = _pad_profile(dz_profile, pad_z_lo, pad_z_hi)
    nz = int(dz_full.shape[0])

    # --- CFL from minimum cell size on every axis ---
    def _axis_min(d_full):
        return jnp.min(d_full) if is_tracer(d_full) else float(np.min(d_full))

    any_traced = (
        is_tracer(dx_full) or is_tracer(dy_full) or is_tracer(dz_full)
    )
    dx_min = _axis_min(dx_full)
    dy_min = _axis_min(dy_full)
    dz_min = _axis_min(dz_full)
    if any_traced:
        dt = 0.99 / (C0 * jnp.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
    else:
        dt = float(
            0.99 / (C0 * np.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
        )

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
        dt=dt, cpml_layers=cpml_layers,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
        pad_x_lo=pad_x_lo, pad_x_hi=pad_x_hi,
        pad_y_lo=pad_y_lo, pad_y_hi=pad_y_hi,
        pad_z_lo=pad_z_lo, pad_z_hi=pad_z_hi,
    )


def _interior_line_positions(
    d_arr_np: np.ndarray, pad_lo: int, pad_hi: int | None = None,
) -> np.ndarray:
    """Return cell-edge positions (0 at first interior face) for a padded
    cell-size array. Length = n_interior + 1.

    ``pad_lo`` and ``pad_hi`` may differ (per-face allocation, 2026-04).
    Back-compat: a single-argument call treats the value as symmetric.
    """
    if pad_hi is None:
        pad_hi = pad_lo
    interior = d_arr_np[pad_lo : len(d_arr_np) - pad_hi]
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    return edges


def _nominal_edges_or_actual(
    d_arr, total_pad: int,
    *, pad_lo: int | None = None,
    fallback_dx: float | None = None,
) -> np.ndarray:
    """Return concrete cell-edge positions for index lookup.

    ``total_pad`` is ``pad_lo + pad_hi`` — the cells removed when
    slicing to the interior. ``pad_lo`` (defaulting to ``total_pad/2``
    for the legacy symmetric case) is needed by the tracer path
    to reconstruct ``n_interior`` and by the concrete path to pick
    the right interior slice.

    When ``d_arr`` is a JAX tracer (mesh-as-design-variable path), we
    fall back to a uniform ``fallback_dx`` reference mesh so that
    physical-coordinate lookup of source / probe / port positions still
    yields a concrete integer index. The traced cell sizes drive the
    FDTD physics downstream; only the structural index is resolved
    from the nominal mesh.
    """
    if pad_lo is None:
        pad_lo = total_pad // 2
    pad_hi = total_pad - pad_lo
    if is_tracer(d_arr):
        if fallback_dx is None or fallback_dx <= 0:
            raise ValueError(
                "tracer-valued cell-size profile requires a concrete "
                "fallback_dx for position->index resolution."
            )
        n_total = int(d_arr.shape[0])
        n_interior = n_total - total_pad
        interior = np.full(n_interior, float(fallback_dx), dtype=np.float64)
        return np.insert(np.cumsum(interior), 0, 0.0)
    return _interior_line_positions(np.asarray(d_arr), pad_lo, pad_hi)


def z_position_to_index(grid: NonUniformGrid, z_phys: float) -> int:
    """Convert physical z-coordinate to (cpml-offset) grid index."""
    edges = _nominal_edges_or_actual(
        grid.dz, grid.pad_z_lo + grid.pad_z_hi,
        pad_lo=grid.pad_z_lo, fallback_dx=float(grid.dx),
    )
    idx = int(np.argmin(np.abs(edges - float(z_phys))))
    return idx + grid.pad_z_lo


def _axis_position_to_index(
    d_arr: jnp.ndarray,
    pad_lo: int,
    pad_hi: int,
    pos: float,
    fallback_dx: float | None = None,
) -> int:
    """Generic non-uniform axis lookup.

    Uses cell-edge positions (same convention as z_position_to_index):
    position 0 is the first interior face, position ``sum(interior)`` is
    the last interior face.
    """
    edges = _nominal_edges_or_actual(
        d_arr, pad_lo + pad_hi, pad_lo=pad_lo, fallback_dx=fallback_dx,
    )
    idx = int(np.argmin(np.abs(edges - float(pos))))
    return idx + pad_lo


def position_to_index(grid: NonUniformGrid, pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to grid indices for NonUniformGrid.

    Accounts for per-face CPML padding (``pad_{axis}_lo`` leading offset).
    All three axes use cumulative cell-size lookup. In the uniform-xy
    case (``dx_arr`` constant) this reduces to the legacy
    ``round(pos[0]/dx) + pad_{axis}_lo`` behaviour within one cell.
    """
    i = _axis_position_to_index(
        grid.dx_arr, grid.pad_x_lo, grid.pad_x_hi, pos[0],
        fallback_dx=float(grid.dx),
    )
    j = _axis_position_to_index(
        grid.dy_arr, grid.pad_y_lo, grid.pad_y_hi, pos[1],
        fallback_dx=float(grid.dy),
    )
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

    # GEO-C3: on the differentiable-material path ``materials.eps_r`` /
    # ``materials.sigma`` are tracers — ``float()`` raised
    # TracerArrayConversionError. Stay in jnp when traced so the gradient
    # propagates into the waveform normalisation; keep the exact ``float()``
    # path otherwise so non-traced output stays bit-identical.
    materials_traced = (
        is_tracer(materials.eps_r) or is_tracer(materials.sigma)
    )
    if materials_traced:
        eps = jnp.asarray(materials.eps_r[i, j, k]) * EPS_0
        sigma = jnp.asarray(materials.sigma[i, j, k])
    else:
        eps = float(materials.eps_r[i, j, k]) * EPS_0
        sigma = float(materials.sigma[i, j, k])
    loss = sigma * grid.dt / (2.0 * eps)

    # Cb = dt / (eps * (1 + loss))
    cb = (grid.dt / eps) / (1.0 + loss)

    # Cell volume: dx_i * dy_j * dz_k (per-cell on each axis).
    # Stay in jnp so a tracer-valued cell-size profile (mesh-as-design
    # variable) propagates the gradient into the waveform normalisation.
    grid_traced = (
        is_tracer(grid.dx_arr) or is_tracer(grid.dy_arr) or is_tracer(grid.dz)
    )
    any_traced = materials_traced or grid_traced
    if grid_traced:
        dx_local = jnp.asarray(grid.dx_arr)[i]
        dy_local = jnp.asarray(grid.dy_arr)[j]
        dz_local = jnp.asarray(grid.dz)[k]
    else:
        dx_local = float(np.asarray(grid.dx_arr)[i])
        dy_local = float(np.asarray(grid.dy_arr)[j])
        dz_local = float(np.asarray(grid.dz)[k])
    dV = dx_local * dy_local * dz_local

    # Normalized waveform: Cb * I(t) / dV
    # This ensures power = ∫(J·E)dV is independent of cell size
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = (cb / dV) * jax.vmap(waveform_fn)(times)

    waveform_out = waveform if any_traced else np.array(waveform)
    return (i, j, k, component, waveform_out)


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
    e_old: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[FDTDState, object | None, object | None]:
    """E-field update with ADE dispersion on non-uniform grid.

    Uses per-axis inverse spacing arrays for curl(H), then applies the
    same ADE coefficient math as the uniform path. The ADE coefficients
    (ca, cb, cc, alpha, beta, etc.) are pre-baked spatial arrays that do
    not depend on dx, so they work unchanged on non-uniform grids.

    Mirrors the structure of ``_update_e_with_optional_dispersion`` in
    ``rfx/simulation.py`` but replaces the uniform ``curl / dx`` with
    non-uniform ``curl * inv_d[axis]``.

    Parameters
    ----------
    e_old : (ex_old, ey_old, ez_old) tuple of arrays, optional
        Explicit pre-step E snapshots to use as ``*_old`` in the ADE
        polarisation update.  When ``None`` (single-device default), the
        helper reads ``state.ex/ey/ez`` directly — these are the fields
        from the start of the step in the single-device runner because
        no E ghost exchange occurs before this call.

        On the distributed NU path (V3 Phase 2D), the caller MUST pass
        explicit snapshots taken BEFORE any ghost exchange so that the
        polarisation update uses pre-exchange E values.  See the ADE
        Ordering Contract documented in ``run_nonuniform_distributed``.
    """
    from rfx.materials.debye import DebyeState
    from rfx.materials.lorentz import LorentzState

    curl_x, curl_y, curl_z = _curl_h_nu(state, inv_dx, inv_dy, inv_dz)
    if e_old is not None:
        ex_old, ey_old, ez_old = e_old
    else:
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


class _NUScanSetup(NamedTuple):
    """Host-side bundle built by :func:`_build_nu_scan` (#383 code motion).

    Carries the scan step function, its initial carry, the stacked source
    table, and the post-scan assembly metadata shared by
    :func:`run_nonuniform` (single ``lax.scan``) and
    :func:`run_nonuniform_until_decay` (chunked host loop). This tuple
    never crosses a JAX transform boundary — ``step_fn`` is a Python
    closure and the ``use_*`` flags are Python bools.
    """
    step_fn: object
    carry_init: dict
    src_waveforms: jnp.ndarray
    dt: object
    sources: list
    probes: list
    wire_ports: list
    dft_planes: list
    flux_monitors: list
    waveguide_meta: tuple
    wp_meta: list
    sp_freqs: object
    use_wire_ports: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_lumped_rlc: bool
    use_ntff: bool
    use_waveguide_ports: bool


def _build_nu_scan(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    pec_mask=None,
    pec_occupancy=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    flux_monitors: list | None = None,
    emit_time_series: bool = True,
    design_mask: jnp.ndarray | None = None,
    aniso_eps: tuple | None = None,
) -> _NUScanSetup:
    """Build the NU scan carry + step function (pure code motion, #383).

    Extracted verbatim from :func:`run_nonuniform` so the until-decay
    entry point can drive the SAME ``step_fn`` / carry through a chunked
    host loop. ``n_steps`` is used only to size the zero-source
    placeholder table; the source waveform arrays themselves define the
    table length when sources are present.
    """
    sources = sources or []
    probes = probes or []
    wire_ports = wire_ports or []
    dft_planes = dft_planes or []
    waveguide_ports = waveguide_ports or []
    flux_monitors = flux_monitors or []
    dt = grid.dt
    use_wire_ports = len(wire_ports) > 0
    use_debye = debye is not None
    use_lorentz = lorentz is not None
    use_dft_planes = len(dft_planes) > 0
    use_flux_monitors = len(flux_monitors) > 0
    use_lumped_rlc = len(rlc_metas) > 0
    use_ntff = ntff_box is not None and ntff_data is not None
    use_waveguide_ports = len(waveguide_ports) > 0
    use_tfsf = tfsf is not None

    # CPML: only initialize when cpml_layers > 0 (skip for PEC boundary)
    use_cpml = grid.cpml_layers > 0

    cpml_params = None
    cpml_state_init = None
    cpml_grid = None
    # Effective CPML axes after PEC/PMC closure. Axes whose lo+hi pad is
    # zero are fully closed and the apply path's `state.e*[:, :, :n]`
    # slices clip to the (small) axis length, breaking the broadcast
    # against the (cpml_layers,) profile coefficients. Drop those axes
    # from `apply_cpml_*` so the no-op branch passes psi through
    # unchanged. Mirrors the uniform runner, which already threads
    # `cpml_axes` from the grid (rfx/runners/uniform.py).
    cpml_axes_eff = "xyz"

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e

        # Pass NonUniformGrid directly — init_cpml duck-types dx/dy/dz.
        # NonUniformGrid does not carry pmc_faces / pec_faces attrs
        # (frozen dataclass, pytree-registered), so the sets must be
        # threaded through from the caller.
        cpml_params, cpml_state_init = init_cpml(
            grid, pec_faces=pec_faces, pmc_faces=pmc_faces,
        )
        cpml_grid = grid
        cpml_axes_eff = "".join(
            ax for ax, lo, hi in (
                ("x", grid.pad_x_lo, grid.pad_x_hi),
                ("y", grid.pad_y_lo, grid.pad_y_hi),
                ("z", grid.pad_z_lo, grid.pad_z_hi),
            )
            if (lo + hi) > 0
        )

    # PMC enforcement (2026-04). The NU scan body previously never
    # zeroed H_tan on PMC faces, so a half-symmetric configuration
    # that relied on the mirror plane was running with an effectively
    # free boundary. Frozen set gives JIT cache a stable hash; empty
    # set short-circuits the apply to a no-op.
    use_pmc_faces = bool(pmc_faces)
    _pmc_faces_frozen = frozenset(pmc_faces) if pmc_faces else frozenset()

    use_pec_mask = pec_mask is not None
    use_pec_occupancy = pec_occupancy is not None

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
    # (defaults bound unconditionally so _NUScanSetup can carry them;
    # both stay unused unless the branch below runs — no behavior change)
    sp_freqs = None
    wp_meta: list = []
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
        # on non-uniform xy meshes. Carry `excite` and `direction` so
        # the post-processing can orient the (a, b) wave decomposition.
        _dx_arr_np = np.asarray(grid.dx_arr)
        _dy_arr_np = np.asarray(grid.dy_arr)
        wp_meta = [(
            wp['mid_i'], wp['mid_j'], wp['mid_k'],
            wp['component'], wp['impedance'],
            float(_dx_arr_np[wp['mid_i']]),
            float(_dy_arr_np[wp['mid_j']]),
            bool(wp.get('excite', True)),
            str(wp.get('direction', '-x')),
        ) for wp in wire_ports]
    else:
        use_wire_ports = False

    # DFT plane probe carry + static metadata
    if use_dft_planes:
        carry_init["dft_planes"] = tuple(probe.accumulator for probe in dft_planes)
        dft_meta = tuple(
            (probe.component, probe.axis, probe.index, probe.freqs)
            for probe in dft_planes
        )
    else:
        dft_meta = ()

    # Flux monitor carry + static metadata (mirrors the uniform scan body
    # in rfx/simulation.py). The NU dt is scalar, so the DFT kernels need
    # no per-axis time weighting; the axis-aware area element dA already
    # lives on each FluxMonitor (handles graded tangential cells).
    if use_flux_monitors:
        from rfx.probes.probes import _FLUX_COMPONENTS as _FC
        flux_meta = tuple(
            (fm.axis, fm.index, fm.freqs, _FC[fm.axis],
             fm.lo1, fm.hi1, fm.lo2, fm.hi2,
             fm.total_steps, fm.window, fm.window_alpha)
            for fm in flux_monitors
        )
        carry_init["flux_monitors"] = tuple(
            (fm.e1_dft, fm.e2_dft, fm.h1_dft, fm.h2_dft)
            for fm in flux_monitors
        )
    else:
        flux_meta = ()

    # Lumped RLC ADE state (one per element) — metas are Python-static
    if use_lumped_rlc:
        carry_init["rlc_states"] = tuple(rlc_states)

    # NTFF accumulators — seeded from caller, updated per step via
    # accumulate_ntff. Box indices and freqs are Python-static.
    if use_ntff:
        carry_init["ntff"] = ntff_data

    # Waveguide-port time-series carry (mirrors uniform path).
    # Phase 2 cleanup (2026-04-25) removed in-scan DFT accumulators;
    # spectra are computed POST-SCAN by a rect full-record DFT on the
    # recorded modal V/I time series.
    if use_waveguide_ports:
        carry_init["waveguide_port_accs"] = tuple(
            (
                cfg.v_probe_t,
                cfg.v_ref_t,
                cfg.i_probe_t,
                cfg.i_ref_t,
                cfg.v_inc_t,
                cfg.n_steps_recorded,
            )
            for cfg in waveguide_ports
        )
        waveguide_meta = tuple(waveguide_ports)
    else:
        waveguide_meta = ()

    # TFSF 1D auxiliary state carry. Injection axis is x (uniform on
    # NU paths we support — dz-only nonuniformity), so the 1D aux runs
    # with grid.dx spacing and the E/H corrections use scalar
    # coeff = dt / (EPS_0 * dx) etc. Oblique / +z,-z cases are
    # rejected upstream (see rfx/runners/nonuniform.py and rfx/api.py).
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d
        tfsf_cfg, tfsf_state = tfsf
        if _is_tfsf_2d(tfsf_cfg):
            raise ValueError(
                "TFSF oblique incidence (2D auxiliary grid) is not yet "
                "supported on nonuniform z mesh. Use angle_deg=0 along x."
            )
        if tfsf_cfg.direction not in ("+x", "-x"):
            raise ValueError(
                "TFSF on nonuniform mesh supports only direction='+x' or "
                f"'-x' (injection along uniform x axis); got {tfsf_cfg.direction!r}."
            )
        carry_init["tfsf"] = tfsf_state

    def step_fn(carry, xs):
        step_idx, src_vals = xs
        st = carry["fdtd"]

        # H update (non-uniform)
        st = update_h_nu(st, materials, dt, inv_dx_h, inv_dy_h, inv_dz_h)
        tfsf_h_state = None
        if use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_h
            st = apply_tfsf_h(st, tfsf_cfg, carry["tfsf"], grid.dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_h as _apply_wg_h_nu
            for cfg_meta in waveguide_meta:
                st = _apply_wg_h_nu(st, cfg_meta, step_idx, dt, grid.dx)
        if use_cpml:
            st, cpml_new = apply_cpml_h(st, cpml_params, carry["cpml"],
                                         cpml_grid, cpml_axes_eff,
                                         materials=materials)
        else:
            cpml_new = None
        if use_pmc_faces:
            from rfx.boundaries.pmc import apply_pmc_faces
            st = apply_pmc_faces(st, _pmc_faces_frozen)
        if use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_h
            tfsf_h_state = update_tfsf_1d_h(tfsf_cfg, carry["tfsf"], grid.dx, dt)

        # E update: use ADE-aware path when dispersive materials are present
        debye_new = None
        lorentz_new = None
        if use_debye or use_lorentz:
            st, debye_new, lorentz_new = _update_e_nu_dispersive(
                st, materials, dt, inv_dx, inv_dy, inv_dz,
                debye=(debye_coeffs, carry["debye"]) if use_debye else None,
                lorentz=(lorentz_coeffs, carry["lorentz"]) if use_lorentz else None,
            )
        elif aniso_eps is not None:
            from rfx.core.yee import update_e_nu_aniso
            _eex, _eey, _eez = aniso_eps
            st = update_e_nu_aniso(
                st, materials, _eex, _eey, _eez, dt,
                inv_dx, inv_dy, inv_dz,
            )
        else:
            st = update_e_nu(st, materials, dt, inv_dx, inv_dy, inv_dz)

        if use_tfsf:
            from rfx.sources.tfsf import apply_tfsf_e
            st = apply_tfsf_e(st, tfsf_cfg, tfsf_h_state, grid.dx, dt)
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import apply_waveguide_port_e as _apply_wg_e_nu
            for cfg_meta in waveguide_meta:
                st = _apply_wg_e_nu(st, cfg_meta, step_idx, dt, grid.dx)
        if use_cpml:
            st, cpml_new = apply_cpml_e(st, cpml_params, cpml_new,
                                         cpml_grid, cpml_axes_eff,
                                         materials=materials)

        # PEC
        st = apply_pec(st)
        if use_pec_mask:
            st = apply_pec_mask(st, pec_mask)
        if use_pec_occupancy:
            st = apply_pec_occupancy(st, pec_occupancy)

        # Lumped RLC ADE update (after E update + boundaries, before sources)
        new_rlc_states = None
        if use_lumped_rlc:
            from rfx.lumped import update_rlc_element
            new_rlc_states = []
            for rlc_st, meta in zip(carry["rlc_states"], rlc_metas):
                st, rlc_st_new = update_rlc_element(st, rlc_st, meta)
                new_rlc_states.append(rlc_st_new)

        # Sources (point sources + wire port excitation)
        for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
            field = getattr(st, sc)
            field = field.at[si, sj, sk].add(src_vals[idx_s])
            st = st._replace(**{sc: field})

        # Waveguide-port injection + DFT probe accumulation. The dx
        # arg is unused by the per-cell-weighted integrals (cfg already
        # stores u_widths/v_widths), but kept in the function signature
        # for back-compat.
        new_waveguide_port_accs = None
        if use_waveguide_ports:
            from rfx.sources.waveguide_port import (
                update_waveguide_port_probe,
            )
            new_waveguide_port_accs = []
            for accs, cfg_meta in zip(
                carry["waveguide_port_accs"], waveguide_meta
            ):
                cfg = cfg_meta._replace(
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                # TFSF-style corrections applied earlier at canonical slots.
                cfg_updated = update_waveguide_port_probe(cfg, st, dt, grid.dx)
                new_waveguide_port_accs.append((
                    cfg_updated.v_probe_t,
                    cfg_updated.v_ref_t,
                    cfg_updated.i_probe_t,
                    cfg_updated.i_ref_t,
                    cfg_updated.v_inc_t,
                    cfg_updated.n_steps_recorded,
                ))

        # Wire port V/I DFT accumulation
        t = step_idx.astype(jnp.float32) * dt
        new_wire_sp = None
        if use_wire_ports:
            new_wire_sp = []
            for (v_dft, i_dft, vinc_dft), (mi, mj, mk, comp, z0, dxi, dyj, _excite, _dir) in \
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

        # DFT plane probe accumulation (identical math to uniform path)
        new_dft_planes = None
        if use_dft_planes:
            t_plane = step_idx.astype(jnp.float32) * dt
            new_dft_planes = []
            for acc, (component, axis, index, freqs) in zip(
                carry["dft_planes"], dft_meta
            ):
                field = getattr(st, component)
                if axis == 0:
                    plane = field[index, :, :]
                elif axis == 1:
                    plane = field[:, index, :]
                else:
                    plane = field[:, :, index]
                phase = jnp.exp(-1j * 2.0 * jnp.pi * freqs * t_plane)
                new_dft_planes.append(
                    acc + plane[None, :, :] * phase[:, None, None] * dt
                )

        # Flux monitor accumulation (mirrors uniform scan body in
        # rfx/simulation.py). H is offset +dx/2 along the normal axis on
        # the Yee grid; average H at idx-1 and idx to co-locate with E at
        # idx for a correct Poynting cross-product. E is sampled at
        # t=step*dt, H at t-dt/2.
        new_flux_monitors = None
        if use_flux_monitors:
            from rfx.core.dft_utils import dft_window_weight as _dft_w
            t_flux = step_idx.astype(jnp.float32) * dt
            new_flux_monitors = []
            for (e1_acc, e2_acc, h1_acc, h2_acc), (
                ax, idx, fqs, comp_names, _lo1, _hi1, _lo2, _hi2,
                _tot_steps, _win_name, _win_alpha,
            ) in zip(carry["flux_monitors"], flux_meta):
                e1n, e2n, h1n, h2n = comp_names
                idx_m1 = max(idx - 1, 0)
                if ax == 0:
                    e1 = getattr(st, e1n)[idx, _lo1:_hi1, _lo2:_hi2]
                    e2 = getattr(st, e2n)[idx, _lo1:_hi1, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h1n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[idx_m1, _lo1:_hi1, _lo2:_hi2] + getattr(st, h2n)[idx, _lo1:_hi1, _lo2:_hi2]) * 0.5
                elif ax == 1:
                    e1 = getattr(st, e1n)[_lo1:_hi1, idx, _lo2:_hi2]
                    e2 = getattr(st, e2n)[_lo1:_hi1, idx, _lo2:_hi2]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h1n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, idx_m1, _lo2:_hi2] + getattr(st, h2n)[_lo1:_hi1, idx, _lo2:_hi2]) * 0.5
                else:
                    e1 = getattr(st, e1n)[_lo1:_hi1, _lo2:_hi2, idx]
                    e2 = getattr(st, e2n)[_lo1:_hi1, _lo2:_hi2, idx]
                    h1 = (getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h1n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                    h2 = (getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx_m1] + getattr(st, h2n)[_lo1:_hi1, _lo2:_hi2, idx]) * 0.5
                t_f64 = t_flux.astype(jnp.float64)
                fqs64 = fqs.astype(jnp.float64)
                _w = _dft_w(step_idx, _tot_steps, _win_name, _win_alpha).astype(jnp.float64)
                phase_e = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * t_f64)
                phase_h = jnp.exp(-1j * 2.0 * jnp.pi * fqs64 * (t_f64 - jnp.float64(dt * 0.5)))
                kernel_e = (phase_e[:, None, None] * dt * _w).astype(jnp.complex128)
                kernel_h = (phase_h[:, None, None] * dt * _w).astype(jnp.complex128)
                new_flux_monitors.append((
                    e1_acc + e1.astype(jnp.float64)[None, :, :] * kernel_e,
                    e2_acc + e2.astype(jnp.float64)[None, :, :] * kernel_e,
                    h1_acc + h1.astype(jnp.float64)[None, :, :] * kernel_h,
                    h2_acc + h2.astype(jnp.float64)[None, :, :] * kernel_h,
                ))

        # NTFF: accumulate tangential E/H DFT on 6 box faces
        new_ntff = None
        if use_ntff:
            from rfx.farfield import accumulate_ntff
            new_ntff = accumulate_ntff(carry["ntff"], st, ntff_box, dt, step_idx)

        # TFSF 1D auxiliary E-field update (mirrors uniform scan body:
        # called AFTER sources, closes the leapfrog step).
        tfsf_new = None
        if use_tfsf:
            from rfx.sources.tfsf import update_tfsf_1d_e
            t_tfsf = step_idx.astype(jnp.float32) * dt
            tfsf_new = update_tfsf_1d_e(tfsf_cfg, tfsf_h_state, grid.dx, dt, t_tfsf)

        # Probes
        if emit_time_series and prb_meta:
            samples = [getattr(st, pc)[pi, pj, pk] for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples)
        else:
            probe_out = jnp.zeros(0)

        # Issue #41: stop_gradient on fields outside the design region so the
        # backward tape does not accumulate entries for cells whose eps does
        # not depend on the optimization variable. Forward physics is
        # unchanged (stop_gradient is identity forward); backward memory +
        # wall-time scale with mask occupancy instead of grid volume.
        if design_mask is not None:
            sg = jax.lax.stop_gradient
            st = st._replace(
                ex=jnp.where(design_mask, st.ex, sg(st.ex)),
                ey=jnp.where(design_mask, st.ey, sg(st.ey)),
                ez=jnp.where(design_mask, st.ez, sg(st.ez)),
                hx=jnp.where(design_mask, st.hx, sg(st.hx)),
                hy=jnp.where(design_mask, st.hy, sg(st.hy)),
                hz=jnp.where(design_mask, st.hz, sg(st.hz)),
            )

        new_carry = {"fdtd": st}
        if use_cpml:
            new_carry["cpml"] = cpml_new
        if use_debye and debye_new is not None:
            new_carry["debye"] = debye_new
        if use_lorentz and lorentz_new is not None:
            new_carry["lorentz"] = lorentz_new
        if use_wire_ports and new_wire_sp is not None:
            new_carry["wire_sparams"] = tuple(new_wire_sp)
        if use_dft_planes and new_dft_planes is not None:
            new_carry["dft_planes"] = tuple(new_dft_planes)
        if use_flux_monitors and new_flux_monitors is not None:
            new_carry["flux_monitors"] = tuple(new_flux_monitors)
        if use_lumped_rlc and new_rlc_states is not None:
            new_carry["rlc_states"] = tuple(new_rlc_states)
        if use_ntff and new_ntff is not None:
            new_carry["ntff"] = new_ntff
        if use_waveguide_ports and new_waveguide_port_accs is not None:
            new_carry["waveguide_port_accs"] = tuple(new_waveguide_port_accs)
        if use_tfsf and tfsf_new is not None:
            new_carry["tfsf"] = tfsf_new
        return new_carry, probe_out

    return _NUScanSetup(
        step_fn=step_fn,
        carry_init=carry_init,
        src_waveforms=src_waveforms,
        dt=dt,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_meta=waveguide_meta,
        wp_meta=wp_meta,
        sp_freqs=sp_freqs,
        use_wire_ports=use_wire_ports,
        use_dft_planes=use_dft_planes,
        use_flux_monitors=use_flux_monitors,
        use_lumped_rlc=use_lumped_rlc,
        use_ntff=use_ntff,
        use_waveguide_ports=use_waveguide_ports,
    )


def run_nonuniform(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    pec_mask=None,
    pec_occupancy=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    flux_monitors: list | None = None,
    checkpoint: bool = False,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    design_mask: jnp.ndarray | None = None,
    aniso_eps: tuple | None = None,
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
    dft_planes : list of DFTPlaneProbe or None
        Frequency-domain plane accumulators. The accumulation is
        identical to the uniform path (acc += field * exp(-j2pi f t) * dt);
        dt is scalar on both paths so no per-axis weighting is required.
    """
    setup = _build_nu_scan(
        grid, materials, n_steps,
        pec_mask=pec_mask,
        pec_occupancy=pec_occupancy,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        design_mask=design_mask,
        aniso_eps=aniso_eps,
    )
    step_fn = setup.step_fn
    carry_init = setup.carry_init
    src_waveforms = setup.src_waveforms

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)

    # ---- n_warmup split --------------------------------------------------
    # When n_warmup > 0, run the first n_warmup steps with the carry
    # stop_gradient'd so AD builds no tape for that transient lead-in
    # (issue #40). Only the trailing n_optimize = n_steps - n_warmup
    # steps participate in reverse-mode autodiff.
    if n_warmup > 0:
        if n_warmup >= n_steps:
            raise ValueError(
                f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
            )
        warmup_steps = jnp.arange(n_warmup, dtype=jnp.int32)
        warmup_xs = (warmup_steps, src_waveforms[:n_warmup])
        warmup_final, warmup_ys = jax.lax.scan(step_fn, carry_init, warmup_xs)
        carry_init = jax.tree_util.tree_map(
            jax.lax.stop_gradient, warmup_final
        )
        warmup_ys = jax.lax.stop_gradient(warmup_ys)
        xs = (
            jnp.arange(n_warmup, n_steps, dtype=jnp.int32),
            src_waveforms[n_warmup:],
        )
        n_steps_opt = n_steps - n_warmup
    else:
        warmup_ys = None
        n_steps_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_steps_opt
    )
    if use_segmented:
        # Scan-of-scan: outer scan over segments wrapped in jax.checkpoint
        # forces XLA to remat the inner scan during backward, so the AD
        # tape only stores carry at segment boundaries (≈ sqrt(n_steps)
        # × carry_size when checkpoint_every ≈ sqrt(n_steps)).
        chunk = int(checkpoint_every)
        n_segments = (n_steps_opt + chunk - 1) // chunk
        pad = n_segments * chunk - n_steps_opt
        opt_steps = xs[0]
        opt_src = xs[1]
        if pad > 0:
            steps_padded = jnp.arange(
                int(opt_steps[0]),
                int(opt_steps[0]) + n_segments * chunk,
                dtype=jnp.int32,
            )
            n_sources = opt_src.shape[1]
            src_pad = jnp.zeros((pad, n_sources), dtype=opt_src.dtype)
            src_padded = jnp.concatenate([opt_src, src_pad], axis=0)
        else:
            steps_padded = opt_steps
            src_padded = opt_src

        seg_steps = steps_padded.reshape(n_segments, chunk)
        seg_src = src_padded.reshape(n_segments, chunk, src_padded.shape[1])

        def segment_body(carry, segment_xs):
            return jax.lax.scan(step_fn, carry, segment_xs)

        seg_body = jax.checkpoint(segment_body)
        final, segment_ys = jax.lax.scan(
            seg_body, carry_init, (seg_steps, seg_src)
        )
        flat = segment_ys.reshape((n_segments * chunk,) + segment_ys.shape[2:])
        opt_ys = flat[:n_steps_opt]
    else:
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        final, opt_ys = jax.lax.scan(body, carry_init, xs)
    # Merge warmup + optimize outputs back into one time_series so the
    # downstream result shape stays (n_steps, n_probes).
    if warmup_ys is not None:
        time_series = jnp.concatenate([warmup_ys, opt_ys], axis=0)
    else:
        time_series = opt_ys

    return _assemble_nu_result(setup, final, time_series)


def _assemble_nu_result(setup: _NUScanSetup, final: dict, time_series) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform,
    #383). Shared by :func:`run_nonuniform` and
    :func:`run_nonuniform_until_decay` so the two paths return the exact
    same schema (state, time_series, dt, conditional dft_planes /
    flux_monitors / ntff_data / waveguide_ports / s_params...)."""
    dt = setup.dt
    dft_planes = setup.dft_planes
    flux_monitors = setup.flux_monitors
    wire_ports = setup.wire_ports
    waveguide_meta = setup.waveguide_meta
    wp_meta = setup.wp_meta
    sp_freqs = setup.sp_freqs
    use_wire_ports = setup.use_wire_ports
    use_dft_planes = setup.use_dft_planes
    use_flux_monitors = setup.use_flux_monitors
    use_lumped_rlc = setup.use_lumped_rlc
    use_ntff = setup.use_ntff
    use_waveguide_ports = setup.use_waveguide_ports

    result = {
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Surface final RLC ADE states (per element).
    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    # Repack DFT plane probes with their final accumulators.
    if use_dft_planes:
        result["dft_planes"] = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    # Repack flux monitors with their final E/H DFT accumulators so the
    # caller can call flux_spectrum() on them (same schema as uniform).
    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            mon._replace(e1_dft=e1, e2_dft=e2, h1_dft=h1, h2_dft=h2)
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    # Surface final NTFF DFT accumulators
    if use_ntff:
        result["ntff_data"] = final["ntff"]

    # Surface final waveguide-port configs (with recorded modal V/I
    # time series; spectra are extracted post-scan via rect-DFT).
    if use_waveguide_ports:
        result["waveguide_ports"] = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(
                waveguide_meta, final["waveguide_port_accs"]
            )
        )

    # ---- Extract full S-matrix column from wire port DFTs ----
    #
    # Each port has a V/I DFT pair, plus a direction that tells us the
    # outward-normal (+x/-x/+y/-y). The wave decomposition is:
    #
    #   direction "-x" (port at left end, outward = -x, inward = +x):
    #       a = (V + Z·I) / 2   (incoming  = +x-moving wave)
    #       b = (V - Z·I) / 2   (outgoing  = -x-moving wave)
    #
    #   direction "+x" (port at right end, outward = +x, inward = -x):
    #       a = (V - Z·I) / 2   (incoming  = -x-moving wave)
    #       b = (V + Z·I) / 2   (outgoing  = +x-moving wave)
    #
    #   direction "-y" / "+y": same idea rotated, with I in the
    #   perpendicular loop direction (already encoded by the runner's
    #   V/I formula via the curl-H integral around the cell).
    #
    # Given these, with ONE excited port `k` the full k-th column of
    # the S-matrix is
    #       S[j, k] = b_j / a_k          (j = every port)
    # which reduces to the familiar S11 = b_k/a_k for j==k (reflection)
    # and to S21 = b_2/a_1 for j != k (transmission). Other columns of
    # the S matrix stay zero — callers need to run additional sims
    # with different excited ports to fill them (reciprocity lets us
    # infer S12 from S21 for passive networks).
    if use_wire_ports and "wire_sparams" in final:
        import numpy as _np
        n_wp = len(wire_ports)
        nf = len(sp_freqs)
        # Tracer-safe accumulation (issue #70): build S with jnp + .at[].set
        # so jax.grad on an objective that pulls through the wire_sparams
        # extractor does not hit TracerArrayConversionError. Concrete-path
        # callers still get a jnp.ndarray that numpy consumers accept via
        # np.asarray(...) (jnp arrays implement __array__).
        S = jnp.zeros((n_wp, n_wp, nf), dtype=jnp.complex64)

        # Pick the first excited port as the "k" column. If no port is
        # excited (all passive), fall back to the legacy diagonal-only
        # extraction (no meaningful S-matrix in that case).
        excited_idx = [idx for idx, meta in enumerate(wp_meta) if meta[7]]
        if not excited_idx:
            excited_idx = list(range(n_wp))   # legacy: treat all as self-excited

        def _ab(v_dft, i_dft, z0, direction):
            """Return (a_incoming, b_outgoing) at one port."""
            v = v_dft
            zi = z0 * i_dft
            if direction in ("-x", "-y"):
                # inward is +x/+y, +x/+y wave is incoming → +sign on I
                return (v + zi) / 2.0, (v - zi) / 2.0
            elif direction in ("+x", "+y"):
                return (v - zi) / 2.0, (v + zi) / 2.0
            else:
                raise ValueError(f"unknown port direction {direction!r}")

        ab_per_port = []
        for (v_dft, i_dft, _), meta in zip(final["wire_sparams"], wp_meta):
            z0 = meta[4]
            direction = meta[8]
            a, b = _ab(v_dft, i_dft, z0, direction)
            ab_per_port.append((a, b))

        for k in excited_idx:
            a_k = ab_per_port[k][0]
            safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
            for j in range(n_wp):
                b_j = ab_per_port[j][1]
                S = S.at[j, k, :].set(b_j / safe_a_k)

        result["s_params"] = S
        result["s_param_freqs"] = _np.array(sp_freqs)

    return result

