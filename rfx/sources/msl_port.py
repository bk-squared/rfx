"""Microstrip line (MSL) port: 2D distributed port with 3-probe de-embedding.

Unlike the 1-cell-transverse ``WirePort``, the MSL port covers the full
trace cross-section (y × z under the trace) and distributes the total
port impedance Z0 as conductivity over the cross-section cells. After
the FDTD run, three downstream probe planes are used in an OpenEMS-style
3-probe recurrence to extract the propagation constant β, characteristic
impedance Z0, and the reflection coefficient at the reference plane.

The math is intentionally numpy-only: extraction runs once per port,
post-simulation, on small per-frequency arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


# Speed of light (used for static-Z0 closed form)
_C0 = 1.0 / np.sqrt(MU_0 * EPS_0)
_ETA0 = np.sqrt(MU_0 / EPS_0)


# ---------------------------------------------------------------------------
# Port description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSLPort:
    """Microstrip line port spanning the full trace cross-section.

    Parameters
    ----------
    feed_x : float
        Feed-plane x-coordinate (metres) where the source and termination
        are placed.
    y_lo, y_hi : float
        Trace lateral extent (metres). ``y_hi - y_lo`` is the trace width.
    z_lo, z_hi : float
        Substrate vertical extent (metres). ``z_lo`` is typically the
        ground plane / substrate bottom; ``z_hi`` the top of the
        substrate (where the trace lies).
    direction : "+x" or "-x"
        Direction the launched wave propagates away from the feed plane.
    impedance : float
        Target characteristic impedance Z0 in ohms (used to set σ).
    excitation : callable or None
        Source waveform ``f(t) -> amplitude`` (e.g. ``GaussianPulse``).
        ``None`` for a passive matched port.
    """

    feed_x: float
    y_lo: float
    y_hi: float
    z_lo: float
    z_hi: float
    direction: str
    impedance: float
    excitation: object = None


# ---------------------------------------------------------------------------
# Cross-section helpers
# ---------------------------------------------------------------------------


def _axis_cell_size(grid, axis: str, idx: int) -> float:
    """Return the cell size at index ``idx`` along ``axis``.

    Supports both uniform Grid and NonUniformGrid via duck typing.
    """
    profile_attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
    profile = getattr(grid, profile_attr, None)
    if profile is not None:
        try:
            n = int(profile.shape[0])
        except Exception:
            return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))
        clamped = max(0, min(idx, n - 1))
        return float(profile[clamped])
    # Fallback for axis-specific scalar (Grid doesn't carry .dy/.dz today)
    return float(getattr(grid, axis if axis != "x" else "dx", grid.dx))


def _msl_yz_cells(grid, port: MSLPort) -> list[tuple[int, int, int]]:
    """Return the (i, j, k) grid indices spanning the MSL cross-section.

    ``i`` is the feed-plane x index. ``j`` ranges over the y-cells from
    ``port.y_lo`` to ``port.y_hi`` (inclusive); ``k`` likewise over z.
    """
    i_feed, j_lo, k_lo = grid.position_to_index((port.feed_x, port.y_lo, port.z_lo))
    _, j_hi, k_hi = grid.position_to_index((port.feed_x, port.y_hi, port.z_hi))
    j_a, j_b = (j_lo, j_hi) if j_lo <= j_hi else (j_hi, j_lo)
    k_a, k_b = (k_lo, k_hi) if k_lo <= k_hi else (k_hi, k_lo)
    cells = []
    for j in range(j_a, j_b + 1):
        for k in range(k_a, k_b + 1):
            cells.append((int(i_feed), int(j), int(k)))
    return cells


# ---------------------------------------------------------------------------
# 2D quasi-TEM mode solver (electrostatic Laplace)
# ---------------------------------------------------------------------------


def _solve_laplace_2d(
    eps_yz: np.ndarray,
    trace_mask: np.ndarray,
    ground_mask: np.ndarray,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Solve ∇·(ε ∇φ) = 0 with Dirichlet trace/ground via 5-point FV.

    Parameters
    ----------
    eps_yz : (n_y, n_z) array of cell-centred relative permittivity.
    trace_mask : (n_y, n_z) bool, True where φ = 1.
    ground_mask : (n_y, n_z) bool, True where φ = 0.
    dy, dz : float, cell sizes.

    Returns
    -------
    phi : (n_y, n_z) electrostatic potential.

    Notes
    -----
    Boundary at far y / top z is implicit Neumann (no flux) by truncating
    coefficients at the array edge. Caller must extend the box well past
    fringing fields (≥ 5W lateral, ≥ 4H above) for that to be accurate.
    """
    try:
        from scipy.sparse import lil_matrix, csr_matrix
        from scipy.sparse.linalg import spsolve
        _have_sparse = True
    except Exception:
        _have_sparse = False

    n_y, n_z = eps_yz.shape
    fixed_mask = trace_mask | ground_mask
    fixed_val = np.where(trace_mask, 1.0, 0.0)

    def _idx(j, k):
        return j * n_z + k

    n_unk = n_y * n_z

    if _have_sparse:
        A = lil_matrix((n_unk, n_unk), dtype=np.float64)
        b = np.zeros(n_unk, dtype=np.float64)
        for j in range(n_y):
            for k in range(n_z):
                p = _idx(j, k)
                if fixed_mask[j, k]:
                    A[p, p] = 1.0
                    b[p] = fixed_val[j, k]
                    continue
                # 5-point ε-weighted Laplacian. Off-diagonal coeffs use
                # harmonic-mean ε at the face (continuous flux).
                diag = 0.0
                # +y neighbour
                if j + 1 < n_y:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j + 1, k] / (
                        eps_yz[j, k] + eps_yz[j + 1, k] + 1e-30
                    )
                    coef = eps_face / (dy * dy)
                    A[p, _idx(j + 1, k)] = coef
                    diag -= coef
                # -y neighbour
                if j - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j - 1, k] / (
                        eps_yz[j, k] + eps_yz[j - 1, k] + 1e-30
                    )
                    coef = eps_face / (dy * dy)
                    A[p, _idx(j - 1, k)] = coef
                    diag -= coef
                # +z neighbour
                if k + 1 < n_z:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k + 1] / (
                        eps_yz[j, k] + eps_yz[j, k + 1] + 1e-30
                    )
                    coef = eps_face / (dz * dz)
                    A[p, _idx(j, k + 1)] = coef
                    diag -= coef
                # -z neighbour
                if k - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k - 1] / (
                        eps_yz[j, k] + eps_yz[j, k - 1] + 1e-30
                    )
                    coef = eps_face / (dz * dz)
                    A[p, _idx(j, k - 1)] = coef
                    diag -= coef
                A[p, p] = diag
                b[p] = 0.0
        phi = spsolve(csr_matrix(A), b)
        return phi.reshape(n_y, n_z)

    # Fallback: Jacobi/Gauss-Seidel iteration.
    phi = np.zeros((n_y, n_z), dtype=np.float64)
    phi[trace_mask] = 1.0
    for _ in range(20000):
        phi_new = phi.copy()
        max_d = 0.0
        for j in range(n_y):
            for k in range(n_z):
                if fixed_mask[j, k]:
                    continue
                num = 0.0
                den = 0.0
                if j + 1 < n_y:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j + 1, k] / (
                        eps_yz[j, k] + eps_yz[j + 1, k] + 1e-30
                    )
                    c = eps_face / (dy * dy)
                    num += c * phi_new[j + 1, k]
                    den += c
                if j - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j - 1, k] / (
                        eps_yz[j, k] + eps_yz[j - 1, k] + 1e-30
                    )
                    c = eps_face / (dy * dy)
                    num += c * phi_new[j - 1, k]
                    den += c
                if k + 1 < n_z:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k + 1] / (
                        eps_yz[j, k] + eps_yz[j, k + 1] + 1e-30
                    )
                    c = eps_face / (dz * dz)
                    num += c * phi_new[j, k + 1]
                    den += c
                if k - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k - 1] / (
                        eps_yz[j, k] + eps_yz[j, k - 1] + 1e-30
                    )
                    c = eps_face / (dz * dz)
                    num += c * phi_new[j, k - 1]
                    den += c
                new_val = num / (den + 1e-30)
                max_d = max(max_d, abs(new_val - phi_new[j, k]))
                phi_new[j, k] = new_val
        phi = phi_new
        if max_d < 1e-9:
            break
    return phi


def compute_msl_mode_profile(
    grid,
    port: MSLPort,
    eps_r_sub: float,
    *,
    pad_y_cells: int | None = None,
    pad_z_cells: int | None = None,
    refine: int = 4,
    fringing_extent_cells: int | None = None,
) -> dict:
    """Solve 2D Laplace for the quasi-TEM microstrip mode at the port plane.

    The output Ez profile is registered ON the FDTD grid cross-section
    cells (extended laterally and upward to capture fringing) so it can
    be applied directly inside ``setup_msl_port`` and
    ``make_msl_port_sources``.

    Parameters
    ----------
    grid : Grid (uniform)
    port : MSLPort
    eps_r_sub : float
        Substrate relative permittivity. The Laplace box uses εr=eps_r_sub
        for cells whose centre falls inside the substrate slab
        ``[port.z_lo, port.z_hi)`` and εr=1 above.
    pad_y_cells, pad_z_cells : int, optional
        Lateral / vertical padding (cells) added beyond the trace
        footprint so the Neumann far-field BC is accurate. Default
        chosen so total box ≥ 5·W laterally and ≥ 4·H above substrate.

    Returns
    -------
    dict
        ``ez_profile``  : (n_y_box, n_z_sub) float — Ez weights at each
                           substrate cell, normalised so that integrating
                           ``Ez·dz`` along z at the trace centre yields 1 V.
        ``cell_indices`` : list of (i_feed, j_grid, k_grid) tuples — the
                           FDTD-grid cells this profile applies to. j_grid
                           covers the extended (fringing) lateral range.
        ``z0_static``    : float — closed-form static Z0 (Ω).
        ``eps_eff``      : float — effective permittivity from twin solve.
        ``j_grid_lo``    : int — leftmost FDTD-grid j of the box.
        ``k_grid_lo``    : int — substrate-bottom FDTD-grid k of the box.
        ``trace_j_lo``   : int — local j-index where trace conductor sits.
        ``trace_j_hi``   : int — inclusive.
    """
    # Cross-section indices on the FDTD grid (substrate only).
    cells = _msl_yz_cells(grid, port)
    j_set = sorted({c[1] for c in cells})
    k_set = sorted({c[2] for c in cells})
    j_trace_lo = j_set[0]
    j_trace_hi = j_set[-1]
    k_sub_lo = k_set[0]
    k_sub_hi = k_set[-1]
    i_feed = cells[0][0]

    n_y_trace = len(j_set)
    n_z_sub = len(k_set)
    dy = float(_axis_cell_size(grid, "y", j_trace_lo))
    dz = float(_axis_cell_size(grid, "z", k_sub_lo))

    H_sub = float(port.z_hi - port.z_lo)
    W_trace = float(port.y_hi - port.y_lo)

    # Box padding: clamp into the available grid footprint so we don't
    # overshoot into CPML regions.
    if pad_y_cells is None:
        # FDTD-grid lateral source extent. Default ≈ 1·H_sub on each
        # side of the trace — enough to inject the dominant fringing
        # tail (where Ez(y) drops to ~10% of trace-centre value) while
        # avoiding the deep fringing region whose source contribution
        # is parasitic. Capped at a minimum of 1 cell.
        target = max(1, int(round(1.0 * H_sub / dy)))
        pad_y_cells = target
    if pad_z_cells is None:
        # ≥ 4·H above substrate top (used inside the Laplace box for
        # static-Z0 fidelity; does not enlarge the FDTD source set).
        target = max(4, int(round(4.0 * H_sub / dz)))
        pad_z_cells = target

    # FDTD-grid cells that will receive an Ez source (clamped into the
    # available grid). Capture the maximum y-padding the grid allows.
    j_grid_lo = max(0, j_trace_lo - pad_y_cells)
    j_grid_hi = min(grid.shape[1] - 1, j_trace_hi + pad_y_cells)
    k_grid_lo = k_sub_lo  # ground at bottom of substrate
    k_grid_hi = min(grid.shape[2] - 1, k_sub_hi + pad_z_cells)
    n_y_grid = j_grid_hi - j_grid_lo + 1
    n_z_grid = k_grid_hi - k_grid_lo + 1

    # Laplace solve box can extend BEYOND the FDTD grid (free-space
    # Neumann decoupled from CPML/PEC) so static Z0 captures fringing
    # accurately. Use a large box: 5·W lateral and 4·H above substrate
    # WITHOUT clamping into the grid. The Ez profile will be sliced back
    # onto the FDTD-grid window after solving.
    #
    # Internal refinement (``refine``×): we solve the Laplace problem on
    # a sub-mesh ``refine`` times finer in y and z so static Z0 / Ez
    # profile shape converge. Profile is averaged back to FDTD cell
    # resolution before injection.
    refine = max(1, int(refine))
    dy_fine = dy / refine
    dz_fine = dz / refine
    n_y_trace_fine = n_y_trace * refine
    n_z_sub_fine = n_z_sub * refine
    laplace_pad_y_fine = max(pad_y_cells * refine,
                             int(round(5.0 * W_trace / dy_fine)))
    laplace_pad_z_fine = max(pad_z_cells * refine,
                             int(round(4.0 * H_sub / dz_fine)))
    n_y_box_fine = n_y_trace_fine + 2 * laplace_pad_y_fine
    n_z_box_fine = n_z_sub_fine + laplace_pad_z_fine
    # Local trace position inside the refined box.
    j_trace_local_lo_box_fine = laplace_pad_y_fine
    j_trace_local_hi_box_fine = laplace_pad_y_fine + (n_y_trace_fine - 1)

    # Substrate occupies the lowest n_z_sub_fine rows of the box.
    eps_yz = np.ones((n_y_box_fine, n_z_box_fine), dtype=np.float64)
    eps_yz[:, : n_z_sub_fine] = float(eps_r_sub)

    # Trace conductor: thin strip at the FIRST air row above substrate.
    trace_mask = np.zeros((n_y_box_fine, n_z_box_fine), dtype=bool)
    if n_z_sub_fine < n_z_box_fine:
        trace_mask[
            j_trace_local_lo_box_fine : j_trace_local_hi_box_fine + 1,
            n_z_sub_fine,
        ] = True
    else:
        trace_mask[
            j_trace_local_lo_box_fine : j_trace_local_hi_box_fine + 1,
            n_z_sub_fine - 1,
        ] = True

    # Ground plane: row k_local = 0 (bottom of substrate sits on PEC).
    ground_mask = np.zeros((n_y_box_fine, n_z_box_fine), dtype=bool)
    ground_mask[:, 0] = True

    # --- Substrate-loaded solve ---
    phi_sub = _solve_laplace_2d(eps_yz, trace_mask, ground_mask, dy_fine, dz_fine)
    # --- Air-only solve (same geometry, εr=1 everywhere) ---
    phi_air = _solve_laplace_2d(
        np.ones_like(eps_yz), trace_mask, ground_mask, dy_fine, dz_fine
    )

    # Capacitance per metre via energy integral W = (1/2) C V², V = 1
    # so C = ε₀ ∫ εr |∇φ|² dA. Compute with central differences.
    def _cap_per_metre(phi: np.ndarray, eps: np.ndarray, ddy: float, ddz: float) -> float:
        gy = np.zeros_like(phi)
        gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * ddy)
        gy[0, :] = (phi[1, :] - phi[0, :]) / ddy
        gy[-1, :] = (phi[-1, :] - phi[-2, :]) / ddy
        gz = np.zeros_like(phi)
        gz[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * ddz)
        gz[:, 0] = (phi[:, 1] - phi[:, 0]) / ddz
        gz[:, -1] = (phi[:, -1] - phi[:, -2]) / ddz
        return float(EPS_0 * np.sum(eps * (gy * gy + gz * gz)) * ddy * ddz)

    C_sub = _cap_per_metre(phi_sub, eps_yz, dy_fine, dz_fine)
    C_air = _cap_per_metre(phi_air, np.ones_like(eps_yz), dy_fine, dz_fine)
    eps_eff = C_sub / C_air if C_air > 0 else 1.0
    z0_static = 1.0 / (_C0 * np.sqrt(C_sub * C_air)) if (C_sub > 0 and C_air > 0) else float(port.impedance)

    # Ez = -∂φ/∂z on the fine substrate cells (k_loc 0..n_z_sub_fine-1).
    ez_fine = np.zeros((n_y_box_fine, n_z_sub_fine), dtype=np.float64)
    for k_loc in range(n_z_sub_fine):
        if k_loc + 1 < phi_sub.shape[1]:
            ez_fine[:, k_loc] = -(phi_sub[:, k_loc + 1] - phi_sub[:, k_loc]) / dz_fine
        else:
            ez_fine[:, k_loc] = -(phi_sub[:, k_loc] - phi_sub[:, k_loc - 1]) / dz_fine

    # Average Ez fine cells back to FDTD coarse cells (refine × refine
    # block average per FDTD cell).
    n_y_box_coarse = n_y_trace + 2 * (laplace_pad_y_fine // refine)
    ez_coarse_full = np.zeros((n_y_box_coarse, n_z_sub), dtype=np.float64)
    laplace_pad_y_coarse = laplace_pad_y_fine // refine
    for j_c in range(n_y_box_coarse):
        j_f0 = j_c * refine
        j_f1 = (j_c + 1) * refine
        if j_f1 > n_y_box_fine:
            continue
        for k_c in range(n_z_sub):
            k_f0 = k_c * refine
            k_f1 = (k_c + 1) * refine
            ez_coarse_full[j_c, k_c] = float(np.mean(ez_fine[j_f0:j_f1, k_f0:k_f1]))

    # Normalise so ∫Ez·dz at the trace centre = 1 V (in COARSE cells,
    # which is what the FDTD source uses).
    j_trace_coarse_lo = laplace_pad_y_coarse
    j_trace_coarse_hi = laplace_pad_y_coarse + (n_y_trace - 1)
    j_centre_coarse = (j_trace_coarse_lo + j_trace_coarse_hi) // 2
    v_centre = float(np.sum(ez_coarse_full[j_centre_coarse, :]) * dz)
    if abs(v_centre) > 1e-30:
        ez_coarse_full = ez_coarse_full / v_centre

    # Slice onto the FDTD-grid lateral window.
    box_offset = laplace_pad_y_coarse - (j_trace_lo - j_grid_lo)
    j_box_start = box_offset
    j_box_end = box_offset + n_y_grid
    j_box_start_clip = max(0, j_box_start)
    j_box_end_clip = min(n_y_box_coarse, j_box_end)
    ez_box_grid = np.zeros((n_y_grid, n_z_sub), dtype=np.float64)
    if j_box_end_clip > j_box_start_clip:
        ez_box_grid[
            (j_box_start_clip - j_box_start) : (j_box_end_clip - j_box_start), :
        ] = ez_coarse_full[j_box_start_clip:j_box_end_clip, :]

    cell_indices = []
    for j_loc in range(n_y_grid):
        j_grid = j_grid_lo + j_loc
        for k_loc in range(n_z_sub):
            k_grid = k_grid_lo + k_loc
            cell_indices.append((int(i_feed), int(j_grid), int(k_grid)))

    return dict(
        ez_profile=ez_box_grid,
        cell_indices=cell_indices,
        z0_static=float(z0_static),
        eps_eff=float(eps_eff),
        j_grid_lo=int(j_grid_lo),
        k_grid_lo=int(k_grid_lo),
        trace_j_lo=int(j_trace_lo),
        trace_j_hi=int(j_trace_hi),
        n_z_sub=int(n_z_sub),
        dy=float(dy),
        dz=float(dz),
    )


# ---------------------------------------------------------------------------
# Material setup: distribute Z0 as σ over the cross-section
# ---------------------------------------------------------------------------


def setup_msl_port(grid, port: MSLPort, materials, *, mode_profile: dict | None = None):
    """Fold port impedance Z0 into σ over the MSL cross-section cells.

    Two modes:

    - **Uniform** (``mode_profile is None``, legacy): cells stacked in z
      are series, cells in y are parallel; for total impedance Z0::

          σ_cell = (N_z · dz_cell) / (Z0 · N_y · dx_cell · dy_cell)

    - **Eigenmode** (``mode_profile`` from
      :func:`compute_msl_mode_profile`): σ ∝ |Ez(y,z)|² with the
      proportionality chosen so the total (y,z)-integrated admittance
      ``Y = ∫∫ σ·dy·dz / dx_feed = 1/Z0``.

    Returns the updated ``materials`` NamedTuple.
    """
    if mode_profile is None:
        cells = _msl_yz_cells(grid, port)
        if not cells:
            return materials
        j_set = sorted({c[1] for c in cells})
        k_set = sorted({c[2] for c in cells})
        n_y = len(j_set)
        n_z = len(k_set)

        sigma = materials.sigma
        for (i, j, k) in cells:
            dx_cell = _axis_cell_size(grid, "x", i)
            dy_cell = _axis_cell_size(grid, "y", j)
            dz_cell = _axis_cell_size(grid, "z", k)
            sigma_cell = (n_z * dz_cell) / (port.impedance * n_y * dx_cell * dy_cell)
            sigma = sigma.at[i, j, k].add(sigma_cell)
        return materials._replace(sigma=sigma)

    # Eigenmode termination: uniform σ across the (extended) port
    # cross-section, magnitude chosen so that the time-averaged power
    # dissipated equals V²/Z0 when V is the TEM voltage.
    #
    #     P_diss  = σ · dx_feed · ∫∫ |Ez(y,z)|² dy dz
    #     V²/Z0   = matched-load power
    #     ⇒ σ    = 1 / (Z0 · dx_feed · ∫∫ |ez_w|² dy dz)
    #
    # ez_w is the normalised mode shape (∫ez_w·dz = 1V at trace centre),
    # so V_TEM = V_src and the integral is taken over the full fringing
    # footprint that compute_msl_mode_profile returned.
    ez_profile = np.asarray(mode_profile["ez_profile"], dtype=np.float64)
    cell_indices = mode_profile["cell_indices"]
    j_box_lo = int(mode_profile["j_grid_lo"])
    k_box_lo = int(mode_profile["k_grid_lo"])
    n_z_sub = int(mode_profile["n_z_sub"])
    dy = float(mode_profile["dy"])
    dz = float(mode_profile["dz"])

    # i_feed is constant across cell_indices.
    i_feed = cell_indices[0][0]
    dx_feed = float(_axis_cell_size(grid, "x", i_feed))

    integrand = float(np.sum(ez_profile * ez_profile) * dy * dz)
    if integrand <= 0:
        return materials
    sigma_uniform = 1.0 / (port.impedance * dx_feed * integrand)

    sigma = materials.sigma
    for (i, j, k) in cell_indices:
        j_loc = j - j_box_lo
        k_loc = k - k_box_lo
        if not (0 <= k_loc < n_z_sub):
            continue
        if not (0 <= j_loc < ez_profile.shape[0]):
            continue
        # Only load cells where the mode actually carries energy. Cells
        # with |Ez|·dz ≪ V_src contribute nothing physical and adding σ
        # there would just damp evanescent fringing.
        if float(ez_profile[j_loc, k_loc]) == 0.0:
            continue
        sigma = sigma.at[i, j, k].add(sigma_uniform)
    return materials._replace(sigma=sigma)


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScaledWaveform:
    """Wrap a base waveform and scale its output by a constant."""

    base: object
    scale: float

    def __call__(self, t):
        return self.base(t) * self.scale


def make_msl_port_sources(grid, port: MSLPort, materials, n_steps,
                          *, mode_profile: dict | None = None):
    """Build the SourceSpec list for an MSL feed plane.

    Two modes:

    - **Uniform** (``mode_profile is None``, legacy): every cell in the
      port cross-section gets an Ez source with amplitude
      ``V_src / N_z`` (voltage division along z).

    - **Eigenmode** (``mode_profile`` from
      :func:`compute_msl_mode_profile`): each cell gets an Ez source
      proportional to the static-Laplace ``Ez(y,z)`` profile. The
      profile is normalised so ``∫Ez·dz`` at the trace centre equals
      ``V_src``, matching the legacy convention. Sources extend
      laterally beyond the trace footprint to inject the fringing field.

    The port impedance must already be folded into ``materials`` via
    :func:`setup_msl_port` (with the same ``mode_profile``).
    """
    if port.excitation is None:
        return []
    from rfx.simulation import SourceSpec  # local import: avoid cycles

    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    base_wave = jax.vmap(port.excitation)(times)

    if mode_profile is None:
        cells = _msl_yz_cells(grid, port)
        if not cells:
            return []
        n_z = len({c[2] for c in cells})
        specs = []
        for (i, j, k) in cells:
            eps = materials.eps_r[i, j, k] * EPS_0
            sigma = materials.sigma[i, j, k]
            loss = sigma * grid.dt / (2.0 * eps)
            cb = (grid.dt / eps) / (1.0 + loss)
            d_par = _axis_cell_size(grid, "z", k)
            waveform = (cb / d_par) * base_wave / float(n_z)
            specs.append(SourceSpec(i=i, j=j, k=k, component="ez", waveform=waveform))
        return specs

    # Eigenmode-shaped Ez source. Profile is normalised so that
    # ∫ Ez·dz at the trace centre = 1 V; multiply by the desired V_src
    # delivered by base_wave (the excitation already carries amplitude).
    ez_profile = np.asarray(mode_profile["ez_profile"], dtype=np.float64)
    cell_indices = mode_profile["cell_indices"]
    j_box_lo = int(mode_profile["j_grid_lo"])
    k_box_lo = int(mode_profile["k_grid_lo"])
    n_z_sub = int(mode_profile["n_z_sub"])

    specs = []
    for (i, j, k) in cell_indices:
        j_loc = j - j_box_lo
        k_loc = k - k_box_lo
        if not (0 <= k_loc < n_z_sub):
            continue
        if not (0 <= j_loc < ez_profile.shape[0]):
            continue
        ez_w = float(ez_profile[j_loc, k_loc])
        if ez_w == 0.0:
            continue
        eps = materials.eps_r[i, j, k] * EPS_0
        sigma = materials.sigma[i, j, k]
        loss = sigma * grid.dt / (2.0 * eps)
        cb = (grid.dt / eps) / (1.0 + loss)
        # ez_w has units V/m (∂φ/∂z with φ ∈ [0,1]V then renormalised so
        # ∫Ez dz = 1V). The legacy uniform path used (cb/d_par)*base/n_z;
        # here we use cb·ez_w·base — the d_par cancels because we
        # injected ε·∂E/∂t = J = -σ_src*Ez_inc with Ez_inc = ez_w·V_src.
        # Equivalent to legacy when ez_w = 1/H_sub and dz uniform.
        waveform = cb * ez_w * base_wave
        specs.append(SourceSpec(i=int(i), j=int(j), k=int(k),
                                component="ez", waveform=waveform))
    return specs


def make_msl_port_sources_jm(
    grid,
    port: MSLPort,
    materials,
    n_steps: int,
    eigenmode_data,
):
    """Build Schelkunoff J+M one-sided source specs for an MSL eigenmode.

    Returns ``(e_specs, h_specs)`` where ``e_specs`` are
    :class:`~rfx.simulation.SourceSpec` objects (added to E-field cells
    at the source plane) and ``h_specs`` are
    :class:`~rfx.simulation.MagneticSourceSpec` objects (added to H-field
    cells at the adjacent half-cell behind the source plane).

    Implementation follows the canonical Taflove TFSF pattern as implemented
    in :func:`rfx.sources.waveguide_port.apply_waveguide_port_h` /
    :func:`apply_waveguide_port_e`.  The two corrections together cancel the
    backward (-x) wave and double the forward (+x) wave:

    **H correction** at ``i_feed - 1`` (for ``+x`` direction):

    * ``Hy[i-1,j,k] += -coeff_H * ez_mode * wave_h``
    * ``Hz[i-1,j,k] += +coeff_H * ey_mode * wave_h``

    where ``coeff_H = dt / (μ₀ · dx)`` (same as waveguide port).

    **E correction** at ``i_feed`` (for ``+x`` direction):

    * ``Ez[i,j,k] += -coeff_E * hy_mode * wave_e``
    * ``Ey[i,j,k] += +coeff_E * hz_mode * wave_e``

    where ``coeff_E = dt / (ε · dx)`` (same as waveguide port).

    Parameters
    ----------
    eigenmode_data : MSLEigenmodeData
        Solved quasi-TEM eigenmode from
        :func:`~rfx.sources.msl_eigenmode.compute_msl_eigenmode_profile`.
    """
    if port.excitation is None:
        return [], []

    from rfx.simulation import SourceSpec, MagneticSourceSpec  # local import: avoid cycles

    dt = grid.dt
    dx = float(grid.dx)
    times_e = jnp.arange(n_steps, dtype=jnp.float32) * dt
    # H lives at half-integer steps: t_{n+1/2} = (n + 0.5) * dt
    times_h = (jnp.arange(n_steps, dtype=jnp.float32) + 0.5) * dt
    base_wave_e = jax.vmap(port.excitation)(times_e)
    base_wave_h = jax.vmap(port.excitation)(times_h)

    em = eigenmode_data
    j_lo = em.j_grid_lo
    k_lo = em.k_grid_lo
    n_y = em.n_y_grid
    n_z = em.n_z_grid

    # TFSF coefficients — identical to waveguide_port.py apply_waveguide_port_h/e
    coeff_H = float(dt / (MU_0 * dx))

    # Direction: +x → H plane behind = i-1; sign = -1 (matches waveguide port)
    #            -x → H plane behind = i;   sign = +1
    if port.direction == "+x":
        h_i_offset = -1
        sign = -1.0
    else:
        h_i_offset = 0
        sign = +1.0

    e_specs: list = []
    h_specs: list = []

    for (i, j, k) in em.cell_indices:
        j_loc = j - j_lo
        k_loc = k - k_lo
        if not (0 <= j_loc < n_y and 0 <= k_loc < n_z):
            continue

        ey_w = float(em.ey[j_loc, k_loc])
        ez_w = float(em.ez[j_loc, k_loc])
        hy_w = float(em.hy[j_loc, k_loc])
        hz_w = float(em.hz[j_loc, k_loc])

        eps = float(materials.eps_r[i, j, k]) * EPS_0
        sigma = float(materials.sigma[i, j, k])
        loss = sigma * dt / (2.0 * eps)
        coeff_E = (dt / (eps * dx)) / (1.0 + loss)

        i_h = int(i) + h_i_offset  # H correction cell index

        # --- E correction at i_feed (driven by H profile) ---
        # Ez += -sign * coeff_E * hy_mode * wave_e
        if hy_w != 0.0:
            e_specs.append(SourceSpec(
                i=int(i), j=int(j), k=int(k),
                component="ez",
                waveform=jnp.array(-sign * coeff_E * hy_w, dtype=jnp.float32) * base_wave_e,
            ))

        # Ey += +sign * coeff_E * hz_mode * wave_e
        if hz_w != 0.0:
            e_specs.append(SourceSpec(
                i=int(i), j=int(j), k=int(k),
                component="ey",
                waveform=jnp.array(sign * coeff_E * hz_w, dtype=jnp.float32) * base_wave_e,
            ))

        # --- H correction at i_feed - 1 (driven by E profile) ---
        if i_h < 0:
            continue  # clamp: no H correction before grid boundary

        # Hy += sign * coeff_H * ez_mode * wave_h   (sign=-1 for +x)
        if ez_w != 0.0:
            h_specs.append(MagneticSourceSpec(
                i=i_h, j=int(j), k=int(k),
                component="hy",
                waveform=jnp.array(sign * coeff_H * ez_w, dtype=jnp.float32) * base_wave_h,
            ))

        # Hz += -sign * coeff_H * ey_mode * wave_h
        if ey_w != 0.0:
            h_specs.append(MagneticSourceSpec(
                i=i_h, j=int(j), k=int(k),
                component="hz",
                waveform=jnp.array(-sign * coeff_H * ey_w, dtype=jnp.float32) * base_wave_h,
            ))

    return e_specs, h_specs


# ---------------------------------------------------------------------------
# Probe-plane locations
# ---------------------------------------------------------------------------


def msl_probe_x_coords(
    grid,
    port: MSLPort,
    n_offset_cells: int = 5,
    n_spacing_cells: int = 3,
) -> tuple[float, float, float]:
    """Return three downstream probe x-coordinates for 3-probe extraction.

    The first probe is ``n_offset_cells`` cells from the feed plane; the
    remaining two are spaced by ``n_spacing_cells`` further along the
    propagation direction.  Indices are clamped into the valid grid
    range so callers always receive in-domain physical coordinates.
    """
    i_feed, _, _ = grid.position_to_index((port.feed_x, port.y_lo, port.z_lo))
    sign = 1 if port.direction == "+x" else -1
    nx = grid.nx

    def _x_for_index(target_i: int) -> float:
        clamped = max(0, min(target_i, nx - 1))
        # User-domain physical x-coord. position_to_index adds pad_x_lo, so
        # subtract it back when rebuilding the coordinate.
        pad = getattr(grid, "pad_x_lo", 0)
        return float((clamped - pad) * grid.dx)

    i1 = i_feed + sign * n_offset_cells
    i2 = i1 + sign * n_spacing_cells
    i3 = i2 + sign * n_spacing_cells
    return _x_for_index(i1), _x_for_index(i2), _x_for_index(i3)


# ---------------------------------------------------------------------------
# 3-probe S-parameter extraction
# ---------------------------------------------------------------------------


def _integrate_v(ez_plane: np.ndarray, j_center: int, z_lo_idx: int, z_hi_idx: int,
                 dz_arr: np.ndarray) -> np.ndarray:
    """V(f) = ∫ Ez dz along the substrate height at y = j_center."""
    total = np.zeros(ez_plane.shape[0], dtype=complex)
    for k in range(z_lo_idx, z_hi_idx + 1):
        total = total + ez_plane[:, j_center, k] * float(dz_arr[k])
    return total


def _integrate_i(hy_plane: np.ndarray, y_lo_idx: int, y_hi_idx: int, z_top_idx: int,
                 dy_arr: np.ndarray) -> np.ndarray:
    """I(f) = ∫ Hy dy across the trace width at z = z_top_idx."""
    total = np.zeros(hy_plane.shape[0], dtype=complex)
    for j in range(y_lo_idx, y_hi_idx + 1):
        total = total + hy_plane[:, j, z_top_idx] * float(dy_arr[j])
    return total


def extract_msl_s_params(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    i1: np.ndarray,
    *,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-probe de-embedding for V/I on a transmission line.

    Given V at three equally-spaced probe planes (separation Δ) and I at
    the first plane, recover the reflection coefficient ``S11`` at probe 1,
    the characteristic impedance ``Z0``, and the per-step phasor
    ``q = exp(-jβΔ)`` (positive root with ``|q| ≤ 1``).

    Parameters
    ----------
    v1, v2, v3 : (n_freqs,) complex
        DFT'd voltage at three probe planes spaced by Δ along the line.
    i1 : (n_freqs,) complex
        DFT'd current at probe plane 1.

    Returns
    -------
    s11, z0, q : (n_freqs,) complex
        ``s11`` is the reflection coefficient at probe 1; ``z0`` the
        extracted characteristic impedance; ``q`` the per-Δ phasor.
    """
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    v3 = np.asarray(v3, dtype=complex)
    i1 = np.asarray(i1, dtype=complex)

    s11, z0, q = _solve_3probe(v1, v2, v3, i1, eps)
    return s11, z0, q


def _solve_3probe(v1, v2, v3, i1, eps):
    """Closed-form 3-probe solver shared by extract_msl_s_params/forward."""
    # q + 1/q = (V1 + V3) / V2  →  q² − coeff·q + 1 = 0
    coeff = (v1 + v3) / (v2 + eps)
    disc = coeff**2 - 4.0 + 0j
    sqrt_disc = np.sqrt(disc)
    q_plus = (coeff + sqrt_disc) / 2.0
    q_minus = (coeff - sqrt_disc) / 2.0

    # Both roots are reciprocals (q_minus = 1/q_plus), so on a lossless
    # line they have |q|=1 exactly and the |q|≤1 selector becomes
    # ambiguous.  The physical forward root must reproduce the observed
    # forward step ratio V2/V1 in the absence of strong reflection; we
    # therefore pick the root whose phase is closer to V2/V1.
    ratio = v2 / (v1 + eps)
    err_plus = np.abs(q_plus - ratio)
    err_minus = np.abs(q_minus - ratio)
    # Tie-breaker: |q| ≤ 1 (decaying) is preferred when both errors match.
    use_plus = (err_plus < err_minus) | (
        (np.isclose(err_plus, err_minus)) & (np.abs(q_plus) <= np.abs(q_minus))
    )
    q = np.where(use_plus, q_plus, q_minus)

    # Forward (alpha) and backward (gamma) wave amplitudes at probe 1
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    gamma = q * (v1 * q - v2) / denom

    z0 = (alpha - gamma) / (i1 + eps)
    s11 = gamma / (alpha + eps)
    return s11, z0, q


def msl_forward_amplitude(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, q)``: forward amplitude at probe 1 and per-Δ phasor.

    Useful on the passive (non-driven) port where only the voltage triple
    is needed to recover the transmitted forward wave.
    """
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    v3 = np.asarray(v3, dtype=complex)
    # Reuse the shared solver (i1 unused for q/alpha — pass v1 as a
    # placeholder; z0/s11 outputs are discarded).
    _, _, q = _solve_3probe(v1, v2, v3, v1, eps)
    denom = (q * q - 1.0) + eps
    alpha = (q * v2 - v1) / denom
    return alpha, q


def compute_s21(alpha_passive: np.ndarray, alpha_driven: np.ndarray,
                *, eps: float = 1e-30) -> np.ndarray:
    """S21 from forward amplitudes on driven (port 1) and passive (port 2)."""
    return np.asarray(alpha_passive, dtype=complex) / (
        np.asarray(alpha_driven, dtype=complex) + eps
    )
