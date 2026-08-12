"""Microstrip line (MSL) port: 2D distributed port geometry, mode-profile
solve, and probe/current primitives.

Unlike the 1-cell-transverse ``WirePort``, the MSL port covers the full
trace cross-section (y × z under the trace) and distributes the total
port impedance Z0 as conductivity over the cross-section cells. After
the FDTD run, downstream probe planes are used to extract the
propagation constant β, characteristic impedance Z0, and the wave
amplitudes.

Production extraction (``Simulation.compute_msl_s_matrix`` /
``compute_mixed_s_matrix``, ``rfx/api/_sparams.py``) places N equally
spaced probes (:func:`msl_probe_x_coords_n`), fits them by SVD
least-squares (issue #80 Fix C), and assembles the S-matrix with the
multi-drive solve (issue #507). The original 3-probe placement
(:func:`msl_probe_x_coords`) and the OpenEMS-style 3-probe recurrence it
was built for are retained as a geometry primitive consumed by
``validation/tmtt_paper/msl_stub_notch_tuning.py`` (via
``rfx.probes.msl_wave_decomp.register_msl_plane_probes``); it is not on
the production S-matrix path.

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
# Axis roles (issue #661)
# ---------------------------------------------------------------------------

#: Axis name -> index into a physical ``(x, y, z)`` tuple.
_MSL_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

#: ``direction`` -> ``(propagation, width, substrate-normal, sign)``.
#: The substrate normal is always z; see :func:`msl_axis_roles`.
_MSL_AXIS_ROLES = {
    "+x": ("x", "y", "z", +1.0),
    "-x": ("x", "y", "z", -1.0),
    "+y": ("y", "x", "z", +1.0),
    "-y": ("y", "x", "z", -1.0),
}

#: Right-handed cyclic transverse pair ``(a, b)`` for a propagation axis
#: ``p``, defined by ``a_hat x b_hat = p_hat``.  This is what makes the
#: closed Ampere loop enclose current along ``+p_hat``: traversing
#: ``+a`` at low ``b``, ``+b`` at high ``a``, ``-a`` at high ``b``,
#: ``-b`` at low ``a`` is a counter-clockwise circuit about ``+p_hat``.
#:
#: Getting this wrong is NOT a loud failure.  A naive ``x <-> y`` axis
#: SWAP (determinant -1, a reflection rather than a rotation) flips the
#: sign of the enclosed current exactly -- measured ``I_swap / I_x =
#: -1.00000009`` on the committed thru fixture's recorded H planes.  That
#: exchanges the wave amplitudes ``a = (V + Z0*I)/2`` and ``b = (V -
#: Z0*I)/2``, which maps the assembled ``S = B A^-1`` to ``A B^-1 =
#: S^-1``.  For the low-loss, nearly-matched line that any MSL lane is
#: validated on, ``S`` is nearly unitary, so ``S^-1 ~ S^dagger``: the
#: MAGNITUDES barely move (measured max ||S| - |S_swapped|| = 1.3e-3,
#: |S11| 0.17905 -> 0.17875) while ``arg(S21)`` is exactly NEGATED
#: (measured: the two angles sum to <= 0.02 deg across the band, i.e. a
#: negative group delay).  Nothing in the lane catches it: |S11| stays
#: far under the 1.05 honesty-guard threshold, column power stays below
#: 1 so the passivity projection is silent, and cond(A) = 1.32.  The
#: complex error is O(1) (max |S - S_swapped| = 1.912).
#: Hence: derive the pair here, never hand-write a per-direction swap,
#: and compare COMPLEX S in any equivalence test.
_MSL_CYCLIC_PAIR = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}

#: Directions ``add_msl_port`` / :class:`MSLPort` accept.
MSL_SUPPORTED_DIRECTIONS = ("+x", "-x", "+y", "-y")

#: Directions that are axis-aligned but cannot be expressed by this port's
#: geometry contract -- see :func:`msl_axis_roles`.
MSL_REJECTED_DIRECTIONS = ("+z", "-z")


def msl_axis_roles(direction: str) -> tuple[str, str, str, float]:
    """Resolve ``direction`` to ``(propagation, width, normal, sign)`` axes.

    A microstrip port has three distinct axis roles:

    * **propagation** -- the axis the launched wave travels along, normal
      to the feed plane and to every N-probe plane;
    * **width** -- the in-board axis the trace width spans;
    * **normal** -- the substrate normal, from ground plane up to the
      trace.

    ``direction`` names the propagation axis and its sign.  The substrate
    normal is **always z**: the port's geometry contract is
    ``position = (x, y, z_lo)`` plus a scalar ``height``, and ``z_lo +
    height`` is what places the substrate.  Nothing in that contract can
    name a different normal, so the board always lies in the xy-plane and
    the free choice is which in-plane axis the feed runs along.

    That is why ``"+z"`` / ``"-z"`` are rejected rather than supported: a
    z-propagating microstrip needs its substrate normal along x or y, and
    the whole normal-axis chain would have to move with it -- the static
    Laplace cross-section solve and its ``ez_profile``, the ``"ez"``
    source component, the modal voltage ``V = sum(Ez*dz)``, and the
    trace-conductor PEC scan that walks upward from the substrate top.
    Accepting ``"+z"`` while leaving those on z would silently return a
    z-normal-flavoured answer for a board that is not oriented that way.

    Returns
    -------
    (propagation, width, normal, sign)
        Axis names ``"x"``/``"y"``/``"z"`` and ``sign`` = +1.0 / -1.0.
    """
    try:
        return _MSL_AXIS_ROLES[direction]
    except (KeyError, TypeError):
        pass
    if direction in MSL_REJECTED_DIRECTIONS:
        raise ValueError(
            f"MSL port direction={direction!r} is not supported. The "
            "microstrip port fixes the substrate normal to z -- its "
            "geometry contract is position=(x, y, z_lo) plus a scalar "
            "height, so the substrate spans z_lo..z_lo+height along z and "
            "the board lies in the xy-plane. A wave propagating along z "
            "would need the substrate normal along x or y, which this "
            "contract cannot express: the Laplace mode solve, the 'ez' "
            "source component, the modal voltage V = sum(Ez*dz) and the "
            "trace-PEC scan all reference the normal axis. Rotate the "
            "board so the feed runs along x or y "
            f"(supported: {', '.join(MSL_SUPPORTED_DIRECTIONS)}), or open "
            "an issue for a substrate-normal parameter."
        )
    raise ValueError(
        f"direction must be one of {MSL_SUPPORTED_DIRECTIONS}, got "
        f"{direction!r}"
    )


def msl_ampere_pair(direction: str) -> tuple[str, str]:
    """Right-handed transverse pair ``(a, b)`` with ``a_hat x b_hat = p_hat``.

    ``a`` and ``b`` name the axes the four closed-Ampere-loop legs run
    along, in the order :func:`msl_loop_current` expects.  Derived from
    :data:`_MSL_CYCLIC_PAIR` -- see the warning recorded there about what
    a hand-written axis swap costs.
    """
    prop, _width, _normal, _sign = msl_axis_roles(direction)
    return _MSL_CYCLIC_PAIR[prop]


# ---------------------------------------------------------------------------
# Port description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSLPort:
    """Microstrip line port spanning the full trace cross-section.

    The field names below are historical (they date from the x-only port)
    and are pinned by the design-IR contract, so they are read in the
    **port frame**, not literally as x/y/z.  :func:`msl_axis_roles`
    resolves ``direction`` to the three axis roles; the mapping is

    ==========  ====================  ====================
    field       ``"+x"`` / ``"-x"``   ``"+y"`` / ``"-y"``
    ==========  ====================  ====================
    ``feed_x``  x (propagation)       y (propagation)
    ``y_lo/hi`` y (trace width)       x (trace width)
    ``z_lo/hi`` z (substrate normal)  z (substrate normal)
    ==========  ====================  ====================

    Use :func:`msl_physical_point` to turn port-frame coordinates back
    into a physical ``(x, y, z)`` tuple.

    Parameters
    ----------
    feed_x : float
        Feed-plane coordinate (metres) along the PROPAGATION axis, where
        the source and termination are placed.
    y_lo, y_hi : float
        Trace extent (metres) along the WIDTH axis. ``y_hi - y_lo`` is
        the trace width.
    z_lo, z_hi : float
        Substrate extent (metres) along the NORMAL axis (always z).
        ``z_lo`` is typically the ground plane / substrate bottom;
        ``z_hi`` the top of the substrate (where the trace lies).
    direction : str
        One of :data:`MSL_SUPPORTED_DIRECTIONS` -- the direction the
        launched wave propagates away from the feed plane.
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


def msl_physical_point(
    direction: str, prop_c: float, width_c: float, normal_c: float
) -> tuple[float, float, float]:
    """Assemble a physical ``(x, y, z)`` point from port-frame coordinates."""
    prop, width, normal, _sign = msl_axis_roles(direction)
    pt = [0.0, 0.0, 0.0]
    pt[_MSL_AXIS_INDEX[prop]] = float(prop_c)
    pt[_MSL_AXIS_INDEX[width]] = float(width_c)
    pt[_MSL_AXIS_INDEX[normal]] = float(normal_c)
    return (pt[0], pt[1], pt[2])


def msl_port_from_entry(pe) -> "MSLPort":
    """Build an :class:`MSLPort` from a ``_MSLPortEntry``.

    ``pe.position`` is a PHYSICAL ``(x, y, z)`` tuple: the feed coordinate
    on the propagation axis, the trace centre on the width axis, and the
    substrate bottom on the normal axis -- each already in its own slot.
    This is the single place that projection happens, so the runners,
    ``compute_msl_s_matrix`` and preflight cannot drift apart on it.
    """
    prop, width, normal, _sign = msl_axis_roles(pe.direction)
    pos = pe.position
    feed = float(pos[_MSL_AXIS_INDEX[prop]])
    centre = float(pos[_MSL_AXIS_INDEX[width]])
    base = float(pos[_MSL_AXIS_INDEX[normal]])
    return MSLPort(
        feed_x=feed,
        y_lo=centre - float(pe.width) / 2.0,
        y_hi=centre + float(pe.width) / 2.0,
        z_lo=base,
        z_hi=base + float(pe.height),
        direction=pe.direction,
        impedance=float(pe.impedance),
        excitation=pe.waveform,
    )


# ---------------------------------------------------------------------------
# Cross-section helpers
# ---------------------------------------------------------------------------


def _axis_cell_size(grid, axis: str, idx: int) -> float:
    """Return the cell size at index ``idx`` along ``axis``.

    Supports both uniform Grid and NonUniformGrid via duck typing.

    On a ``NonUniformGrid`` the per-cell sizes live in ``dx_arr``/``dy_arr``/
    ``dz`` (the NamedTuple has NO ``*_profile`` attribute), so the duck-typed
    ``getattr(grid, "dx_profile")`` path below would fall through to the scalar
    BOUNDARY ``grid.dx`` for every cell — the wrong-cell bug. Read the real
    per-cell array first. On a uniform ``Grid`` (not a ``NonUniformGrid``) this
    branch is skipped and the legacy behaviour is byte-identical.
    """
    from rfx.nonuniform import NonUniformGrid
    if isinstance(grid, NonUniformGrid):
        per_cell = np.asarray({"x": grid.dx_arr, "y": grid.dy_arr, "z": grid.dz}[axis])
        clamped = max(0, min(int(idx), int(per_cell.shape[0]) - 1))
        return float(per_cell[clamped])
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

    The cross-section is the plane normal to the PROPAGATION axis at the
    feed coordinate, spanning the trace WIDTH axis and the substrate
    NORMAL axis. Indices are returned in physical ``(i, j, k)`` order, and
    the iteration order is width-major then normal -- for a ``"+x"`` port
    that is exactly the historical ``for j: for k:`` order over y then z.
    """
    prop, width, normal, _sign = msl_axis_roles(port.direction)
    ip = _MSL_AXIS_INDEX[prop]
    iw = _MSL_AXIS_INDEX[width]
    inr = _MSL_AXIS_INDEX[normal]

    lo_idx = _msl_position_to_index(
        grid, msl_physical_point(port.direction, port.feed_x, port.y_lo, port.z_lo)
    )
    hi_idx = _msl_position_to_index(
        grid, msl_physical_point(port.direction, port.feed_x, port.y_hi, port.z_hi)
    )
    i_feed = int(lo_idx[ip])
    w_a, w_b = sorted((int(lo_idx[iw]), int(hi_idx[iw])))
    n_a, n_b = sorted((int(lo_idx[inr]), int(hi_idx[inr])))

    cells = []
    for w in range(w_a, w_b + 1):
        for n in range(n_a, n_b + 1):
            cell = [0, 0, 0]
            cell[ip] = i_feed
            cell[iw] = int(w)
            cell[inr] = int(n)
            cells.append((cell[0], cell[1], cell[2]))
    return cells


def msl_cross_section_span(grid, port: MSLPort) -> dict:
    """Feed index plus the width / normal cell spans of the cross-section.

    Returns a dict with ``i_feed``, ``w_lo``/``w_hi``, ``w_centre``,
    ``n_lo``/``n_hi`` (all ints, inclusive spans) and the resolved axis
    names/indices. Every consumer that used to unpack ``c[0]``/``c[1]``/
    ``c[2]`` from :func:`_msl_yz_cells` should read this instead, so the
    axis projection lives in one place.
    """
    prop, width, normal, sign = msl_axis_roles(port.direction)
    ip = _MSL_AXIS_INDEX[prop]
    iw = _MSL_AXIS_INDEX[width]
    inr = _MSL_AXIS_INDEX[normal]
    cells = _msl_yz_cells(grid, port)
    ws = sorted({c[iw] for c in cells})
    ns = sorted({c[inr] for c in cells})
    return dict(
        prop_axis=prop, width_axis=width, normal_axis=normal, sign=sign,
        prop_idx=ip, width_idx=iw, normal_idx=inr,
        i_feed=int(cells[0][ip]),
        w_lo=int(ws[0]), w_hi=int(ws[-1]),
        w_centre=int((ws[0] + ws[-1]) // 2),
        n_lo=int(ns[0]), n_hi=int(ns[-1]),
        cells=cells,
    )


def msl_cell(direction: str, i_prop: int, i_width: int, i_normal: int
             ) -> tuple[int, int, int]:
    """Assemble a physical ``(i, j, k)`` index from port-frame indices."""
    prop, width, normal, _sign = msl_axis_roles(direction)
    cell = [0, 0, 0]
    cell[_MSL_AXIS_INDEX[prop]] = int(i_prop)
    cell[_MSL_AXIS_INDEX[width]] = int(i_width)
    cell[_MSL_AXIS_INDEX[normal]] = int(i_normal)
    return (cell[0], cell[1], cell[2])


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
    # "y" below is the trace-WIDTH axis and "z" the substrate-NORMAL axis
    # (issue #661): for a "+x"/"-x" port those are literally y and z, for
    # a "+y"/"-y" port the width axis is x. The solve itself is a pure
    # (width, normal) cross-section problem -- measured independent of the
    # propagation axis beyond ``i_feed`` (identical ez_profile/z0_static/
    # eps_eff across two feed planes and opposite directions).
    span = msl_cross_section_span(grid, port)
    width_axis = span["width_axis"]
    normal_axis = span["normal_axis"]
    j_trace_lo = span["w_lo"]
    j_trace_hi = span["w_hi"]
    k_sub_lo = span["n_lo"]
    i_feed = span["i_feed"]

    n_y_trace = j_trace_hi - j_trace_lo + 1
    n_z_sub = span["n_hi"] - k_sub_lo + 1
    dy = float(_axis_cell_size(grid, width_axis, j_trace_lo))
    dz = float(_axis_cell_size(grid, normal_axis, k_sub_lo))

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
    j_grid_hi = min(grid.shape[span["width_idx"]] - 1, j_trace_hi + pad_y_cells)
    k_grid_lo = k_sub_lo  # ground at bottom of substrate
    n_y_grid = j_grid_hi - j_grid_lo + 1

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
            cell_indices.append(
                msl_cell(port.direction, i_feed, j_grid, k_grid)
            )

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
        # Issue #661: which physical axis each of the two cross-section
        # indices belongs to, so consumers can project a physical
        # (i, j, k) cell back onto (width, normal) without re-deriving it.
        direction=str(port.direction),
        prop_idx=int(span["prop_idx"]),
        width_idx=int(span["width_idx"]),
        normal_idx=int(span["normal_idx"]),
        prop_axis=str(span["prop_axis"]),
        width_axis=str(width_axis),
        normal_axis=str(normal_axis),
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
        span = msl_cross_section_span(grid, port)
        cells = span["cells"]
        if not cells:
            return materials
        n_y = span["w_hi"] - span["w_lo"] + 1
        n_z = span["n_hi"] - span["n_lo"] + 1
        # Issue #661: these three cell sizes are the PROPAGATION, WIDTH
        # and NORMAL cell sizes -- not literally dx/dy/dz. sigma scales as
        # 1/d_prop (measured: halving the propagation-axis cell doubles
        # sum(sigma), ratio 2.0000), so reading dx here for a "+y" port
        # would be wrong by dy/dx -- and EXACTLY RIGHT on a cubic grid,
        # which is why no cubic-cell equivalence test can catch it.
        ax_p = span["prop_axis"]
        ax_w = span["width_axis"]
        ax_n = span["normal_axis"]
        ip, iw, inr = span["prop_idx"], span["width_idx"], span["normal_idx"]
        sigma = materials.sigma
        for cell in cells:
            i, j, k = cell
            d_prop = _axis_cell_size(grid, ax_p, cell[ip])
            d_width = _axis_cell_size(grid, ax_w, cell[iw])
            d_norm = _axis_cell_size(grid, ax_n, cell[inr])
            sigma_cell = (
                (n_z * d_norm) / (port.impedance * n_y * d_prop * d_width)
            )
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
    # Issue #661: project physical cells onto (width, normal) indices.
    ip = int(mode_profile["prop_idx"])
    iw = int(mode_profile["width_idx"])
    inr = int(mode_profile["normal_idx"])
    ax_p = str(mode_profile["prop_axis"])

    # i_feed is constant across cell_indices, on the PROPAGATION axis.
    i_feed = cell_indices[0][ip]
    dx_feed = float(_axis_cell_size(grid, ax_p, i_feed))

    integrand = float(np.sum(ez_profile * ez_profile) * dy * dz)
    if integrand <= 0:
        return materials
    sigma_uniform = 1.0 / (port.impedance * dx_feed * integrand)

    sigma = materials.sigma
    for cell in cell_indices:
        i, j, k = cell
        j_loc = cell[iw] - j_box_lo
        k_loc = cell[inr] - k_box_lo
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
        span = msl_cross_section_span(grid, port)
        cells = span["cells"]
        if not cells:
            return []
        n_z = span["n_hi"] - span["n_lo"] + 1
        ax_n = span["normal_axis"]
        inr = span["normal_idx"]
        specs = []
        for cell in cells:
            i, j, k = cell
            eps = materials.eps_r[i, j, k] * EPS_0
            sigma = materials.sigma[i, j, k]
            loss = sigma * grid.dt / (2.0 * eps)
            cb = (grid.dt / eps) / (1.0 + loss)
            # d_par is the cell size along the SUBSTRATE NORMAL (always z,
            # the axis "ez" points along) -- not the propagation axis.
            d_par = _axis_cell_size(grid, ax_n, cell[inr])
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
    iw = int(mode_profile["width_idx"])
    inr = int(mode_profile["normal_idx"])

    specs = []
    for cell in cell_indices:
        i, j, k = cell
        j_loc = cell[iw] - j_box_lo
        k_loc = cell[inr] - k_box_lo
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

    # Issue #661: this launch is x-only and stays that way. It hard-codes
    # the (hy, hz) / (ey, ez) TFSF correction pair and ``coeff_H = dt /
    # (MU_0 * dx)`` to the x propagation axis, and its eigenmode source
    # (``rfx.sources.msl_eigenmode``) is itself a documented falsified
    # dead-end kept only under a strict xfail. Generalising it would mean
    # re-deriving a solver that is not on the supported lane, so a non-x
    # direction is refused rather than half-rotated.
    if port.direction not in ("+x", "-x"):
        raise NotImplementedError(
            f"MSL mode='eigenmode' supports direction '+x'/'-x' only, got "
            f"{port.direction!r}. The Schelkunoff J+M launch hard-codes the "
            "x-axis TFSF correction pair and rides on the FDFD eigenmode "
            "solver, which is not a supported lane. Use mode='laplace' (the "
            "add_msl_port default) for a y-directed port."
        )

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


def _msl_position_to_index(grid, pos):
    """Grid-agnostic physical-position -> index lookup.

    ``rfx.grid.Grid`` exposes ``position_to_index`` as a method; the
    ``NonUniformGrid`` NamedTuple does NOT — its lookup is the free
    function ``rfx.nonuniform.position_to_index`` (cumulative cell-edge,
    so it is graded-aware). Duck-types over both.
    """
    method = getattr(grid, "position_to_index", None)
    if callable(method):
        return method(pos)
    from rfx.nonuniform import position_to_index as _nu_position_to_index
    return _nu_position_to_index(grid, pos)


def _msl_coord_for_index(grid, axis: str, target_i: int) -> float:
    """Physical coordinate along ``axis`` of a (clamped) grid index.

    Graded-mesh aware. On a uniform ``Grid`` this is the legacy
    ``(clamped - pad_lo) * grid.dx`` and is byte-identical (``Grid`` has
    no ``*_arr``). On a ``NonUniformGrid`` ``grid.dx`` is ONLY the
    boundary cell, so the index must be converted to a physical
    coordinate via the cumulative cell-edge sum of the interior per-cell
    profile (mirroring ``_range_to_slice_nu`` in
    ``rfx/runners/nonuniform.py``). Indices are clamped into the
    in-domain range. (In the leading-CPML region ``u<0`` the uniform
    branch returns a negative coord while the NU branch clamps to
    ``edges[0]=0``; valid probe placement never targets that region.)
    """
    n_axis = int(getattr(grid, {"x": "nx", "y": "ny", "z": "nz"}[axis]))
    pad = int(getattr(grid, f"pad_{axis}_lo", 0))
    clamped = max(0, min(int(target_i), n_axis - 1))
    u = clamped - pad  # user-domain (non-CPML) interior index
    arr_attr = {"x": "dx_arr", "y": "dy_arr", "z": "dz"}[axis]
    from rfx.nonuniform import NonUniformGrid
    if not isinstance(grid, NonUniformGrid):
        # Uniform Grid — legacy scalar spacing (byte-identical).
        return float(u * grid.dx)
    per_cell = getattr(grid, arr_attr, None)
    if per_cell is None:
        return float(u * grid.dx)
    # NonUniformGrid — cumulative interior cell-edge positions.
    pad_hi = int(getattr(grid, f"pad_{axis}_hi", 0))
    from rfx.nonuniform import interior_cells
    interior = interior_cells(np.asarray(per_cell, dtype=float), pad, pad_hi)
    edges = np.insert(np.cumsum(interior), 0, 0.0)
    u_c = max(0, min(u, len(edges) - 1))
    return float(edges[u_c])


def _msl_x_for_index(grid, target_i: int) -> float:
    """x-axis alias of :func:`_msl_coord_for_index` (external callers)."""
    return _msl_coord_for_index(grid, "x", target_i)


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
    Graded-mesh aware (see :func:`_msl_x_for_index`).
    """
    prop, _w, _n, sign_f = msl_axis_roles(port.direction)
    idx = _msl_position_to_index(
        grid, msl_physical_point(port.direction, port.feed_x, port.y_lo, port.z_lo)
    )
    i_feed = int(idx[_MSL_AXIS_INDEX[prop]])
    sign = int(sign_f)
    i1 = i_feed + sign * n_offset_cells
    i2 = i1 + sign * n_spacing_cells
    i3 = i2 + sign * n_spacing_cells
    return (
        _msl_coord_for_index(grid, prop, i1),
        _msl_coord_for_index(grid, prop, i2),
        _msl_coord_for_index(grid, prop, i3),
    )


def msl_probe_x_coords_n(
    grid,
    port: MSLPort,
    n_probes: int,
    n_offset_cells: int = 5,
    n_spacing_cells: int = 3,
) -> tuple[float, ...]:
    """Return ``n_probes`` downstream probe x-coordinates (issue #80 Fix C).

    Generalises :func:`msl_probe_x_coords` to N >= 3 probes for the
    N-probe least-squares wave-decomposition extractor. Probe ``n`` sits
    ``n_offset_cells + n * n_spacing_cells`` cells from the feed plane,
    along the propagation direction. Indices are clamped into the valid
    grid range so callers always receive in-domain physical coordinates.
    """
    if n_probes < 3:
        raise ValueError(f"n_probes must be >= 3, got {n_probes}")
    prop, _w, _n, sign_f = msl_axis_roles(port.direction)
    idx = _msl_position_to_index(
        grid, msl_physical_point(port.direction, port.feed_x, port.y_lo, port.z_lo)
    )
    i_feed = int(idx[_MSL_AXIS_INDEX[prop]])
    sign = int(sign_f)
    return tuple(
        _msl_coord_for_index(
            grid, prop, i_feed + sign * (n_offset_cells + n * n_spacing_cells)
        )
        for n in range(n_probes)
    )


# ---------------------------------------------------------------------------
# Closed Ampère-loop current (∮H·dl) for S-parameter extraction
# ---------------------------------------------------------------------------


def msl_loop_current(
    hy_plane,
    hz_plane,
    *,
    j_lo: int,
    j_hi: int,
    k_trace_lo: int,
    k_trace_hi: int,
    dy_arr,
    dz_arr,
    direction: str,
):
    """Longitudinal trace current via the closed Ampere loop ``∮H·dl``.

    The microstrip trace conductor occupies the grid cells
    ``y ∈ [j_lo, j_hi]``, ``z ∈ [k_trace_lo, k_trace_hi]``.  The line
    current ``I(f)`` is the closed contour integral of the transverse H
    field around that conductor in the x-normal probe plane.  By the
    discrete Ampere identity on the Yee grid this contour integral
    equals the longitudinal current threading the trace::

        I = ∮ H·dl
          = + Σ_j  Hy[j, k_trace_lo-1]·dy     (bottom leg, below trace)
            − Σ_j  Hy[j, k_trace_hi  ]·dy     (top leg,    above trace)
            + Σ_k  Hz[j_hi,   k]·dz           (right leg)
            − Σ_k  Hz[j_lo-1, k]·dz           (left leg)

    with ``j`` spanning ``[j_lo, j_hi]`` on the horizontal legs and ``k``
    spanning ``[k_trace_lo, k_trace_hi]`` on the side legs.

    Yee staggering (rfx convention — ``rfx/core/yee.py``): ``Hy[i,j,k]``
    sits at ``z=(k+½)dz``, so ``Hy[·,·,k_trace_lo-1]`` is the edge on the
    bottom face of the trace block and ``Hy[·,·,k_trace_hi]`` the edge on
    the top face.  ``Hz[i,j,k]`` sits at ``y=(j+½)dy``, so
    ``Hz[·,j_hi,·]`` / ``Hz[·,j_lo-1,·]`` are the right / left side edges.

    The pre-issue-#80 current (a bottom-leg-only Hy integral) undercounted
    ``I`` by ~1.5x — it dropped the air-side return Hy and the trace-edge
    Hz — which inflated the de-embedded Z0 to ~74 ohm vs the ~48 ohm
    analytic Hammerstad-Jensen value.  Closing the loop restores Ampere's
    law.

    Parameters
    ----------
    hy_plane, hz_plane : (n_freqs, ny, nz) complex
        x-normal DFT-plane accumulators for Hy and Hz at the probe plane.
    j_lo, j_hi : int
        Trace-conductor y-cell span (inclusive).
    k_trace_lo, k_trace_hi : int
        Trace-conductor z-cell span (inclusive).
    dy_arr, dz_arr : array
        Per-cell sizes along the ``a`` / ``b`` axes (uniform or
        non-uniform).
    direction : str
        Propagation direction, one of :data:`MSL_SUPPORTED_DIRECTIONS`.
        The raw contour integral is negated for POSITIVE-going ports
        (``"+x"``, ``"+y"``) and left as-is for negative-going ones.

        **Sign convention, corrected (see issue #524).** This docstring
        used to claim without qualification that "the returned ``I`` is
        positive for a forward quasi-TEM wave (``Z0 = V/I > 0``)". That
        is true only for a POSITIVE-going port. The contour is traversed
        right-handedly about ``+p_hat`` regardless of the direction sign,
        so the raw integral is the current along ``+p_hat`` for both
        signs; negating it only on positive-going ports leaves a
        negative-going port's ``I`` referenced the opposite way round.
        Measured on the committed thru fixture, each port on its OWN
        drive: ``Re((alpha - gamma)/I1)`` = **+57.52 ohm** for the
        ``"+x"`` port and **-57.56 ohm** for the ``"-x"`` port — same
        magnitude to 0.08 %, opposite sign.

        That is not a defect, it is the #140 convention, and the lane is
        self-consistent about it: ``compute_msl_s_matrix`` multiplies the
        REPORTED ``Z0`` by ``msl_axis_roles(direction)[3]`` so both ports
        report positive, while the wave split ``a = (V + Z0*I)/2``,
        ``b = (V - Z0*I)/2`` consumes the un-normalised current at every
        port, which keeps ``S = B A^-1`` physical (same fixture:
        ``|S11| = |S22|`` to 5 decimals, reciprocity
        ``max ||S21| - |S12|| = 1.27e-05``, column power <= 0.99998).
        The mechanism behind the flip is that ``extract_msl_nprobe`` is
        fed raw physical coordinates, so ``alpha`` is the ``+p_hat``
        wave at BOTH ports and the ``(alpha - gamma)`` numerator swaps
        role with the direction — visible in the same measurement as
        ``|alpha| >> |gamma|`` at both ports on drive 0 and the reverse
        on drive 1.

        Only the docstring was wrong; no code changed for this. The
        other two items on issue #524 — the passive port's ~30 ohm
        termination reading and the 0.194-vs-0.073 drive asymmetry — are
        untouched here and keep that issue open.

    Axis generality (issue #661)
    ----------------------------
    The parameter names above are the ``"+x"``-frame names, but what the
    body actually needs is the right-handed transverse pair ``(a, b)``
    with ``a_hat x b_hat = p_hat`` for propagation axis ``p``
    (:func:`msl_ampere_pair`).  Read the arguments in that frame:

    * ``hy_plane`` / ``hz_plane`` are ``H_a`` / ``H_b``, each indexed
      ``[freq, a_index, b_index]``;
    * ``j_lo``/``j_hi`` is the trace span along ``a``, ``k_trace_lo``/
      ``k_trace_hi`` the span along ``b``;
    * ``dy_arr``/``dz_arr`` are the per-cell sizes along ``a`` / ``b``.

    For ``"+x"`` that is ``(a, b) = (y, z)``, i.e. literally the legacy
    meaning — this function's arithmetic is unchanged.  For ``"+y"`` it
    is ``(a, b) = (z, x)``: ``H_a`` is **Hz** and ``H_b`` is **Hx**, and
    the caller must transpose the recorded y-normal plane (which the DFT
    probe stores as ``[freq, x, z]``) into ``[freq, z, x]``.

    Do not "generalise" this by swapping x and y.  That is a reflection,
    not a rotation: measured on the committed thru fixture's own recorded
    H planes, the cyclic pair reproduces the x-port current to 9.2e-8
    relative while the swap returns exactly ``-I`` (ratio -1.00000009),
    which silently inverts the S-matrix.  See :data:`_MSL_CYCLIC_PAIR`.

    Returns
    -------
    (n_freqs,) complex line current ``I(f)``.
    """
    n_y = int(hy_plane.shape[1])
    n_z = int(hy_plane.shape[2])
    if k_trace_lo < 1 or j_lo < 1 or k_trace_hi >= n_z or j_hi >= n_y:
        raise ValueError(
            "msl_loop_current: the Ampere loop runs one cell outside the "
            "trace block, so the trace needs >=1 cell of margin below it, "
            "above it, and on each side. Got "
            f"j=[{j_lo},{j_hi}], k_trace=[{k_trace_lo},{k_trace_hi}] "
            f"on a ({n_y},{n_z}) y-z plane."
        )
    dy = np.asarray(dy_arr, dtype=float)
    dz = np.asarray(dz_arr, dtype=float)
    js = slice(j_lo, j_hi + 1)
    ks = slice(k_trace_lo, k_trace_hi + 1)

    # Horizontal legs — Hy integrated across the trace width.
    bottom = (hy_plane[:, js, k_trace_lo - 1] * dy[js]).sum(axis=1)
    top = (hy_plane[:, js, k_trace_hi] * dy[js]).sum(axis=1)
    # Vertical legs — Hz integrated over the trace height.
    right = (hz_plane[:, j_hi, ks] * dz[ks]).sum(axis=1)
    left = (hz_plane[:, j_lo - 1, ks] * dz[ks]).sum(axis=1)

    i_loop = bottom - top + right - left
    # Negate for positive-going ports (issue #140 convention, generalised
    # over the axis roles in issue #661): "+x" and "+y" both negate, "-x"
    # and "-y" both do not. Resolving through msl_axis_roles also makes an
    # unsupported direction fail loudly here rather than silently skipping
    # the negation, which is what a bare ``== "+x"`` test would have done
    # for a "+y" port.
    _prop, _w, _n, sign = msl_axis_roles(direction)
    if sign > 0:
        i_loop = -i_loop
    return i_loop
