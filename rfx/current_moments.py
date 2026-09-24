"""The current flowing in a structure, reduced to a few numbers per block
while the solve is still running.

An antenna radiates because current flows in it: on the metal foils, as
polarization and loss current in the dielectric, and at the feed. On the Yee
lattice that current is not modelled and not fitted — Ampere's law is an
identity the solver enforces at every E edge on every timestep, so the total
current through an edge is known exactly from the two arrays the step already
holds::

    J^{n+1/2} = curl_h H^{n+1/2} - eps0 (E^{n+1} - E^n) / dt

On a vacuum edge the right-hand side is the E update rearranged, so it is
exactly zero there; on a conductor edge, where E is held at zero, it is the
discrete curl of H alone; inside the substrate it is the polarization and
loss current; at the feed it is the impressed source current. Nothing about
the structure has to be known to read it.

A block of that current a small fraction of a wavelength across cannot be
resolved by the radiation integral. Only a few numbers of it reach the far
field: the total current moment ``P``, its first spatial moment ``Q`` about
the block's own centre, and its second ``T``. This module sums the edge
currents of a slab into those numbers, block by block, inside the time loop,
and runs a small complex accumulator per frequency::

    acc[f, block, slot] += M[block, slot] * exp(-j w_f (n + 1/2) dt) * dt

The spatial reduction happens FIRST and does not depend on how many
frequencies are asked for; only the small (blocks x 30) vector is multiplied
by the per-frequency phases.

Relation to the post-processing route
-------------------------------------
The same current can be built after the run from recorded frequency-domain
planes (DFT plane probes), in the frequency domain::

    J_dft = curl(H_dft) - j w_tilde eps0 E_dft exp(-j w dt / 2),
    j w_tilde = (2j/dt) sin(w dt / 2)

and drops the field left standing at the end of the record. The time-domain
form here is the same quantity with a different time stamp and that one term
kept. Writing ``A`` for this module's accumulator and ``E^N`` for the final
electric field, the algebra is exact::

    sum_n (E^{n+1} - E^n) e^{-j w (n+1/2) dt}
        = 2j sin(w dt/2) E_tilde + E^N e^{-j w (N+1/2) dt}

so that

    post_J_dft = exp(-j w dt (s - 1/2)) * ( A + eps0 E^N e^{-j w (N+1/2) dt} )

where ``s`` is the time, in units of dt, that the recording runner's plane
probes wrote on the state they sampled — 1 on the uniform lane, 0 on the
graded-mesh lane, which is a measured difference between the two runners and
not a choice (see :data:`UNIFORM_PLANE_STAMP_STEPS`).
:func:`to_post_processing_convention` applies exactly that, and
:func:`end_of_record_moments` supplies the end term from the final state.
The stamp this module uses, ``(n + 1/2) dt``, is the physical one: the
current is a half-step quantity because it sits between the two electric
fields it differences.

Vocabulary: a ``slab`` is the index window of E edges the monitor reads, a
``block`` is one in-plane square of that slab taken through its whole
thickness, ``P``/``Q``/``T`` are the three expansion orders, and a ``slot``
is one of the 30 numbers a block keeps (3 components x 10 weights).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from rfx.core.yee import EPS_0

C0 = 299_792_458.0
MU_0 = 1.25663706212e-6
ETA_0 = float(np.sqrt(MU_0 / EPS_0))

# Weight slots per component. 0 is the plain edge volume (-> P), 1..3 carry
# one displacement factor (-> Q), 4..9 the six independent products of two
# (-> T, symmetric in its two position indices).
_W_P = 1
_W_Q = 4
_W_T = 10
NUMBERS_PER_BLOCK = {0: 3, 1: 12, 2: 30}

# (a, b) position-index pair behind each second-order slot.
_T_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _n_weights(order: int) -> int:
    return {0: _W_P, 1: _W_Q, 2: _W_T}[int(order)]


class CurrentMomentMonitor(NamedTuple):
    """Everything the in-loop reduction needs, built once before the run.

    The index fields are Python ints so every slice in
    :func:`accumulate_current_moments` resolves at trace time; the weights,
    spacings, block map and centres are arrays. Nothing here is read from a
    grid object inside the scan.

    Attributes
    ----------
    i_lo, i_hi, j_lo, j_hi : int
        Half-open in-plane index window of the slab's E edges.
    k_lo, k_hi : int
        Node-plane index range, inclusive. ``ex``/``ey`` live on every node
        plane ``k_lo..k_hi``; ``ez`` on the half planes ``k_lo..k_hi-1``.
    freqs : (n_freqs,) array
    w_ex, w_ey, w_ez : (n_w, ni, nj, nk) arrays
        Edge volume times the displacement products, per weight slot.
    seg : (ni*nj,) int32
        Block index of each in-plane position. Blocks are in-plane only, so
        this does not depend on the plane or on the component.
    centres : (n_blocks, 3) float64 numpy array
        Each block's expansion centre — the unweighted mean of the positions
        of the edges in it. Fixed geometry: it never moves with the
        solution.
    inv_dx_e, inv_dy_e, inv_dz_e : (ni,), (nj,), (nk,) arrays
        The E-update inverse spacings the curl divides by, cropped to the
        slab. On a uniform grid every entry is ``1/dx``.
    curl_signs : tuple of 6 floats
        ``(+1, -1, +1, -1, +1, -1)`` — the sign of each of the six curl
        terms, in the order they appear in the three components. A field
        rather than a literal so the mutation harness can flip one and show
        that the declared checks catch it; production never passes anything
        else.
    half_step : float
        Where in the step the current is stamped, in units of ``dt``. The
        derived value is 0.5: ``J^{n+1/2}`` sits between ``E^n`` and
        ``E^{n+1}``. The mutation harness sets 0.0.
    """

    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    k_lo: int
    k_hi: int
    order: int
    n_blocks: int
    block_cells: int
    freqs: jnp.ndarray
    w_ex: jnp.ndarray
    w_ey: jnp.ndarray
    w_ez: jnp.ndarray
    seg: jnp.ndarray
    centres: np.ndarray
    inv_dx_e: jnp.ndarray
    inv_dy_e: jnp.ndarray
    inv_dz_e: jnp.ndarray
    block_key: np.ndarray
    n_edges: int
    curl_signs: tuple = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    half_step: float = 0.5

    @property
    def n_weights(self) -> int:
        return _n_weights(self.order)

    @property
    def numbers_per_block(self) -> int:
        return 3 * self.n_weights

    @property
    def accumulated_numbers(self) -> int:
        return int(len(self.freqs)) * self.n_blocks * self.numbers_per_block


# ---------------------------------------------------------------------------
# Lattice metrics
# ---------------------------------------------------------------------------

def e_dual_spacings(cells):
    """Distance between the two H cell centres the E update differences.

    ``dual[0] = d[0]``, ``dual[k] = (d[k-1] + d[k]) / 2`` — the reciprocal of
    the E-update inverse spacing (``rfx.nonuniform.e_node_dual_spacings``,
    reproduced here in host numpy so a monitor can be built from plain cell
    arrays without a grid object). It is the side of the dual face the
    conduction current crosses, so it is also what turns a current density
    into a current.
    """
    d = np.asarray(cells, dtype=np.float64)
    dual = np.empty_like(d)
    dual[0] = d[0]
    dual[1:] = 0.5 * (d[:-1] + d[1:])
    return dual


def _axis_arrays(grid, axis: int):
    """``(node_positions, cell_widths, pad_lo, pad_hi, n)`` for one axis.

    The node line comes from the repo's own coordinate producer
    (``rfx.geometry.rasterize_grid.coords_from_nonuniform_grid``) whenever the
    grid carries per-cell arrays, so the frame is bit-for-bit the one the NTFF
    box and the post-processing route use. Recomputing it here instead put the
    nodes 2e-16 m away, which is invisible until a declared board edge lands
    exactly halfway between two nodes — the tutorial patch's 60 mm ground
    plane on a 2 mm mesh does, on both axes — and then the nearest-node search
    lands one cell apart on the two routes.

    A uniform ``Grid`` has no per-cell arrays; there the nodes are
    ``(i - pad_lo) * dx`` and the cells are all ``dx``.
    """
    name = ("x", "y", "z")[axis]
    n = int(grid.shape[axis])
    pad_lo = int(getattr(grid, f"pad_{name}_lo", 0))
    pad_hi = int(getattr(grid, f"pad_{name}_hi", 0))

    arr_name = {0: "dx_arr", 1: "dy_arr", 2: "dz"}[axis]
    exact_name = {0: "dx_arr_f64", 1: "dy_arr_f64", 2: "dz_f64"}[axis]
    widths = getattr(grid, arr_name, None)
    if widths is None:
        cells = np.asarray(grid.cells(axis), dtype=np.float64)[:n]
        nodes = (np.arange(n, dtype=np.float64) - pad_lo) * float(cells[0])
        return nodes, cells, pad_lo, pad_hi, n

    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    coords = coords_from_nonuniform_grid(grid)
    nodes = np.asarray(getattr(coords, name), dtype=np.float64)[:n]
    exact = getattr(grid, exact_name, None)
    if exact is not None and np.asarray(exact).dtype == np.float64:
        cells = np.asarray(exact, dtype=np.float64)
    else:
        cells = np.asarray(widths, dtype=np.float64)
    cells = (cells[:n] if cells.size >= n
             else np.pad(cells, (0, n - cells.size), mode="edge"))
    return nodes, cells, pad_lo, pad_hi, n


# ---------------------------------------------------------------------------
# Building the monitor
# ---------------------------------------------------------------------------

def build_current_moment_monitor(
    *,
    node_x, node_y, node_z,
    cell_x, cell_y, cell_z,
    i_range, j_range, k_node_range,
    block_cells: int,
    freqs,
    order: int = 2,
    off_cells: int = 0,
    pads=None,
    curl_spacings=None,
    volume_spacings=None,
    centres_override=None,
    curl_signs=None,
    half_step: float = 0.5,
    dtype=np.float32,
) -> CurrentMomentMonitor:
    """Assemble the weight arrays and the block map for one slab.

    ``i_range``/``j_range`` are half-open index windows; ``k_node_range`` is
    ``(k_lo, k_hi)`` INCLUSIVE node planes. ``block_cells`` is the in-plane
    block side in cells; ``off_cells`` shifts the partition origin (0 or half
    a block are the two the block rule was measured with).

    ``pads`` is ``(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)``. When given, a slab
    that reaches into the absorber is refused: inside the CPML the E update
    is not Ampere's law, so a current read there is the absorber's fiction
    and not the structure's current.

    ``curl_spacings``, ``volume_spacings``, ``centres_override``,
    ``curl_signs`` and ``half_step`` exist so the mutation harness can put a
    deliberately wrong metric, centre, sign or time stamp in and measure what
    the declared checks then read. The defaults are the derived values and
    are what every result uses.
    """
    order = int(order)
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    node_z = np.asarray(node_z, dtype=np.float64)
    cell_x = np.asarray(cell_x, dtype=np.float64)
    cell_y = np.asarray(cell_y, dtype=np.float64)
    cell_z = np.asarray(cell_z, dtype=np.float64)

    i0, i1 = int(i_range[0]), int(i_range[1])
    j0, j1 = int(j_range[0]), int(j_range[1])
    k0, k1 = int(k_node_range[0]), int(k_node_range[1])
    if i0 < 1 or j0 < 1 or k0 < 1:
        raise ValueError(
            "the slab must start at index 1 or above on every axis: the E "
            "curl reads index-1 and the zero-padded edge of that stencil is "
            "not a backward difference "
            f"(got i_lo={i0}, j_lo={j0}, k_lo={k0})")
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        raise ValueError(
            f"empty slab: i {i0}..{i1}, j {j0}..{j1}, k nodes {k0}..{k1}")
    if int(block_cells) < 1:
        raise ValueError(f"block_cells must be >= 1, got {block_cells}")

    if pads is not None:
        px_lo, px_hi, py_lo, py_hi, pz_lo, pz_hi = (int(v) for v in pads)
        nx, ny, nz = node_x.size, node_y.size, node_z.size
        # The curl reads one index back on each axis, so the window that has
        # to clear the absorber starts one cell below the slab itself.
        for name, lo, hi, n, p_lo, p_hi in (
                ("x", i0 - 1, i1, nx, px_lo, px_hi),
                ("y", j0 - 1, j1, ny, py_lo, py_hi),
                ("z", k0 - 1, k1 + 1, nz, pz_lo, pz_hi)):
            if lo < p_lo or hi > n - p_hi:
                raise ValueError(
                    f"the slab reaches the {name} absorber: the monitor reads "
                    f"[{lo}, {hi}) against pads {p_lo}/{p_hi} of {n} cells. "
                    "Inside the CPML the E update is not Ampere's law, so the "
                    "current read there is not the structure's current.")

    ii = np.arange(i0, i1)
    jj = np.arange(j0, j1)
    kk = np.arange(k0, k1 + 1)          # node planes: ex, ey
    kz = np.arange(k0, k1)              # half planes: ez
    ni, nj, nk, nkz = ii.size, jj.size, kk.size, kz.size

    cs = (curl_spacings if curl_spacings is not None
          else (e_dual_spacings(cell_x), e_dual_spacings(cell_y),
                e_dual_spacings(cell_z)))
    vs = (volume_spacings if volume_spacings is not None
          else (e_dual_spacings(cell_x), e_dual_spacings(cell_y),
                e_dual_spacings(cell_z)))
    inv_dx_e = 1.0 / np.asarray(cs[0], dtype=np.float64)[ii]
    inv_dy_e = 1.0 / np.asarray(cs[1], dtype=np.float64)[jj]
    inv_dz_e = 1.0 / np.asarray(cs[2], dtype=np.float64)[kk]
    dual_x = np.asarray(vs[0], dtype=np.float64)
    dual_y = np.asarray(vs[1], dtype=np.float64)
    dual_z = np.asarray(vs[2], dtype=np.float64)

    # ---- edge positions, the true Yee ones -------------------------------
    # ex(i+1/2, j, k), ey(i, j+1/2, k), ez(i, j, k+1/2).
    px = np.empty((ni, nj, nk, 3))
    px[..., 0] = (node_x[ii] + 0.5 * cell_x[ii])[:, None, None]
    px[..., 1] = node_y[jj][None, :, None]
    px[..., 2] = node_z[kk][None, None, :]
    py = np.empty((ni, nj, nk, 3))
    py[..., 0] = node_x[ii][:, None, None]
    py[..., 1] = (node_y[jj] + 0.5 * cell_y[jj])[None, :, None]
    py[..., 2] = node_z[kk][None, None, :]
    pz = np.empty((ni, nj, nkz, 3))
    pz[..., 0] = node_x[ii][:, None, None]
    pz[..., 1] = node_y[jj][None, :, None]
    pz[..., 2] = (node_z[kz] + 0.5 * cell_z[kz])[None, None, :]

    # ---- edge volumes: dual x dual x primal, axis by axis -----------------
    vx = (cell_x[ii][:, None, None] * dual_y[jj][None, :, None]
          * dual_z[kk][None, None, :]) * np.ones((ni, nj, nk))
    vy = (dual_x[ii][:, None, None] * cell_y[jj][None, :, None]
          * dual_z[kk][None, None, :]) * np.ones((ni, nj, nk))
    vz = (dual_x[ii][:, None, None] * dual_y[jj][None, :, None]
          * cell_z[kz][None, None, :]) * np.ones((ni, nj, nkz))

    # ---- block map: in-plane only, whole slab thickness in one block ------
    d = int(block_cells)
    off = int(off_cells)
    bi = (ii - i0 + off) // d
    bj = (jj - j0 + off) // d
    key = np.stack(np.broadcast_arrays(bi[:, None], bj[None, :]), axis=-1)
    key2 = key.reshape(-1, 2)
    uniq, seg = np.unique(key2, axis=0, return_inverse=True)
    seg = np.asarray(seg, dtype=np.int64).ravel()
    n_blocks = int(uniq.shape[0])

    # ---- centres: unweighted mean of the positions of the edges in the
    # block, over every component and every plane.
    if centres_override is not None:
        centres = np.asarray(centres_override, dtype=np.float64)
        if centres.shape != (n_blocks, 3):
            raise ValueError(
                f"centres_override must be ({n_blocks}, 3), got {centres.shape}")
    else:
        sums = np.zeros((n_blocks, 3))
        counts = np.zeros(n_blocks)
        for pos, nkc in ((px, nk), (py, nk), (pz, nkz)):
            flat = pos.reshape(ni * nj, nkc, 3).sum(axis=1)
            for a in range(3):
                sums[:, a] += np.bincount(seg, weights=flat[:, a],
                                          minlength=n_blocks)
            counts += np.bincount(seg, weights=np.full(ni * nj, float(nkc)),
                                  minlength=n_blocks)
        centres = sums / counts[:, None]

    # ---- weights: volume times the displacement products ------------------
    n_w = _n_weights(order)

    def _weights(pos, vol, nkc):
        delta = pos - centres[seg].reshape(ni, nj, 1, 3)
        w = np.empty((n_w, ni, nj, nkc))
        w[0] = vol
        if order >= 1:
            for a in range(3):
                w[1 + a] = delta[..., a] * vol
        if order >= 2:
            for s, (a, b) in enumerate(_T_PAIRS):
                w[4 + s] = delta[..., a] * delta[..., b] * vol
        return w

    w_ex = _weights(px, vx, nk)
    w_ey = _weights(py, vy, nk)
    w_ez = _weights(pz, vz, nkz)

    signs = tuple(float(s) for s in (curl_signs if curl_signs is not None
                                     else (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)))
    if len(signs) != 6:
        raise ValueError(f"curl_signs must have 6 entries, got {len(signs)}")

    return CurrentMomentMonitor(
        i_lo=i0, i_hi=i1, j_lo=j0, j_hi=j1, k_lo=k0, k_hi=k1,
        order=order, n_blocks=n_blocks, block_cells=d,
        freqs=jnp.asarray(np.asarray(freqs, dtype=np.float64)),
        w_ex=jnp.asarray(w_ex, dtype=dtype),
        w_ey=jnp.asarray(w_ey, dtype=dtype),
        w_ez=jnp.asarray(w_ez, dtype=dtype),
        seg=jnp.asarray(seg, dtype=jnp.int32),
        # Host float64: the centres never enter the scan (they are already
        # baked into the weights), and a float32 copy moved a block centre by
        # 4e-10 m against the post-processing route's own.
        centres=np.asarray(centres, dtype=np.float64),
        inv_dx_e=jnp.asarray(inv_dx_e, dtype=dtype),
        inv_dy_e=jnp.asarray(inv_dy_e, dtype=dtype),
        inv_dz_e=jnp.asarray(inv_dz_e, dtype=dtype),
        block_key=uniq,
        n_edges=int(ni * nj * (2 * nk + nkz)),
        curl_signs=signs,
        half_step=float(half_step),
    )


def _index_window(nodes, lo_m, hi_m, n_axis, margin_cells=0):
    """Nearest node index to each corner, plus a cell margin, inclusive.

    ``np.argmin`` at BOTH ends, so a corner that lands exactly halfway
    between two nodes resolves to the lower one at either end. Breaking that
    tie outwards instead is defensible on its own and WOULD put the window
    one cell wider on an exact tie at a high corner. On the tutorial patch
    the exact ties are at the low corners and the high corners are strict
    minima by 2.8e-17 m, so the slab there is 25,920 edges at 2 mm and
    84,500 at 1 mm.

    This function does not clamp to ``[1, n_axis - 1]``; the builder below
    refuses such a window instead of moving it.

    Away from an exact tie what decides the window is that ``nodes`` comes
    from ``coords_from_nonuniform_grid`` (see :func:`_axis_arrays`) rather
    than from a second cumsum here. A node line recomputed locally landed
    2e-16 m away, which turned the tutorial patch's halfway board edge into
    a strict minimum on the other side and moved the slab by a cell — 24,500
    edges instead of 25,920.
    """
    nodes = np.asarray(nodes)[:n_axis]
    lo = int(np.argmin(np.abs(nodes - float(lo_m))))
    hi = int(np.argmin(np.abs(nodes - float(hi_m))))
    return lo - int(margin_cells), hi + int(margin_cells)


def current_moment_monitor_from_grid(
    grid,
    *,
    corner_lo,
    corner_hi,
    block_size: float,
    freqs,
    order: int = 2,
    margin_cells=0,
    off_cells: int = 0,
    periodic=None,
    **kwargs,
) -> CurrentMomentMonitor:
    """Build the monitor from a grid and two corners in metres.

    ``dtype`` is the storage type of the weights and the spacings. It has to
    follow the FIELD dtype, not be pinned: the weights multiply the current
    inside the reduction, so float32 weights quantize the block moments at
    float32 no matter how wide the accumulator is.

    ``margin_cells`` is a scalar or a per-axis triple of extra cells around
    the declared corners. ``block_size`` is the in-plane block side in metres; it is rounded to a
    whole number of cells on the x axis and the realized side is what the
    monitor reports, never the declared one. ``corner_lo``/``corner_hi`` are
    in the same frame as ``add_ntff_box``: physical zero at the inner edge of
    the lo-face absorber.

    Refuses what it has not been shown to be right on.
    """
    if periodic is not None and any(bool(p) for p in periodic):
        raise NotImplementedError(
            "the current-moment monitor reads Ampere's law at the slab's own "
            "edges; on a periodic or Bloch axis the curl stencil wraps and "
            "carries a phase, which this reduction does not apply. Use a "
            "non-periodic run.")
    nodes = []
    cells = []
    pads = []
    for axis in range(3):
        nd, cl, p_lo, p_hi, _n = _axis_arrays(grid, axis)
        nodes.append(nd)
        cells.append(cl)
        pads.extend([p_lo, p_hi])
    pads = (pads[0], pads[1], pads[2], pads[3], pads[4], pads[5])

    nx, ny, nz = (int(v) for v in grid.shape)
    mx, my, mz = ((margin_cells,) * 3 if np.isscalar(margin_cells)
                  else tuple(int(v) for v in margin_cells))
    i0, i1 = _index_window(nodes[0], corner_lo[0], corner_hi[0], nx, mx)
    j0, j1 = _index_window(nodes[1], corner_lo[1], corner_hi[1], ny, my)
    k0, k1 = _index_window(nodes[2], corner_lo[2], corner_hi[2], nz, mz)

    # The window has to exist inside the array before anything reads a cell
    # width out of it: a margin that pushes an index below 1 used to die in
    # numpy ("zero-size array to reduction operation maximum") instead of here.
    for name, lo_i, hi_i, n_ax in (("x", i0, i1, nx), ("y", j0, j1, ny),
                                   ("z", k0, k1, nz)):
        if lo_i < 1 or hi_i > n_ax - 1 or hi_i <= lo_i:
            raise ValueError(
                f"the current-moment slab's {name} window [{lo_i}, {hi_i}] does "
                f"not fit in a {n_ax}-node axis: the slab must start at index 1 "
                "or above and end at least one node inside the array (the curl "
                "reads one H sample back). Shrink the slab or the margin.")

    # The block side is a whole number of cells, and the in-plane mesh has to
    # be uniform for one number to describe it: the block map is an index
    # rule, so a graded x or y axis would make blocks of different physical
    # size carry one declared side, and a dx != dy mesh would make them
    # rectangles reported as a square.
    cx = np.asarray(cells[0][i0:i1 + 1], dtype=np.float64)
    cy = np.asarray(cells[1][j0:j1 + 1], dtype=np.float64)
    in_plane = np.concatenate([cx, cy])
    if float(in_plane.max() - in_plane.min()) > 1e-12 * float(in_plane.max()):
        raise NotImplementedError(
            "the current-moment monitor's block side is an index rule, so it "
            "needs a uniform in-plane mesh with dx == dy over the slab. This "
            f"slab spans x cells {cx.min():.6g}..{cx.max():.6g} m and y cells "
            f"{cy.min():.6g}..{cy.max():.6g} m. The z axis may be graded. "
            "Build the monitor with build_current_moment_monitor() and an "
            "explicit index window if a graded in-plane block rule is wanted.")
    d_cells = max(1, int(round(float(block_size) / float(cells[0][i0]))))

    return build_current_moment_monitor(
        node_x=nodes[0], node_y=nodes[1], node_z=nodes[2],
        cell_x=cells[0], cell_y=cells[1], cell_z=cells[2],
        i_range=(i0, i1 + 1), j_range=(j0, j1 + 1), k_node_range=(k0, k1),
        block_cells=d_cells, freqs=freqs, order=order, off_cells=off_cells,
        pads=pads, **kwargs)


# ---------------------------------------------------------------------------
# The scan-body half
# ---------------------------------------------------------------------------

def current_moment_accum_dtype(field_dtype=jnp.float32):
    """Complex dtype for the accumulator, given the field dtype.

    Same policy as ``rfx.farfield.ntff_accum_dtype`` and for the same reason:
    the carry has to close under ``jax_enable_x64``, and a float16 field must
    not drag a recursive accumulator into float16.
    """
    return jnp.result_type(
        jnp.promote_types(jnp.dtype(field_dtype), jnp.float32), jnp.complex64)


def init_current_moment_data(monitor: CurrentMomentMonitor, *,
                             field_dtype=jnp.float32):
    """Zeroed ``(accumulator, kahan_compensation)``.

    The accumulator is ``(n_freqs, n_blocks, 3, n_weights)``. Kahan
    compensation costs one array of the same (small) shape and keeps the
    running DFT near float64 precision in complex64 over tens of thousands of
    steps — the same reason ``init_ntff_data`` carries one.
    """
    cdtype = current_moment_accum_dtype(field_dtype)
    shape = (int(len(monitor.freqs)), int(monitor.n_blocks), 3,
             monitor.n_weights)
    return (jnp.zeros(shape, dtype=cdtype), jnp.zeros(shape, dtype=cdtype))


def slab_e_snapshot(state, monitor: CurrentMomentMonitor):
    """``E^n`` on the slab only — the one extra array the monitor stores.

    Called at the top of the step, before the E update, where the state still
    holds the electric field at ``n dt``. Slab-sized rather than full-grid so
    the reverse-mode tape carries the slab and not the domain.
    """
    m = monitor
    return (
        state.ex[m.i_lo:m.i_hi, m.j_lo:m.j_hi, m.k_lo:m.k_hi + 1],
        state.ey[m.i_lo:m.i_hi, m.j_lo:m.j_hi, m.k_lo:m.k_hi + 1],
        state.ez[m.i_lo:m.i_hi, m.j_lo:m.j_hi, m.k_lo:m.k_hi],
    )


def slab_curl_h(state, monitor: CurrentMomentMonitor):
    """``curl_h H`` at the slab's E edges, the same stencil the E update uses.

    Backward staggered differences divided by the E-update spacing — the
    slab-local spelling of ``rfx.core.yee.curl_h`` (uniform) and
    ``curl_h_nu`` (graded). No index-0 case appears because the builder
    refuses a slab that starts at index 0.
    """
    m = monitor
    i0, i1, j0, j1, k0, k1 = m.i_lo, m.i_hi, m.j_lo, m.j_hi, m.k_lo, m.k_hi
    s = m.curl_signs
    hx, hy, hz = state.hx, state.hy, state.hz
    ix = m.inv_dx_e[:, None, None]
    iy = m.inv_dy_e[None, :, None]
    iz = m.inv_dz_e[None, None, :]

    # ex(i+1/2, j, k): dHz/dy - dHy/dz
    curl_x = (
        s[0] * (hz[i0:i1, j0:j1, k0:k1 + 1] - hz[i0:i1, j0 - 1:j1 - 1, k0:k1 + 1]) * iy
        + s[1] * (hy[i0:i1, j0:j1, k0:k1 + 1] - hy[i0:i1, j0:j1, k0 - 1:k1]) * iz
    )
    # ey(i, j+1/2, k): dHx/dz - dHz/dx
    curl_y = (
        s[2] * (hx[i0:i1, j0:j1, k0:k1 + 1] - hx[i0:i1, j0:j1, k0 - 1:k1]) * iz
        + s[3] * (hz[i0:i1, j0:j1, k0:k1 + 1] - hz[i0 - 1:i1 - 1, j0:j1, k0:k1 + 1]) * ix
    )
    # ez(i, j, k+1/2): dHy/dx - dHx/dy
    curl_z = (
        s[4] * (hy[i0:i1, j0:j1, k0:k1] - hy[i0 - 1:i1 - 1, j0:j1, k0:k1]) * ix
        + s[5] * (hx[i0:i1, j0:j1, k0:k1] - hx[i0:i1, j0 - 1:j1 - 1, k0:k1]) * iy
    )
    return curl_x, curl_y, curl_z


def _reduce_to_blocks(monitor: CurrentMomentMonitor, jx, jy, jz):
    """``(n_blocks, 3, n_w)`` — the spatial reduction, frequency-independent.

    Contract over the plane index first, then segment-sum the in-plane
    positions into their blocks. This step does not grow with the number of
    frequencies, because no frequency appears in it.
    """
    m = monitor
    out = []
    for w, j in ((m.w_ex, jx), (m.w_ey, jy), (m.w_ez, jz)):
        per_ij = jnp.einsum("wijk,ijk->ijw", w, j.astype(w.dtype))
        out.append(jax.ops.segment_sum(
            per_ij.reshape(-1, w.shape[0]), m.seg,
            num_segments=m.n_blocks, indices_are_sorted=False))
    return jnp.stack(out, axis=1)          # (n_blocks, 3, n_w)


def accumulate_current_moments(data, state, e_prev_slab,
                               monitor: CurrentMomentMonitor, dt, step_idx):
    """One timestep of the block-moment DFT.

    ``state`` is the post-E-update, post-source-injection state — E at
    ``(n+1) dt`` and H at ``(n+1/2) dt``, the same slot the NTFF box
    accumulates at. ``e_prev_slab`` is :func:`slab_e_snapshot` taken at the
    top of the same step, so the difference is exactly one timestep of the
    electric field and the current it gives is the half-step current.

    The source injection happens before this slot, so the feed's impressed
    current is part of ``J`` and not something added afterwards.
    """
    m = monitor
    acc, comp = data
    cdtype = acc.dtype
    rdtype = jnp.finfo(cdtype).dtype

    curl_x, curl_y, curl_z = slab_curl_h(state, m)
    ex_new, ey_new, ez_new = slab_e_snapshot(state, m)
    ex_old, ey_old, ez_old = e_prev_slab
    _fd = curl_x.dtype
    coef = jnp.asarray(EPS_0, dtype=_fd) / jnp.asarray(dt, dtype=_fd)
    jx = curl_x - coef * (ex_new.astype(_fd) - ex_old.astype(_fd))
    jy = curl_y - coef * (ey_new.astype(_fd) - ey_old.astype(_fd))
    jz = curl_z - coef * (ez_new.astype(_fd) - ez_old.astype(_fd))

    moments = _reduce_to_blocks(m, jx, jy, jz).astype(cdtype)

    _dt = jnp.asarray(dt, dtype=rdtype)
    t = (jnp.asarray(step_idx, dtype=rdtype)
         + jnp.asarray(m.half_step, dtype=rdtype)) * _dt
    omega = jnp.asarray(2.0 * jnp.pi, dtype=rdtype) * jnp.asarray(
        m.freqs, dtype=rdtype)
    phase = jnp.exp(jnp.asarray(-1j, dtype=cdtype) * omega * t) * _dt

    val = moments[None, :, :, :] * phase[:, None, None, None]
    # Kahan compensated summation, as in ``accumulate_ntff``.
    y = val - comp
    total = acc + y
    comp_new = (total - acc) - y
    return (total, comp_new)


# ---------------------------------------------------------------------------
# Reading the accumulator
# ---------------------------------------------------------------------------

def moments_to_PQT(accumulator, monitor: CurrentMomentMonitor):
    """Split the slots into ``(P, Q, T)``, shaped as the far field wants them.

    ``P`` is ``(..., n_blocks, 3)`` with the component last, ``Q``
    ``(..., n_blocks, 3, 3)`` indexed ``[a, c]`` (position axis first,
    component last) and ``T`` ``(..., n_blocks, 3, 3, 3)`` indexed
    ``[a, b, c]`` and symmetric in ``a, b`` — the index order
    :func:`block_far_field_np` consumes.
    """
    a = np.asarray(accumulator)
    order = monitor.order
    P = a[..., 0]
    Q = T = None
    if order >= 1:
        Q = np.moveaxis(a[..., 1:4], -1, -2)
    if order >= 2:
        T = np.zeros(a.shape[:-2] + (3, 3, 3), dtype=a.dtype)
        for s, (i, j) in enumerate(_T_PAIRS):
            val = a[..., 4 + s]                 # (..., n_blocks, 3)
            T[..., i, j, :] = val
            T[..., j, i, :] = val
    return P, Q, T


def moments_to_PQT_jax(accumulator, monitor: CurrentMomentMonitor):
    """:func:`moments_to_PQT` without leaving the trace.

    The host version writes T with in-place assignment, which a tracer
    cannot carry. This one stacks instead, so a differentiable objective can
    take a far field from the accumulator. Same index order.
    """
    a = jnp.asarray(accumulator)
    order = monitor.order
    P = a[..., 0]
    Q = T = None
    if order >= 1:
        Q = jnp.moveaxis(a[..., 1:4], -1, -2)
    if order >= 2:
        slot = {}
        for s, (i, j) in enumerate(_T_PAIRS):
            slot[(i, j)] = a[..., 4 + s]
            slot[(j, i)] = a[..., 4 + s]
        T = jnp.stack([jnp.stack([slot[(i, j)] for j in range(3)], axis=-2)
                       for i in range(3)], axis=-3)
    return P, Q, T


def end_of_record_moments(monitor: CurrentMomentMonitor, state, n_steps, dt):
    """The one term a finite record leaves behind, as block moments.

    The post-processing route's frequency-domain current drops
    ``eps0 E^N exp(-j w (N + 1/2) dt)`` — the electric field still standing
    when the record stopped. The time-domain accumulator keeps it, so this is
    what has to be added to the post-processed moments (not subtracted from
    these) before the two are compared.
    """
    m = monitor
    ex, ey, ez = slab_e_snapshot(state, m)
    mom = np.asarray(_reduce_to_blocks(
        m, EPS_0 * ex, EPS_0 * ey, EPS_0 * ez), dtype=np.complex128)
    freqs = np.asarray(m.freqs, dtype=np.float64)
    ph = np.exp(-2j * np.pi * freqs * (float(n_steps) + 0.5) * float(dt))
    return mom[None, :, :, :] * ph[:, None, None, None]


# Where each runner's DFT plane probe stamps the state it samples, in units
# of dt, counting the scan step as n. Both runners sample the SAME state (E at
# (n+1)dt, H at (n+1/2)dt) at the same slot, and then stamp it differently:
#
#   rfx/simulation.py      t = st.step * dt   -> (n+1) dt   (st.step is n+1
#                                                 after update_e)
#   rfx/nonuniform.py      t = step_idx * dt  ->  n dt
#
# One full timestep apart, measured 2026-09-21 on the patch fixture: the block
# moments extracted from the two disagreed by exactly |exp(-j w dt) - 1|
# (1.527e-2 at 2.0 GHz, 1.832e-2 at 2.4, 2.137e-2 at 2.8, against w*dt =
# 1.527e-2 / 1.832e-2 / 2.137e-2). It cancels out of a magnitude spectrum,
# which is why it has gone unnoticed, but any quantity that combines E with H
# carries it. The constants below let a caller say which record it is holding
# rather than guess.
UNIFORM_PLANE_STAMP_STEPS = 1.0
NONUNIFORM_PLANE_STAMP_STEPS = 0.0


def plane_stamp_steps(grid) -> float:
    """Which of the two stamps the DFT plane probes of this grid's runner use."""
    return (NONUNIFORM_PLANE_STAMP_STEPS if getattr(grid, "dx_arr", None)
            is not None else UNIFORM_PLANE_STAMP_STEPS)


def to_post_processing_convention(accumulator, end_term, dt, freqs, *,
                                  plane_stamp_steps=UNIFORM_PLANE_STAMP_STEPS):
    """This module's convention mapped onto the DFT-plane route's.

    The two differ by where the current is stamped and by the end-of-record
    term, and by nothing else; both are exact algebra, not corrections. The
    current here sits at ``(n + 1/2) dt`` because it is the difference of two
    electric fields; the post-processing route inherits whatever time its
    plane probes wrote on the state, so::

        post_J = exp(-j w dt (s - 1/2)) * (A + eps0 E^N e^{-j w (N+1/2) dt})

    with ``s = plane_stamp_steps``. Both terms of the post-processing formula
    carry the same factor, so a run recorded with the other stamp is only
    globally phased — nothing inside it is mis-registered.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    s = float(plane_stamp_steps)
    factor = np.exp(-2j * np.pi * freqs * float(dt) * (s - 0.5))
    a = np.asarray(accumulator, dtype=np.complex128)
    e = (0.0 if end_term is None
         else np.asarray(end_term, dtype=np.complex128))
    return (a + e) * factor[:, None, None, None]


def _direction_frame_np(theta, phi):
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    st, ct = np.sin(TH), np.cos(TH)
    sp, cp = np.sin(PH), np.cos(PH)
    r_hat = np.stack([st * cp, st * sp, ct], axis=-1).reshape(-1, 3)
    th_hat = np.stack([ct * cp, ct * sp, -st], axis=-1).reshape(-1, 3)
    ph_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1).reshape(-1, 3)
    return r_hat, th_hat, ph_hat, TH.shape


def block_far_field_jax(theta, phi, centres, P, Q, T, k, order, eta=ETA_0):
    """The far field of a set of blocks, in JAX, for the differentiable arm.

    Each block contributes ``exp(+jk rhat.c) [P + jk (rhat.Q)
    - (k^2/2)(rhat rhat : T)]`` to the electric radiation vector — the Taylor
    expansion of ``exp(+jk rhat.r)`` about the block's own centre. The same
    expression as :func:`block_far_field_np`; the tests pin the two together
    rather than trusting that two spellings agree.
    """
    r_hat, th_hat, ph_hat, shape = _direction_frame_np(theta, phi)
    r_hat = jnp.asarray(r_hat)
    th_hat = jnp.asarray(th_hat)
    ph_hat = jnp.asarray(ph_hat)
    centres = jnp.asarray(centres)
    ph = jnp.exp(1j * k * (r_hat @ centres.T))          # (n_dir, n_blocks)
    N = ph @ jnp.asarray(P)
    if order >= 1 and Q is not None:
        Qj = jnp.asarray(Q)
        for a in range(3):
            N = N + 1j * k * r_hat[:, a, None] * (ph @ Qj[:, a, :])
    if order >= 2 and T is not None:
        Tj = jnp.asarray(T)
        for a in range(3):
            for b in range(3):
                N = N + (-(k ** 2) / 2.0) * (
                    r_hat[:, a, None] * r_hat[:, b, None] * (ph @ Tj[:, a, b, :]))
    jk4pi = 1j * k / (4.0 * np.pi)
    e_th = -jk4pi * eta * jnp.sum(N * th_hat, axis=1)
    e_ph = -jk4pi * eta * jnp.sum(N * ph_hat, axis=1)
    return e_th.reshape(shape), e_ph.reshape(shape)


def block_far_field_np(theta, phi, centres, P, Q, T, k, order, eta=ETA_0):
    """Host twin of :func:`block_far_field_jax`, same expression."""
    r_hat, th_hat, ph_hat, shape = _direction_frame_np(theta, phi)
    centres = np.asarray(centres, dtype=np.float64)
    ph = np.exp(1j * k * (r_hat @ centres.T))
    N = ph @ np.asarray(P, dtype=np.complex128)
    if order >= 1 and Q is not None:
        Q = np.asarray(Q, dtype=np.complex128)
        for a in range(3):
            N += 1j * k * r_hat[:, a, None] * (ph @ Q[:, a, :])
    if order >= 2 and T is not None:
        T = np.asarray(T, dtype=np.complex128)
        for a in range(3):
            for b in range(3):
                N += (-(k ** 2) / 2.0) * (r_hat[:, a, None]
                                          * r_hat[:, b, None] * (ph @ T[:, a, b, :]))
    jk4pi = 1j * k / (4.0 * np.pi)
    e_th = -jk4pi * eta * np.sum(N * th_hat, axis=1)
    e_ph = -jk4pi * eta * np.sum(N * ph_hat, axis=1)
    return e_th.reshape(shape), e_ph.reshape(shape)


def far_field_all_freqs(accumulator, monitor: CurrentMomentMonitor,
                        theta, phi, *, backend="numpy"):
    """The pattern at every monitored frequency, shaped ``(nf, nth, nph)``."""
    P, Q, T = moments_to_PQT(accumulator, monitor)
    freqs = np.asarray(monitor.freqs, dtype=np.float64)
    centres = np.asarray(monitor.centres)
    fn = block_far_field_np if backend == "numpy" else block_far_field_jax
    th_out, ph_out = [], []
    for f_i, f in enumerate(freqs):
        k = 2.0 * np.pi * float(f) / C0
        a, b = fn(theta, phi, centres, P[f_i],
                  None if Q is None else Q[f_i],
                  None if T is None else T[f_i], k, monitor.order)
        th_out.append(np.asarray(a))
        ph_out.append(np.asarray(b))
    return np.stack(th_out), np.stack(ph_out)


def current_moment_far_field(result, theta, phi):
    """The radiation pattern of the block current moments a run accumulated.

    ``result`` is what ``Simulation.run()`` or ``Simulation.forward()``
    returned for a simulation that declared
    :meth:`~rfx.Simulation.add_current_moment_monitor`. Returns the same
    :class:`rfx.farfield.FarFieldResult` that :func:`rfx.compute_far_field`
    returns for an NTFF box, in the same convention (``E_theta``, ``E_phi``
    omitting the ``1/r`` factor, ``exp(+jk rhat.r)`` phase), so
    :func:`rfx.directivity`, :func:`rfx.radiation_pattern` and
    :func:`rfx.axial_ratio` take it unchanged. The pattern is evaluated at
    every monitored frequency.

    Inside ``jax.grad`` or ``jax.jit`` (the accumulator is a tracer) the
    pattern is built with ``jnp`` throughout and stays differentiable, as
    :func:`rfx.compute_far_field` dispatches to its JAX twin.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.farfield import FarFieldResult

    if isinstance(result, dict):
        data = result.get("current_moment_data")
        monitor = result.get("current_moment_monitor")
    else:
        data = getattr(result, "current_moment_data", None)
        monitor = getattr(result, "current_moment_monitor", None)
    if data is None or monitor is None:
        raise ValueError(
            "this result carries no block current moments; declare them with "
            "Simulation.add_current_moment_monitor() before run()/forward().")
    acc = data[0] if isinstance(data, (tuple, list)) else data

    if not is_tracer(acc):
        e_th, e_ph = far_field_all_freqs(acc, monitor, theta, phi)
        return FarFieldResult(E_theta=e_th, E_phi=e_ph,
                              theta=np.asarray(theta, dtype=np.float64),
                              phi=np.asarray(phi, dtype=np.float64),
                              freqs=np.asarray(monitor.freqs,
                                               dtype=np.float64))

    # Under an outer jax.jit the monitor was built inside the trace, so its
    # frequencies are a tracer too: k stays a jnp scalar here.
    freqs = jnp.asarray(monitor.freqs)
    P, Q, T = moments_to_PQT_jax(acc, monitor)
    th_out, ph_out = [], []
    for f_i in range(int(freqs.shape[0])):
        k = 2.0 * jnp.pi * freqs[f_i] / C0
        a, b = block_far_field_jax(theta, phi, monitor.centres, P[f_i],
                                   None if Q is None else Q[f_i],
                                   None if T is None else T[f_i], k,
                                   monitor.order)
        th_out.append(a)
        ph_out.append(b)
    return FarFieldResult(E_theta=jnp.stack(th_out), E_phi=jnp.stack(ph_out),
                          theta=theta, phi=phi, freqs=freqs)


def weight_dtype_for(sim):
    """Storage type for the monitor's weights, from the run's precision.

    Measured 2026-09-22: with ``precision="float64"`` the block
    accumulator is complex128 while these weights stayed float32, and the
    reduction they multiply quantized the moments at float32. The far field
    taken from them then matched a float64 central difference only to 1e-6 to
    2e-4, with the difference ladder's plateau stuck at a step of 1e-3, while
    the Huygens box on the same runs reached 1e-9 with a plateau at 1e-5. The
    default is unchanged for a float32 run, so those stay bit-identical.
    """
    return (np.float64
            if getattr(sim, "_precision", "float32") == "float64"
            else np.float32)


def monitor_for_simulation(sim, grid, periodic=None):
    """The monitor a ``Simulation`` declared, realized against the built grid.

    ``Simulation.add_current_moment_monitor`` only records the declaration;
    the slab window, the block map and the centres come from the grid the
    solve actually builds, so they are realized here — the repo's
    assert-realized-not-declared rule. Returns ``None`` when nothing was
    declared, which is what keeps every existing run unchanged.
    """
    spec = getattr(sim, "_current_moments", None)
    if spec is None:
        return None
    corner_lo, corner_hi, block_size, freqs, order, extra = spec
    from rfx.core.jax_utils import is_tracer
    for _name in ("dx_arr", "dy_arr", "dz"):
        if is_tracer(getattr(grid, _name, None)):
            raise NotImplementedError(
                "the current-moment monitor needs concrete cell sizes: the "
                "edge positions, volumes and block centres are baked into "
                "fixed weights before the run, and a traced mesh profile "
                "(mesh-as-design-variable) would move them. Differentiate "
                "the materials, not the mesh, while this monitor is on.")
    if int(getattr(sim, "_stencil_order", 2)) != 2:
        raise NotImplementedError(
            "the current-moment monitor reads Ampere's law with the "
            "second-order curl; with stencil_order=4 the E update differences "
            "H over the wider fourth-order stencil, so the current read at a "
            "vacuum edge would not be zero. Use stencil_order=2 while this "
            "monitor is on.")
    if getattr(sim, "_tfsf", None) is not None:
        raise NotImplementedError(
            "the current-moment monitor and a TFSF source are not supported "
            "together: inside the total-field region the E update carries the "
            "incident-field correction, so the current read there would "
            "include the injection and not just the structure's current.")
    return current_moment_monitor_from_grid(
        grid, corner_lo=corner_lo, corner_hi=corner_hi,
        block_size=block_size, freqs=freqs, order=order,
        periodic=periodic, dtype=weight_dtype_for(sim), **(extra or {}))


# Lanes whose scan body accumulates the monitor. Every other entry point has
# to say so rather than return a result with the field quietly set to None.
SUPPORTED_LANES = ("uniform run()/run_until_decay", "graded-mesh run()",
                   "uniform forward()", "graded-mesh forward()")


def require_accumulated_current_moments(sim, result, context: str) -> None:
    """A declared monitor must come back filled, whatever lane ran.

    The per-lane refusals (:func:`refuse_current_moment_monitor`) name the
    lanes known not to accumulate the monitor, and a list like that is only as
    complete as the last person to add a runner made it: the ADI lane was
    missing from it until a reviewer ran one. This is the check that does not
    need the list. It sits where ``run()`` and ``forward()`` hand their result
    back, and refuses a result whose ``current_moment_data`` is empty although
    a monitor was declared. It costs the whole run before it fires, so the
    per-lane refusals stay as the early, cheap answer; this one guarantees
    that the silent case cannot exist.
    """
    if getattr(sim, "_current_moments", None) is None:
        return
    if not hasattr(result, "current_moment_data"):
        return  # e.g. the all-port S-parameter dict of forward()
    if result.current_moment_data is None:
        raise NotImplementedError(
            f"add_current_moment_monitor() was declared, but the lane that "
            f"{context}() took returned no block current moments. That scan "
            "body does not accumulate them; an empty field handed back "
            "silently is worse than a refusal. Supported: "
            + ", ".join(SUPPORTED_LANES) + ".")


def refuse_current_moment_monitor(sim, lane: str) -> None:
    """Refuse a declared monitor on a lane that would not accumulate it.

    The monitor is read out of the simulation in exactly three places
    (``rfx/runners/uniform.py``, ``rfx/runners/nonuniform.py``,
    ``rfx/api/_execute.py``). Every other runner steps its own scan body and
    never looks, so a declaration reaching one of those used to produce a
    finished result whose ``current_moment_data`` was ``None`` with nothing
    said. Same shape as the distributed lane's refusal of
    ``add_flux_monitor()``.
    """
    if getattr(sim, "_current_moments", None) is None:
        return
    raise NotImplementedError(
        f"add_current_moment_monitor() is not supported on the {lane}; that "
        "scan body does not accumulate block current moments, and a result "
        "with the field silently empty is worse than a refusal. Supported: "
        + ", ".join(SUPPORTED_LANES) + ".")
