"""Far-field transforms of near-to-far-field surface records."""

from __future__ import annotations

from rfx._precision import HIGHEST
from rfx.farfield import ETA_0
from rfx.farfield import FarFieldResult
from rfx.farfield import NTFFBox
from rfx.farfield import NTFFData
from rfx.farfield import _AXIS_NAMES
from rfx.farfield import _face_positions
from rfx.farfield import _ntff_face_pads
from rfx.farfield import _surface_currents
from rfx.grid import C0
from rfx.grid import Grid
import jax.numpy as jnp
import numpy as np


def compute_far_field(
    ntff_data: NTFFData,
    box: NTFFBox,
    grid: Grid,
    theta: np.ndarray,
    phi: np.ndarray,
) -> FarFieldResult:
    """Compute far-field radiation pattern from NTFF DFT data.

    Every face cell contributes one sample of the equivalent surface
    currents J = n x H and M = -n x E, weighted by the cell's area and
    phased by its position. Where that sample is taken is what decides the
    order of the rule: with ``box.collocation == "face_centre"`` it sits at
    the centre of the cell, matching what ``accumulate_ntff`` stored there,
    and the sum is a midpoint rule — second order in the cell size. With
    ``"node"`` (the legacy layout, and the default for a hand-built box) it
    sits at the cell's lower-corner node, which is a left-endpoint
    rectangle rule — first order.

    Automatically dispatches to the JAX-differentiable implementation
    when called inside ``jax.grad`` or other JAX tracing contexts.
    Use this function for both post-processing and optimization objectives.

    Parameters
    ----------
    ntff_data : NTFFData
        Accumulated frequency-domain tangential fields.
    box : NTFFBox
    grid : Grid
    theta : (n_theta,) array in radians [0, pi]
    phi : (n_phi,) array in radians [0, 2*pi]

    Returns
    -------
    FarFieldResult
    """
    # Dispatch on the inputs themselves: any traced leaf (the six faces, the
    # box's frequencies, the angles, a graded grid's spacings) takes the
    # jnp implementation, and an error there is the caller's to see. The
    # host path below reads every one of them with numpy (#1364).
    import jax
    from rfx.core.jax_utils import is_tracer
    if any(is_tracer(leaf) for leaf in
           jax.tree_util.tree_leaves((ntff_data, box, grid, theta, phi))):
        return compute_far_field_jax(ntff_data, box, grid, theta, phi)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    freqs = np.asarray(box.freqs, dtype=np.float64)
    nf = len(freqs)
    k_arr = 2 * np.pi * freqs / C0  # (nf,)

    dx = grid.dx
    dy = getattr(grid, 'dy', dx)
    dz_arr = getattr(grid, 'dz', None)  # (nz,) for NonUniformGrid, None for Grid
    # In-plane grading was previously ignored here: dx/dy above are the
    # BOUNDARY cell sizes on a NonUniformGrid, so a mesh graded in x or y had
    # its surface elements and face coordinates computed with the wrong
    # spacing — silently, with a pattern returned (issue #743). The z axis
    # already carried per-cell spacing; x and y now do too. When neither is
    # graded, every path below is the scalar one, bit-for-bit.
    dx_arr = getattr(grid, 'dx_arr', None)
    dy_arr = getattr(grid, 'dy_arr', None)
    pads = _ntff_face_pads(grid, box)
    cpml_lo_x, cpml_lo_y, cpml_lo_z = (pads[f"{axis}_lo"] for axis in _AXIS_NAMES)
    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi
    face_centre = bool(getattr(box, "face_centre", False))

    # Build edge positions for graded axes (physical origin at the inner edge
    # of the lo-face CPML, matching the uniform formula (idx - cpml_lo) * d).
    def _edges(d_arr, cpml_lo):
        if d_arr is None:
            return None
        e = np.concatenate([[0.0], np.cumsum(np.asarray(d_arr, dtype=np.float64))])
        return e - e[cpml_lo]

    z_edges = _edges(dz_arr, cpml_lo_z)
    x_edges = _edges(dx_arr, cpml_lo_x)
    y_edges = _edges(dy_arr, cpml_lo_y)
    inplane_graded = x_edges is not None or y_edges is not None

    def _cells(d_arr, scalar, lo, hi):
        if d_arr is None:
            return np.full(hi - lo, float(scalar))
        return np.asarray(d_arr, dtype=np.float64)[lo:hi]

    # Per-face dS. Returns a scalar when nothing is graded (bit-identical to
    # the pre-#743 code), a (n1, n2) area array otherwise.
    def _face_dS_full(axis, other_ranges):
        (a0, a1), (b0, b1) = other_ranges
        if axis == 0:      # x face: spanned by y and z
            d1 = _cells(dy_arr, dy, a0, a1)
            d2 = _cells(dz_arr, dx, b0, b1)
        elif axis == 1:    # y face: spanned by x and z
            d1 = _cells(dx_arr, dx, a0, a1)
            d2 = _cells(dz_arr, dx, b0, b1)
        else:              # z face: spanned by x and y
            d1 = _cells(dx_arr, dx, a0, a1)
            d2 = _cells(dy_arr, dy, b0, b1)
        return d1[:, None] * d2[None, :]

    def _face_dS(axis, k_lo, k_hi):
        if dz_arr is not None and axis in (0, 1):
            dz_face = np.asarray(dz_arr[k_lo:k_hi], dtype=np.float64)
            d_perp = dy if axis == 0 else dx
            return d_perp * dz_face  # (nk,)
        # A uniform Grid is cubic in z (see _face_positions, which measures
        # z in units of dx), so dz == dx here.
        return _scalar_face_dS(axis, dx, dy, dx)

    # Observation direction unit vectors
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sth, cth = np.sin(TH), np.cos(TH)
    sph, cph = np.sin(PH), np.cos(PH)

    r_hat = np.stack([sth * cph, sth * sph, cth], axis=-1)     # (nθ, nφ, 3)
    th_hat = np.stack([cth * cph, cth * sph, -sth], axis=-1)
    ph_hat = np.stack([-sph, cph, np.zeros_like(sth)], axis=-1)

    n_th, n_ph = len(theta), len(phi)
    r_flat = r_hat.reshape(-1, 3)      # (n_dir, 3)
    n_dir = r_flat.shape[0]

    N_total = np.zeros((nf, n_dir, 3), dtype=np.complex128)
    L_total = np.zeros((nf, n_dir, 3), dtype=np.complex128)

    # Process each face
    face_specs = [
        # (data, axis, sign, face_idx, other_ranges)
        (ntff_data.x_lo, 0, -1, i0, ((j0, j1), (k0, k1))),
        (ntff_data.x_hi, 0, +1, i1, ((j0, j1), (k0, k1))),
        (ntff_data.y_lo, 1, -1, j0, ((i0, i1), (k0, k1))),
        (ntff_data.y_hi, 1, +1, j1, ((i0, i1), (k0, k1))),
        (ntff_data.z_lo, 2, -1, k0, ((i0, i1), (j0, j1))),
        (ntff_data.z_hi, 2, +1, k1, ((i0, i1), (j0, j1))),
    ]

    for face_dft, axis, sign, face_idx, other_ranges in face_specs:
        face_np = np.asarray(face_dft, dtype=np.complex128)  # (nf, n1, n2, 4)
        n1, n2 = face_np.shape[1], face_np.shape[2]
        if n1 == 0 or n2 == 0:
            continue

        pos = _face_positions(axis, face_idx, other_ranges, dx,
                              cpml_lo_x, cpml_lo_y, cpml_lo_z,
                              dy=dy, z_edges=z_edges,
                              x_edges=x_edges, y_edges=y_edges,
                              centre=face_centre)
        pos_flat = pos.reshape(-1, 3)     # (nc, 3)
        fields_flat = face_np.reshape(nf, -1, 4)  # (nf, nc, 4)

        J, M = _surface_currents(fields_flat, axis, sign)  # (nf, nc, 3)

        # Per-cell dS for non-uniform z on x/y faces
        if inplane_graded:
            dS_flat = _face_dS_full(axis, other_ranges).reshape(-1)
        else:
            k_range = other_ranges[1] if axis in (0, 1) else None
            if k_range is not None:
                dS_k = _face_dS(axis, k_range[0], k_range[1])
                if np.ndim(dS_k) > 0:
                    # Tile along the non-k dimension to match the cell count
                    dS_flat = np.tile(dS_k, n1)  # (n1*nk,) = (nc,)
                else:
                    dS_flat = dS_k
            else:
                dS_flat = _face_dS(axis, 0, 0)

        dot = r_flat @ pos_flat.T  # (n_dir, nc)

        for fi in range(nf):
            phase = np.exp(1j * k_arr[fi] * dot)  # (n_dir, nc)
            if np.ndim(dS_flat) > 0:
                J_weighted = J[fi] * dS_flat[:, None]  # (nc, 3)
                M_weighted = M[fi] * dS_flat[:, None]
                N_total[fi] += np.einsum("dc,cj->dj", phase, J_weighted)
                L_total[fi] += np.einsum("dc,cj->dj", phase, M_weighted)
            else:
                N_total[fi] += np.einsum("dc,cj->dj", phase, J[fi]) * dS_flat
                L_total[fi] += np.einsum("dc,cj->dj", phase, M[fi]) * dS_flat

    # Project onto theta and phi unit vectors
    th_flat = th_hat.reshape(-1, 3)
    ph_flat = ph_hat.reshape(-1, 3)

    N_th = np.sum(N_total * th_flat[None, :, :], axis=-1)  # (nf, n_dir)
    N_ph = np.sum(N_total * ph_flat[None, :, :], axis=-1)
    L_th = np.sum(L_total * th_flat[None, :, :], axis=-1)
    L_ph = np.sum(L_total * ph_flat[None, :, :], axis=-1)

    # Far-field components
    E_theta = np.zeros((nf, n_dir), dtype=np.complex128)
    E_phi = np.zeros((nf, n_dir), dtype=np.complex128)
    for fi in range(nf):
        jk = 1j * k_arr[fi]
        E_theta[fi] = -jk / (4 * np.pi) * (L_ph[fi] + ETA_0 * N_th[fi])
        E_phi[fi] = jk / (4 * np.pi) * (L_th[fi] - ETA_0 * N_ph[fi])

    return FarFieldResult(
        E_theta=E_theta.reshape(nf, n_th, n_ph),
        E_phi=E_phi.reshape(nf, n_th, n_ph),
        theta=theta,
        phi=phi,
        freqs=freqs,
    )


def compute_far_field_jax(
    ntff_data,
    box,
    grid,
    theta,
    phi,
    *,
    max_phase_bytes: float = 4e9,
):
    """JAX-differentiable far-field computation for use inside jax.grad.

    Same physics as ``compute_far_field`` — including its face-cell sample
    placement, which is the midpoint rule when the box says
    ``collocation="face_centre"`` and the legacy corner-node rule when it
    says ``"node"`` — but uses ``jnp`` throughout, enabling end-to-end
    differentiation for far-field optimization.

    ``max_phase_bytes`` bounds the (n_freqs, n_directions, n_cells) phase
    array the transform materializes per face; the direction grid is
    split to stay under it. Splitting theta is EXACT — the sum over
    surface cells is independent per direction — and the chunked result
    is bit-identical to the single-shot one. Pass ``float("inf")`` to
    force a single pass.
    """
    # The transform below materializes a (n_freqs, n_directions, n_cells)
    # phase array per face. All three factors are unbounded, and a board
    # pattern run (13 freqs, 181x72 directions, 3.3e5 cells/face) asked for
    # 214 GB and died AFTER an 8-hour solve (issue #727). The sum over
    # surface cells is independent per direction, so splitting theta is
    # EXACT — the chunked and unchunked results agree to the bit (locked by
    # tests/unit/farfield/test_farfield_chunking.py). max_phase_bytes bounds that array;
    # raise it to trade memory for fewer passes, or set it to inf to force
    # the single-shot path.
    # The DFT accumulators carry the simulation's realized precision.
    faces = [getattr(ntff_data, name) for name in
             ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
             if getattr(ntff_data, name, None) is not None]
    cdtype = jnp.result_type(*faces, 1j)
    rdtype = jnp.finfo(cdtype).dtype
    theta = jnp.asarray(theta, dtype=rdtype)
    n_th_total = int(theta.shape[0])
    if n_th_total > 1 and np.isfinite(max_phase_bytes):
        # Size from the actual face arrays. The first version of this
        # scanned for attributes starting with "J" — a name no NTFFData
        # field has (they are x_lo/x_hi/y_lo/y_hi/z_lo/z_hi, each
        # (n_freqs, n1, n2, 4)). n_cells stayed 0, the computed budget was
        # never exceeded, and the default path never chunked: a board
        # pattern run died on the same 214 GB allocation this function was
        # changed to prevent, after an 8.5-hour solve. The unit test passed
        # because it forced a tiny max_phase_bytes — exercising the
        # mechanism but never the sizing.
        n_cells = 0
        for _name in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
            _arr = getattr(ntff_data, _name, None)
            _shape = getattr(_arr, "shape", None)
            if _shape is not None and len(_shape) >= 3:
                n_cells = max(n_cells, int(_shape[1]) * int(_shape[2]))
        if n_cells > 0:
            # The count is static even when the frequencies are traced
            # (a box built inside the caller's jit, #1364).
            n_freqs = int(np.shape(box.freqs)[0])
            n_ph = int(jnp.asarray(phi).shape[0])
            per_theta = max(1.0, n_freqs * n_ph * n_cells * 16.0)
            # At least one theta per pass: when a single direction already
            # exceeds the budget, chunking to 1 is the smallest the split
            # can do, and it is still the difference between running and
            # dying.
            chunk = max(1, int(max_phase_bytes // per_theta))
            if chunk < n_th_total:
                parts = [
                    compute_far_field_jax(
                        ntff_data, box, grid, theta[i:i + chunk], phi,
                        max_phase_bytes=float("inf"))
                    for i in range(0, n_th_total, chunk)
                ]
                return FarFieldResult(
                    E_theta=jnp.concatenate([p.E_theta for p in parts], axis=1),
                    E_phi=jnp.concatenate([p.E_phi for p in parts], axis=1),
                    theta=theta,
                    phi=parts[0].phi,
                    freqs=parts[0].freqs,
                )
    theta = jnp.asarray(theta, dtype=rdtype)
    phi = jnp.asarray(phi, dtype=rdtype)
    freqs = jnp.asarray(box.freqs, dtype=rdtype)
    nf = freqs.shape[0]
    k_arr = 2 * jnp.pi * freqs / C0

    dx = grid.dx
    dy = getattr(grid, 'dy', dx)
    dz_arr = getattr(grid, 'dz', None)
    dx_arr = getattr(grid, 'dx_arr', None)   # in-plane grading (#743)
    dy_arr = getattr(grid, 'dy_arr', None)
    pads = _ntff_face_pads(grid, box)
    cpml_lo_x, cpml_lo_y, cpml_lo_z = (pads[f"{axis}_lo"] for axis in _AXIS_NAMES)
    i0, i1 = box.i_lo, box.i_hi
    j0, j1 = box.j_lo, box.j_hi
    k0, k1 = box.k_lo, box.k_hi
    face_centre = bool(getattr(box, "face_centre", False))

    # Edge positions for graded axes (#743: x and y were previously read as
    # the boundary scalar, so a graded in-plane mesh was integrated with the
    # wrong dS and face coordinates, silently).
    def _edges_j(d_arr, cpml_lo):
        if d_arr is None:
            return None
        dj = jnp.asarray(d_arr, dtype=rdtype)
        e = jnp.concatenate([jnp.zeros(1, dtype=rdtype), jnp.cumsum(dj)])
        return e - e[cpml_lo]

    dz_jnp = jnp.asarray(dz_arr, dtype=rdtype) if dz_arr is not None else None
    z_edges = _edges_j(dz_arr, cpml_lo_z)
    x_edges = _edges_j(dx_arr, cpml_lo_x)
    y_edges = _edges_j(dy_arr, cpml_lo_y)
    inplane_graded = x_edges is not None or y_edges is not None

    def _cells_j(d_arr, scalar, lo, hi):
        if d_arr is None:
            return jnp.full((hi - lo,), float(scalar), dtype=rdtype)
        return jnp.asarray(d_arr, dtype=rdtype)[lo:hi]

    def _face_dS_full_j(axis, other_ranges):
        (a0, a1), (b0, b1) = other_ranges
        if axis == 0:
            d1, d2 = _cells_j(dy_arr, dy, a0, a1), _cells_j(dz_arr, dx, b0, b1)
        elif axis == 1:
            d1, d2 = _cells_j(dx_arr, dx, a0, a1), _cells_j(dz_arr, dx, b0, b1)
        else:
            d1, d2 = _cells_j(dx_arr, dx, a0, a1), _cells_j(dy_arr, dy, b0, b1)
        return d1[:, None] * d2[None, :]

    # Per-face dS helper — the same axis-aware area element as the numpy twin.
    def _face_dS_jax(axis, k_lo, k_hi):
        if dz_arr is not None and axis in (0, 1):
            dz_face = dz_jnp[k_lo:k_hi]
            d_perp = dy if axis == 0 else dx
            return d_perp * dz_face  # (nk,)
        return _scalar_face_dS(axis, dx, dy, dx)

    TH, PH = jnp.meshgrid(theta, phi, indexing="ij")
    sth, cth = jnp.sin(TH), jnp.cos(TH)
    sph, cph = jnp.sin(PH), jnp.cos(PH)

    r_hat = jnp.stack([sth * cph, sth * sph, cth], axis=-1)
    th_hat = jnp.stack([cth * cph, cth * sph, -sth], axis=-1)
    ph_hat = jnp.stack([-sph, cph, jnp.zeros_like(sth)], axis=-1)

    n_th, n_ph = theta.shape[0], phi.shape[0]
    r_flat = r_hat.reshape(-1, 3)
    n_dir = r_flat.shape[0]

    N_total = jnp.zeros((nf, n_dir, 3), dtype=cdtype)
    L_total = jnp.zeros((nf, n_dir, 3), dtype=cdtype)

    face_specs = [
        (ntff_data.x_lo, 0, -1, i0, ((j0, j1), (k0, k1))),
        (ntff_data.x_hi, 0, +1, i1, ((j0, j1), (k0, k1))),
        (ntff_data.y_lo, 1, -1, j0, ((i0, i1), (k0, k1))),
        (ntff_data.y_hi, 1, +1, j1, ((i0, i1), (k0, k1))),
        (ntff_data.z_lo, 2, -1, k0, ((i0, i1), (j0, j1))),
        (ntff_data.z_hi, 2, +1, k1, ((i0, i1), (j0, j1))),
    ]

    for face_dft, axis, sign, face_idx, other_ranges in face_specs:
        face = jnp.asarray(face_dft)
        n1, n2 = face.shape[1], face.shape[2]
        if n1 == 0 or n2 == 0:
            continue

        pos = _face_positions_jax(axis, face_idx, other_ranges, dx,
                                  cpml_lo_x, cpml_lo_y, cpml_lo_z,
                                  dy=dy, z_edges=z_edges,
                                  x_edges=x_edges, y_edges=y_edges,
                                  centre=face_centre)
        pos_flat = pos.reshape(-1, 3)
        fields_flat = face.reshape(nf, -1, 4)

        J, M = _surface_currents_jax(fields_flat, axis, sign)

        if inplane_graded:
            dS_flat = _face_dS_full_j(axis, other_ranges).reshape(-1)
        else:
            # Per-cell dS for non-uniform z on x/y faces
            k_range = other_ranges[1] if axis in (0, 1) else None
            if k_range is not None:
                dS_k = jnp.asarray(_face_dS_jax(axis, k_range[0], k_range[1]), dtype=rdtype)
                if dS_k.ndim > 0:
                    dS_flat = jnp.tile(dS_k, n1)  # (n1*nk,) = (nc,)
                else:
                    dS_flat = dS_k
            else:
                dS_flat = _face_dS_jax(axis, 0, 0)

        dot = jnp.matmul(r_flat, pos_flat.T, precision=HIGHEST)  # (n_dir, nc)

        phase = jnp.exp(1j * k_arr[:, None, None] * dot[None, :, :])
        dS_flat = jnp.asarray(dS_flat, dtype=rdtype)  # may be Python float
        if jnp.ndim(dS_flat) > 0:
            J_w = J * dS_flat[None, :, None]
            M_w = M * dS_flat[None, :, None]
            N_total = N_total + jnp.einsum("fdc,fcj->fdj", phase, J_w, precision=HIGHEST)
            L_total = L_total + jnp.einsum("fdc,fcj->fdj", phase, M_w, precision=HIGHEST)
        else:
            N_total = N_total + jnp.einsum("fdc,fcj->fdj", phase, J, precision=HIGHEST) * dS_flat
            L_total = L_total + jnp.einsum("fdc,fcj->fdj", phase, M, precision=HIGHEST) * dS_flat

    th_flat = th_hat.reshape(-1, 3)
    ph_flat = ph_hat.reshape(-1, 3)

    N_th = jnp.sum(N_total * th_flat[None, :, :], axis=-1)
    N_ph = jnp.sum(N_total * ph_flat[None, :, :], axis=-1)
    L_th = jnp.sum(L_total * th_flat[None, :, :], axis=-1)
    L_ph = jnp.sum(L_total * ph_flat[None, :, :], axis=-1)

    jk = 1j * k_arr[:, None]  # (nf, 1)
    E_theta = -jk / (4 * jnp.pi) * (L_ph + ETA_0 * N_th)
    E_phi = jk / (4 * jnp.pi) * (L_th - ETA_0 * N_ph)

    return FarFieldResult(
        E_theta=E_theta.reshape(nf, n_th, n_ph),
        E_phi=E_phi.reshape(nf, n_th, n_ph),
        theta=theta,
        phi=phi,
        freqs=freqs,
    )


def _scalar_face_dS(axis, dx, dy, dz):
    """Area of one face cell: the product of the two widths that SPAN it.

    An x face is spanned by y and z, a y face by x and z, a z face by x and
    y. This used to be written as ``dx*dy`` for all three, which is right by
    coincidence for the x and z faces of a cubic grid and wrong for the y
    face of any grid whose y cells differ from its z cells.
    """
    if axis == 0:
        return dy * dz
    if axis == 1:
        return dx * dz
    return dx * dy


def _surface_currents_jax(fields, axis, sign):
    """JAX version of _surface_currents."""
    f0, f1, f2, f3 = (fields[..., i] for i in range(4))
    z = jnp.zeros_like(f0)
    s = sign
    if axis == 0:
        J = jnp.stack([z, -s * f3, s * f2], axis=-1)
        M = jnp.stack([z, s * f1, -s * f0], axis=-1)
    elif axis == 1:
        J = jnp.stack([s * f3, z, -s * f2], axis=-1)
        M = jnp.stack([-s * f1, z, s * f0], axis=-1)
    else:
        J = jnp.stack([-s * f3, s * f2, z], axis=-1)
        M = jnp.stack([s * f1, -s * f0, z], axis=-1)
    return J, M


def _face_positions_jax(axis, idx, other_ranges, dx, cpml_lo_x, cpml_lo_y, cpml_lo_z,
                        dy=None, z_edges=None, x_edges=None, y_edges=None,
                        centre=False):
    """JAX version of _face_positions with non-uniform z support.

    ``cpml_lo_*`` are per-axis low-face CPML thicknesses (see the numpy
    twin ``_face_positions`` for semantics). Under symmetric face_layers
    they all equal ``grid.cpml_layers``. ``centre`` places each sample at
    the centre of its face cell (the midpoint rule) instead of the cell's
    lower-corner node.
    """
    if dy is None:
        dy = dx

    def _z_pos(k):
        if z_edges is not None:
            return z_edges[k]
        return (k - cpml_lo_z) * dx

    def _x_pos(i):
        if x_edges is not None:
            return x_edges[i]
        return (i - cpml_lo_x) * dx

    def _y_pos(j):
        if y_edges is not None:
            return y_edges[j]
        return (j - cpml_lo_y) * dy

    def _axis_pos(lo, hi, edges, cpml_lo, d):
        if edges is not None:
            if centre:
                return 0.5 * (edges[lo:hi] + edges[lo + 1:hi + 1])
            return edges[lo:hi]
        base = (jnp.arange(lo, hi) - cpml_lo) * d
        return base + 0.5 * d if centre else base

    if axis == 0:
        j_range, k_range = other_ranges
        x_fixed = _x_pos(idx)
        y = _axis_pos(j_range[0], j_range[1], y_edges, cpml_lo_y, dy)
        z = _axis_pos(k_range[0], k_range[1], z_edges, cpml_lo_z, dx)
        Y, Z = jnp.meshgrid(y, z, indexing="ij")
        X = jnp.full_like(Y, x_fixed)
    elif axis == 1:
        i_range, k_range = other_ranges
        y_fixed = _y_pos(idx)
        x = _axis_pos(i_range[0], i_range[1], x_edges, cpml_lo_x, dx)
        z = _axis_pos(k_range[0], k_range[1], z_edges, cpml_lo_z, dx)
        X, Z = jnp.meshgrid(x, z, indexing="ij")
        Y = jnp.full_like(X, y_fixed)
    else:
        i_range, j_range = other_ranges
        x = _axis_pos(i_range[0], i_range[1], x_edges, cpml_lo_x, dx)
        y = _axis_pos(j_range[0], j_range[1], y_edges, cpml_lo_y, dy)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = jnp.full_like(X, _z_pos(idx))
    return jnp.stack([X, Y, Z], axis=-1)
