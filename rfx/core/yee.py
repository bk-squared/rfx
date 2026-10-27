"""Yee cell FDTD update equations — pure JAX functions.

All functions are jax.jit-compatible and operate on explicit state arrays.
No hidden mutable state.
"""

from __future__ import annotations

from functools import partial, wraps
from typing import NamedTuple

from rfx import _realized

import jax
import jax.numpy as jnp
import numpy as np


class FDTDState(NamedTuple):
    """Complete FDTD simulation state at one timestep."""

    # Electric field components (Nx, Ny, Nz)
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    # Magnetic field components (Nx, Ny, Nz)
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    # Timestep counter
    step: jnp.ndarray


class MaterialArrays(NamedTuple):
    """Material property arrays on the grid.

    ``eps_r`` and ``sigma`` are VOLUME properties of the cell, and since #1210
    an E component takes the mean of them over the four cells its edge touches.
    A lumped element is not a volume: a port's 50 ohm load, an RLC resistor or
    capacitor is a device across ONE edge, folded into the cell's sigma / eps_r
    as an equivalent so that the old cell-owned update produced the right edge
    conductance. Averaging it would divide it among four cells and spread it
    over twelve edges (measured: a 50 ohm wire port then read |S11| = 0.20
    instead of 1/3, and a lumped port in a CPML dielectric read |S11| = 1.86,
    non-physical).

    So the stamp is recorded twice: inside ``eps_r`` / ``sigma``, where every
    reader that wants the cell's total still finds it, and again in the two
    ``*_lumped`` records, PER E COMPONENT (#1236). A record is ``None`` (no
    stamp anywhere) or a 3-tuple ``(x, y, z)`` whose entry ``c`` is ``None``
    or a grid-shaped array holding the stamps of the elements that sit on
    component ``c``'s edge. :func:`component_e_materials` averages the volume
    (``eps_r`` minus every stamp) and adds each stamp back at its own cell, to
    ITS OWN component only. Until #1236 the record had no component and was
    added back to all three: a 50 ohm port on Ez was also a 50 ohm resistor on
    the Ex and Ey edges leaving its node (a centre-fed dipole's resonance
    +0.34 % at lambda/43, the feed-tip radial field pinned to 0.03 of the gap
    field on the two loaded sides against 0.37 on the free one).
    ``None`` is bit-identical to averaging ``eps_r`` and ``sigma`` outright.

    ``sigma_film`` is the edge-owned conductivity of declared films. It is
    not contained in ``sigma``: it is added after the edge mean and never
    subtracted. Its optional per-component tuple has the lumped record's form.

    Every RLC element's ``D0`` (series and parallel) reads its own edge
    through ``rfx.lumped.edge_update_denominator`` -> ``cell_component_e_materials``.
    The lanes whose volume is still CELL-owned (the distributed slab update,
    UPML) take :func:`cell_owned_component_materials`: the cell total minus
    the other components' stamps. The dispersive Debye/Lorentz coefficients
    take :func:`component_e_materials` per component like the plain update
    (#1260; until then they applied the cell total to all three).
    """

    # Relative permittivity (Nx, Ny, Nz) — used in E update
    eps_r: jnp.ndarray
    # Conductivity S/m (Nx, Ny, Nz) — used in E update (lossy)
    sigma: jnp.ndarray
    # Relative permeability (Nx, Ny, Nz) — used in H update
    mu_r: jnp.ndarray
    # The edge-owned part of ``sigma`` / ``eps_r`` (lumped stamps): None, or
    # a 3-tuple (x, y, z) of per-component arrays / None (#1236)
    sigma_lumped: object = None
    eps_r_lumped: object = None
    # H-component-owned permeability increments (port and PEC filament contours).
    # None means no contour increment beyond the volume face mean.
    mu_r_wire: object = None
    # Set only after final cell/stamp assembly on migrated single-device paths.
    components: object = None
    # Realized relaxed-conductor factors; lane steps remain independently pinned.
    edge_keep: object = None
    # Edge-owned conductivity of declared films: never contained in sigma,
    # added after the edge mean, never subtracted. None or (x, y, z).
    sigma_film: object = None


def component_h_materials(materials, periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool], cell_sizes=None):
    """Relative permeability at Hx, Hy, Hz, including local wire contours.

    Each component lies on the face between its own-axis backward cell and
    the cell at its index. Average their inverse permeabilities with their
    cell lengths. ``cell_sizes`` contains the three primal-cell width arrays;
    omitted widths mean a uniform mesh. Equal cells retain their exact bits.
    Wire contour increments are added after this volume-material mean.
    """
    parts = getattr(materials, "mu_r_wire", None)
    mu = materials.mu_r
    @jax.custom_jvp
    def exact_equal(lo, hi, mean):
        return jnp.where(lo == hi, hi, mean)

    @exact_equal.defjvp
    def exact_equal_jvp(primals, tangents):
        # Equality changes only rounding, not the material law's derivative:
        # independent equal-valued neighbours still both influence the face.
        return exact_equal(*primals), tangents[2]

    if jnp.ndim(mu) == 0:
        volume = (mu,) * 3
    else:
        high_wall = _walls_off_periodic(high_wall, periodic)
        mu = _material_wall_view(mu, high_wall)
        volume = []
        for axis in range(3):
            lo = _material_bwd_neighbour(mu, axis, periodic, high_wall=high_wall)
            if cell_sizes is None:
                mean = 2 / (1 / lo + 1 / mu)
            else:
                widths = jnp.asarray(cell_sizes[axis])
                # Cell metrics omit the terminal node; material arrays include it.
                if widths.shape[0] == mu.shape[axis] - 1:
                    widths = jnp.concatenate((widths, widths[-1:]))
                shape = [1, 1, 1]
                shape[axis] = widths.shape[0]
                hi_d = widths.reshape(shape)
                hi_d = _material_wall_view(hi_d, high_wall)
                lo_d = _material_bwd_neighbour(hi_d, axis, periodic, high_wall=high_wall)
                mean = (lo_d + hi_d) / (lo_d / lo + hi_d / mu)
            volume.append(exact_equal(lo, mu, mean))
        volume = tuple(volume)
    if parts is None:
        return volume
    if not isinstance(parts, (tuple, list)) or len(parts) != 3:
        raise TypeError("mu_r_wire must be None or a (Hx, Hy, Hz) tuple of arrays/None")
    return tuple(m if p is None else m + p for m, p in zip(volume, parts))


_LUMPED_AXIS = {"ex": 0, "ey": 1, "ez": 2}


def lumped_axis(component) -> int:
    """The axis index ``0/1/2`` of an E component name (``"ex"/"ey"/"ez"``)."""
    try:
        return _LUMPED_AXIS[str(component).lower()]
    except KeyError:
        raise ValueError(
            f"a lumped stamp sits on one E edge: component must be 'ex', 'ey' "
            f"or 'ez', got {component!r}") from None


def lumped_components(record):
    """The ``(x, y, z)`` entries of a lumped record (#1236).

    ``None`` (no stamp anywhere) gives ``(None, None, None)``. A 3-tuple (or
    list) is returned as a tuple. Anything else -- in particular a bare array,
    the component-less record that loaded all three E edges before #1236 -- is
    refused, so a caller written for the old record cannot silently keep the
    old rule.
    """
    if record is None:
        return (None, None, None)
    if isinstance(record, (tuple, list)) and len(record) == 3:
        return tuple(record)
    raise TypeError(
        "a lumped record is per E component since #1236: None or a 3-tuple "
        "(x, y, z) of arrays / None, one entry per component the stamp loads; "
        f"got {type(record).__name__}. Stamp through "
        "rfx.sources.sources.stamp_lumped_sigma / stamp_lumped_eps with the "
        "element's component.")


def lumped_total(record):
    """The sum of a record's per-component stamps (what sits in the cell
    total), or ``None`` when the record holds no stamp. One component is
    returned as-is, so a single-component record costs no arithmetic."""
    total = None
    for part in lumped_components(record):
        if part is not None:
            total = part if total is None else total + part
    return total


def permittivity_without_lumped(materials):
    """Volume relative permittivity with every recorded capacitor removed."""
    stamp = lumped_total(getattr(materials, "eps_r_lumped", None))
    return materials.eps_r if stamp is None else materials.eps_r - stamp


def add_lumped_eps(eps_components, record, *, inverse=False):
    """Add each capacitor to its own E edge after volume averaging (#1263).

    ``eps_components`` must contain only the volume contribution, after
    any smoothing or conformal correction. Call once for the completed
    array, so neither the volume average nor a conformal weight acts on
    the lumped device. ``None`` leaves the ordinary material update active.

    For an inverse tensor, inv/(1 + inv*stamp) = 1/(1/inv + stamp).
    This adds to permittivity before inversion and leaves PEC inv=0 at zero.
    """
    if eps_components is None:
        return None
    out = []
    for eps, stamp in zip(eps_components, lumped_components(record)):
        if stamp is None:
            out.append(eps)
        elif inverse:
            out.append(eps / (1.0 + eps * stamp))
        else:
            out.append(eps + stamp)
    return tuple(out)


def map_lumped(record, fn):
    """Apply ``fn`` to every array of a lumped record, keeping its shape:
    ``None`` stays ``None``, a ``None`` component stays ``None``. For readers
    that slice, pad, shard or cast the material arrays."""
    if record is None:
        return None
    return tuple(None if part is None else fn(part)
                 for part in lumped_components(record))


def add_film(sigma_components, materials):
    """Add each declared film to its own E component after the volume mean."""
    return tuple(s if f is None else s + f for s, f in zip(
        sigma_components, lumped_components(materials.sigma_film)))


def cell_owned_component_materials(materials):
    """Per-component ``(eps_r, sigma)`` for a CELL-OWNED lane (#1236).

    A lane that takes every coefficient from the cell that owns the edge (the
    distributed slab update, UPML's ``init_upml``) reads ``materials.sigma`` for all three
    components, so a stamp in that total loads all three edges at its node.
    This returns, per component ``c``, the cell total minus the stamps that
    belong to the OTHER two components: the volume stays cell-owned (that
    lane's rule, #1210 not converted there) and each lumped element loads its
    own edge only. With no record it returns ``(eps_r,)*3, (sigma,)*3``.
    """
    eps_parts = lumped_components(getattr(materials, "eps_r_lumped", None))
    sig_parts = lumped_components(getattr(materials, "sigma_lumped", None))

    def per_component(total, parts):
        out = []
        for c in range(3):
            other = None
            for c2, part in enumerate(parts):
                if c2 != c and part is not None:
                    other = part if other is None else other + part
            out.append(total if other is None else total - other)
        return tuple(out)

    return (per_component(materials.eps_r, eps_parts),
            add_film(per_component(materials.sigma, sig_parts), materials))


def init_state(shape: tuple[int, int, int], *, field_dtype=jnp.float32) -> FDTDState:
    """Initialize zero-valued FDTD state.

    Parameters
    ----------
    shape : (Nx, Ny, Nz)
    field_dtype : jnp dtype
        Data type for field arrays.  Use ``jnp.float16`` for mixed-
        precision mode (material coefficients and accumulators stay
        float32; only field storage is reduced).
    """
    zeros = jnp.zeros(shape, dtype=field_dtype)
    return FDTDState(
        ex=zeros, ey=zeros, ez=zeros,
        hx=zeros, hy=zeros, hz=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )


def ade_state_dtype(field_dtype=None):
    """Dtype for a dispersion ADE polarization carry, given the field dtype.

    One policy, shared by ``rfx.materials.debye.init_debye`` and
    ``rfx.materials.lorentz.init_lorentz`` (allocation); the matching scan
    bodies derive their output dtype from the carry they were handed, so the
    ``lax.scan`` carry closes (issue #656, same family as #630/#644/#646).

    ``promote_types(field_dtype, float32)`` — PROMOTE, never pin, never a bare
    copy of the field dtype:

    * float16 (``precision="mixed"``) -> float32.  The float32 floor is
      mandatory: P is a recursive accumulator (``P^{n+1}`` is built from
      ``P^n``/``P^{n-1}``), and float16's 11-bit mantissa would destroy it.
      ``precision="mixed"`` promises float16 FIELDS with float32 accumulation.
    * float32 (the default)             -> float32.  Unchanged, x64 or not.
    * float64 (``precision="float64"``) -> float64.  The hard float32 pin this
      replaces is what made any Debye/Lorentz/Drude pole crash there.
    * complex64                         -> complex64.  A flat float32 pin
      would silently drop the imaginary part; ``promote_types`` preserves it.

    ``field_dtype=None`` (a caller that does not thread the field dtype)
    falls back to the ambient default float with the same float32 floor —
    the ``rfx.lumped.init_rlc_state`` precedent from #646 — so the carry
    tracks ``jax_enable_x64`` instead of pinning under it.

    The float64 row needs x64 actually enabled: the ``jnp.result_type``
    wrapper clamps an unavailable float64 back to float32, so this returns a
    dtype that is allocatable rather than one ``jnp.zeros`` would silently
    truncate. Same clamp, and same wrapper, as ``farfield.ntff_accum_dtype``;
    it is the documented ``precision="float64"``-without-x64 footgun that
    ``preflight()`` reports as ``precision_float64_without_x64``.

    NOTE the complex row is a policy statement, not a supported lane: the
    dispersive branches of ``update_e_dispersive`` ignore the Bloch phase
    (#404), so oblique-Bloch + dispersion is unsupported and still raises a
    carry-dtype mismatch rather than closing on wrong physics.
    """
    base = jnp.result_type(float) if field_dtype is None else jnp.dtype(field_dtype)
    return jnp.result_type(jnp.promote_types(base, jnp.float32))


def init_materials(shape: tuple[int, int, int]) -> MaterialArrays:
    """Initialize free-space material arrays."""
    ones = jnp.ones(shape, dtype=jnp.float32)
    return MaterialArrays(
        eps_r=ones,
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=ones,
    )


# Physical constants — post-2019 SI / CODATA. EPS_0 and MU_0 form a
# mutually consistent pair: 1/sqrt(MU_0 * EPS_0) == c (299792458 m/s),
# so the FDTD's effective speed of light is exact. Changing one without
# the other breaks that consistency (see the TE10-cutoff gate).
EPS_0 = 8.8541878128e-12  # F/m — post-2019 SI / CODATA value
MU_0 = 1.25663706212e-6  # H/m — post-2019 SI / CODATA value


def _shift_fwd(arr, axis):
    """arr[i+1] with zero at the last position (replaces roll(arr, -1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, 1)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(1, None)
    return padded[tuple(slices)]


def _shift_bwd(arr, axis):
    """arr[i-1] with zero at the first position (replaces roll(arr, +1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, -1)
    return padded[tuple(slices)]


# --- (2,4) fourth-order-in-space staggered stencil (Fang 1996 / standard) ---
# A 4th-order option for the smooth-bulk lever (analytic gate-1: ~2.52x fewer
# cells/wavelength than (2,2) for the same dispersion error; gate-2 spike:
# this advantage is a SMOOTH-PROPAGATION property and does NOT extend to PEC /
# geometric features, which staircase at 2nd order regardless — see
# docs/research_notes/20260627_memory_efficiency_techniques_exploration.md).
# The two coefficients on the near (1/2-cell) and far (3/2-cell) staggered
# differences. At a non-periodic boundary the wider stencil reaches outside the
# domain, so the first/last TWO gaps revert to 2nd order (the 2-cell PEC/edge
# ribbon — a 1-cell ribbon leaves the i=N-2 fwd / i=1 bwd gap malformed).
_C4_NEAR = 9.0 / 8.0
_C4_FAR = -1.0 / 24.0


def _ribbon_2nd(d4, d2, axis):
    """Revert the first TWO and last TWO slices along ``axis`` to the 2nd-order
    difference. The (2,4) far term f[i+2]-f[i-1] (fwd) / f[i+1]-f[i-2] (bwd)
    reaches outside the domain for any gap within 2 cells of a non-periodic face;
    those zero-padded far terms form a malformed stencil, so revert them. Width-2
    covers every boundary-affected gap for BOTH directions (over-reverts at most
    one interior-valid cell — negligible in the large domains 4th order targets;
    for tiny axes it safely degenerates to all-2nd-order)."""
    lead = [slice(None)] * d4.ndim
    lead[axis] = slice(0, 2)
    trail = [slice(None)] * d4.ndim
    trail[axis] = slice(-2, None)
    d4 = d4.at[tuple(lead)].set(d2[tuple(lead)])
    d4 = d4.at[tuple(trail)].set(d2[tuple(trail)])
    return d4


def _diff_fwd_o(arr, axis, periodic, order, bloch=None):
    """Forward staggered first difference (f[i+1]-f[i] family), at i+1/2; the
    caller divides by dx. order=2 is byte-identical to ``_shift_fwd(arr)-arr``
    when ``bloch is None``.

    ``bloch`` (oblique-periodic Bloch field-transformation, #404): a length-3
    tuple of per-axis complex phases ``exp(-j·k_axis·dx)``.  On a PERIODIC axis
    the forward-rolled neighbour is multiplied by ``bloch[axis]`` so the plain
    ``jnp.roll`` wrap represents the exact discrete Yee derivative of a wave with
    transverse wavenumber ``k_axis``.  ``bloch=None`` (default) leaves the real
    path untouched; ``bloch`` is only threaded on order-2 (guarded upstream)."""
    if order == 2:
        if periodic[axis]:
            nxt = jnp.roll(arr, -1, axis)
            if bloch is not None:
                nxt = nxt * bloch[axis]
            return nxt - arr
        return _shift_fwd(arr, axis) - arr
    # order == 4:  c1*(f[i+1]-f[i]) + c2*(f[i+2]-f[i-1])
    if periodic[axis]:
        near = jnp.roll(arr, -1, axis) - arr
        far = jnp.roll(arr, -2, axis) - jnp.roll(arr, 1, axis)
        return _C4_NEAR * near + _C4_FAR * far
    near = _shift_fwd(arr, axis) - arr
    far = _shift_fwd(_shift_fwd(arr, axis), axis) - _shift_bwd(arr, axis)
    return _ribbon_2nd(_C4_NEAR * near + _C4_FAR * far, near, axis)


def _diff_bwd_o(arr, axis, periodic, order, bloch=None, *, boundary=None, frame=None, owned=None):
    """Backward staggered first difference (f[i]-f[i-1] family), at i-1/2; the
    caller divides by dx. order=2 is byte-identical to ``arr-_shift_bwd(arr)``
    when ``bloch is None``.

    For the Bloch field-transformation (#404) the BACKWARD-rolled neighbour
    carries the CONJUGATE phase ``exp(+j·k_axis·dx)`` (``bloch[axis].conjugate()``),
    the exact discrete adjoint of the forward stagger.  See ``_diff_fwd_o``."""
    if boundary is not None:
        periodic = boundary.periodic
    if order != 2 and boundary is not None and boundary.pmc_faces:
        raise NotImplementedError("PMC magnetic image: yee._diff_bwd_o far neighbors "
                                  "require stencil_order=2")
    back = h_neighbor(arr, axis, periodic=periodic, boundary=boundary, frame=frame, owned=owned)
    if order == 2:
        if periodic[axis] and bloch is not None:
            back = back * bloch[axis].conjugate()
        return arr - back

    # order == 4:  c1*(f[i]-f[i-1]) + c2*(f[i+1]-f[i-2])
    if periodic[axis]:
        near = arr - back
        far = jnp.roll(arr, -1, axis) - jnp.roll(arr, 2, axis)
        return _C4_NEAR * near + _C4_FAR * far
    near = arr - back
    far = _shift_fwd(arr, axis) - _shift_bwd(_shift_bwd(arr, axis), axis)
    return _ribbon_2nd(_C4_NEAR * near + _C4_FAR * far, near, axis)


class CurlBoundary(NamedTuple):
    """Realized terminal walls and periodic axes of an E-update stencil.

    Built from ``resolve_wall_faces`` by each single-device step setup.
    Tangential H is read with an odd image across each magnetic face.
    """

    pec_faces: frozenset = frozenset()
    pmc_faces: frozenset = frozenset()
    periodic: tuple = (False, False, False)


def h_neighbor(h, axis, *, boundary=None, periodic=(False, False, False),
               index=None, frame=None, owned=None):
    """Backward operand for a tangential-H difference at an E node.

    With N cells, physical H indices are 0..N-1 and the stored high ghost
    is N. PMC requires D_H[0]=2H[0] and D_H[N]=-2H[N-1]. The high operand
    includes H[N] to cancel the stored ghost at consumption. The zero update
    coefficient holds that ghost at zero; no physical H is zeroed.
    Periodic and indexed invariant-axis conventions take precedence.
    ``frame`` bounds this axis's physical rows; ``owned`` selects global faces.
    Neither argument applies to an indexed read.
    """
    if index is not None and (frame is not None or owned is not None):
        raise ValueError("indexed h_neighbor does not accept frame or owned")
    if boundary is not None:
        periodic = boundary.periodic
    n = h.shape[axis]
    from rfx.boundaries.pmc import magnetic_image_faces
    faces = magnetic_image_faces(boundary.pmc_faces, h.shape, periodic) if boundary is not None else frozenset()
    lo = f"{'xyz'[axis]}_lo" in faces
    hi = f"{'xyz'[axis]}_hi" in faces
    if index is None:
        back = jnp.roll(h, 1, axis) if periodic[axis] else _shift_bwd(h, axis)
        edge = [slice(None)] * h.ndim
        lo_owner = True if owned is None else owned.get(f"{'xyz'[axis]}_lo", True)
        hi_owner = True if owned is None else owned.get(f"{'xyz'[axis]}_hi", True)
        if lo and lo_owner is not False:
            edge[axis] = 0 if frame is None else frame[0]
            value = -h[tuple(edge)]
            if lo_owner is not True:
                value = jnp.where(lo_owner, value, back[tuple(edge)])
            back = back.at[tuple(edge)].set(value)
        if hi and hi_owner is not False:
            edge[axis] = -1 if frame is None else frame[1] - 1
            inner = list(edge)
            inner[axis] = -2 if frame is None else frame[1] - 2
            value = h[tuple(edge)] + 2 * h[tuple(inner)]
            if hi_owner is not True:
                value = jnp.where(hi_owner, value, back[tuple(edge)])
            back = back.at[tuple(edge)].set(value)
        return back
    i = int(index[axis])
    if n == 1 or periodic[axis]:
        back = list(index)
        back[axis] = (i - 1) % n
        return h[tuple(back)]
    if i == 0:
        return -h[index] if lo else jnp.zeros_like(h[index])
    back = list(index)
    back[axis] = i - 1
    if hi and i == n - 1:
        return h[index] + 2 * h[tuple(back)]
    return h[tuple(back)]


def curl_h(hx, hy, hz, dx: float | None, periodic: tuple,
           stencil_order: int = 2, bloch: tuple | None = None, *,
           boundary=None, scale_after=False, inv_spacing=None, frame=None, owned=None):
    """The uniform E-update curl, with the realized terminal boundary.

    ``scale_after`` preserves ADE's subtraction-before-division arithmetic;
    ``dx=None`` preserves baked coefficients (which already contain 1/dx).
    UPML passes its broadcast inverse metrics via ``inv_spacing``. These
    forms preserve the pre-B3a floating-point operation order, not different
    boundary stencils. H is cast to the caller's compute dtype beforehand.
    """
    if boundary is not None:
        periodic = boundary.periodic

    def component(ha, a, hb, b):
        da = _diff_bwd_o(ha, a, periodic, stencil_order, bloch, boundary=boundary,
                         frame=None if frame is None else frame.get("xyz"[a]), owned=owned)
        db = _diff_bwd_o(hb, b, periodic, stencil_order, bloch, boundary=boundary,
                         frame=None if frame is None else frame.get("xyz"[b]), owned=owned)
        if inv_spacing is not None:
            return da * inv_spacing[a] - db * inv_spacing[b]
        if dx is None:
            return da - db
        if scale_after:
            return (da - db) / dx
        return da / dx - db / dx

    return (component(hz, 1, hy, 2), component(hx, 2, hz, 0),
            component(hy, 0, hx, 1))


def curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, *, boundary=None, frame=None, owned=None):
    """Graded E-update curl using the same boundary neighbour convention.

    Inverse spacings are the E-update dual metrics, unchanged by B3a.
    """
    return curl_h(
        hx, hy, hz, None, (False, False, False), boundary=boundary, frame=frame, owned=owned,
        inv_spacing=(inv_dx[:, None, None], inv_dy[None, :, None],
                     inv_dz[None, None, :]))


def _walls_off_periodic(high_wall, periodic):
    """A run-level periodic axis has no wall, even where the grid lists none (#1966)."""
    return tuple(False if p else w for w, p in zip(high_wall, periodic))


def _material_wall_view(arr, high_wall, *, array_module=jnp):
    """Read the last inside cell at global high walls, without changing cells."""
    for axis, wall in enumerate(high_wall):
        if arr.shape[axis] < 2 or wall is False:
            continue
        indices = array_module.arange(arr.shape[axis])
        indices = array_module.where(wall & (indices == arr.shape[axis] - 1),
                                     indices - 1, indices)
        arr = array_module.take(arr, indices, axis=axis)
    return arr


def _material_bwd_neighbour(arr, ax, periodic, *, high_wall: tuple[bool, bool, bool], array_module=jnp):
    """The cell one step back along ``ax``, with the MATERIAL's outside rule.

    ``jnp.roll(arr, 1, axis=ax)`` on a periodic axis and on a length-1 axis
    (the 2-D lane's self-adjacency) — value for value what
    ``rfx.boundaries.pec._shift(arr, ax, periodic, +1)`` returns there, so the
    edge-average below and the PEC incidence rule wrap the same way (#689/#931,
    pinned by ``test_edge_averaged_materials``'s periodic test).

    Off a periodic axis the two rules differ, and must: ``_shift`` pads with
    ZERO because "no conductor outside the lattice" is the right reading of an
    occupancy, while a permittivity of zero outside the lattice is a division
    by zero and a conductivity of zero is a declaration about a region the
    lattice does not have. The material is edge-replicated instead: the cells
    outside take the boundary cell's own value, so a boundary edge averages
    over the cells that exist.
    """
    arr = _material_wall_view(arr, _walls_off_periodic(high_wall, periodic), array_module=array_module)
    if arr.shape[ax] == 1 or periodic[ax]:
        return array_module.roll(arr, 1, axis=ax)
    first = [slice(None)] * arr.ndim
    first[ax] = slice(0, 1)
    body = [slice(None)] * arr.ndim
    body[ax] = slice(0, arr.shape[ax] - 1)
    return array_module.concatenate([arr[tuple(first)], arr[tuple(body)]], axis=ax)


def edge_averaged_materials(eps_r, sigma, periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool], cell_sizes=None):
    """Per-E-component ``(eps_r, sigma)``: the mean over the edge's four cells.

    THE one spelling of the material-to-edge rule (#1210). Every lane that
    turns a cell-centred permittivity or conductivity into an E update
    coefficient goes through here; a hand-copied second rule is this repo's
    recurring defect.

    **The physics.** A Yee E component does not live inside a cell — it lies on
    an edge of the primal lattice, shared by the four cells around it:
    ``Ex(i+½, j, k)`` by cells ``(i, j−1..j, k−1..k)``, ``Ey(i, j+½, k)`` by
    ``(i−1..i, j, k−1..k)``, ``Ez(i, j, k+½)`` by ``(i−1..i, j−1..j, k)``. The
    field on that edge is TANGENTIAL to every interface between those cells, so
    its constitutive parameters are the tangential averages: the four
    conduction paths lie in parallel along the edge, which adds their
    conductances (σ arithmetic), and the tangential component of E is
    continuous across the interface, which makes the effective permittivity the
    arithmetic mean too (the tangential entry of the Kottke/Taflove §10 subcell
    average; the harmonic mean is the NORMAL entry and belongs to a different
    component). Taking the owning cell's value instead puts a conductor's plus
    faces one cell inside the drawn body and makes a dielectric interface
    first-order.

    ``eps_r``, ``sigma`` : cell-centred ``(nx, ny, nz)`` arrays. A conductivity
    that is already per E edge (#1216's design-box tuple) is NOT a cell
    quantity and does not come through here.
    ``periodic`` : per-axis flags; a periodic axis wraps, a non-periodic one
    edge-replicates (see :func:`_material_bwd_neighbour`).

    Returns ``((eps_x, eps_y, eps_z), (sig_x, sig_y, sig_z))``, grid-shaped.

    **Homogeneous regions are bit-identical** to the cell-owned rule: the four
    summands are the same float, and ``((a + a) + (a + a)) * 0.25 == a``
    exactly in binary floating point. That is what keeps every vacuum fixture
    and every uniform-dielectric lock byte-for-byte where it was.
    """
    return (edge_mean_components(eps_r, periodic, high_wall=high_wall, cell_sizes=cell_sizes),
            edge_mean_components(sigma, periodic, high_wall=high_wall, cell_sizes=cell_sizes))


def _edge_mean_four(values):
    """The unchanged equal-transverse-cell arithmetic and operation order."""
    return ((values[0] + values[1]) + (values[2] + values[3])) * 0.25


# Leading significant bits kept per part, (difference, fraction), by dtype:
# every part product then fits the significand and is exact.
_EXACT_PARTS = {"float32": ((12,), (12,), "uint32", 32, 24),
                "float64": ((26,), (18, 18), "uint64", 64, 53)}


def _exact_parts(x, leading, word, width, digits, xp):
    """``x`` as a sum of parts with few significant bits each; exact."""
    parts = []
    for keep in leading:
        mask = np.dtype(word).type(((1 << width) - 1) ^ ((1 << (digits - keep)) - 1))
        if xp is np:
            head = (np.ascontiguousarray(x).view(word) & mask).view(x.dtype)
        else:
            head = jax.lax.stop_gradient(jax.lax.bitcast_convert_type(
                jax.lax.bitcast_convert_type(x, word) & mask, x.dtype))
        parts.append(head)
        x = x - head
    return (*parts, x)


def _edge_mean_pair(lo, hi, fraction):
    """One pass of the separable primal-area rule for E materials and poles.

    ``lo + (hi - lo) * fraction``, with the product written as a sum of exact
    part products added in a fixed order. A multiply that feeds an add may be
    fused by the compiler, and the plain form then rounds differently eager,
    under jit and per device (one last bit on a graded mesh). A fused exact
    product rounds the same sum the same way, so this value has one bit
    pattern in every compile context, also when a stamp is added after it.

    That holds for operands that arrive as rounded arrays. A caller that
    forms the cells as a product in the same compiled program (a traced
    ``background + rho * contrast``) can still have THAT multiply fused into
    the subtraction here; non-finite cells and differences below about 1e-28
    (float32) are outside it as well.
    """
    spec = _EXACT_PARTS.get(str(getattr(hi, "dtype", "")))
    if spec is None or not hasattr(fraction, "dtype") or fraction.dtype != hi.dtype:
        # A Python 0.5 (an ungraded axis) is a power of two: already exact.
        return lo + (hi - lo) * fraction
    d_bits, f_bits, word, width, digits = spec
    xp = np if all(isinstance(v, (np.ndarray, np.generic)) for v in (lo, hi, fraction)) else jnp
    if xp is jnp:
        lo, hi, fraction = jnp.broadcast_arrays(jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(fraction))
    else:
        lo, hi, fraction = np.broadcast_arrays(lo, hi, fraction)
    d_parts = _exact_parts(hi - lo, d_bits, word, width, digits, xp)
    f_parts = _exact_parts(fraction, f_bits, word, width, digits, xp)
    result = lo
    for rank in range(len(d_parts) + len(f_parts) - 1):
        for i, d in enumerate(d_parts):
            j = rank - i
            if 0 <= j < len(f_parts):
                result = result + d * f_parts[j]
    return result


def _width_fraction(lo, d, array_module=jnp):
    """``d / (lo + d)`` for the pair mean, divided on the HOST when the widths
    are concrete.

    A GPU's float division is not correctly rounded, and a compiled program
    folds a division of captured constants on the host: the same widths gave
    one quotient in an eager call on a GPU and another under ``jax.jit``
    (RTX 4070 SUPER: 3 of 9 distinct quotients one bit apart). Concrete widths
    are therefore divided once, in NumPy, wherever the mean is formed. Traced
    widths are divided where they live; a lane that wants the same bits as the
    others keeps its widths concrete.
    """
    if array_module is np or isinstance(lo, jax.core.Tracer) or isinstance(d, jax.core.Tracer):
        return d / (lo + d)
    lo, d = np.asarray(lo), np.asarray(d)
    return jnp.asarray(d / (lo + d))


def _edge_mean_component(arr, t1, t2, periodic, cell_sizes, array_module=jnp, *, high_wall: tuple[bool, bool, bool]):
    high_wall = _walls_off_periodic(high_wall, periodic)
    arr = _material_wall_view(arr, high_wall, array_module=array_module)
    if cell_sizes is not None and any(cell_sizes[t] is not None for t in (t1, t2)):
        result = arr
        for t in (t1, t2):
            d = cell_sizes[t]
            if d is None:
                fraction = 0.5
            else:
                shape = [1, 1, 1]
                shape[t] = arr.shape[t]
                d = array_module.asarray(d, dtype=arr.dtype).reshape(shape)
                d = _material_wall_view(d, high_wall, array_module=array_module)
                lo = _material_bwd_neighbour(d, t, periodic, high_wall=high_wall, array_module=array_module)
                fraction = _width_fraction(lo, d, array_module)
            lo = _material_bwd_neighbour(result, t, periodic, high_wall=(False,) * 3, array_module=array_module)
            result = _edge_mean_pair(lo, result, fraction)
        return result
    a1 = _material_bwd_neighbour(arr, t1, periodic, high_wall=(False,) * 3, array_module=array_module)
    a2 = _material_bwd_neighbour(arr, t2, periodic, high_wall=(False,) * 3, array_module=array_module)
    a12 = _material_bwd_neighbour(a1, t2, periodic, high_wall=(False,) * 3, array_module=array_module)
    return _edge_mean_four((arr, a1, a2, a12))


def edge_mean_components(arr, periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool], cell_sizes=None, array_module=jnp):
    """``(mean_x, mean_y, mean_z)``: a cell array averaged over each E edge's
    four incident cells -- the arithmetic of :func:`edge_averaged_materials`,
    which calls it for ``eps_r`` and ``sigma``.

    Also the rule for a dispersive pole (#1260): a pole's susceptibility is
    parallel along the edge exactly as a conductivity is, so the edge carries
    ``delta_eps`` times the fraction of its four cells that hold the pole. The
    caller passes the pole's cell mask as a float array and gets that fraction
    back, per component. Equal cells give exact quarters; graded cells use
    the same primal-area fractions as epsilon and sigma. ``cell_sizes`` is
    a tuple of one-dimensional widths, with None for constant axes.
    """
    def mean4(t1, t2):
        return _edge_mean_component(arr, t1, t2, periodic, cell_sizes, array_module, high_wall=high_wall)

    return tuple(mean4(*[t for t in range(3) if t != c]) for c in range(3))


def component_e_materials(materials, periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool], cell_sizes=None):
    """Per-component ``(eps_r, sigma)`` of a :class:`MaterialArrays` (#1210).

    The volume part is edge-averaged; the lumped stamps
    (``sigma_lumped`` / ``eps_r_lumped``, see :class:`MaterialArrays`) are
    removed before the average and added back at their own cell, because a
    lumped device lives on an edge and not in a cell volume -- and only to
    the component whose edge it sits on (#1236): a port on Ez loads the Ez
    edge at its node, not the Ex and Ey edges that leave the same node.

    With no stamps this is :func:`edge_averaged_materials` on
    ``materials.eps_r`` and ``materials.sigma`` and nothing else.
    """
    eps_v = permittivity_without_lumped(materials)
    sig_v = materials.sigma
    sig_parts = lumped_components(getattr(materials, "sigma_lumped", None))
    sig_l = lumped_total(sig_parts)
    if sig_l is not None:
        sig_v = sig_v - sig_l
    eps_c, sig_c = edge_averaged_materials(
        eps_v, sig_v, periodic, high_wall=high_wall, **({} if cell_sizes is None else dict(cell_sizes=cell_sizes)))
    # Each stamp back on its OWN component (#1236).
    eps_c = add_lumped_eps(eps_c, getattr(materials, "eps_r_lumped", None))
    sig_c = tuple(s if part is None else s + part
                  for s, part in zip(sig_c, sig_parts))
    return eps_c, add_film(sig_c, materials)


_E_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}


def cell_component_e_materials(materials, cell, component,
                               periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool], cell_sizes=None):
    """``(eps_r, sigma)`` the E update uses for ONE component at ONE cell.

    :func:`component_e_materials` restricted to a single node, by indexing the
    four incident cells instead of building grid-sized arrays — what a source
    or port DRIVE coefficient needs, since it is built once per run and may be
    built from traced materials.

    The rule is the same rule: the mean of the VOLUME material over the four
    cells incident to that component's edge, plus the lumped stamps at this
    cell that sit on THIS component's edge (#1236). The out-of-domain
    convention is
    :func:`_material_bwd_neighbour`'s — wrap on a periodic or length-1 axis,
    edge-replicate otherwise. ``test_the_cell_helper_agrees_with_the_grid_wide_one``
    pins the two against each other, so this is not a second spelling.

    Why a drive coefficient needs it: ``make_j_source`` turns a current into a
    field increment through the SAME Cb the update multiplies the curl by, and
    a cell-centred Cb on a material step transverse to the injected component
    is off by the ratio of the two permittivities (measured: 0.556x the
    declared current on an eps 1|9 step).
    """
    axis = _E_COMPONENT_AXIS[str(component).lower()]
    t1, t2 = [a for a in range(3) if a != axis]
    cell = tuple(int(c) for c in cell)
    shape = tuple(materials.eps_r.shape)

    def back(idx, ax):
        if idx > 0:
            return idx - 1
        # index 0: wrap on a periodic or length-1 axis, replicate otherwise.
        return shape[ax] - 1 if (periodic[ax] or shape[ax] == 1) else 0

    idxs = []
    for d1 in (0, 1):
        for d2 in (0, 1):
            c = list(cell)
            if d1:
                c[t1] = back(cell[t1], t1)
            if d2:
                c[t2] = back(cell[t2], t2)
            idxs.append(tuple(n - 2 if wall and n > 1 and i == n - 1 else i
                              for i, n, wall in zip(c, shape, high_wall)))

    eps_parts = lumped_components(getattr(materials, "eps_r_lumped", None))
    sig_parts = lumped_components(getattr(materials, "sigma_lumped", None))

    def lumped_at(parts, idx):
        # ``lumped_total(parts)[idx]`` without a grid-sized sum: the parts are
        # added in the same x, y, z order, so the float is the same.
        total = None
        for part in parts:
            if part is not None:
                total = part[idx] if total is None else total + part[idx]
        return total

    def mean4(arr, parts):
        # The same arithmetic as the grid-wide rule: subtract every
        # component's stamps from each incident cell, average, add back only
        # this component's own stamp.
        v = [arr[i] for i in idxs]
        if any(part is not None for part in parts):
            v = [a - lumped_at(parts, i) for a, i in zip(v, idxs)]
        if cell_sizes is not None and any(cell_sizes[t] is not None for t in (t1, t2)):
            fractions = []
            for t in (t1, t2):
                if cell_sizes[t] is None:
                    fractions.append(0.5)
                else:
                    d = jnp.asarray(cell_sizes[t], dtype=arr.dtype)
                    hi = cell[t] - int(high_wall[t] and shape[t] > 1 and cell[t] == shape[t] - 1)
                    fractions.append(_width_fraction(d[back(cell[t], t)], d[hi]))
            hi = _edge_mean_pair(v[2], v[0], fractions[0])
            lo = _edge_mean_pair(v[3], v[1], fractions[0])
            m = _edge_mean_pair(lo, hi, fractions[1])
        else:
            # the grid-wide order: (cell + back t1) + (back t2 + both)
            m = _edge_mean_four((v[0], v[2], v[1], v[3]))
        own = parts[axis]
        return m if own is None else m + own[cell]

    film_at = materials._replace(sigma_film=map_lumped(
        materials.sigma_film, lambda part: part[cell]))
    sigma = mean4(materials.sigma, sig_parts)
    return (mean4(materials.eps_r, eps_parts),
            add_film((sigma,) * 3, film_at)[axis])


def cell_component_e_coeffs(materials, cell, component, dt,
                            periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool]):
    """``(Ca, Cb)`` of the E update for one component at one cell (#1210)."""
    eps_r, sigma = cell_component_e_materials(materials, cell, component,
                                              periodic, high_wall=high_wall)
    return e_update_coeffs(eps_r, sigma, dt)


def e_component_coeffs(materials, dt, periodic=(False, False, False), *, cell_sizes=None, high_wall: tuple[bool, bool, bool]):
    """Per-component E coefficients differentiated through their stored operands."""
    eps, sig = ((materials.components.eps_update, materials.components.sigma_update)
                if materials.components is not None
                else component_e_materials(materials, periodic,
                    **({} if cell_sizes is None else dict(cell_sizes=cell_sizes)), high_wall=high_wall))
    if _realized.ACTIVE is not None:
        eps, sig = _realized.electric(materials, eps, sig, "yee.E", periodic=periodic)
    pairs = [e_update_coeffs(e, s, dt) for e, s in zip(eps, sig)]
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def edge_averaged_e_update_coeffs(eps_r, sigma, dt,
                                  periodic=(False, False, False), *, high_wall: tuple[bool, bool, bool]):
    """``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` for the edge-averaged rule.

    :func:`edge_averaged_materials` followed by :func:`e_update_coeffs`, which
    stays the single spelling of the coefficient formula itself (the bit-
    identity lock ``tests/locks/test_sheet_refactor_bit_identity.py`` watches
    that function).
    """
    eps, sig = edge_averaged_materials(eps_r, sigma, periodic, high_wall=high_wall)
    pairs = [e_update_coeffs(e, s, dt) for e, s in zip(eps, sig)]
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def si_value_eps_r_grad(si_fn, eps_r_fn, *args):
    """``si_fn(*args)``, differentiated as ``eps_r_fn(*args)`` (#1357).

    The E coefficients are written in SI units, ``eps = eps_r*EPS_0`` and a
    division by it. The VALUE is fine; the REVERSE pass is not. The VJP of
    ``x/eps`` multiplies the cotangent by ``eps**-2`` (about 1.3e22 in vacuum),
    so a float32 cotangent above about 2.6e16 on Cb overflows to inf, and
    ``0*inf`` then gives NaN on the lossless cells. Whether it fires depends
    only on the objective's scale: a sum of squared fields from a
    current-driven graded-mesh run is about 1e17. The ADI lane's
    ``(MU_0*eps*dx**2)**-2`` overflows even at a cotangent of 1.

    ``eps_r_fn`` computes the same coefficients written in eps_r, with the SI
    constants grouped as Python floats such as ``dt/EPS_0``
    (:func:`e_coeffs_eps_r_units`), so its derivative stays in range. A
    ``jax.custom_jvp`` returns ``si_fn``'s value and takes the tangent from
    ``jax.jvp(eps_r_fn)``. The primal is ``si_fn`` itself, so a run that is
    not differentiated compiles exactly the graph it compiled before. That
    matters because XLA's algebraic simplifier rewrites the SI arithmetic
    (measured on the 3-D ADI lane), so any extra op in the forward graph,
    even one whose value is exactly 0, can move the bits.

    ``args`` are pytrees. Their ``jax.Array`` leaves (concrete or traced) are
    the differentiable inputs, and every other leaf (a Python-float ``dt``, a
    NumPy array) is held fixed. Every array the coefficients depend on must
    therefore be an ARGUMENT: differentiating with respect to a value that
    ``si_fn`` or ``eps_r_fn`` closes over raises ``UnexpectedTracerError``
    (JAX 0.6.2 and 0.10.2). Closing over a traced value that is not
    differentiated works. With no ``jax.Array`` leaf the call is plain host
    arithmetic and returns ``si_fn(*args)``.
    """
    leaves, treedef = jax.tree.flatten(args)
    dyn = [i for i, leaf in enumerate(leaves) if isinstance(leaf, jax.Array)]
    if not dyn:
        return si_fn(*args)

    # The closures below outlive this call (JAX caches a custom_jvp function
    # with what it closes over), so they must not hold the argument arrays:
    # the array slots are filled from the call's own operands each time.
    fixed = [None if isinstance(leaf, jax.Array) else leaf for leaf in leaves]

    def _args(values):
        full = list(fixed)
        for i, v in zip(dyn, values):
            full[i] = v
        return jax.tree.unflatten(treedef, full)

    @jax.custom_jvp
    def _coeffs(*values):
        return si_fn(*_args(values))

    @_coeffs.defjvp
    def _coeffs_jvp(primals, tangents):
        out = _coeffs(*primals)
        _, t_out = jax.jvp(lambda *v: eps_r_fn(*_args(v)), primals, tangents)
        return out, jax.tree.map(lambda t, o: t.astype(o.dtype), t_out, out)

    return _coeffs(*(leaves[i] for i in dyn))


def e_coeffs_eps_r_units(eps_r, sigma, dt, *, inverse=False):
    """``(loss, Cb)`` of the lossy E update written in eps_r units (#1357).

    ``k = dt/(2*EPS_0)``, ``loss = sigma*k/eps_r``, ``Cb = (dt/EPS_0) /
    (eps_r + sigma*k)``. These are the SI ``sigma*dt/(2*eps)`` and ``(dt/eps)
    / (1 + loss)`` with ``eps = eps_r*EPS_0`` cancelled, and ``Ca = (1 -
    loss)/(1 + loss)``. When ``dt`` is a number, ``dt/EPS_0`` and ``k`` are
    Python floats, so no SI-sized factor ever meets a cotangent. With
    ``inverse=True`` the first argument is ``1/eps_r`` (the Stage-2 subpixel
    tensor, 0 on a frozen PEC edge): ``loss = sigma*k*inv``, ``Cb =
    (dt/EPS_0)*inv/(1 + loss)``.

    This is the derivative half of :func:`si_value_eps_r_grad` for every
    Ca/Cb builder.
    """
    if not isinstance(dt, jax.Array):
        dt = float(dt)
    k = dt / (2.0 * EPS_0)
    dt_eps0 = dt / EPS_0
    if inverse:
        loss = (sigma * k) * eps_r
        return loss, (dt_eps0 * eps_r) / (1.0 + loss)
    return (sigma * k) / eps_r, dt_eps0 / (eps_r + sigma * k)


def _ca_cb_eps_r_units(eps_r, sigma, dt, inverse=False):
    """``(Ca, Cb)`` from :func:`e_coeffs_eps_r_units`."""
    loss, cb = e_coeffs_eps_r_units(eps_r, sigma, dt, inverse=inverse)
    return (1.0 - loss) / (1.0 + loss), cb


def _e_update_coeffs_si(eps_r, sigma, dt):
    eps = eps_r * EPS_0
    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)
    return ca, cb


def e_update_coeffs(eps_r, sigma, dt):
    """``(Ca, Cb)`` of the lossy E update for a permittivity and conductivity.

    Ca = (1 - σ·dt/(2ε)) / (1 + σ·dt/(2ε)),  Cb = (dt/ε) / (1 + σ·dt/(2ε)).

    One spelling, shared by :func:`update_e` — which builds the pair over the
    whole grid from ``materials`` — and :func:`update_e_box`, which builds it
    over one box from arrays that need not come from ``materials`` (#1179).
    The arguments are arrays of any shape (or scalars); nothing here is
    grid-sized by construction.

    Issue #1357: the values are the SI spelling's bits and the derivative is the
    eps_r spelling's (:func:`si_value_eps_r_grad`). A traced ``eps_r`` or
    ``sigma`` therefore gets a finite gradient at any objective scale.
    """
    return si_value_eps_r_grad(_e_update_coeffs_si, _ca_cb_eps_r_units,
                               eps_r, sigma, dt)


# ---------------------------------------------------------------------------
# Non-uniform mesh updates
# ---------------------------------------------------------------------------

def _nu_e_metric_kwargs(materials, inverse_duals, cell_sizes):
    """Raw NU cells require primal metrics, including an explicit constant axis.

    Rounded inverse duals cannot establish primal-cell constancy. Grid callers
    supply electric_cell_sizes(grid), whose None axes use grid.is_constant.
    """
    if materials.components is not None:
        return {}
    if cell_sizes is not None:
        return dict(cell_sizes=cell_sizes)
    raise ValueError(
        "non-uniform E update requires realized component materials or primal "
        "cell_sizes; unequal or traced dual metrics cannot select the E-edge material rule")


# ---------------------------------------------------------------------------
# Pre-computed update coefficients for high-throughput scan loops
# ---------------------------------------------------------------------------

class UpdateCoeffs(NamedTuple):
    """Pre-computed FDTD update coefficients.

    Eliminates per-step coefficient recomputation and optionally bakes in
    PEC boundary enforcement (zero coefficients at boundary cells) to
    remove the need for separate ``apply_pec()`` calls.

    Use :func:`precompute_coeffs` to build.
    """
    # H coefficient: one grid array, or (Hx, Hy, Hz) arrays with wire stamps.
    ch: object
    # E-field decay coefficient  — per-component (Nx, Ny, Nz)
    ca_ex: jnp.ndarray
    ca_ey: jnp.ndarray
    ca_ez: jnp.ndarray
    # E-field curl coefficient (includes 1/dx)  — per-component (Nx, Ny, Nz)
    cb_ex: jnp.ndarray
    cb_ey: jnp.ndarray
    cb_ez: jnp.ndarray


def precompute_coeffs(
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    pec_axes: str = "",
    pec_faces=(),
    periodic: tuple = (False, False, False),
    high_wall: tuple[bool, bool, bool],
) -> UpdateCoeffs:
    """Pre-compute all FDTD update coefficients.

    Parameters
    ----------
    materials : MaterialArrays
    dt, dx : float
    pec_axes : str
        Legacy spelling: axes on which to bake PEC (zero tangential E)
        into the coefficients on BOTH faces.
    pec_faces : iterable of str
        Face labels (``"x_lo"`` ...) to bake, the per-face spelling that
        ``resolve_wall_faces`` produces (#1164); a magnetic face is not in
        it. Union with ``pec_axes``. With these baked, ``apply_pec_faces``
        is no longer needed for those faces.
    periodic : tuple of 3 bools
        Passed to :func:`edge_averaged_materials` for the wrap convention of
        the edge average (#1210). The GPU fast lane this bakes for is gated
        on all-False today; the parameter exists so the rule has one spelling.

    Returns
    -------
    UpdateCoeffs
    """
    mu = (materials.components.mu_update if materials.components is not None
          else component_h_materials(materials, periodic, high_wall=high_wall))
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "precompute.H", periodic=periodic)
    # Preserve the historical scalar-array bake, including its f32 numerator.
    ch = tuple(jnp.float32(dt / (MU_0 * dx)) / m for m in mu)
    if materials.mu_r_wire is not None:
        ch = tuple((dt / (MU_0 * dx)) / m for m in mu)
    elif _realized.ACTIVE is not None:
        # Replay each component with the no-record float32 bake.
        ch = tuple(jnp.float32(dt / (MU_0 * dx)) / m for m in mu)

    # #1210: per-component eps/sigma, the mean over the four cells incident
    # to each component's edge. The arithmetic below is unchanged, so a
    # homogeneous grid bakes the same bits it baked before.
    _eps_c, _sig_c = ((materials.components.eps_update, materials.components.sigma_update) if materials.components is not None
                      else component_e_materials(materials, periodic, high_wall=high_wall))
    if _realized.ACTIVE is not None:
        _eps_c, _sig_c = _realized.electric(
            materials, _eps_c, _sig_c, "precompute.E", periodic=periodic)

    def _bake_si(eps_r_c, sigma_c, dt, dx):
        eps = eps_r_c * jnp.float32(EPS_0)
        loss = sigma_c * jnp.float32(dt) / (jnp.float32(2.0) * eps)
        denom = jnp.float32(1.0) + loss
        return ((jnp.float32(1.0) - loss) / denom,
                (jnp.float32(dt) / eps) / (denom * jnp.float32(dx)))

    def _bake_eps_r(eps_r_c, sigma_c, dt, dx):
        ca, cb = _ca_cb_eps_r_units(eps_r_c, sigma_c, dt)
        return ca, cb / dx

    def _bake(eps_r_c, sigma_c):
        # #1357: these bits, the eps_r-unit derivative.
        return si_value_eps_r_grad(_bake_si, _bake_eps_r,
                                   eps_r_c, sigma_c, dt, dx)

    (ca_ex, cb_ex) = _bake(_eps_c[0], _sig_c[0])
    (ca_ey, cb_ey) = _bake(_eps_c[1], _sig_c[1])
    (ca_ez, cb_ez) = _bake(_eps_c[2], _sig_c[2])

    # Bake the electric walls into the coefficients by zeroing Ca and Cb
    # of the tangential E components on each wall face -- the same planes
    # ``apply_pec_faces`` zeroes.
    faces = set(pec_faces)
    for a in pec_axes:
        faces.update({f"{a}_lo", f"{a}_hi"})
    if faces:
        lo, hi = 0, -1
        if "x_lo" in faces:
            ca_ey = ca_ey.at[lo, :, :].set(0.0)
            ca_ez = ca_ez.at[lo, :, :].set(0.0)
            cb_ey = cb_ey.at[lo, :, :].set(0.0)
            cb_ez = cb_ez.at[lo, :, :].set(0.0)
        if "x_hi" in faces:
            ca_ey = ca_ey.at[hi, :, :].set(0.0)
            ca_ez = ca_ez.at[hi, :, :].set(0.0)
            cb_ey = cb_ey.at[hi, :, :].set(0.0)
            cb_ez = cb_ez.at[hi, :, :].set(0.0)
        if "y_lo" in faces:
            ca_ex = ca_ex.at[:, lo, :].set(0.0)
            ca_ez = ca_ez.at[:, lo, :].set(0.0)
            cb_ex = cb_ex.at[:, lo, :].set(0.0)
            cb_ez = cb_ez.at[:, lo, :].set(0.0)
        if "y_hi" in faces:
            ca_ex = ca_ex.at[:, hi, :].set(0.0)
            ca_ez = ca_ez.at[:, hi, :].set(0.0)
            cb_ex = cb_ex.at[:, hi, :].set(0.0)
            cb_ez = cb_ez.at[:, hi, :].set(0.0)
        if "z_lo" in faces:
            ca_ex = ca_ex.at[:, :, lo].set(0.0)
            ca_ey = ca_ey.at[:, :, lo].set(0.0)
            cb_ex = cb_ex.at[:, :, lo].set(0.0)
            cb_ey = cb_ey.at[:, :, lo].set(0.0)
        if "z_hi" in faces:
            ca_ex = ca_ex.at[:, :, hi].set(0.0)
            ca_ey = ca_ey.at[:, :, hi].set(0.0)
            cb_ex = cb_ex.at[:, :, hi].set(0.0)
            cb_ey = cb_ey.at[:, :, hi].set(0.0)

    return UpdateCoeffs(
        ch=ch,
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex, cb_ey=cb_ey, cb_ez=cb_ez,
    )


def update_he_fast(state: FDTDState, coeffs: UpdateCoeffs, *, boundary=None) -> FDTDState:
    """Combined H + E update using pre-computed :class:`UpdateCoeffs`.

    Performs a full leapfrog step (H half-step then E full-step) with
    PEC baked into the coefficients.  This is the fastest path for the
    common case of non-periodic boundaries with uniform mesh.
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630). This is
    # the GPU-only fast lane (rfx/simulation.py's update_he_fast dispatch);
    # the uniform CPU lane routes through update_h/update_e above instead.
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    # --- H update (upcast to _cdtype for arithmetic) ---
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    ch = coeffs.ch
    cx, cy, cz = ch if isinstance(ch, tuple) else (ch,) * 3
    hx = (state.hx.astype(_cdtype) - cx * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - cy * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - cz * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    # --- E update (with PEC baked into coefficients) ---
    # Upcast newly computed H fields back to _cdtype for curl computation
    hx_f = hx.astype(_cdtype)
    hy_f = hy.astype(_cdtype)
    hz_f = hz.astype(_cdtype)
    curl_x, curl_y, curl_z = curl_h(
        hx_f, hy_f, hz_f, None, (False, False, False), boundary=boundary)
    ex = (coeffs.ca_ex * ex + coeffs.cb_ex * curl_x).astype(_fdtype)
    ey = (coeffs.ca_ey * ey + coeffs.cb_ey * curl_y).astype(_fdtype)
    ez = (coeffs.ca_ez * ez + coeffs.cb_ez * curl_z).astype(_fdtype)
    return FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                     step=state.step + 1)


def _require_nu_aniso_metrics(update):
    """Validate raw-cell metrics before the anisotropic builder is traced."""
    @wraps(update)
    def checked(state, materials, eps_ex, eps_ey, eps_ez, dt,
                inv_dx, inv_dy, inv_dz, *, high_wall, boundary=None, cell_sizes=None):
        metrics = _nu_e_metric_kwargs(materials, (inv_dx, inv_dy, inv_dz), cell_sizes)
        return update(state, materials, eps_ex, eps_ey, eps_ez, dt,
                      inv_dx, inv_dy, inv_dz, high_wall=high_wall, boundary=boundary, **metrics)
    return checked


def _e_update_coeffs_inv_si(inv, sigma, dt):
    """``(Ca, Cb)`` in inverse-eps form, ``inv = 1/eps_r``.

    ``loss = σ · dt · μ / (2 · ε₀)`` is finite for any (σ, μ) ≥ 0, and the
    ``1 + loss`` denominator is ≥ 1, so there is no division hazard.
    """
    inv_eps0 = 1.0 / EPS_0
    loss = 0.5 * sigma * dt * inv * inv_eps0
    ca = (1.0 - loss) / (1.0 + loss)
    cb = (dt * inv * inv_eps0) / (1.0 + loss)
    return ca, cb


def _ca_cb_inv_eps_r_units(inv, sigma, dt):
    return _ca_cb_eps_r_units(inv, sigma, dt, inverse=True)


# Moved to the stepping layer; the old import path resolves lazily (a module-level import would be circular).
_LAZY_EXPORTS = {
    "update_e": "rfx.stepping.yee",
    "update_e_aniso": "rfx.stepping.yee",
    "update_e_aniso_inv": "rfx.stepping.yee",
    "update_e_box": "rfx.stepping.yee",
    "update_e_fast": "rfx.stepping.yee",
    "update_e_nu": "rfx.stepping.yee",
    "update_e_nu_aniso": "rfx.stepping.yee",
    "update_h": "rfx.stepping.yee",
    "update_h_fast": "rfx.stepping.yee",
    "update_h_nu": "rfx.stepping.yee",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(_LAZY_EXPORTS[name]), name)
