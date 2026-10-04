"""Yee cell FDTD update equations — pure JAX functions.

All functions are jax.jit-compatible and operate on explicit state arrays.
No hidden mutable state.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

from rfx import _realized

import jax
import jax.numpy as jnp


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


def component_h_materials(materials, periodic=(False, False, False), *, cell_sizes=None):
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
        volume = []
        for axis in range(3):
            lo = _material_bwd_neighbour(mu, axis, periodic)
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
                lo_d = _material_bwd_neighbour(hi_d, axis, periodic)
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
            per_component(materials.sigma, sig_parts))


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


def _diff_bwd_o(arr, axis, periodic, order, bloch=None, *, boundary=None):
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
    back = h_neighbor(arr, axis, periodic=periodic, boundary=boundary)
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


@partial(jax.jit, static_argnums=(4, 5, 6))
def update_h(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False),
             stencil_order: int = 2,
             bloch: tuple | None = None) -> FDTDState:
    """Magnetic field half-step update (Faraday's law).

    H^{n+1/2} = H^{n-1/2} - (dt / μ) * curl(E^n)

    Note: uses a single ``dx`` for all three axes (cubic cells: dx=dy=dz).
    For non-uniform z grids, use ``update_h_nu`` / ``update_e_nu`` instead.

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    stencil_order: 2 (default, byte-identical) or 4 (Fang (2,4) wide stencil,
        reverting to 2nd order in the 1-cell ribbon at non-periodic faces).
    bloch: None (default, real path, byte-identical) or a length-3 tuple of
        per-axis complex phases ``exp(-j·k_axis·dx)`` for the oblique-periodic
        Bloch field-transformation (#404).  When set, the fields must be a
        complex dtype (the transformed envelope P); requires stencil_order=2.
    """
    so = stencil_order
    if so not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {so}")
    if materials.mu_r_wire is not None and so != 2:
        raise NotImplementedError("wire-port radius requires the second-order Yee stencil")
    if bloch is not None and so != 2:
        raise ValueError(
            f"bloch phase (oblique-periodic BC, #404) requires stencil_order=2, got {so}"
        )

    # Compute dtype: the oblique-periodic Bloch path carries a complex field
    # envelope P; the real path computes at ``max(state dtype, float32)`` --
    # upcast for reduced-precision (float16) fields (byte-identical for
    # float32 storage), and PROMOTED for float64 storage so the arithmetic
    # does not re-quantize a higher-precision field back to float32 every
    # timestep (issue #630 -- the prior hard `jnp.float32` pin here silently
    # discarded float64 field precision regardless of storage dtype).
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    mu = component_h_materials(materials, periodic)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", periodic=periodic)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    # curl E components via forward staggered differences (order=2 byte-identical)
    # dEz/dy - dEy/dz
    curl_x = (
        _diff_fwd_o(ez, 1, periodic, so, bloch) / dx
        - _diff_fwd_o(ey, 2, periodic, so, bloch) / dx
    )
    # dEx/dz - dEz/dx
    curl_y = (
        _diff_fwd_o(ex, 2, periodic, so, bloch) / dx
        - _diff_fwd_o(ez, 0, periodic, so, bloch) / dx
    )
    # dEy/dx - dEx/dy
    curl_z = (
        _diff_fwd_o(ey, 0, periodic, so, bloch) / dx
        - _diff_fwd_o(ex, 1, periodic, so, bloch) / dx
    )

    hx = (state.hx.astype(_cdtype) - ch_x * curl_x).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - ch_y * curl_y).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - ch_z * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


class CurlBoundary(NamedTuple):
    """Realized terminal walls and periodic axes of an E-update stencil.

    Built from ``resolve_wall_faces`` by each single-device step setup.
    Tangential H is read with an odd image across each magnetic face.
    """

    pec_faces: frozenset = frozenset()
    pmc_faces: frozenset = frozenset()
    periodic: tuple = (False, False, False)


def h_neighbor(h, axis, *, boundary=None, periodic=(False, False, False),
               index=None):
    """Backward operand for a tangential-H difference at an E node.

    With N cells, physical H indices are 0..N-1 and the stored high ghost
    is N. PMC requires D_H[0]=2H[0] and D_H[N]=-2H[N-1]. The high operand
    includes H[N] to cancel the stored ghost at consumption, including any
    source/absorber change since canonicalization. No physical H is zeroed.
    Periodic and indexed invariant-axis conventions take precedence.
    """
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
        if lo:
            edge[axis] = 0
            back = back.at[tuple(edge)].set(-h[tuple(edge)])
        if hi:
            edge[axis] = -1
            inner = list(edge)
            inner[axis] = -2
            back = back.at[tuple(edge)].set(h[tuple(edge)] + 2 * h[tuple(inner)])
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
           boundary=None, scale_after=False, inv_spacing=None):
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
        da = _diff_bwd_o(ha, a, periodic, stencil_order, bloch, boundary=boundary)
        db = _diff_bwd_o(hb, b, periodic, stencil_order, bloch, boundary=boundary)
        if inv_spacing is not None:
            return da * inv_spacing[a] - db * inv_spacing[b]
        if dx is None:
            return da - db
        if scale_after:
            return (da - db) / dx
        return da / dx - db / dx

    return (component(hz, 1, hy, 2), component(hx, 2, hz, 0),
            component(hy, 0, hx, 1))


def curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, *, boundary=None):
    """Graded E-update curl using the same boundary neighbour convention.

    Inverse spacings are the E-update dual metrics, unchanged by B3a.
    """
    return curl_h(
        hx, hy, hz, None, (False, False, False), boundary=boundary,
        inv_spacing=(inv_dx[:, None, None], inv_dy[None, :, None],
                     inv_dz[None, None, :]))


def _material_bwd_neighbour(arr, ax, periodic):
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
    if arr.shape[ax] == 1 or periodic[ax]:
        return jnp.roll(arr, 1, axis=ax)
    first = [slice(None)] * arr.ndim
    first[ax] = slice(0, 1)
    body = [slice(None)] * arr.ndim
    body[ax] = slice(0, arr.shape[ax] - 1)
    return jnp.concatenate([arr[tuple(first)], arr[tuple(body)]], axis=ax)


def edge_averaged_materials(eps_r, sigma, periodic=(False, False, False)):
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
    return (edge_mean_components(eps_r, periodic),
            edge_mean_components(sigma, periodic))


def edge_mean_components(arr, periodic=(False, False, False)):
    """``(mean_x, mean_y, mean_z)``: a cell array averaged over each E edge's
    four incident cells -- the arithmetic of :func:`edge_averaged_materials`,
    which calls it for ``eps_r`` and ``sigma``.

    Also the rule for a dispersive pole (#1260): a pole's susceptibility is
    parallel along the edge exactly as a conductivity is, so the edge carries
    ``delta_eps`` times the fraction of its four cells that hold the pole. The
    caller passes the pole's cell mask as a float array and gets that fraction
    back, per component (0, 0.25, 0.5, 0.75 or 1, each exact).
    """
    def mean4(t1, t2):
        a1 = _material_bwd_neighbour(arr, t1, periodic)
        a2 = _material_bwd_neighbour(arr, t2, periodic)
        a12 = _material_bwd_neighbour(a1, t2, periodic)
        # Pairwise, so the homogeneous sum is exact (F4/#1210).
        return ((arr + a1) + (a2 + a12)) * 0.25

    return tuple(mean4(*[t for t in range(3) if t != c]) for c in range(3))


def component_e_materials(materials, periodic=(False, False, False)):
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
    eps_c, sig_c = edge_averaged_materials(eps_v, sig_v, periodic)
    # Each stamp back on its OWN component (#1236).
    eps_c = add_lumped_eps(eps_c, getattr(materials, "eps_r_lumped", None))
    sig_c = tuple(s if part is None else s + part
                  for s, part in zip(sig_c, sig_parts))
    return eps_c, sig_c


_E_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}


def cell_component_e_materials(materials, cell, component,
                               periodic=(False, False, False)):
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
            idxs.append(tuple(c))

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
        m = ((v[0] + v[1]) + (v[2] + v[3])) * 0.25
        own = parts[axis]
        return m if own is None else m + own[cell]

    return (mean4(materials.eps_r, eps_parts),
            mean4(materials.sigma, sig_parts))


def cell_component_e_coeffs(materials, cell, component, dt,
                            periodic=(False, False, False)):
    """``(Ca, Cb)`` of the E update for one component at one cell (#1210)."""
    eps_r, sigma = cell_component_e_materials(materials, cell, component,
                                              periodic)
    return e_update_coeffs(eps_r, sigma, dt)


def e_component_coeffs(materials, dt, periodic=(False, False, False)):
    """``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` for a MaterialArrays.

    :func:`component_e_materials` then :func:`e_update_coeffs`. This is the
    entry every grid-wide E update uses.
    """
    eps, sig = component_e_materials(materials, periodic)
    if _realized.ACTIVE is not None:
        eps, sig = _realized.electric(materials, eps, sig, "yee.E", periodic=periodic)
    pairs = [e_update_coeffs(e, s, dt) for e, s in zip(eps, sig)]
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def edge_averaged_e_update_coeffs(eps_r, sigma, dt,
                                  periodic=(False, False, False)):
    """``((ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z))`` for the edge-averaged rule.

    :func:`edge_averaged_materials` followed by :func:`e_update_coeffs`, which
    stays the single spelling of the coefficient formula itself (the bit-
    identity lock ``tests/locks/test_sheet_refactor_bit_identity.py`` watches
    that function).
    """
    eps, sig = edge_averaged_materials(eps_r, sigma, periodic)
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

    def _args(values):
        full = list(leaves)
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

    #1357: the values are the SI spelling's bits and the derivative is the
    eps_r spelling's (:func:`si_value_eps_r_grad`). A traced ``eps_r`` or
    ``sigma`` therefore gets a finite gradient at any objective scale.
    """
    return si_value_eps_r_grad(_e_update_coeffs_si, _ca_cb_eps_r_units,
                               eps_r, sigma, dt)


def update_e_box(state: FDTDState, prev: FDTDState, box: tuple,
                 ca, cb, dx: float,
                 periodic: tuple = (False, False, False),
                 stencil_order: int = 2,
                 bloch: tuple | None = None,
                 inv_d: tuple | None = None, *, boundary=None) -> FDTDState:
    """Redo the E update inside one index box with its own Ca/Cb (#1179).

    ``prev`` is the state BEFORE the grid-wide E update of this timestep. It
    carries the same H^{n+1/2} that update consumed (nothing between the two
    touches H), so this recomputes ``E^{n+1} = Ca·E^n + Cb·curl(H^{n+1/2})``
    at the box cells exactly as :func:`update_e` would — from coefficients
    that do not have to come from ``materials``.

    Why it exists: the Yee E update is LINEAR in the fields, so reverse-mode
    AD needs the primal field only where a coefficient is a design variable.
    With ``materials`` left constant and only this box recomputed from a
    traced permittivity, every array the backward pass has to keep is
    box-shaped — ``prev.E`` and ``curl(H)`` enter only through the sliced
    products, and slicing and scatter are both linear (they save nothing).
    The grid-sized alternative is a traced ``materials.eps_r``, which puts
    six grid-sized arrays per step on the tape.

    ``ca`` and ``cb`` are box-shaped (or scalar) and built once, outside the
    time loop, by :func:`e_update_coeffs`. Each may instead be a 3-TUPLE
    (or list) of box-shaped arrays, one per E component in x, y, z order:
    component ``c`` is then updated with ``ca[c]``/``cb[c]``. Two things
    produce that form. The #1210 edge average makes every coefficient
    per-component — an E component takes the mean of eps and sigma over the
    four cells its edge touches — which is what
    :func:`rfx.simulation._design_box_edge_coeffs` builds. And a conducting
    SHEET (#1216) gives each component its own conductivity: its current runs
    along the two in-plane edges only, so the third component must keep the
    background coefficient while the other two carry the design conductivity.
    ``(v, v, v)`` is the scalar/array form cell for cell. Both sequence types
    are accepted here because :func:`rfx.simulation._resolve_design_box`
    accepts either for ``DesignBoxSpec.sigma``; it normalises what it builds
    to a tuple, so a list reaches this kernel only from a direct call.

    ``inv_d`` selects the GRADED-MESH curl (#1183). ``None`` (default) is the
    uniform :func:`curl_h` at spacing ``dx``; a tuple
    ``(inv_dx, inv_dy, inv_dz)`` of the E-update inverse spacings is
    :func:`curl_h_nu` instead, the curl :func:`update_e_nu` uses — so the box
    reproduces the non-uniform update the same way it reproduces the uniform
    one. ``dx``, ``periodic``, ``stencil_order`` and ``bloch`` are then
    unused: the non-uniform lane installs no periodic BC, has no
    fourth-order stencil and no Bloch phase.

    Cost: one extra whole-grid ``curl_h`` per step, the same shared stencil
    helper the E update and the #677 sheet operator use. Slicing H to the box
    plus a halo would avoid it, at the price of a second spelling of the
    stencil; measured on CPU the extra curl is repaid by the smaller tape.
    """
    i0, i1, j0, j1, k0, k1 = box
    sl = (slice(i0, i1), slice(j0, j1), slice(k0, k1))

    # Same compute/storage dtype policy as update_e -- the two results are
    # compared cell-for-cell by the equality gate, so they must round alike.
    # update_e_nu's policy is the same expression without the complex branch
    # (it has no Bloch path), so one spelling serves both lanes.
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    if inv_d is None:
        curl_x, curl_y, curl_z = curl_h(
            prev.hx.astype(_cdtype), prev.hy.astype(_cdtype),
            prev.hz.astype(_cdtype), dx, periodic, stencil_order, bloch,
            boundary=boundary)
    else:
        curl_x, curl_y, curl_z = curl_h_nu(
            prev.hx.astype(_cdtype), prev.hy.astype(_cdtype),
            prev.hz.astype(_cdtype), *inv_d, boundary=boundary)

    def _comp(v, c):
        """``v`` for every component, or the c-th entry of a 3-sequence."""
        return v[c] if isinstance(v, (tuple, list)) else v

    ex = (_comp(ca, 0) * prev.ex[sl].astype(_cdtype)
          + _comp(cb, 0) * curl_x[sl]).astype(_fdtype)
    ey = (_comp(ca, 1) * prev.ey[sl].astype(_cdtype)
          + _comp(cb, 1) * curl_y[sl]).astype(_fdtype)
    ez = (_comp(ca, 2) * prev.ez[sl].astype(_cdtype)
          + _comp(cb, 2) * curl_z[sl]).astype(_fdtype)

    return state._replace(
        ex=state.ex.at[sl].set(ex),
        ey=state.ey.at[sl].set(ey),
        ez=state.ez.at[sl].set(ez),
    )


@partial(jax.jit, static_argnames=("periodic", "stencil_order", "bloch", "boundary"))
def update_e(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False),
             stencil_order: int = 2,
             bloch: tuple | None = None, *, boundary=None) -> FDTDState:
    """Electric field full-step update (Ampere's law).

    For lossy media with conductivity σ:
    E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})

    Ca = (1 - σ*dt/(2ε)) / (1 + σ*dt/(2ε))
    Cb = (dt/ε) / (1 + σ*dt/(2ε))

    ε and σ are per COMPONENT (#1210): a Yee E component sits on an edge
    shared by four cells, and takes the mean of their ε and σ — see
    :func:`edge_averaged_materials` for the physics and the boundary
    convention. Homogeneous regions are bit-identical to the cell-owned rule
    this replaces.

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    stencil_order: 2 (default, byte-identical) or 4 (Fang (2,4) wide stencil,
        reverting to 2nd order in the 1-cell ribbon at non-periodic faces).
    bloch: None (default, real path, byte-identical) or a length-3 tuple of
        per-axis complex phases ``exp(-j·k_axis·dx)`` for the oblique-periodic
        Bloch field-transformation (#404); requires stencil_order=2 and complex
        fields.  The backward differences use the conjugate phase automatically.
    """
    so = stencil_order
    if so not in (2, 4):
        raise ValueError(f"stencil_order must be 2 or 4, got {so}")
    if bloch is not None and so != 2:
        raise ValueError(
            f"bloch phase (oblique-periodic BC, #404) requires stencil_order=2, got {so}"
        )

    # Compute dtype: complex for the Bloch envelope path, ``max(state dtype,
    # float32)`` for the real path (byte-identical for float32 storage;
    # upcast covers reduced-precision fields; promoted for float64 storage
    # -- see the matching comment in update_h, issue #630).
    _fdtype = state.ex.dtype
    _cdtype = (
        jnp.complex64 if jnp.iscomplexobj(state.ex)
        else jnp.promote_types(state.ex.dtype, jnp.float32)
    )
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: the coefficients are per COMPONENT, from the mean of eps_r and
    # sigma over the four cells incident to that component's edge.
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_component_coeffs(
        materials, dt, periodic)

    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, so, bloch, boundary=boundary)

    ex = (ca_x * state.ex.astype(_cdtype) + cb_x * curl_x).astype(_fdtype)
    ey = (ca_y * state.ey.astype(_cdtype) + cb_y * curl_y).astype(_fdtype)
    ez = (ca_z * state.ez.astype(_cdtype) + cb_z * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


# ---------------------------------------------------------------------------
# Non-uniform mesh updates
# ---------------------------------------------------------------------------

def update_h_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx_h: jnp.ndarray, inv_dy_h: jnp.ndarray, inv_dz_h: jnp.ndarray,
                *, cell_sizes=None,
                ) -> FDTDState:
    """H update for non-uniform Yee grid.

    The H-curl differences two E nodes that straddle a single cell, so
    it divides by the LOCAL cell width (CORE-C2 fix, 2026-05-16).

    Parameters
    ----------
    inv_dx_h, inv_dy_h, inv_dz_h : (N,) arrays
        H-update inverse spacing ``inv_d_h[k] = 1/d[k]`` for k<N-1,
        ``inv_d_h[N-1] = 0``. Built by
        ``rfx.nonuniform._profile_to_inv_arrays`` (second return value).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    if cell_sizes is None:
        # Compatibility for direct kernel callers supplying inverse primal
        # metrics. Production grids pass their original cell lengths.
        cell_sizes = tuple(1 / jnp.where(v > 0, v, v[-2])
                           for v in (inv_dx_h, inv_dy_h, inv_dz_h))
    mu = component_h_materials(materials, cell_sizes=cell_sizes)
    if _realized.ACTIVE is not None:
        mu = _realized.magnetic(materials, mu, "yee.H", cell_sizes=cell_sizes)
    mu_x, mu_y, mu_z = (m * MU_0 for m in mu)
    ch_x, ch_y, ch_z = dt / mu_x, dt / mu_y, dt / mu_z

    # Forward differences with same shape (zero-pad via _shift_fwd)
    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h[None, :, None]
    )

    hx = (state.hx.astype(_cdtype) - ch_x * curl_x).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - ch_y * curl_y).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - ch_z * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                *, boundary=None) -> FDTDState:
    """E update for non-uniform Yee grid.

    The E-curl differences two H cell-centres, whose separation is the
    MEAN of the two adjacent cell widths (CORE-C2 fix, 2026-05-16).

    Parameters
    ----------
    inv_dx, inv_dy, inv_dz : (N,) arrays
        E-update inverse spacing ``inv_d_e[k] = 2/(d[k-1]+d[k])`` for
        k>=1, ``inv_d_e[0] = 1/d[0]``. Built by
        ``rfx.nonuniform._profile_to_inv_arrays`` (first return value).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)

    # One spelling with the uniform lane and with update_e_box, so a design
    # box on a graded mesh builds the SAME Ca/Cb from its own permittivity
    # (#1183). #1210: per component, from the mean over the four cells
    # incident to the edge. Non-periodic: the graded-mesh lane installs no
    # periodic BC (the NU boundary carries no periodic axes), the same assumption
    # ``update_e_box``'s ``inv_d`` branch documents.
    (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = e_component_coeffs(
        materials, dt, (False, False, False))

    # Backward differences with same shape (zero-pad via _shift_bwd)
    curl_x, curl_y, curl_z = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, boundary=boundary)

    ex = (ca_x * state.ex.astype(_cdtype) + cb_x * curl_x).astype(_fdtype)
    ey = (ca_y * state.ey.astype(_cdtype) + cb_y * curl_y).astype(_fdtype)
    ez = (ca_z * state.ez.astype(_cdtype) + cb_z * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


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
    mu = component_h_materials(materials, periodic)
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
    _eps_c, _sig_c = component_e_materials(materials, periodic)
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
            ca_ey = ca_ey.at[lo, :, :].set(0.0); ca_ez = ca_ez.at[lo, :, :].set(0.0)
            cb_ey = cb_ey.at[lo, :, :].set(0.0); cb_ez = cb_ez.at[lo, :, :].set(0.0)
        if "x_hi" in faces:
            ca_ey = ca_ey.at[hi, :, :].set(0.0); ca_ez = ca_ez.at[hi, :, :].set(0.0)
            cb_ey = cb_ey.at[hi, :, :].set(0.0); cb_ez = cb_ez.at[hi, :, :].set(0.0)
        if "y_lo" in faces:
            ca_ex = ca_ex.at[:, lo, :].set(0.0); ca_ez = ca_ez.at[:, lo, :].set(0.0)
            cb_ex = cb_ex.at[:, lo, :].set(0.0); cb_ez = cb_ez.at[:, lo, :].set(0.0)
        if "y_hi" in faces:
            ca_ex = ca_ex.at[:, hi, :].set(0.0); ca_ez = ca_ez.at[:, hi, :].set(0.0)
            cb_ex = cb_ex.at[:, hi, :].set(0.0); cb_ez = cb_ez.at[:, hi, :].set(0.0)
        if "z_lo" in faces:
            ca_ex = ca_ex.at[:, :, lo].set(0.0); ca_ey = ca_ey.at[:, :, lo].set(0.0)
            cb_ex = cb_ex.at[:, :, lo].set(0.0); cb_ey = cb_ey.at[:, :, lo].set(0.0)
        if "z_hi" in faces:
            ca_ex = ca_ex.at[:, :, hi].set(0.0); ca_ey = ca_ey.at[:, :, hi].set(0.0)
            cb_ex = cb_ex.at[:, :, hi].set(0.0); cb_ey = cb_ey.at[:, :, hi].set(0.0)

    return UpdateCoeffs(
        ch=ch,
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex, cb_ey=cb_ey, cb_ez=cb_ez,
    )


def update_h_fast(state: FDTDState, ch: jnp.ndarray | tuple) -> FDTDState:
    """H update using pre-computed coefficient ``ch = dt/(mu*dx)``.

    Avoids recomputing material coefficients each timestep.
    Non-periodic boundaries only (uses ``_shift_fwd``).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    ex = state.ex.astype(_cdtype)
    ey = state.ey.astype(_cdtype)
    ez = state.ez.astype(_cdtype)
    cx, cy, cz = ch if isinstance(ch, tuple) else (ch,) * 3
    hx = (state.hx.astype(_cdtype) - cx * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(_cdtype) - cy * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(_cdtype) - cz * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_fast(
    state: FDTDState,
    ca_ex: jnp.ndarray, ca_ey: jnp.ndarray, ca_ez: jnp.ndarray,
    cb_ex: jnp.ndarray, cb_ey: jnp.ndarray, cb_ez: jnp.ndarray,
    *, boundary=None,
) -> FDTDState:
    """E update using pre-computed per-component coefficients.

    ``ca_*`` and ``cb_*`` already include PEC zeroing when built with
    :func:`precompute_coeffs` and ``pec_axes``, so no separate
    ``apply_pec()`` call is needed.

    Non-periodic boundaries only (uses ``_shift_bwd``).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, None, (False, False, False), boundary=boundary)
    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)
    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


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


@partial(jax.jit, static_argnames=("boundary",))
def update_e_nu_aniso(state: FDTDState, materials: MaterialArrays,
                      eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                      dt: float,
                      inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                      *, boundary=None) -> FDTDState:
    """Non-uniform E update with per-component anisotropic permittivity.

    Same backward-difference structure as :func:`update_e_nu` (E-update
    mean spacing via ``inv_dx``/``inv_dy``/``inv_dz``) but uses three separate
    permittivity arrays (Ex, Ey, Ez) so subpixel-smoothing can be applied
    on a non-uniform mesh. ``materials.sigma`` is still applied
    isotropically (matches the uniform-path :func:`update_e_aniso`).
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes. The graded-mesh lane installs no periodic BC.
    sigma_ex, sigma_ey, sigma_ez = component_e_materials(
        materials, (False, False, False))[1]

    # The same arithmetic as update_e's, so one spelling (and its #1357
    # eps_r-unit derivative) serves both.
    ca_ex, cb_ex = e_update_coeffs(eps_ex, sigma_ex, dt)
    ca_ey, cb_ey = e_update_coeffs(eps_ey, sigma_ey, dt)
    ca_ez, cb_ez = e_update_coeffs(eps_ez, sigma_ez, dt)

    # Backward differences with per-cell inv-spacing (mirrors update_e_nu)
    curl_x, curl_y, curl_z = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


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


@partial(jax.jit, static_argnames=("periodic", "boundary"))
def update_e_aniso_inv(state: FDTDState, materials: MaterialArrays,
                       inv_xx: jnp.ndarray, inv_yy: jnp.ndarray, inv_zz: jnp.ndarray,
                       dt: float, dx: float,
                       periodic: tuple = (False, False, False), *, boundary=None) -> FDTDState:
    """Electric field update with per-component **inverse** permittivity.

    Stage 2 production form: takes the diagonal of the inverse-eps
    tensor (``inv_xx``, ``inv_yy``, ``inv_zz``) and uses it directly in
    the Yee update via *multiplication*. Numerically stable in the PEC
    limit (``inv = 0``): the Ca/Cb coefficients reduce to ``Ca = 1``,
    ``Cb = 0`` cleanly, freezing the field. Compare to
    :func:`update_e_aniso` (forward-eps form), which would compute
    ``1/(eps + 1e-30)`` and produce a huge but finite scaling — the
    NaN-trap path the original Stage 1 implementation hit.

    Derivation: ``stage2_ca_cb_derivation.md`` §5. Let ``μ = 1/ε_r``
    (per-component dimensionless inverse permittivity); then
    ``ε_abs = ε₀/μ``, so:

        loss = σ · dt · μ / (2 · ε₀)
        Ca   = (1 − loss) / (1 + loss)
        Cb   = (dt · μ / ε₀) / (1 + loss)
        E^{n+1} = Ca · E^n + Cb · curl(H^{n+1/2})

    For PEC tangential (``μ = 0``): loss = 0 → Ca = 1, Cb = 0 → field
    frozen. For dielectric (μ = 1/ε_r): same numerical form as the
    legacy ``update_e_aniso`` to within float-arithmetic ordering
    (~5 ULP). For partial-PEC perpendicular (``μ = (1−f)/ε_out``):
    finite scaling, no division hazard.

    Parameters
    ----------
    state : FDTDState
    materials : MaterialArrays
        Used only for ``sigma`` (conductivity, isotropic per cell).
        ``materials.eps_r`` is **not** consulted — the per-component
        inverse permittivity passed via ``inv_xx``/``inv_yy``/``inv_zz``
        is the source of truth.
    inv_xx, inv_yy, inv_zz : jnp.ndarray
        Per-component inverse-permittivity arrays (shape grid.shape,
        dtype float32). Typical sources:
        :func:`rfx.geometry.smoothing.compute_inv_eps_tensor_diag`.
    dt, dx : float
    periodic : tuple of 3 bools
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes.
    sigma_ex, sigma_ey, sigma_ez = component_e_materials(materials, periodic)[1]

    # Per-component lossy update coefficients in inv-eps form
    # (:func:`_e_update_coeffs_inv_si`). #1357: those bits, the eps_r-unit
    # derivative (``* inv_eps0`` handed the cotangent a factor 1/EPS_0).
    ca_ex, cb_ex = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_xx, sigma_ex, dt)
    ca_ey, cb_ey = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_yy, sigma_ey, dt)
    ca_ez, cb_ez = si_value_eps_r_grad(
        _e_update_coeffs_inv_si, _ca_cb_inv_eps_r_units, inv_zz, sigma_ez, dt)

    # curl H (identical to update_e and update_e_aniso).
    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


def update_e_aniso(state: FDTDState, materials: MaterialArrays,
                   eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                   dt: float, dx: float,
                   periodic: tuple = (False, False, False), *, boundary=None) -> FDTDState:
    """Electric field update with per-component anisotropic permittivity.

    Same as :func:`update_e` but uses separate permittivity arrays for
    each E-field component (Ex, Ey, Ez) to support subpixel smoothing.

    The conductivity from ``materials.sigma`` is still applied isotropically.

    Parameters
    ----------
    state : FDTDState
    materials : MaterialArrays
        Used only for ``sigma`` (conductivity).
    eps_ex, eps_ey, eps_ez : jnp.ndarray
        Per-component relative permittivity arrays, each of shape (Nx, Ny, Nz).
    dt, dx : float
    periodic : tuple of 3 bools
    """
    # Compute dtype: max(state dtype, float32) -- byte-identical for float32
    # storage, upcast for float16, promoted for float64 (issue #630).
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    hx = state.hx.astype(_cdtype)
    hy = state.hy.astype(_cdtype)
    hz = state.hz.astype(_cdtype)
    # #1210: sigma is a VOLUME conductivity, so it takes the same
    # per-component edge average the subpixel lane already gives eps -- the
    # conduction paths of the four cells lie in parallel along the edge. The
    # permittivity here stays the Kottke/subpixel tensor, which is already
    # per-component by construction. Where sigma is uniform (every lossless
    # subpixel fixture) the mean of four equal floats is that float, so those
    # runs keep their bytes.
    sigma_ex, sigma_ey, sigma_ez = component_e_materials(materials, periodic)[1]

    if _realized.ACTIVE is not None:
        (eps_ex, eps_ey, eps_ez), (sigma_ex, sigma_ey, sigma_ez) = _realized.electric(
            materials, (eps_ex, eps_ey, eps_ez),
            (sigma_ex, sigma_ey, sigma_ez), "aniso.E", periodic=periodic)

    # Per-component lossy update coefficients: the arithmetic of
    # e_update_coeffs, so one spelling (and its #1357 eps_r-unit
    # derivative) serves both.
    ca_ex, cb_ex = e_update_coeffs(eps_ex, sigma_ex, dt)
    ca_ey, cb_ey = e_update_coeffs(eps_ey, sigma_ey, dt)
    ca_ez, cb_ez = e_update_coeffs(eps_ez, sigma_ez, dt)

    # curl H (same as update_e)
    curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, boundary=boundary)

    ex = (ca_ex * state.ex.astype(_cdtype) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(_cdtype) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(_cdtype) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )
