"""Reference-plane port waves for the lumped/wire S-matrix path (issue #313).

Phase-1 implementation of the REFERENCE-PLANE architecture promoted by the
issue #313 Phase-0 verdict (closed-box flux referee, 2026-07-10): the
port-cell wave definitions at a wire port are near-field-dominated (0 cells
from the discontinuity) and do not conserve power against a leak-free
closed-box referee, while line V/I measured a few cells outboard on the
uniform line does — the zero-free-parameter conservation identity
``(|f|^2 - |b|^2)/Re(Zc)`` closed at +1.19% -> -0.28% at all 9 bins and the
implied |S21| landed within ~2.5% of the box referee at all bins (shipped
port-cell extractor: 33-47% off).  This module measures port waves at
reference planes instead of at the port cells, for the OFF-DIAGONAL
S-matrix entries only.

Method (all conventions measured/anchored in the Phase-0 lane; constants in
the issue #313 Phase-0 comment — do not re-derive):

* Plane V = vertical-E GAP line integral at an integer Yee plane N cells
  outboard (into the DUT) along the line axis — over the port's
  component-axis cell range TRIMMED to the cells not PEC-masked at the
  plane column.  Measured witness (A/B vs point probes, 2026-07-10): the
  in-trace Ez edge of a one-cell-thick PEC trace is NOT zeroed by the
  staircase mask convention and carries ~0.21 of the gap field, so an
  untrimmed port-extent integral mis-scales V by ~0.89 (which cancels in
  S but corrupts the reported Zc; Phase-0 measured 47.9-48.6 ohm with
  the 2-cell gap integral).
* Plane I = average of the two adjacent Ampere loops around the signal
  trace at the half-planes (plane -/+ dx/2), centring I on the V plane
  (2nd-order de-stagger).  Loop legs sit half a cell outside the PEC
  trace bounding box (right-hand rule: positive current along the
  +line-axis direction).
* ``exp(+j*omega*dt/2)`` is applied to the H-derived (current) phasors at
  extraction — the Yee leapfrog half-step.  Comparator anchor: this
  collapses the receive-cell Ohm-law phase from +0.008..+0.018 to ~0
  (measured, Phase 0).
* Forward/backward split with a MEASURED two-plane line-invariant
  ``Zc^2 = (V1^2 - V2^2)/(I1^2 - I2^2)`` (exact for any load on a uniform
  line; measured 47.9-48.6 ohm on the canonical thru — never assume the
  port's nominal 50 ohm).  The opt-in therefore registers TWO planes per
  port: the reference plane at N cells outboard and a second plane at 2N
  cells outboard.
* Phase-only de-embedding to the port plane with MEASURED beta from the
  same two planes (slow-wave beta/(w/c) = 1.048-1.061 measured on the
  canonical thru — far above numerical-dispersion class, so an analytic
  beta would be wrong).  Magnitude-neutral by construction.
* Off-diagonals: ``S_ij = b_inc_plane(i) / a_fwd_plane(j)`` where both
  ports opted in.  DIAGONAL ``S_jj`` always stays on the byte-frozen
  legacy path (``decompose_wire_s_matrix`` — issue #308 conventions),
  untouched.

The scan-side DFT accumulators mirror the coax plane-V/I machinery class
(``extract_coaxial_plane_vi_from_dft`` and the wire/lumped
``*_sparam_accs`` carry channels in ``rfx.simulation``): static geometry is
precomputed here into :class:`WireRefPlaneSpec`, and the production
``jax.lax.scan`` body accumulates the rect-DFT phasors inline (same
pre-source-injection sample point and DFT kernel as the port-cell
accumulators).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax.numpy as jnp

_COMP_AXIS = {"ex": 0, "ey": 1, "ez": 2}
_H_OF_AXIS = {0: "hx", 1: "hy", 2: "hz"}
_AXIS_OF_NAME = {"x": 0, "y": 1, "z": 2}


class WireRefPlaneSpec(NamedTuple):
    """Static reference-plane V/I DFT spec for the compiled scan.

    One spec per (port, plane slot); an opted port registers TWO
    (``plane_slot`` 0 at ``n_cells_outboard = N`` and slot 1 at ``2N``).
    All index fields are Python ints (static under ``jax.jit``);
    ``freqs`` is the DFT frequency array.
    """

    port_index: int          # index into the sparam-eligible port order
    plane_slot: int          # 0 = reference plane (N cells), 1 = Zc plane (2N)
    n_cells_outboard: int    # this plane's offset from the port plane (cells)
    outboard_sign: int       # +1/-1: DUT direction along the line axis
    line_axis: int           # 0/1/2 — the transmission-line axis
    plane_index: int         # integer Yee plane index along line_axis
    e_component: str         # port E component ("ex"/"ey"/"ez")
    comp_axis: int           # E-component axis
    e_lo: int                # component-axis cell range of the gap V integral
    e_hi: int                # (exclusive)
    third_index: int         # fixed index on the remaining axis
    hu_component: str        # H along u  ((a, u, v) right-handed cyclic)
    hv_component: str        # H along v
    u_axis: int
    v_axis: int
    u_lo_leg: int            # Hv legs (columns) just outside the trace bbox
    u_hi_leg: int
    v_lo_leg: int            # Hu legs (rows) just outside the trace bbox
    v_hi_leg: int
    u_span_lo: int           # Hu leg span (exclusive hi)
    u_span_hi: int
    v_span_lo: int           # Hv leg span (exclusive hi)
    v_span_hi: int
    freqs: object            # (n_freqs,) jnp.ndarray
    impedance: float         # port Z0 (bookkeeping only — never used as Zc)


def _tri(by_axis: dict) -> tuple:
    """Assemble an (i, j, k) index tuple from an {axis: index-or-slice} map."""
    return (by_axis[0], by_axis[1], by_axis[2])


def _trace_bbox_at_plane(pec2d: np.ndarray, seed_uv: tuple[int, int]):
    """Bounding box of the signal-trace PEC connected component in a slice.

    ``pec2d`` is the (nu, nv) boolean PEC cross-section at the reference
    plane; ``seed_uv`` is the transverse cell of the port's extent END
    (which sits in/at the trace conductor).  If the seed cell itself is
    not PEC, the nearest PEC cell (Euclidean) seeds the component.
    4-connected BFS; returns (u_lo, u_hi, v_lo, v_hi) inclusive.
    """
    nu, nv = pec2d.shape
    su, sv = seed_uv
    if not (0 <= su < nu and 0 <= sv < nv):
        raise ValueError(
            f"reference-plane seed cell {seed_uv} outside the grid slice "
            f"({nu}x{nv})")
    if not pec2d[su, sv]:
        cand = np.argwhere(pec2d)
        if cand.size == 0:
            raise ValueError(
                "reference_plane_cells: no PEC conductor found in the "
                "reference-plane cross-section — the plane V/I method "
                "requires a PEC signal trace (uniform transmission line) "
                "at the plane. Check the port direction and N.")
        d2 = (cand[:, 0] - su) ** 2 + (cand[:, 1] - sv) ** 2
        su, sv = (int(x) for x in cand[int(np.argmin(d2))])
    # 4-connected BFS
    seen = np.zeros_like(pec2d, dtype=bool)
    stack = [(su, sv)]
    seen[su, sv] = True
    u_lo = u_hi = su
    v_lo = v_hi = sv
    while stack:
        u, v = stack.pop()
        u_lo, u_hi = min(u_lo, u), max(u_hi, u)
        v_lo, v_hi = min(v_lo, v), max(v_hi, v)
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            uu, vv = u + du, v + dv
            if 0 <= uu < nu and 0 <= vv < nv and pec2d[uu, vv] \
                    and not seen[uu, vv]:
                seen[uu, vv] = True
                stack.append((uu, vv))
    return u_lo, u_hi, v_lo, v_hi


def build_wire_refplane_specs(
    *,
    grid,
    port_cells,
    e_component: str,
    impedance: float,
    direction: str,
    n_cells: int,
    freqs,
    port_index: int,
    pec_mask,
) -> tuple:
    """Build the two per-port :class:`WireRefPlaneSpec` entries.

    Parameters
    ----------
    grid : Grid
    port_cells : list of (i, j, k)
        The wire port's rasterized cells (``_wire_port_cells`` output).
    e_component : str
        Port E component.
    direction : str
        The port's OUTWARD-normal direction ("+x"/"-x"/"+y"/"-y" —
        ``add_port`` semantics).  The reference planes go the OPPOSITE
        way (into the DUT along the line axis).
    n_cells : int
        N — the reference-plane offset in cells; the second (Zc) plane
        registers at 2N.
    pec_mask : array-like of bool
        The geometry PEC mask BEFORE port-cell clearing (the trace
        conductor must be present at the plane cross-sections).

    Returns
    -------
    (WireRefPlaneSpec, WireRefPlaneSpec)
        Plane slot 0 (N cells outboard) and slot 1 (2N cells outboard).

    Raises
    ------
    ValueError
        On geometry that cannot support the method: plane out of bounds,
        no PEC trace at a plane, or a NON-uniform line (the PEC trace
        cross-section bounding boxes at the two planes differ — the
        two-plane Zc invariant requires a uniform line between them).
    """
    if direction is None or direction not in ("+x", "-x", "+y", "-y"):
        raise ValueError(
            "reference_plane_cells requires an explicit port "
            f"direction ('+x'/'-x'/'+y'/'-y'), got {direction!r}")
    if pec_mask is None:
        raise ValueError(
            "reference_plane_cells: the simulation has no PEC geometry — "
            "the plane V/I method requires a PEC signal trace (uniform "
            "transmission line).")
    n_cells = int(n_cells)
    if n_cells < 1:
        raise ValueError(f"reference_plane_cells must be >= 1, got {n_cells}")

    line_axis = _AXIS_OF_NAME[direction[1]]
    outboard_sign = -1 if direction[0] == "+" else +1
    comp_axis = _COMP_AXIS[e_component]
    if comp_axis == line_axis:
        raise ValueError(
            f"reference_plane_cells: port component {e_component!r} lies "
            f"along the line axis {direction!r} — the gap-V line integral "
            "must be transverse to the line.")
    third_axis = 3 - line_axis - comp_axis

    cells = [tuple(int(x) for x in c) for c in port_cells]
    if not cells:
        raise ValueError("reference_plane_cells: empty wire port cell list")
    i_port = cells[0][line_axis]
    third_index = cells[0][third_axis]
    comp_ids = sorted(c[comp_axis] for c in cells)
    e_lo, e_hi = comp_ids[0], comp_ids[-1] + 1
    # The extent END cell (top of the wire — sits in/at the trace) seeds
    # the trace connected component.  _wire_port_cells spans ascending
    # component-axis indices from the base; the end is the last cell.
    end_cell = max(cells, key=lambda c: c[comp_axis])

    # (a, u, v) right-handed cyclic axes for the Ampere loop.
    u_axis = (line_axis + 1) % 3
    v_axis = (line_axis + 2) % 3
    hu_component = _H_OF_AXIS[u_axis]
    hv_component = _H_OF_AXIS[v_axis]

    pec3d = np.asarray(pec_mask, dtype=bool)
    shape = pec3d.shape

    specs = []
    bboxes = []
    for slot, off in enumerate((n_cells, 2 * n_cells)):
        plane_index = i_port + outboard_sign * off
        if not (1 <= plane_index < shape[line_axis]):
            raise ValueError(
                f"reference_plane_cells: plane {slot} (index {plane_index} "
                f"along axis {line_axis}) is outside the grid "
                f"(n={shape[line_axis]}; the dual Ampere loops need "
                "plane_index-1 >= 0). Reduce N or move the port.")
        # 2D PEC cross-section with axes ordered (u, v).
        sl = np.take(pec3d, plane_index, axis=line_axis)
        # np.take drops line_axis; remaining axes keep their relative order.
        rem = [a for a in (0, 1, 2) if a != line_axis]
        pos = {a: rem.index(a) for a in rem}
        pec2d = np.transpose(sl, (pos[u_axis], pos[v_axis]))
        seed_uv = (end_cell[u_axis], end_cell[v_axis])
        u_lo, u_hi, v_lo, v_hi = _trace_bbox_at_plane(pec2d, seed_uv)
        bboxes.append((u_lo, u_hi, v_lo, v_hi))
        # GAP trim for the V line integral: keep only the port-extent
        # cells NOT PEC-masked at the plane column.  The in-conductor Ez
        # edge of a one-cell-thick trace is not zeroed by the staircase
        # mask convention and carries ~0.21 of the gap field (measured
        # A/B vs point probes) — including it re-scales V (cancels in S,
        # corrupts the measured Zc).  The gap must be one contiguous run.
        gap = [c for c in range(e_lo, e_hi)
               if not pec3d[_tri({line_axis: plane_index,
                                  comp_axis: c,
                                  third_axis: third_index})]]
        if not gap:
            raise ValueError(
                "reference_plane_cells: every port-extent cell at the "
                f"reference plane (index {plane_index}) is inside PEC — "
                "no gap for the V line integral.")
        if gap[-1] - gap[0] + 1 != len(gap):
            raise ValueError(
                "reference_plane_cells: the non-PEC gap cells at the "
                f"reference plane are not contiguous ({gap}) — the gap-V "
                "line integral is ill-defined for this geometry.")
        gap_lo, gap_hi = gap[0], gap[-1] + 1
        # Loop legs: Hv columns half a cell outside the bbox u-faces,
        # Hu rows half a cell outside the bbox v-faces (Yee stagger).
        u_lo_leg, u_hi_leg = u_lo - 1, u_hi + 1
        v_lo_leg, v_hi_leg = v_lo - 1, v_hi + 1
        u_span_lo, u_span_hi = u_lo, u_hi + 2
        v_span_lo, v_span_hi = v_lo, v_hi + 2
        nu, nv = shape[u_axis], shape[v_axis]
        if u_lo_leg < 0 or u_hi_leg >= nu or v_lo_leg < 0 or v_hi_leg >= nv \
                or u_span_hi > nu or v_span_hi > nv:
            raise ValueError(
                "reference_plane_cells: the Ampere loop around the trace "
                f"bbox u[{u_lo},{u_hi}] v[{v_lo},{v_hi}] leaves the grid "
                f"({nu}x{nv} transverse) — the trace touches the domain "
                "boundary at the reference plane.")
        specs.append(WireRefPlaneSpec(
            port_index=int(port_index),
            plane_slot=slot,
            n_cells_outboard=off,
            outboard_sign=outboard_sign,
            line_axis=line_axis,
            plane_index=int(plane_index),
            e_component=e_component,
            comp_axis=comp_axis,
            e_lo=int(gap_lo),
            e_hi=int(gap_hi),
            third_index=int(third_index),
            hu_component=hu_component,
            hv_component=hv_component,
            u_axis=u_axis,
            v_axis=v_axis,
            u_lo_leg=int(u_lo_leg),
            u_hi_leg=int(u_hi_leg),
            v_lo_leg=int(v_lo_leg),
            v_hi_leg=int(v_hi_leg),
            u_span_lo=int(u_span_lo),
            u_span_hi=int(u_span_hi),
            v_span_lo=int(v_span_lo),
            v_span_hi=int(v_span_hi),
            freqs=freqs,
            impedance=float(impedance),
        ))
    if bboxes[0] != bboxes[1]:
        raise ValueError(
            "reference_plane_cells: the PEC trace cross-section differs "
            f"between the two planes (bbox {bboxes[0]} at N={n_cells} vs "
            f"{bboxes[1]} at 2N={2*n_cells}) — the two-plane Zc invariant "
            "requires a UNIFORM line spanning both planes. Reduce N or "
            "move the planes onto the uniform section.")
    return tuple(specs)


def wire_refplane_step_vi(st, spec: WireRefPlaneSpec, dx):
    """Per-step (V, I_minus, I_plus) at one reference plane.

    Called inside the production scan body at the same pre-source-injection
    sample point as the port-cell V/I accumulators.  The reference plane
    sits >= 1 cell away from every source cell, so pre- vs post-injection
    sampling is identical here (injection only touches the driven port
    cell within a step) — this is exactly what the plane architecture
    buys over port-cell sampling.

    V = -sum(E over the gap cells) * dx (FDTD sign, potential of the
    trace relative to the return).  I_minus / I_plus are the Ampere-loop
    line integrals around the trace at the half-planes plane -/+ dx/2;
    the caller averages them and applies the exp(+j*w*dt/2) half-step
    correction AT EXTRACTION (raw phasors are accumulated uncorrected).
    """
    a, u, v = spec.line_axis, spec.u_axis, spec.v_axis
    e_field = getattr(st, spec.e_component)
    e_idx = _tri({a: spec.plane_index,
                  spec.comp_axis: slice(spec.e_lo, spec.e_hi),
                  3 - a - spec.comp_axis: spec.third_index})
    v_plane = -jnp.sum(e_field[e_idx]) * dx

    hv = getattr(st, spec.hv_component)
    hu = getattr(st, spec.hu_component)
    v_span = slice(spec.v_span_lo, spec.v_span_hi)
    u_span = slice(spec.u_span_lo, spec.u_span_hi)

    def _loop(ah):
        s_hv = (jnp.sum(hv[_tri({a: ah, u: spec.u_hi_leg, v: v_span})])
                - jnp.sum(hv[_tri({a: ah, u: spec.u_lo_leg, v: v_span})]))
        s_hu = (jnp.sum(hu[_tri({a: ah, v: spec.v_hi_leg, u: u_span})])
                - jnp.sum(hu[_tri({a: ah, v: spec.v_lo_leg, u: u_span})]))
        return (s_hv - s_hu) * dx

    return v_plane, _loop(spec.plane_index - 1), _loop(spec.plane_index)


# ---------------------------------------------------------------------------
# Extraction-side math (pure NumPy, complex128)
# ---------------------------------------------------------------------------

def refplane_centered_current(i_minus, i_plus, freqs, dt):
    """Plane-centred, half-step-corrected current phasor.

    Averages the two adjacent Ampere loops (centres I on the V plane) and
    applies the Yee leapfrog ``exp(+j*w*dt/2)`` correction to the
    H-derived phasor (comparator anchor: collapses the receive-cell
    Ohm-law phase from +0.008..+0.018 to ~0 — Phase 0, issue #313).
    """
    w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    hcorr = np.exp(+1j * w * dt / 2.0)
    return 0.5 * (np.asarray(i_minus, dtype=np.complex128)
                  + np.asarray(i_plus, dtype=np.complex128)) * hcorr


def refplane_zc_two_plane(v1, i1, v2, i2):
    """Measured line impedance from the exact two-plane invariant.

    ``V^2 - Zc^2 I^2 = 4 f b`` is x-invariant on a uniform line for ANY
    load, so ``Zc^2 = (V1^2 - V2^2)/(I1^2 - I2^2)`` — parameter-free, no
    fit.  Root sign fixed to Re(Zc) >= 0.
    """
    v1 = np.asarray(v1, dtype=np.complex128)
    v2 = np.asarray(v2, dtype=np.complex128)
    i1 = np.asarray(i1, dtype=np.complex128)
    i2 = np.asarray(i2, dtype=np.complex128)
    den = i1 ** 2 - i2 ** 2
    den = np.where(np.abs(den) > 0.0, den, np.ones_like(den))
    zc = np.sqrt((v1 ** 2 - v2 ** 2) / den)
    return np.where(zc.real < 0.0, -zc, zc)


def refplane_split(v, i_corr, zc, outboard_sign):
    """(outgoing, incoming) wave pair at a plane.

    With V = trace-vs-return potential and I positive along the +line
    axis, ``(V + Zc I)/2`` travels +axis and ``(V - Zc I)/2`` travels
    -axis.  ``outgoing`` travels outboard (away from the port, into the
    DUT); ``incoming`` travels toward the port.
    """
    v = np.asarray(v, dtype=np.complex128)
    w_plus = 0.5 * (v + zc * i_corr)
    w_minus = 0.5 * (v - zc * i_corr)
    if outboard_sign > 0:
        return w_plus, w_minus
    return w_minus, w_plus


def refplane_beta(out_near, out_far, separation_m):
    """Measured per-bin propagation constant from the outgoing wave.

    ``out_far = out_near * exp(-j*beta*d)`` for the outboard-travelling
    wave between the plane pair (d = N*dx), so
    ``beta = -angle(out_far/out_near)/d`` — real, per frequency bin.
    """
    near = np.asarray(out_near, dtype=np.complex128)
    far = np.asarray(out_far, dtype=np.complex128)
    safe = np.where(np.abs(near) > 0.0, near, np.ones_like(near))
    return -np.angle(far / safe) / float(separation_m)


def decompose_wire_s_matrix_with_reference_planes(
    v_all,
    i_all,
    z0,
    port_cell_counts,
    *,
    plane_v,
    plane_im,
    plane_ip,
    plane_enabled,
    plane_offsets,
    outboard_signs,
    freqs,
    dt,
    dx,
    return_line_diagnostics: bool = False,
):
    """Shared plane-aware wire S-matrix decomposition (driver + eager entry).

    Starts from the byte-frozen legacy decomposition
    (:func:`rfx.probes.probes.decompose_wire_s_matrix` — diagonals AND any
    off-diagonal whose port pair is not fully opted in), then replaces
    ``S[i, j]`` (i != j, both ports opted in) with the reference-plane
    waves::

        a_j (drive)   = outgoing(plane j0) * exp(+j*beta_j*N_j*dx) / sqrt(Re Zc_j)
        b_i (receive) = incoming(plane i0) * exp(-j*beta_i*N_i*dx) / sqrt(Re Zc_i)
        S[i, j]       = b_i / a_j

    Zc_p and beta_p are measured per port from its OWN drive pass (the
    strongest local signal; the two-plane invariant holds for any load).
    Diagonals are never touched.

    Parameters
    ----------
    v_all, i_all : (n_ports, n_ports, n_freqs) complex
        Port-cell midpoint V/I phasors (legacy accumulators) —
        ``[drive j, receive i]``.
    plane_v, plane_im, plane_ip : (n_ports, n_ports, 2, n_freqs) complex
        Raw plane accumulators ``[drive j, port p, plane slot]``; zeros
        where a port did not opt in.
    plane_enabled : (n_ports,) bool
    plane_offsets : (n_ports,) int — N per opted port (slot 1 is at 2N).
    outboard_signs : (n_ports,) int
    dt, dx : float — timestep and cell size (uniform lane).
    return_line_diagnostics : bool
        When True also return a dict with per-port measured ``zc`` and
        ``beta`` arrays plus the raw at-plane wave pairs (R5 inspection
        surface): ``diagnostics["plane_waves"][(j, p, slot)] =
        (outgoing, incoming)`` for drive pass ``j``, port ``p``'s plane
        ``slot`` (0 = reference plane, 1 = 2N plane; drive-port slot-1
        pairs and all slot-0 pairs are recorded).  These are the
        NON-de-embedded waves at the plane — the zero-free-parameter
        conservation identity ``(|out|^2 - |inc|^2)/Re(Zc)`` equals the
        net line power there (Phase-0 falsifier F2).
    """
    from rfx.probes.probes import decompose_wire_s_matrix

    # np.array (copy): the jnp output buffer is read-only and the plane
    # path overwrites the opted off-diagonals in place below.
    S = np.array(
        decompose_wire_s_matrix(v_all, i_all, z0, port_cell_counts),
        dtype=np.complex64,
    )
    n_ports = S.shape[0]
    opted = [p for p in range(n_ports) if bool(plane_enabled[p])]
    diagnostics: dict = {"zc": {}, "beta": {}, "opted": tuple(opted),
                         "plane_waves": {}}
    if len(opted) >= 2:
        zc = {}
        beta = {}
        for p in opted:
            i1 = refplane_centered_current(
                plane_im[p, p, 0], plane_ip[p, p, 0], freqs, dt)
            i2 = refplane_centered_current(
                plane_im[p, p, 1], plane_ip[p, p, 1], freqs, dt)
            v1 = np.asarray(plane_v[p, p, 0], dtype=np.complex128)
            v2 = np.asarray(plane_v[p, p, 1], dtype=np.complex128)
            zc[p] = refplane_zc_two_plane(v1, i1, v2, i2)
            out1, in1 = refplane_split(v1, i1, zc[p], outboard_signs[p])
            out2, in2 = refplane_split(v2, i2, zc[p], outboard_signs[p])
            beta[p] = refplane_beta(out1, out2,
                                    int(plane_offsets[p]) * float(dx))
            diagnostics["zc"][p] = zc[p]
            diagnostics["beta"][p] = beta[p]
            diagnostics["plane_waves"][(p, p, 0)] = (out1, in1)
            diagnostics["plane_waves"][(p, p, 1)] = (out2, in2)
        for j in opted:
            d_j = int(plane_offsets[j]) * float(dx)
            vj = np.asarray(plane_v[j, j, 0], dtype=np.complex128)
            ij = refplane_centered_current(
                plane_im[j, j, 0], plane_ip[j, j, 0], freqs, dt)
            out_j, _ = refplane_split(vj, ij, zc[j], outboard_signs[j])
            a_port = (out_j * np.exp(+1j * beta[j] * d_j)
                      / np.sqrt(zc[j].real))
            safe_a = np.where(np.abs(a_port) > 0.0, a_port,
                              np.ones_like(a_port))
            for i in opted:
                if i == j:
                    continue  # DIAGONAL: byte-frozen legacy path, always.
                d_i = int(plane_offsets[i]) * float(dx)
                vi_ = np.asarray(plane_v[j, i, 0], dtype=np.complex128)
                ii = refplane_centered_current(
                    plane_im[j, i, 0], plane_ip[j, i, 0], freqs, dt)
                out_i, in_i = refplane_split(vi_, ii, zc[i],
                                             outboard_signs[i])
                diagnostics["plane_waves"][(j, i, 0)] = (out_i, in_i)
                b_port = (in_i * np.exp(-1j * beta[i] * d_i)
                          / np.sqrt(zc[i].real))
                S[i, j, :] = (b_port / safe_a).astype(np.complex64)
    if return_line_diagnostics:
        return S, diagnostics
    return S
