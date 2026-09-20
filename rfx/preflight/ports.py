"""Lumped/wire-port and coaxial-port preflight, moved verbatim out of
``rfx.api._preflight``.

Issue #980 Phase 3, leg 6. The ports family asks what the LATTICE does to a
port after it is drawn: whether a coaxial pin meets registered PEC at its
junction plane and is shorted by geometry rather than by the port (#589),
whether a lumped RLC element is being illuminated by a TFSF plane wave (the
#425 divergence), where ``reference_plane_cells`` puts the measurement planes
(#313), which of a port's, source's or probe's field components the realized
PEC edge set has FROZEN (#929 / #931 §1.7, with the #314/#319 dead-cell
advisories, the #556 end-gap finding and the #544 non-uniform skip), and
whether a single-cell port is floating mid-substrate with nothing to drive
against (#71). Everything here was relocated byte for byte out of
``rfx/api/_preflight.py`` -- same text, same order, same indentation, same
docstrings, nothing renamed, reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first. The extension was chosen by a CALL CENSUS, as legs 3 to 5 were. All
six bodies below were entered -- the family sits on
``_validate_simulation_config``'s unconditional spine, so 54 or 55 of the 57
fixtures called each, and ``_wire_port_cell_centers`` twice as the helper it
is -- while THREE emitted nothing: the coax junction check (the only body of
the ports_coax family, and the corpus had no shorted junction), both
reference-plane advisories, and the floating-port advisory, whose predicate
wants a driven single-cell port inside a dielectric with NO conductor one
cell away along its own component axis -- the configuration every
patch-antenna fixture in the suite deliberately does not have. Five new
fixtures close those, plus the two codes of ``_validate_cfg_port_inside_pec``
the corpus had never reached. The lock module's own docstring carries the
measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

The one module-level name this leg moved, ``_component_is_dead`` with its
``_H_LOOP`` curl-loop table, is NOT here: it went to
:mod:`rfx.preflight._common`, because its second reader
(``_RealizedPEC.component_is_dead``) stays in the facade until the
realization leg and a leg module may not import from the facade. That is leg
2's ``_sorted_box_corners`` decision, re-derived on this leg's own AST scope
walk rather than inherited from the split inventory -- which is what leg 5's
``_local_cell`` finding made mandatory.

Two cross-family edges leave this module and both are ``self.`` lookups on
the composed ``Simulation``, so neither costs an import:
``_check_coaxial_port_junction_aperture`` calls ``self._port_realized_edges``
and ``_validate_cfg_port_inside_pec`` calls ``self._campaign_ctx``, both of
which stay in the facade with the realization family. The third,
``self._wire_port_cell_centers``, is intra-module.
"""

from __future__ import annotations

import math

import numpy as np

from rfx.preflight._common import (
    _component_is_dead,
    _fmt_len,
    PreflightWarning,
)


def _check_coaxial_port_junction_aperture(self) -> None:
    """Advise when a coaxial port's pin meets REGISTERED PEC at its
    junction plane — a short by geometry, not by the port (issue #589).

    Root cause this names: ``_assemble_materials`` is PEC-OR-only
    (``pec_mask = pec_mask | mask``, rfx/api/_compile.py) and there is
    no CSG subtraction shape, so a ground plane declared as a full PEC
    sheet with a dielectric "clearance hole" declared AFTER it stays
    solid. The committed coax-MSL junction fixture was built exactly
    that way; its settled run measured S00 = (-0.9928, -0.0048) at
    6 GHz — the pin was terminated in a short by the ground sheet, and
    no check said so (the fixture's only structural test asserted
    pin-column PEC continuity, trivially true through a solid sheet).

    What is measured: on the PRODUCTION assembly's REALIZED conductor
    set (:meth:`_port_realized_edges`, sim.add geometry and thin
    conductors only — the coax stub the compute_coaxial_* /
    compute_coax_msl_transition methods stamp into eps/sigma is not
    registered geometry and is not read here), at the port's
    junction plane ``k = position_to_index(position)[axis]``, the
    count of NODES CARRYING A TANGENTIAL WALL in the FIRST dielectric
    ring outside the pin. "Carries a wall" is the lattice ownership
    contract's own test (#931 §1.9): some E edge tangential to the
    port axis and incident to the node is PEC — which is what a sheet
    ground at plane ``k`` and a volume ground whose face lies on
    plane ``k`` both put there, and what a primal CELL mask never
    marked for a volume's far face (the #868 class). The ring is
    defined ON THE LATTICE: ``a + dx/sqrt(2) < r <= min(a +
    dx/sqrt(2) + dx, shell_inner)`` with ``r`` the node distance from
    the port centre and ``shell_inner = b - min(dx, (b - a)/2)`` the
    radius ``stamp_coaxial_line`` realizes for the shell. The inner
    bound is the pin's own reach: a PEC volume is centre-sampled
    (§1.1) and every node of an occupied cell carries its wall, so
    the pin's OWN nodes extend to at most ``a + dx/sqrt(2)`` (half
    the cell diagonal past its radius); a node beyond that cannot be
    a corner of a pin cell, so anything there is OTHER registered
    conductor. The earlier ``a + dx/2`` bound was calibrated to the
    pre-#931 node-sampled knife-edge footprint and would count the
    pin's own realized rim.

    Deliberately NOT the full ``a < r < shell_inner`` annulus: a
    clearance hole narrower than the shell (the fixture's predeclared
    0.4 mm hole under a 0.5 mm shell_inner leaves a one-cell ground
    lip, 32/68 of that annulus PEC) is a valid launch and the wide
    rule would flag the fix.

    Report-only (severity "warning", no refusal): a calibration short
    built from REGISTERED PEC would trip it and is the user's call.
    The repo's own calibration short is stamped via
    ``stamp_coaxial_short_plane`` (sigma), so no current lane does.
    Not audited (silent by construction, disclosed here): the
    non-uniform lane, 2-D mode, and a port whose position does not map
    into the grid (the compiler rejects that separately). Measured on
    the example snapshot (tests/contracts/test_example_fidelity_contract.py):
    the one coaxial variant has 0 PEC cells in the ring, so no
    snapshot row changes.
    """
    import warnings as _w

    if not self._coaxial_ports:
        return
    if (self._dx_profile is not None or self._dy_profile is not None
            or self._dz_profile is not None):
        return
    from rfx.sources.coaxial_port import _FACE_CONFIG

    grid = self._build_grid()
    if getattr(grid, "is_2d", False):
        return
    realized = self._port_realized_edges(grid)
    if realized is None:
        return
    dx = float(grid.dx)
    pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
    axis_names = ("x", "y", "z")
    for n, port in enumerate(self._coaxial_ports):
        cfg = _FACE_CONFIG.get(str(port.face))
        if cfg is None:
            continue
        axis = axis_names.index(cfg[0])
        pos = tuple(float(v) for v in port.position)
        try:
            idx = grid.position_to_index(pos)
        except ValueError:
            continue
        k = int(idx[axis])
        t1, t2 = [a for a in range(3) if a != axis]
        a = float(port.pin_radius)
        b = float(port.outer_radius)
        shell_inner = b - min(dx, 0.5 * (b - a))
        c1 = (np.arange(grid.shape[t1]) - pads[t1]) * dx - pos[t1]
        c2 = (np.arange(grid.shape[t2]) - pads[t2]) * dx - pos[t2]
        r = np.hypot(c1[:, None], c2[None, :])
        if not (0 <= k < grid.shape[axis]):
            continue
        plane = realized.wall_nodes_on_plane(axis, k)
        r_lo = a + dx / math.sqrt(2.0)
        r_hi = min(r_lo + dx, shell_inner)
        ring = (r > r_lo) & (r <= r_hi)
        n_ring = int(np.count_nonzero(ring))
        if n_ring == 0:
            continue
        n_pec = int(np.count_nonzero(plane & ring))
        if n_pec == 0:
            continue
        outside = plane & (r > r_lo) & (r <= b)
        r_first = float(r[outside].min()) if outside.any() else float("nan")
        _w.warn(
            PreflightWarning(
                f"Coaxial port {n} (face='{port.face}', pin r="
                f"{a * 1e6:.0f} um, outer r={b * 1e6:.0f} um): at its "
                f"junction plane ({axis_names[axis]}="
                f"{pos[axis] * 1e3:.3f} mm, node {k - pads[axis]}) "
                f"{n_pec}/{n_ring} nodes of the FIRST dielectric ring "
                f"outside the pin ({r_lo * 1e6:.0f} < r <= "
                f"{r_hi * 1e6:.0f} um; lattice-based bounds, so the "
                f"pin's own realized rim, at most r = {a * 1e6:.0f} um "
                f"+ dx/sqrt(2), is excluded) carry a REALIZED PEC wall "
                f"— the first registered conductor outside the pin "
                f"sits at r = {r_first * 1e6:.1f} um. The pin is "
                f"terminated in a short by registered geometry at this "
                f"plane. The assembly is PEC-OR-only: a ground sheet "
                f"declared as a full plane with a dielectric 'hole' "
                f"declared AFTER it stays solid (issue #589: S00 = "
                f"-0.9928 at 6 GHz measured on exactly that fixture). "
                f"If a clearance aperture was intended, build the "
                f"conductor WITH the hole (a patterned sheet shape via "
                f"add_thin_conductor, or an annular volume); if this "
                f"short is the intended calibration standard, no "
                f"action is needed (report-only).",
                code="coaxial_port_junction_short",
                source="_check_coaxial_port_junction_aperture",
                loc=f"coaxial_port[{n}] face={port.face}",
            ),
            stacklevel=4,
        )


def _validate_cfg_tfsf_with_lumped_rlc(self, _w) -> None:
    """Warn: a lumped RLC element illuminated by a TFSF plane wave is unstable.

    A ``add_lumped_rlc(...)`` element driven by a TFSF plane wave diverges
    (measured: blow-up to ~1e35 by ~250 steps, C-independent). The root cause is
    the TFSF total/scattered-field decomposition coupling into the lumped ADE
    current, NOT a missing circuit path: embedding the element in a PEC-gap
    structure does NOT cure it (tested 2026-07-22, #425 — a two-electrode PEC gap
    still grows 0.1→1e25→NaN over ~800 steps; see
    docs/research_notes/experiments/tfsf_lumped_pec_gap_stability.py). So there is
    no geometry fix at the API level; a stable plane-wave lumped lane needs a
    solver-level fix to the TFSF↔lumped coupling. The tunable-load (varactor)
    gradient IS validated on the PORT-fed lane (``add_port`` +
    ``forward(rlc_values_override=...)`` — tests/unit/autodiff/test_lumped_rlc_ad.py); use that
    for varactor/RIS design. See the tracking issue (#425).
    """
    if self._tfsf is None or not self._lumped_rlc:
        return
    _w.warn(PreflightWarning(
        "add_lumped_rlc(...) + a TFSF plane-wave source is numerically unstable "
        "(fields diverge, C-independent). This is the TFSF↔lumped-ADE coupling, "
        "not a missing circuit path — a PEC-gap structure does NOT cure it "
        "(tested, #425). Use the validated PORT-fed lane (add_port + "
        "forward(rlc_values_override=...)) for varactor/tunable-load design.",
        code="tfsf_lumped_rlc_unstable",
        source="_validate_cfg_tfsf_with_lumped_rlc",
    ))


def _validate_cfg_refplane_placement(self, _w) -> None:
    """Advisories for ``add_port(reference_plane_cells=...)`` (issue #313).

    (a) ``reference_plane_cells < 10`` puts the measurement planes in the
    port near field.  Measured on the canonical 16 mm thru (dx = 0.5 mm,
    gap-trimmed V, 2026-07-10 battery): N=3 planes read Zc = 52-53 ohm
    with |Im/Re| up to 8.2%, beta/(w/c) = 1.16-1.20, and a -3.1%
    closed-box-referee |S21| residual, while N=10 planes read the clean
    mid-line constants.  The Phase-0 pre-registration rule places BOTH
    planes (N and 2N cells) >= 10 cells from every port.

    (b) When SOME but not ALL impedance-carrying wire ports opt in, the
    off-diagonal S entries involving a non-opted port silently stay on
    the legacy port-cell path (only pairs where both ports opt in use
    the plane waves) — surface that at preflight instead of letting the
    mixed matrix pass unremarked.
    """
    wire_ports = [
        pe for pe in self._ports
        if pe.impedance != 0.0 and pe.extent is not None
    ]
    opted = [
        pe for pe in wire_ports
        if getattr(pe, "reference_plane_cells", None) is not None
    ]
    if not opted:
        return
    near = [pe for pe in opted if pe.reference_plane_cells < 10]
    if near:
        _w.warn(
            PreflightWarning(
                f"{len(near)} reference-plane port(s) use "
                "reference_plane_cells < 10 — planes this close sit in "
                "the port near field. Measured on the canonical thru "
                "(dx = 0.5 mm, 2026-07-10 battery): N=3 planes read "
                "Zc = 52-53 ohm with |Im(Zc)/Re(Zc)| up to 8.2% and "
                "beta/(w/c) = 1.16-1.20 (vs the clean mid-line "
                "constants at N=10), and the closed-box-referee |S21| "
                "residual was -3.1%. The Phase-0 pre-registration rule "
                "(issue #313) places BOTH planes (N and 2N cells) >= 10 "
                "cells from every port — prefer "
                "reference_plane_cells >= 10 when the line length "
                "allows it.",
                code="refplane_near_field",
                source="_validate_cfg_refplane_placement",
            ),
            stacklevel=2,
        )
    if len(opted) != len(wire_ports):
        _w.warn(
            PreflightWarning(
                f"{len(opted)} of {len(wire_ports)} impedance-carrying "
                "wire ports opt into reference_plane_cells — "
                "off-diagonal S entries involving a non-opted port "
                "SILENTLY stay on the legacy port-cell path (only "
                "pairs where BOTH ports opt in use the plane waves). "
                "Opt in every wire port of the S-matrix, or none.",
                code="refplane_partial_optin",
                source="_validate_cfg_refplane_placement",
            ),
            stacklevel=2,
        )


def _validate_cfg_port_inside_pec(self, _w, dx: float) -> None:
    """P1.8: Port/source/probe frozen by realized PEC.

    Lattice ownership contract (#931 §1.7 / #929): every question this
    check asks is answered by the run's REALIZED edge set, read once
    from the production assembly with the sheet/wire collectors:

    * a wire-port extent cell is DEAD iff the port component's own E
      edge at that index is PEC (``_wire_port_live_cells``, the same
      primitive the assembler uses — issue #544 made the two share
      it, and #931 made the primitive read edges, so a sheet trace,
      which owns no cell, is visible here);
    * a point port / source / probe on an E component is dead iff its
      own edge is PEC; on an H component iff all four E edges of its
      curl loop are PEC (``curl E = 0`` freezes it). Half a cell above
      a SHEET only one loop edge is PEC, so an MSL diagnostic Hy probe
      at the trace plane stays live; inside a one-cell VOLUME all four
      are PEC and H is frozen, which is what a volume declaration
      means. The former ``<= 1.5 dx`` thickness exemption inferred
      sheet-ness from a bounding box — the #929 class — and is gone;
    * the #556/#929 end-gap advisory measures the separation from
      the actual source-end node to the nearest conductor on its own
      column. A volume face, sheet plane, or axial PEC edge ending at
      that node supplies contact. Contact is silent; a separation
      is reported without guessing the intended coupling mechanism.

    Non-uniform lane: the wire-port primitive cannot index a
    ``NonUniformGrid`` (no ``position_to_index``), so the wire-port
    advisories emit a "classification unavailable" note there rather
    than guess (issue #544 review item 6; #303 class); the point
    port/probe rule runs on both lanes through the shared context.
    """
    ctx = self._campaign_ctx()
    is_nonuniform = ctx.lane == "nonuniform"
    realized = None
    classification_unavailable_reason: str | None = None
    if ctx.error == "traced-mesh":
        classification_unavailable_reason = (
            "traced mesh (mesh-as-design-variable) -- no concrete "
            "node positions")
    elif ctx.error is not None:
        classification_unavailable_reason = ctx.error
    elif is_nonuniform:
        classification_unavailable_reason = (
            "non-uniform mesh (dz_profile/dx_profile/"
            "dy_profile set) -- the shared wire-port primitive "
            "only covers the uniform-grid path (issue #544)"
        )
    # A model with no conductor declaration has nothing that can
    # freeze a port: skip the production assembly (the expensive part
    # of the context) rather than run it to read an all-False set.
    has_conductor = bool(getattr(self, "_thin_conductors", None))
    if not has_conductor:
        try:
            has_conductor = any(
                self._resolve_material(e.material_name).sigma
                >= self._PEC_SIGMA_THRESHOLD for e in self._geometry)
        except KeyError:
            has_conductor = False
    if ctx.error is None and has_conductor:
        realized = ctx.realized()
        if realized is None:
            classification_unavailable_reason = ctx.assembly_error
    grid = ctx.grid
    entries = ctx.entry_realizations() if ctx.error is None else []
    shape = tuple(grid.shape) if grid is not None else None

    def _owners(component, idx):
        """Names of the declarations whose OWN realized edges freeze
        ``component`` at ``idx`` (attribution through the same
        function, never through a per-entry cell mask)."""
        names = []
        for e in entries:
            if not e.is_pec:
                continue
            if _component_is_dead(e.edges(ctx.periodic, shape),
                                  component, idx):
                if e.name not in names:
                    names.append(e.name)
        return names or ["unknown"]

    for pe in self._ports:
        if not getattr(pe, "extent", None):
            continue
        rasterized = self._wire_port_cell_centers(pe)
        if rasterized is None:
            continue
        centers, mid_idx = rasterized
        mid_center = centers[mid_idx]

        dead_indices: list[int] = []
        dead_names: list[str] = []
        if realized is not None and not is_nonuniform:
            from rfx.sources.sources import (
                WirePort, _wire_port_cells, _wire_port_live_cells,
            )
            axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
            end = list(pe.position)
            end[axis] += pe.extent
            wp = WirePort(
                start=tuple(pe.position), end=tuple(end),
                component=pe.component, impedance=pe.impedance,
            )
            try:
                cells, live_flags, _ = _wire_port_live_cells(
                    grid, wp, realized.edges)
            except ValueError:
                # Every extent cell is dead: _wire_port_live_cells
                # raises there (issue #318 — such a port has no live
                # cell to terminate or drive) instead of returning a
                # degenerate split. Report all cells dead.
                cells = _wire_port_cells(grid, wp)
                live_flags = [False] * len(cells)
            dead_indices = [
                idx for idx, live in enumerate(live_flags) if not live
            ]
            if dead_indices:
                for idx in dead_indices:
                    for name in _owners(pe.component, cells[idx]):
                        if name not in dead_names:
                            dead_names.append(name)

            # Issue #556 (D5 follow-up, #488 arc): the OPPOSITE
            # failure mode of the #314/#319 advisories above. There,
            # a port extent cell lands ON PEC (dead cell). Here, the
            # port terminates SHORT of a conductor. On the
            # D5 "end-fed trace" fixture (dx=80um, h_sub=254um) the
            # trace's realization landed one full cell above the
            # wire's top. That historical D5 report described
            # |S21| rising with frequency (docs/research_notes/
            # 20260728_i488_falsifier_ledger.md). It motivated this
            # check, but a geometric gap alone does not identify its
            # electromagnetic coupling mechanism. No cell is dead in
            # that fixture, so #314/#319 are correctly silent.
            #
            # Contact (#931 §1.9, #929) is read from realized walls
            # and PEC source-axis edges on the terminal's own column.
            # The wire's last edge on the + side runs
            # from node ``cells[-1]`` to ``cells[-1] + 1``, so its end
            # node is ``cells[-1] + 1``; on the - side the end node
            # is ``cells[0]``. "Fires" = the end edge is live, the
            # end node has no conductor contact but a further node
            # on that column does. Search every realized candidate
            # (#929), not only a one-cell neighbour. An axial PEC edge
            # can supply contact without a tangential wall (filaments).
            # A dipole ending in open vacuum stays silent. A measured
            # separation alone does not determine the intended coupling.
            if cells and live_flags:
                axis_letter = "xyz"[axis]
                for end_idx, step in ((0, -1), (len(cells) - 1, +1)):
                    if not live_flags[end_idx]:
                        continue
                    end_node = list(cells[end_idx])
                    if step > 0:
                        end_node[axis] += 1
                    ij = tuple(end_node[t] for t in range(3)
                               if t != axis)
                    if not all(0 <= end_node[t] < shape[t]
                               for t in range(3)):
                        continue
                    planes = set(realized.wall_planes(axis, ij=ij))
                    line_index = tuple(slice(None) if a == axis else end_node[a]
                                       for a in range(3))
                    axial_edges = np.flatnonzero(
                        np.asarray(realized.edges[axis])[line_index])
                    # Looking downward, edge k ends at node k+1;
                    # looking upward, it begins at node k.
                    edge_nodes = {int(k)+(1 if step < 0 else 0)
                                  for k in axial_edges}
                    contacts = planes | edge_nodes
                    if end_node[axis] in contacts:
                        continue            # galvanic contact
                    candidates = [k for k in contacts
                                  if 0 <= k < shape[axis]
                                  and (k-end_node[axis])*step > 0]
                    if not candidates:
                        continue
                    beyond = min(candidates, key=lambda k: abs(k-end_node[axis]))
                    edge_index = list(end_node)
                    edge_index[axis] = beyond-(1 if step < 0 else 0)
                    adj_names = []
                    for e in entries:
                        if not e.is_pec:
                            continue
                        if beyond in e.wall_planes(axis, ctx.periodic,
                                                   shape):
                            foot = e.footprint_on_plane(
                                axis, beyond, ctx.periodic, shape)
                            if bool(foot[ij]) and e.name not in adj_names:
                                adj_names.append(e.name)
                    if beyond in edge_nodes:
                        for name in _owners(pe.component, tuple(edge_index)):
                            if name not in adj_names:
                                adj_names.append(name)
                    adj_names = adj_names or ["unknown"]
                    side = ("+" if step > 0 else "-") + axis_letter
                    nodes_ax = ctx.nodes[axis]
                    gap_cells = abs(beyond-end_node[axis])
                    gap_m = abs(float(nodes_ax[beyond])-float(nodes_ax[end_node[axis]]))
                    kind = "wall plane" if beyond in planes else f"{pe.component}-edge endpoint"
                    _w.warn(
                        PreflightWarning(
                            f"Wire port at {pe.position} (extent "
                            f"{pe.extent}, component {pe.component}): "
                            f"its {side}-side end node "
                            f"{tuple(end_node)} ({axis_letter} = "
                            f"{_fmt_len(float(nodes_ax[end_node[axis]]))}) "
                            f"carries no realized PEC contact, but the "
                            f"nearest outward conductor node {beyond} "
                            f"({axis_letter} = "
                            f"{_fmt_len(float(nodes_ax[beyond]))}) is a "
                            f"realized {kind} of {adj_names}. The "
                            f"port end is "
                            f"{gap_cells} cell(s) (gap = {gap_m:g} m) short "
                            f"of that conductor. This is a geometric "
                            f"separation; it does not determine the "
                            f"intended electromagnetic coupling. If "
                            f"galvanic contact is intended, adjust the "
                            f"position and extent together so its end node "
                            f"lands on that wall plane or conductor endpoint, "
                            f"or correct the conductor declaration.",
                            code="wire_port_end_gap_to_conductor",
                            source="_validate_cfg_port_inside_pec",
                        ),
                        stacklevel=3,
                    )
        elif classification_unavailable_reason is not None:
            _w.warn(
                PreflightWarning(
                    f"Wire port at {pe.position} (extent {pe.extent}): "
                    f"dead-cell classification unavailable "
                    f"({classification_unavailable_reason}). The "
                    f"#314/#319 PEC-overlap advisories and the #556 "
                    f"end-gap-to-conductor advisory are skipped "
                    f"for this port -- inspect the realized live/dead "
                    f"source edges and the intended terminal contacts "
                    f"on that mesh. Conductor overlap at an endpoint "
                    f"can be intentional; do not infer contact from "
                    f"the absence of this advisory.",
                    code="wire_port_dead_cell_classification_unavailable",
                    source="_validate_cfg_port_inside_pec",
                ),
                stacklevel=3,
            )

        if mid_idx in dead_indices:
            # Kept verbatim from the #314 fix (PR #317): probe-cell
            # corruption is the stronger, measured failure mode.
            name = dead_names[0] if dead_names else "an assembled PEC region"
            _w.warn(
                PreflightWarning(
                    f"Wire port at {pe.position} (extent "
                    f"{pe.extent}): its MIDPOINT V/I probe cell "
                    f"(center {tuple(round(x, 6) for x in mid_center)}) "
                    f"lands inside PEC geometry "
                    f"'{name}'. S-parameters from "
                    f"this port are silently corrupted (measured: "
                    f"near-null forward transmission + over-unity "
                    f"reverse). Shorten/lengthen the extent or move "
                    f"the port so the midpoint cell sits in "
                    f"dielectric (issue #314).",
                    code="wire_port_midpoint_in_pec",
                    source="_validate_cfg_port_inside_pec",
                ),
                stacklevel=3,
            )

        non_midpoint_dead = [i for i in dead_indices if i != mid_idx]
        if non_midpoint_dead:
            n = len(centers)
            n_live = n - len(dead_indices)
            z0 = getattr(pe, "impedance", 0.0) or 0.0
            z_eff = z0 * n_live / n
            _w.warn(
                PreflightWarning(
                    f"Wire port at {pe.position} (extent {pe.extent}) "
                    f"rasterizes to n={n} cells of which "
                    f"{len(dead_indices)} have their {pe.component} "
                    f"edge inside realized PEC "
                    f"{dead_names} (n_live/n = {n_live}/{n}). Dead "
                    f"cells are shorted by the PEC and are excluded "
                    f"from the port's resistance distribution, drive "
                    f"injection, and wave normalization (issue #318 "
                    f"fix): the port terminates at {z0:g} ohm across "
                    f"its {n_live} live cells. (rfx versions before "
                    f"the #318 fix counted all {n} cells and "
                    f"physically terminated at Z0*(n_live/n) = "
                    f"{z_eff:.1f} ohm — the issue-#313 finding.) "
                    f"Verify the extent was MEANT to end on/inside "
                    f"the conductor, and keep the midpoint V/I probe "
                    f"cell live; to silence, shorten the extent or "
                    f"move the port so none of its cells has its "
                    f"{pe.component} edge inside a conductor (per the "
                    f"realized edge set -- a volume shorts every edge "
                    f"between its two faces, a sheet shorts only the "
                    f"edges IN its plane and leaves the normal edge "
                    f"through it live).",
                    code="wire_port_dead_extent_cells",
                    source="_validate_cfg_port_inside_pec",
                ),
                stacklevel=3,
            )

    if realized is None:
        return
    _internal = getattr(self, "_internal_probe_indices", frozenset())
    _probe_entries = [
        pe for _pi, pe in enumerate(self._probes) if _pi not in _internal
    ]  # skip library-internal witness probes (issue #470; see
    #    _validate_cfg_absorber_placement for the rationale)
    for pe in list(self._ports) + _probe_entries:
        if getattr(pe, "extent", None):
            continue        # wire ports: the per-cell rule above
        pos = tuple(float(v) for v in pe.position)
        component = (getattr(pe, "component", "") or "").lower()
        if component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
            continue
        try:
            if is_nonuniform:
                from rfx.nonuniform import position_to_index as _nu_p2i
                idx = tuple(int(v) for v in _nu_p2i(grid, pos))
            else:
                idx = tuple(int(v) for v in grid.position_to_index(pos))
        except (ValueError, TypeError, IndexError, AttributeError):
            continue
        if not realized.component_is_dead(component, idx):
            continue
        names = _owners(component, idx)
        what = ("its own E edge is a realized PEC edge"
                if component[0] == "e" else
                "all four E edges of its curl loop are realized PEC "
                "edges, so curl E = 0 and the component never moves")
        _w.warn(
            PreflightWarning(
                f"Port/source/probe at {pos} ({component}) is frozen "
                f"by realized PEC geometry {names}: {what} (lattice "
                f"ownership contract, #931). Field will be zero. "
                f"Move it off the conductor, or drive/read a "
                f"component the conductor leaves live (the normal E "
                f"through a sheet plane; tangential H beside a sheet).",
                code="port_in_pec",
                source="_validate_cfg_port_inside_pec",
            ),
            stacklevel=3,
        )

def _wire_port_cell_centers(self, pe):
    """Per-cell physical sample centers of a wire port's rasterization.

    Uses the production endpoint snap and shared half-open edge span on
    either grid lane. Returns ``(centers, midpoint_index)`` with one
    physical E-edge center per driven cell and the midpoint V/I probe
    at ``cells[len(cells) // 2]``. Returns None when the declaration
    cannot be rasterized (diagnostics must not crash a run).
    """
    try:
        from rfx.sources.sources import (
            WirePort, _wire_port_cells, wire_port_edge_span,
        )
        from rfx.nonuniform import NonUniformGrid, position_to_index
        from rfx.geometry.rasterize_grid import (
            coords_from_nonuniform_grid, coords_from_uniform_grid,
        )
        axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
        end = list(pe.position)
        end[axis] += pe.extent
        grid = self._build_realized_grid()
        if isinstance(grid, NonUniformGrid):
            start_idx = position_to_index(grid, pe.position)
            end_idx = position_to_index(grid, tuple(end))
            lo, hi = sorted((start_idx[axis], end_idx[axis]))
            first, last = wire_port_edge_span(
                grid, axis, lo, hi, float(pe.position[axis]),
                float(end[axis]))
            cells = []
            for k in range(first, last + 1):
                cell = list(start_idx)
                cell[axis] = k
                cells.append(tuple(cell))
            coords = coords_from_nonuniform_grid(grid)
        else:
            wp = WirePort(start=tuple(pe.position), end=tuple(end),
                          component=pe.component, impedance=pe.impedance)
            cells = _wire_port_cells(grid, wp)
            coords = coords_from_uniform_grid(grid)
        if not cells:
            return None
        nodes = [np.asarray(line, dtype=float)
                 for line in (coords.x, coords.y, coords.z)]
        centers = []
        for cell in cells:
            # An E edge is at its node on the transverse axes and at
            # the midpoint of its bounding nodes on its own axis.
            pos = [float(nodes[ax][cell[ax]]) for ax in range(3)]
            pos[axis] = 0.5 * (pos[axis] + nodes[axis][cell[axis] + 1])
            centers.append(tuple(pos))
        return centers, len(cells) // 2
    except Exception:
        return None

def _validate_cfg_floating_single_cell_port(self, _w) -> None:
    """P1.9: Single-cell port in dielectric with no adjacent PEC pin
    (issue #71). A single-cell LumpedPort placed mid-substrate with
    no conducting pin or microstrip does not couple to patch-antenna
    TM modes — the optimiser reads a nonsense loss from the
    floating Ez source. Recommend extent=<substrate_height> to
    promote to a WirePort spanning ground → patch.
    """
    _PORT_COMP_AXIS = {"ex": 0, "ey": 1, "ez": 2}
    for pe in self._ports:
        # Filter: only true ports (impedance > 0), single-cell
        # (extent is None), actively excited (excite is True).
        # add_source() creates _PortEntry with impedance=0.0 and is
        # intentionally a soft source — not a port footgun.
        if not pe.impedance or pe.impedance <= 0.0:
            continue
        if pe.extent is not None:
            continue
        if not pe.excite:
            continue
        pos = pe.position
        # Find the dielectric geometry enclosing the port cell.
        enclosing_eps_r = None
        enclosing_name = None
        for entry in self._geometry:
            if entry.material_name == "pec":
                continue
            if not hasattr(entry.shape, "bounding_box"):
                continue
            try:
                c1, c2 = entry.shape.bounding_box()
            except (NotImplementedError, TypeError):
                continue
            inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
            if not inside:
                continue
            mspec = self._materials.get(entry.material_name)
            if mspec is not None and float(mspec.eps_r) > 1.0 + 1e-3:
                enclosing_eps_r = float(mspec.eps_r)
                enclosing_name = entry.material_name
                break
        if enclosing_eps_r is None:
            continue
        # Check for a PEC geometry one cell away along the port's
        # component axis (coax-style pin or microstrip feed edge).
        # Without such a pin, the port cell cannot drive a vertical
        # current that couples to the patch TM mode.
        comp_axis = _PORT_COMP_AXIS.get(pe.component)
        if comp_axis is None:
            continue
        nudge = float(self._dx or 0.0) * 1.01
        adj_positions = (
            tuple(pos[i] + (nudge if i == comp_axis else 0.0) for i in range(3)),
            tuple(pos[i] - (nudge if i == comp_axis else 0.0) for i in range(3)),
        )
        # #931: a pin / ground may be a SHEET declaration
        # (add_thin_conductor); sheets are conductors in this census.
        from rfx.materials.thin_conductor import sheet_bounds as _sb
        adjacent_bounds = []
        for entry in self._geometry:
            if entry.material_name != "pec":
                continue
            if not hasattr(entry.shape, "bounding_box"):
                continue
            try:
                c1, c2 = entry.shape.bounding_box()
            except (NotImplementedError, TypeError):
                continue
            adjacent_bounds.append((c1, c2))
        for tc in getattr(self, "_thin_conductors", ()):
            if not getattr(tc, "is_pec", False):
                continue
            try:
                c1, c2 = _sb(tc.shape)
            except (NotImplementedError, TypeError, AttributeError):
                continue
            if c1 is not None and c2 is not None:
                adjacent_bounds.append((tuple(c1), tuple(c2)))
        has_adjacent_pec = False
        for apos in adj_positions:
            for c1, c2 in adjacent_bounds:
                if all(c1[ax] <= apos[ax] <= c2[ax] for ax in range(3)):
                    has_adjacent_pec = True
                    break
            if has_adjacent_pec:
                break
        if has_adjacent_pec:
            continue
        _w.warn(
            PreflightWarning(
                f"Single-cell port at {pos} ({pe.component}) sits inside "
                f"dielectric '{enclosing_name}' (eps_r={enclosing_eps_r:.2f}) "
                f"with no adjacent PEC along the {pe.component[1]}-axis. A "
                f"floating single-cell port inside substrate does not "
                f"couple to patch-antenna TM modes. Pass "
                f"extent=<substrate_height> to create a WirePort spanning "
                f"ground → patch plane (issue #71).",
                code="floating_port",
                source="_validate_cfg_floating_single_cell_port",
            ),
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the six functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all six names here are private, so
# ``tests/locks/test_preflight_split_snapshot.py`` pins these six directly.
#
# None of the six was a ``@staticmethod`` -- the one staticmethod this leg
# moves is ``_validate_tfsf_vacuum_boundary``, which went to
# ``rfx/preflight/sources.py`` -- so this module has no decorator the facade
# has to re-apply and every restored qualname below becomes
# ``Simulation.<name>`` after composition.
# ---------------------------------------------------------------------------
_check_coaxial_port_junction_aperture.__qualname__ = (
    "_PreflightMixin._check_coaxial_port_junction_aperture"
)
_validate_cfg_tfsf_with_lumped_rlc.__qualname__ = (
    "_PreflightMixin._validate_cfg_tfsf_with_lumped_rlc"
)
_validate_cfg_refplane_placement.__qualname__ = (
    "_PreflightMixin._validate_cfg_refplane_placement"
)
_validate_cfg_port_inside_pec.__qualname__ = (
    "_PreflightMixin._validate_cfg_port_inside_pec"
)
_wire_port_cell_centers.__qualname__ = (
    "_PreflightMixin._wire_port_cell_centers"
)
_validate_cfg_floating_single_cell_port.__qualname__ = (
    "_PreflightMixin._validate_cfg_floating_single_cell_port"
)
