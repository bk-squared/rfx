"""The mixed-family S-matrix calculator, moved verbatim out of ``rfx.api._sparams``.

Issue #980 Phase 2. ``compute_mixed_s_matrix`` — the lumped/wire + MSL
two-family driver of issue #488 — is relocated here byte for byte from the
``_SparamMixin`` class body: same text, same order, same docstring, dedented by
exactly four spaces and nothing else. ``rfx/api/_sparams.py`` binds the function
back onto ``_SparamMixin`` at the line the ``def`` used to occupy, so
``Simulation.compute_mixed_s_matrix`` keeps its name, signature, ``__doc__`` and
bound-method behaviour, and every call site is unaffected.

``self`` is still the :class:`rfx.api.Simulation` instance: the function is a
module-level ``def`` whose first parameter is ``self``, not a free function with
a different contract. It writes ``self._ports`` / ``self._probes`` /
``self._dft_planes`` / ``self._flux_monitors`` / ``self._msl_ports`` /
``self._internal_probe_indices`` and calls ``self.preflight()``,
``self.add_flux_monitor()`` and ``self._resolve_msl_probe_entries()`` exactly as
it did as a method. A thin class wrapper may follow in a later #980 step; this
step adds none.

The move is gated on bit identity of the S arrays the leg returns
(``tests/locks/test_sparams_split_bit_identity.py``, ``mixed`` leg), not on a
tolerance.

Import contract, inherited from ``rfx.api._sparams``: import only
``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy — never
``from rfx.api import ...`` the package, which would make
``rfx/api/__init__.py`` stop being the sole composition point. The function-local
``from rfx.api._preflight import ...`` imports inside the body are inside the
body, exactly as they were, and so do not run at import time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.sources.sources import GaussianPulse

from rfx.api._spec import MixedSMatrixResult

from rfx.sparams._common import (
    _msl_cell_profile,
    msl_modal_voltage,
    _msl_wave_split_reliability,
    _warn_msl_wave_split_unreliable,
    _warn_msl_beta_scan_railed,
    _warn_if_ringdown_truncated,
    _project_passive,
    _warn_if_passivity_projected,
    _warn_if_nonpassive_smatrix,
    _assemble_mixed_power_wave_s,
    _mixed_reciprocity_deviation,
    _mixed_flux_magnitude_override,
    _register_msl_h_planes,
    _collocated_msl_h,
)


def compute_mixed_s_matrix(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 40.0,
    freqs: "jnp.ndarray | None" = None,
    n_freqs: int = 100,
    strict_extractor: bool = False,
    enforce_passivity: bool = True,
    skip_preflight: bool = False,
    return_diagnostics: bool = False,
    magnitude_channel: str = "flux",
    reciprocity_tol: float = 0.06,
) -> "MixedSMatrixResult":
    """Mixed-family S-matrix: lumped/wire ports + MSL ports (issue #488).

    End-to-end S-parameters on ONE structure carrying two port
    families — the first supported pair is a homogeneous lumped OR
    wire set (``add_port``) together with MSL ports (``add_msl_port``),
    e.g. a vertical probe feed launching onto a microstrip line.

    Each port is driven in turn (lumped/wire ports first, then MSL
    ports, registration order within each family); the non-driven
    lumped/wire ports remain physical matched resistor loads and the
    non-driven MSL ports are passive probe columns. Extraction reuses
    the validated per-family wave machinery unchanged and combines the
    waves in the **Kurokawa power-wave convention** (each wave divided
    by ``sqrt(Re Z0)`` of its own port) — with unequal reference
    impedances across families a pseudo-wave ratio would be off by
    ``sqrt(Z_j/Z_i)`` (issue #460); reciprocity of a reciprocal
    structure is the committed internal falsifier for this choice.

    MAGNITUDE CHANNEL (``magnitude_channel``, default ``"flux"``):
    the #488 falsifier battery + an independent Poynting-flux referee
    measured the port-cell V*I accounting undercounting delivered
    power ~3x at a wire probe feed (the OPEN issue-#313 class) and
    the analytic Hammerstad-Jensen Z0 anchor diverging from the
    measured line ratio on an interface-aligned mesh. Off-diagonal
    MAGNITUDES are therefore taken from Poynting flux by default:
    auto-registered surfaces (a closed 5-face box around each
    lumped/wire port over the z-lo ground; a full cross-section plane
    at each MSL port's probe-0 x) give per-drive net powers, and

        |S_ij|^2 = P_arrive,i / (P_net,j / (1 - |S_jj|^2))

    where ``S_jj`` is the per-family diagonal supplying the reflection
    correction; no Z0 anchor enters the magnitude. Off-diagonal PHASE
    still comes from the wave channel (see below).
    ``magnitude_channel="wave"`` keeps the raw power-wave magnitudes
    (diagnostic; carries the #313 deflation at lumped/wire ports).

    HONESTY NOTES (read before quoting numbers):

    * **Neither diagonal is verified on this lane, and the returned
      diagonal is not always the measured one.** Two separate
      findings: (a) the wire port-cell V*I accounting was measured
      undercounting delivered power ~3x against an independent
      Poynting referee (the open issue #313 reaching the diagonal at
      a near-field-dominated vertical probe), and on an end-fed
      fixture the MSL probe plane's local ``V/I`` was ~591 ohm and
      strongly reactive while the reported ``|S22|`` was 0.03 — those
      cannot both be right. (b) With ``enforce_passivity=True`` (the
      default), ``_project_passive`` is a JOINT SVD clip: when any
      entry is non-passive it rewrites others as a side effect.
      Measured on the committed test fixture, the shipped MSL
      diagonal came out ~4x its unprojected value. So ``result.S``'s
      diagonal is a projected quantity, not a raw per-family
      measurement — read ``S_raw`` and ``passivity_correction`` for
      what was actually measured, and treat any diagonal-derived
      conclusion accordingly.

    * The flux magnitudes inherit a flux-accounting envelope (box
      leakage + finite DFT; the Phase-0 referee class measured ~1.3%
      on the canonical thru) and an ill-conditioning guard: when
      ``1 - |S_jj|^2`` is small (near-total reflection at the driven
      port) the normalization is unreliable and a warning names the
      column.
    * **A per-column power sum near 1 is NOT evidence on this
      channel.** Substituting the definition gives
      ``sum_i |S_ij|^2 = |S_jj|^2 + (P_arr/P_net)(1 - |S_jj|^2)``,
      which is identically 1 whenever the arriving power equals the
      net launched power — for ANY value of the diagonal. The flux
      normalization makes passivity an identity, not a measurement,
      so do not quote column power as a passivity check here (it
      still detects power GAIN, i.e. ``P_arr > P_net``). The
      independent internal check on this channel is RECIPROCITY:
      ``S_ij`` and ``S_ji`` come from different runs with different
      normalizations, so their agreement is real evidence. It runs
      automatically on every extraction (see ``reciprocity_tol``),
      audits the RAW matrix, and is the check that a wrong diagonal
      trips.
    * With ``magnitude_channel="wave"``, off-diagonal magnitudes that
      RECEIVE at a lumped/wire port cell inherit the issue-#313
      near-field deflation of the default port-cell waves. The
      returned ``s21_power_witness`` cross-checks the MSL-receiving
      direction against a delivered-power normalization; quote it
      alongside ``|S|``.
    * Cross-family off-diagonal PHASE mixes two reference-plane
      conventions (port cell vs de-embedded MSL probe plane) and a
      component-mixing ±1 (probes.py sign-convention fence);
      magnitude is the validated observable.
    * ``settling_db`` is the ring-down witness (above −40 dB =
      truncation suspect); preflight output is part of the result.

    v1 restrictions (loud ``NotImplementedError``): uniform mesh only,
    no waveguide/Floquet/coax/TFSF registrations, no bare sources or
    0-ohm ports (they would fire in every drive run), no
    ``reference_plane_cells`` wire ports, no mixed lumped+wire set
    (same fence as the production scan driver), imperative only (no
    ``eps_override`` AD channel). The default ``"flux"`` channel adds
    two more, because the per-port flux box omits its bottom face and
    treats the port extent as a height: a **PEC ``z_lo`` boundary**
    and **vertical (``component="ez"``) lumped/wire ports** are
    required. ``magnitude_channel="wave"`` makes neither assumption.

    Parameters
    ----------
    reciprocity_tol : float, default 0.06
        Relative ``|S_ij|`` vs ``|S_ji|`` disagreement above which the
        reciprocity witness warns. Deliberately set BELOW the 9%
        residual measured on this lane's own reference fixture, so
        that fixture warns rather than passing silently: a tolerance
        above the known residual would document the check and never
        fire it.

    Returns
    -------
    MixedSMatrixResult
    """
    import dataclasses as _dc

    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.sources.msl_port import (
        MSLPort,
        _msl_yz_cells,
        msl_cross_section_span,
        msl_h_plane_stencil,
        msl_loop_current,
        msl_probe_x_coords_n,
    )

    # ---- Registration guards (v1 envelope) --------------------------
    if not self._msl_ports:
        raise ValueError(
            "compute_mixed_s_matrix() needs at least one add_msl_port() "
            "registration (for a pure lumped/wire multiport use the "
            "production scan driver / extract_s_matrix)."
        )
    lw_entries = [pe for pe in self._ports if pe.impedance != 0.0]
    if not lw_entries:
        raise ValueError(
            "compute_mixed_s_matrix() needs at least one sparam-eligible "
            "add_port() lumped/wire port (impedance != 0). For a pure "
            "MSL multiport use compute_msl_s_matrix()."
        )
    if any(pe.impedance == 0.0 for pe in self._ports):
        raise NotImplementedError(
            "compute_mixed_s_matrix() does not support bare sources / "
            "0-ohm ports (add_source or add_port(impedance=0)): they "
            "are not excite-gated and would fire in EVERY drive run, "
            "contaminating the single-drive S-parameter contract."
        )
    if self._waveguide_ports or self._floquet_ports:
        raise NotImplementedError(
            "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
            "waveguide/Floquet ports are not part of the validated "
            "mixed lane (issue #488)."
        )
    _mixed_non_x = [
        pe.name for pe in self._msl_ports
        if pe.direction not in ("+x", "-x")
    ]
    if _mixed_non_x:
        raise NotImplementedError(
            "compute_mixed_s_matrix() v1 covers '+x'/'-x' MSL ports "
            f"only; {_mixed_non_x} are not x-directed. The y-directed "
            "MSL lane landed in issue #661 for compute_msl_s_matrix(); "
            "this mixed lane is itself EXPERIMENTAL (issue #488) with "
            "both diagonals unverified, so it is fenced rather than "
            "extended untested. Use compute_msl_s_matrix() for a pure "
            "MSL multiport."
        )
    if self._coaxial_ports:
        raise NotImplementedError(
            "compute_mixed_s_matrix() v1 covers lumped/wire + MSL only; "
            "coaxial ports need a separate calibration contract."
        )
    if self._tfsf is not None:
        raise NotImplementedError(
            "compute_mixed_s_matrix() is not supported together with "
            "TFSF; TFSF is a plane-wave source, not a port."
        )
    is_wire = [pe.extent is not None for pe in lw_entries]
    if any(is_wire) and not all(is_wire):
        raise NotImplementedError(
            "compute_mixed_s_matrix(): mixed lumped + wire port sets "
            "are not supported (the off-diagonal wave-decomposition "
            "conventions differ — same fence as "
            "compute_lumped_wire_s_matrix_via_scan)."
        )
    wire_mode = all(is_wire)
    if any(getattr(pe, "reference_plane_cells", None) for pe in lw_entries):
        raise NotImplementedError(
            "compute_mixed_s_matrix() v1 does not support "
            "add_port(reference_plane_cells=...); the mixed lane uses "
            "the delivered-power witness for magnitude honesty instead."
        )
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        raise NotImplementedError(
            "compute_mixed_s_matrix() v1 supports the uniform mesh "
            "only (issue #488 scope: NU is explicitly out until the "
            "first pair ships)."
        )
    if self._refinement is not None:
        raise NotImplementedError(
            "compute_mixed_s_matrix() is not supported with SBP-SAT "
            "subgridding."
        )
    if self._solver == "adi":
        raise NotImplementedError(
            "compute_mixed_s_matrix() is not supported with "
            "solver='adi'; use the uniform Yee solver."
        )

    n_lw = len(lw_entries)
    grid = self._build_grid()

    if freqs is None:
        freqs_arr = np.asarray(
            jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs)
        )
    else:
        freqs_arr = np.asarray(freqs)
    n_freqs_used = int(freqs_arr.shape[0])
    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)

    # ---- MSL geometry prep (mirrors compute_msl_s_matrix; uniform
    # lane only, so the NU grid/dz-profile machinery is not needed) ----
    entries = self._resolve_msl_probe_entries(grid)
    n_msl = len(entries)
    msl_ports: list[MSLPort] = []
    for pe in entries:
        x_feed, y_centre, z_lo = pe.position
        msl_ports.append(MSLPort(
            feed_x=float(x_feed),
            y_lo=float(y_centre - pe.width / 2),
            y_hi=float(y_centre + pe.width / 2),
            z_lo=float(z_lo),
            z_hi=float(z_lo + pe.height),
            direction=pe.direction,
            impedance=pe.impedance,
            excitation=pe.waveform,
        ))
    # Probe-0 x-coordinate per port: the S1 V*I wave split lives at
    # probe 0 (the de-embedding reference plane); the mixed lane does
    # not run the N-probe spatial fit, so only probe 0 is recorded.
    probe_xs = [
        msl_probe_x_coords_n(
            grid, mp,
            n_probes=int(pe.n_probes),
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        for mp, pe in zip(msl_ports, entries)
    ]
    h_stencils = [msl_h_plane_stencil(grid, mp, xs[0])
                  for mp, xs in zip(msl_ports, probe_xs)]
    from rfx.api._preflight import msl_probe_clearance_for_port
    probe_clearance = tuple(msl_probe_clearance_for_port(
        self, pe, grid, probe_coordinates=xs,
    ) for pe, xs in zip(entries, probe_xs))
    # Probe-LADDER validation (issue #488 attempt-1 defect D3). The S1
    # V*I split records probe 0 only, but v1 must not silently accept
    # a ladder the validated MSL lane rejects: attempt 1 registered a
    # "+x"-facing port whose default ladder (offset 31 + 4x12 cells)
    # ran past the declared domain; msl_probe_x_coords_n CLAMPS such
    # coordinates, and the surviving probe-0 plane sat 1.1 mm from
    # the trace's open end at the domain edge (a Box is rasterized
    # only inside the declared domain — the trace does NOT continue
    # into the CPML padding, so every boundary-touching trace has an
    # OPEN end there). The plane then measures the open-stub
    # standing-wave impedance -j*Z0*cot(beta*d) ~ 1/f (~1400 ohm at
    # 1 GHz vs the 48 ohm line) instead of a travelling wave.
    # Hard guard: every ladder coordinate strictly inside (0, lx)
    # and strictly monotonic (clamp shows up as duplicates).
    # Advisory: probe-0 closer than lambda_g/4 to a domain x-edge.
    from rfx.api._preflight import msl_min_probe_clearance
    _lx_dom = float(self._domain[0])
    _clear = msl_min_probe_clearance(float(self._freq_max))
    for pe, pxs in zip(entries, probe_xs):
        xs = [float(x) for x in pxs]
        mono = all(
            (xs[q + 1] - xs[q]) * (1 if pe.direction == "+x" else -1)
            > 0.5 * float(grid.dx)
            for q in range(len(xs) - 1)
        )
        if (not mono) or min(xs) <= 0.0 or max(xs) >= _lx_dom:
            raise ValueError(
                f"compute_mixed_s_matrix: MSL port {pe.name!r} probe "
                f"ladder ({', '.join(f'{x * 1e3:.2f}' for x in xs)} mm) "
                f"leaves the declared x-domain (0, {_lx_dom * 1e3:.2f}) "
                "mm or was clamped at its edge — the equivalent "
                "registration is rejected by compute_msl_s_matrix, and "
                "a plane near the trace's open end at the domain edge "
                "measures the stub standing wave, not the line. Face "
                "the port toward the DUT (direction), reduce "
                "n_probe_offset/n_probe_spacing, or enlarge the domain."
            )
        _edge_d = min(xs[0], _lx_dom - xs[0])
        if _edge_d < _clear:
            import warnings as _w488
            _w488.warn(
                f"compute_mixed_s_matrix: MSL port {pe.name!r} probe-0 "
                f"plane is {_edge_d * 1e3:.2f} mm from a domain x-edge "
                f"(< lambda_g/4 = {_clear * 1e3:.2f} mm at freq_max). "
                "This is a layout recommendation near a possible "
                "trace discontinuity, not an accuracy bound. Check "
                "the realized termination and observation-plane "
                "sensitivity; reliable only screens relative low signal.",
                stacklevel=2,
            )

    dy_arr = _msl_cell_profile(grid, "y", grid.ny)
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    port_idx_meta = []
    for mp in msl_ports:
        span = msl_cross_section_span(grid, mp)
        port_idx_meta.append(dict(
            j_lo=span["w_lo"], j_hi=span["w_hi"],
            k_lo=span["n_lo"], k_hi=span["n_hi"],
            j_centre=span["w_centre"], k_top=span["n_hi"],
        ))

    # One materials assembly shared by the HJ eps anchor AND every
    # drive run (materials do not depend on excite flags).
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0_hj
    _refuse_f0_hj(self._thin_conductors, "MSL junction S-parameter")
    _mx_pec_sheets: list = []
    _mx_pec_wires: list = []
    materials, debye_spec, lorentz_spec, pec_mask, _, _, _ = \
        self._assemble_materials(
            grid, pec_sheets=_mx_pec_sheets, pec_wires=_mx_pec_wires)
    # #931 §1.9: realized wall planes locate the trace, not cells.
    from rfx.boundaries.pec import (
        realized_pec_edge_masks as _rpem_mx,
    )
    from rfx.probes.msl_wave_decomp import (
        realized_trace_planes_on_column as _trace_planes_mx,
    )
    _mx_pec_edge_masks = None
    if pec_mask is not None or _mx_pec_sheets or _mx_pec_wires:
        _mx_pec_edge_masks = _rpem_mx(
            pec_mask, sheets=tuple(_mx_pec_sheets),
            wires=tuple(_mx_pec_wires),
            periodic=self._periodic_flags())

    # Analytic Hammerstad-Jensen anchor per MSL port (eps precedence
    # mirrors compute_msl_s_matrix: explicit eps_r_sub > rasterised
    # eps_r at the trace-centre substrate cell).
    z0_hj_per_port: list[float] = []
    beta0_per_port: list[np.ndarray] = []
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    _c0_mixed = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    for p_idx, pe in enumerate(entries):
        meta = port_idx_meta[p_idx]
        if pe.eps_r_sub is not None:
            eps_r_ref = float(pe.eps_r_sub)
        else:
            k_mid = (meta["k_lo"] + meta["k_hi"]) // 2
            i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
            eps_r_ref = float(np.asarray(
                materials.eps_r[i_feed_p, meta["j_centre"], k_mid]
            ))
        z0_hj, eps_eff_hj = hammerstad_jensen_z0_eps_eff(
            pe.width, pe.height, eps_r_ref
        )
        z0_hj_per_port.append(float(z0_hj))
        beta0_per_port.append(
            2.0 * np.pi * freqs_arr * float(np.sqrt(eps_eff_hj))
            / _c0_mixed
        )

    # Trace-conductor z-cell span (closed Ampere loop needs the PEC
    # trace; mirrors compute_msl_s_matrix issue #80 stage S1).
    trace_k_per_port: list[tuple[int, int]] = []
    for p_idx in range(n_msl):
        meta = port_idx_meta[p_idx]
        i_feed_p = _msl_yz_cells(grid, msl_ports[p_idx])[0][0]
        _k_lo_tr, _k_hi_tr = _trace_planes_mx(
            _mx_pec_edge_masks, 2, (i_feed_p, meta["j_centre"]),
            meta["k_top"], periodic=self._periodic_flags())
        if _k_lo_tr is None:
            raise RuntimeError(
                "compute_mixed_s_matrix: no realized PEC trace "
                "conductor found above the substrate top for MSL port "
                f"{entries[p_idx].name!r}; the closed Ampere-loop "
                "current needs the trace. Declare the microstrip trace "
                "as a Box(material='pec') (a volume) or as a "
                "zero-thickness Box / add_thin_conductor (a sheet, "
                "#931)."
            )
        trace_k_per_port.append((_k_lo_tr, _k_hi_tr))

    # Wire live-cell counts for the per-cell impedance normalization
    # (mirrors compute_lumped_wire_s_matrix_via_scan, issue #318).
    n_live_lw = np.ones(n_lw, dtype=np.int64)
    if wire_mode:
        from rfx.sources.sources import WirePort, _wire_port_live_cells
        axis_map = {"ex": 0, "ey": 1, "ez": 2}
        for idx, pe in enumerate(lw_entries):
            end = list(pe.position)
            end[axis_map[pe.component]] += pe.extent
            wp = WirePort(
                start=pe.position, end=tuple(end),
                component=pe.component, impedance=pe.impedance,
                excitation=pe.waveform,
            )
            n_live_lw[idx] = _wire_port_live_cells(
                grid, wp, _mx_pec_edge_masks)[2]

    if not skip_preflight:
        # One preflight for the full registration (run() would fire it
        # per drive run — 2*n_ports repeats of the same advisories).
        self.preflight()

    if magnitude_channel not in ("flux", "wave"):
        raise ValueError(
            "compute_mixed_s_matrix: magnitude_channel must be 'flux' "
            f"(default) or 'wave', got {magnitude_channel!r}."
        )
    if magnitude_channel == "flux":
        # The per-port flux box is CLOSED by the z-lo PEC ground: it
        # has five faces and omits the bottom because flux through a
        # PEC face is identically zero. It also treats pe.extent as a
        # VERTICAL height. Both are physical preconditions, so check
        # them instead of assuming (review finding: a user with an
        # open z-lo boundary or a horizontal port would otherwise get
        # a silently wrong P_net on the DEFAULT channel).
        _pec_faces = (
            self._boundary_spec.pec_faces()
            if self._boundary_spec is not None else set()
        )
        if "z_lo" not in _pec_faces:
            raise NotImplementedError(
                "compute_mixed_s_matrix(magnitude_channel='flux') "
                "requires a PEC z_lo boundary: the per-port flux box "
                "omits its bottom face because flux through a PEC "
                "ground is identically zero. Your z_lo face is "
                f"{sorted(_pec_faces) or 'not PEC'}, so the box would "
                "not be closed and P_net would be wrong. Use "
                "BoundarySpec(z=Boundary(lo='pec', ...)), or pass "
                "magnitude_channel='wave' (which carries the #313 "
                "port-cell deflation instead)."
            )
        _bad = [pe for pe in lw_entries if pe.component != "ez"]
        if _bad:
            raise NotImplementedError(
                "compute_mixed_s_matrix(magnitude_channel='flux') "
                "supports vertical (component='ez') lumped/wire ports "
                "only: the flux box is built assuming the port extent "
                "is a z height above the ground plane. Offending "
                f"component(s): {sorted({pe.component for pe in _bad})}."
            )

    drive_plan = [("lw", j) for j in range(n_lw)] + \
                 [("msl", d) for d in range(n_msl)]
    n_runs = len(drive_plan)
    _complex_dtype = (
        jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    )

    saved_dft = list(self._dft_planes)
    saved_msl = list(self._msl_ports)
    saved_ports = list(self._ports)
    saved_probes = list(self._probes)
    saved_internal = set(self._internal_probe_indices)
    saved_flux = list(self._flux_monitors)
    try:
        # Ring-down settling witness probes at every MSL probe plane,
        # mid-substrate (worst plane wins — a single plane is
        # standing-wave-node sensitive; mirrors compute_msl_s_matrix).
        _witness_base = len(self._probes)
        _witness_total = 0
        for pe_w, pxs_w in zip(entries, probe_xs):
            for _x_w in pxs_w:
                self.add_probe(
                    position=(
                        float(_x_w),
                        float(pe_w.position[1]),
                        float(pe_w.position[2]) + 0.5 * float(pe_w.height),
                    ),
                    component="ez",
                )
                _witness_total += 1
        self._internal_probe_indices.update(
            range(_witness_base, _witness_base + _witness_total)
        )
        witness_probes = list(self._probes)

        # Flux surfaces for the magnitude channel (registered once —
        # geometry is drive-independent). Lumped/wire port p: a closed
        # 5-face box over the z-lo ground plane (flux through the PEC
        # ground face is identically zero); outward-signed face list.
        # MSL port p: one full-cross-section x-plane at probe 0 (its
        # de-embedding reference plane).
        flux_faces_lw: list[list[tuple[str, float]]] = []
        flux_names_msl: list[str] = []
        if magnitude_channel == "flux":
            _m = 3.0 * float(grid.dx)
            for p, pe in enumerate(lw_entries):
                x0_p, y0_p, _z0p = (float(c) for c in pe.position)
                z_top = float(pe.extent or grid.dx) + _m
                faces = []
                for ax, coord, size, center, sgn in (
                    ("x", x0_p - _m, (2 * _m, z_top), (y0_p, z_top / 2), -1.0),
                    ("x", x0_p + _m, (2 * _m, z_top), (y0_p, z_top / 2), +1.0),
                    ("y", y0_p - _m, (2 * _m, z_top), (x0_p, z_top / 2), -1.0),
                    ("y", y0_p + _m, (2 * _m, z_top), (x0_p, z_top / 2), +1.0),
                    ("z", z_top, (2 * _m, 2 * _m), (x0_p, y0_p), +1.0),
                ):
                    nm = f"_mixed_flux_lw{p}_{ax}{'+' if sgn > 0 else '-'}{coord:.6g}"
                    self.add_flux_monitor(
                        axis=ax, coordinate=float(coord),
                        freqs=jnp.asarray(freqs_arr),
                        size=(float(size[0]), float(size[1])),
                        center=(float(center[0]), float(center[1])),
                        name=nm,
                    )
                    faces.append((nm, sgn))
                flux_faces_lw.append(faces)
            for p, pxs in enumerate(probe_xs):
                nm = f"_mixed_flux_msl{p}"
                self.add_flux_monitor(
                    axis="x", coordinate=float(pxs[0]),
                    freqs=jnp.asarray(freqs_arr), name=nm,
                )
                flux_names_msl.append(nm)

        v_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
        vref_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
        i_lw = np.zeros((n_runs, n_lw, n_freqs_used), dtype=np.complex128)
        v0_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
        i_msl = np.zeros((n_runs, n_msl, n_freqs_used), dtype=np.complex128)
        _n_probes_max = max(len(pxs) for pxs in probe_xs)
        v_lad = np.zeros(
            (n_runs, n_msl, _n_probes_max, n_freqs_used),
            dtype=np.complex128,
        )
        settling_db_runs = np.full(n_runs, np.nan)
        # Signed per-run flux accountings (magnitude_channel="flux"):
        # box_lw[run, p]  = net OUTWARD box flux at lumped/wire port p
        # plane_msl[run, p] = raw +x-directed flux at MSL port p's plane
        box_lw = np.zeros((n_runs, n_lw, n_freqs_used))
        plane_msl = np.zeros((n_runs, n_msl, n_freqs_used))

        for run_idx, (fam, loc) in enumerate(drive_plan):
            # Excite exactly one port; every other port keeps its
            # physical termination (matched resistor cells for
            # lumped/wire, passive probe column for MSL). The driven
            # lumped/wire port's default waveform is synthesised by
            # the runner (issue #322); the driven MSL port mirrors the
            # compute_msl_s_matrix default.
            self._ports = [
                _dc.replace(pe, excite=(fam == "lw" and k == loc))
                for k, pe in enumerate(saved_ports)
            ]
            # Rebuild from the RESOLVED entries (auto probe offset /
            # spacing solved above, the ladder ``probe_xs`` was built
            # from), not from the raw registration ``saved_msl`` — so
            # ``self._msl_ports`` inside ``_forward_from_materials``
            # describes the ladder this run actually probes, as
            # compute_msl_s_matrix already does. ``saved_msl`` is
            # restored in ``finally``.
            run_msl = []
            for k, pe in enumerate(entries):
                driven = fam == "msl" and k == loc
                wf = (
                    (pe.waveform if pe.waveform is not None else
                     GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8))
                    if driven else None
                )
                run_msl.append(_dc.replace(pe, excite=driven, waveform=wf))
            self._msl_ports = run_msl
            self._probes = list(witness_probes)

            # Per-run named DFT planes: Ez at probe 0 (line voltage),
            # Hy+Hz at probe 0 (closed Ampere-loop current legs).
            self._dft_planes = list(saved_dft)
            names = []
            h_names = []
            for p_idx, pxs in enumerate(probe_xs):
                nm = f"_mixed_run{run_idx}_p{p_idx}"
                # Full ez ladder: probe 0 feeds the V*I wave split;
                # the N-probe least-squares fit (measured line Zc for
                # the MSL diagonal — see the z0_fit block below) needs
                # every plane.
                for q_idx, x_c in enumerate(pxs):
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(x_c), component="ez",
                        freqs=jnp.asarray(freqs_arr),
                        name=nm + f"_ez{q_idx}",
                    )
                h_names.append(_register_msl_h_planes(
                    self, nm, h_stencils[p_idx], ("hy", "hz"),
                    jnp.asarray(freqs_arr),
                ))
                names.append(nm)

            raw = self._forward_from_materials(
                grid, materials, debye_spec, lorentz_spec,
                n_steps=n_steps, checkpoint=False, pec_mask=pec_mask,
                pec_sheets=tuple(_mx_pec_sheets),
                pec_wires=tuple(_mx_pec_wires),
                port_s11_freqs=freqs_arr,
                _return_raw_port_sparams=True,
            )
            accs = raw["wire"] if wire_mode else raw["lumped"]
            if accs is None or len(accs) != n_lw:
                raise RuntimeError(
                    "compute_mixed_s_matrix: production scan returned "
                    f"{0 if accs is None else len(accs)} "
                    f"{'wire' if wire_mode else 'lumped'} accumulators "
                    f"for run {run_idx}, expected {n_lw}."
                )
            for i_port in range(n_lw):
                _spec_i, vi = accs[i_port]
                v_lw[run_idx, i_port, :] = np.asarray(vi[0])
                i_lw[run_idx, i_port, :] = np.asarray(vi[1])
                if wire_mode:
                    # Issue #683 x #764: pre-injection drive-sample
                    # reference channel (vi[4]) — keeps this lane's
                    # byte-frozen #488 wave algebra frozen through the
                    # wire sampling flip (only the drive port's own
                    # sample moved; passive samples are slot-invariant).
                    # A shorter tuple marks a lane without a separate
                    # v_ref channel (the NU lane): it has ALWAYS sampled
                    # POST-injection and its decomposer was calibrated in
                    # that frame, so vi[0] is that lane's correct drive
                    # reference — not a pre-injection sample.
                    vref_lw[run_idx, i_port, :] = np.asarray(
                        vi[4] if len(vi) > 4 else vi[0])

            planes = raw.get("dft_planes")
            if not planes:
                raise RuntimeError(
                    "compute_mixed_s_matrix: the production scan "
                    "returned no DFT planes — the issue-#488 raw hook "
                    "is out of sync with _forward_from_materials."
                )

            if magnitude_channel == "flux":
                from rfx.probes.probes import flux_spectrum
                fmon = raw.get("flux_monitors")
                if not fmon:
                    raise RuntimeError(
                        "compute_mixed_s_matrix: the production scan "
                        "returned no flux monitors — the issue-#488 "
                        "flux hook is out of sync with "
                        "_forward_from_materials."
                    )
                for p, faces in enumerate(flux_faces_lw):
                    acc = np.zeros(n_freqs_used)
                    for nm, sgn in faces:
                        acc = acc + sgn * np.asarray(
                            flux_spectrum(fmon[nm]), dtype=np.float64
                        )
                    box_lw[run_idx, p, :] = acc
                for p, nm in enumerate(flux_names_msl):
                    plane_msl[run_idx, p, :] = np.asarray(
                        flux_spectrum(fmon[nm]), dtype=np.float64
                    )

            # Settling witness from the probe time series (worst
            # end/peak Ez^2 across the MSL probe planes).
            _ts = raw.get("time_series")
            if _ts is not None and not is_tracer(_ts):
                _ts_np = np.asarray(
                    _ts[:, _witness_base:_witness_base + _witness_total],
                    dtype=float,
                )
                if (_ts_np.shape[0] >= 10
                        and _ts_np.shape[1] == _witness_total):
                    _p = _ts_np ** 2
                    _tail = max(1, _p.shape[0] // 10)
                    _end = _p[-_tail:, :].mean(axis=0)
                    _peak = _p.max(axis=0)
                    _tiny = np.finfo(float).tiny
                    settling_db_runs[run_idx] = float(np.max(
                        10.0 * np.log10((_end + _tiny) / (_peak + _tiny))
                    ))

            # MSL line V (probe-0 plane) + closed-loop I, with the
            # leapfrog E/H half-step correction (mirrors
            # compute_msl_s_matrix line-for-line; see that method for
            # the full derivation comments).
            _hs_phase = jnp.exp(
                1j * 2.0 * jnp.pi * jnp.asarray(freqs_arr)
                * (float(grid.dt) * 0.5)
            )
            for p_idx, meta in enumerate(port_idx_meta):
                nm = names[p_idx]
                for q_idx in range(len(probe_xs[p_idx])):
                    ez_plane = jnp.asarray(
                        planes[nm + f"_ez{q_idx}"].accumulator
                    )
                    # Anchor the V-span top on the rasterized trace
                    # node, not round(h_sub/dx) — see the sibling
                    # comment in compute_msl_s_matrix (PR #516 F2).
                    v_q = msl_modal_voltage(
                        ez_plane, j_centre=meta["j_centre"],
                        k_lo=meta["k_lo"],
                        k_hi=trace_k_per_port[p_idx][0],
                        dz_arr=dz_arr, dtype=_complex_dtype,
                    )
                    v_lad[run_idx, p_idx, q_idx, :] = np.asarray(v_q)
                v_f = jnp.asarray(v_lad[run_idx, p_idx, 0, :])
                hy_plane, hz_plane = _collocated_msl_h(
                    planes, h_names[p_idx], h_stencils[p_idx]["weights"],
                )
                hy_plane = hy_plane * _hs_phase[:, None, None].astype(hy_plane.dtype)
                hz_plane = hz_plane * _hs_phase[:, None, None].astype(hz_plane.dtype)
                k_tr_lo, k_tr_hi = trace_k_per_port[p_idx]
                i_f = msl_loop_current(
                    hy_plane, hz_plane,
                    j_lo=meta["j_lo"], j_hi=meta["j_hi"],
                    k_trace_lo=k_tr_lo, k_trace_hi=k_tr_hi,
                    dy_arr=dy_arr, dz_arr=dz_arr,
                    direction=msl_ports[p_idx].direction,
                )
                v0_msl[run_idx, p_idx, :] = np.asarray(v_f)
                i_msl[run_idx, p_idx, :] = np.asarray(i_f)

        # ---- Power-wave assembly (pure; unit-tested separately) ----
        S, s21_power = _assemble_mixed_power_wave_s(
            v_lw, i_lw, v0_msl, i_msl,
            np.asarray([pe.impedance for pe in lw_entries]),
            n_live_lw, np.asarray(z0_hj_per_port),
            wire_mode, drive_plan,
            v_ref_lw=(vref_lw if wire_mode else None),
        )
        S = jnp.asarray(S, dtype=_complex_dtype)

        # ---- N-probe line Zc: DIAGNOSTIC ONLY (issue #488) ---------
        # This lane does NOT substitute a fitted Zc into the MSL
        # diagonal. An earlier revision did, and it was withdrawn
        # after review: the fit's sign is not stable here, so the
        # substitution silently fell back to the analytic anchor and
        # looked like a confirmed measurement.
        #
        # Mechanism (why a sign fix would not be enough): the model
        # V_n = alpha*exp(-j beta x_n) + gamma*exp(+j beta x_n) is
        # invariant under (beta -> -beta, alpha <-> gamma), so a beta
        # scan that lands on the wrong branch SWAPS the two wave
        # roles and flips the sign of (alpha - gamma) — and therefore
        # of z0 = (alpha - gamma)/I1. Measured on one fixture: the
        # fitted sign differs between num_periods=4 and 20.
        #
        # The SECOND half of what this comment used to allege — that
        # `msl_loop_current`'s docstring ("the returned I is positive
        # for a forward quasi-TEM wave") and compute_msl_s_matrix's
        # #140 dir_sign comment ("a -x port's fitted z0 inherits a
        # negative sign") describe OPPOSITE conventions — was
        # measured during issue #661 and is a PROSE defect only, now
        # corrected in msl_loop_current's docstring. Own-drive
        # diagonal on the committed thru fixture:
        # Re((alpha-gamma)/I1) = +57.52 ohm at the "+x" port and
        # -57.56 ohm at the "-x" port (same magnitude to 0.08%), so
        # the #140 comment described the code and the docstring's
        # blanket claim was scoped wrong. The lane is self-consistent
        # about it: |S11|=|S22| to 5 decimals, reciprocity 1.27e-05,
        # column power <= 0.99998, both reported Z0 positive. No code
        # changed. The beta-branch instability above is separate and
        # still stands.
        #
        # Issue #524 now carries ONE open item, not two. The
        # 2026-08-30 re-measurement on current main (VESSL
        # 369367257265, PR #799, driver
        # scripts/diagnostics/msl_passive_port_reflection.py) closed
        # the drive asymmetry: the two ports read |Gamma| 0.1799 and
        # 0.1759 on the shipped R = 50 ohm fixture, and the
        # cell-aligned and +/-y-rotated variants reproduce that to 4
        # decimals, so the July 0.194-vs-0.073 split does not exist on
        # this tree. The same battery WITHDREW the "~30 ohm
        # termination" inference: it was built on a July line Zc of
        # 38.75 ohm, while the fitted Zc here is 41.9 ohm against the
        # Hammerstad-Jensen 47.9 ohm -- a dx bias (#487 class), not a
        # termination error. What stays open on #524 is the passive
        # port's reflection itself: |Gamma_passive| = 0.176-0.180 at
        # R = 50 ohm with no mechanism named. Both single-reflector
        # models were falsified over a 25 / Zc / 50 / open R sweep
        # (shunt Zc/(2R+Zc) misses by up to 0.119, end |R-Zc|/(R+Zc)
        # by 0.242; implied k runs 1.91 -> 1.03, so the shape does not
        # follow 1/(2R+Zc)); reported and stopped there, no two-
        # parameter fit.
        #
        # The fit is still computed and EXPOSED (return_diagnostics)
        # because it is the only handle on the measured-vs-analytic Zc
        # gap (41.9 ohm fitted vs 47.9 ohm Hammerstad-Jensen on this
        # mesh, 2026-08-30; #487), but it never feeds a shipped
        # number, and its magnitude is reported without a sign claim.
        from rfx.probes.msl_wave_decomp import extract_msl_nprobe
        z0_msl_fit = np.full((n_msl, n_freqs_used), np.nan)
        beta_railed_msl = np.zeros((n_msl, n_freqs_used), dtype=bool)
        for d in range(n_msl):
            run_d = n_lw + d
            n_p = len(probe_xs[d])
            res_fit = extract_msl_nprobe(
                jnp.asarray(v_lad[run_d, d, :n_p, :].T),
                jnp.asarray(np.asarray(probe_xs[d], dtype=float)),
                jnp.asarray(i_msl[run_d, d, :]),
                jnp.asarray(beta0_per_port[d]),
                z0_hj=z0_hj_per_port[d],
            )
            z0_msl_fit[d, :] = np.abs(np.asarray(
                jax.lax.stop_gradient(res_fit["z0"])
            ).astype(np.complex128))
            beta_railed_msl[d, :] = np.asarray(
                jax.lax.stop_gradient(res_fit["beta_railed"])
            ).astype(bool)
            _dev = float(np.max(
                np.abs(z0_msl_fit[d, :] - z0_hj_per_port[d])
                / z0_hj_per_port[d]
            ))
            if _dev > 0.10:
                import warnings as _wz
                _wz.warn(
                    f"compute_mixed_s_matrix: the N-probe |Zc| fit for "
                    f"MSL port {entries[d].name!r} deviates up to "
                    f"{_dev * 100:.1f}% from analytic Hammerstad-Jensen "
                    f"{z0_hj_per_port[d]:.2f} ohm. The MSL diagonal "
                    "uses the analytic anchor; this fitted value is "
                    "not used in S. That does not certify the shared "
                    "V/I inputs. Check settling_db, the relative "
                    "low-signal mask, probe_clearance and observation-"
                    "plane sensitivity before interpreting S.",
                    stacklevel=2,
                )
        # β-scan rail flags for the diagnostic N-probe fit (issue
        # #681): a railed bin's |Zc|/β above are the ±35% scan-window
        # limit, not a measurement. S is unaffected (the MSL diagonal
        # uses the analytic HJ anchor).
        _warn_msl_beta_scan_railed(
            beta_railed_msl, freqs_arr,
            tuple(pe.name for pe in entries),
        )
        s_wave_full = None

        if magnitude_channel == "flux":
            import warnings as _wf
            s_wave_full = np.asarray(jax.lax.stop_gradient(S))
            msl_away = [
                (+1.0 if pe.direction == "+x" else -1.0) for pe in entries
            ]
            S, _ill_cond, _neg_power = _mixed_flux_magnitude_override(
                S, box_lw, plane_msl, drive_plan, msl_away, n_lw,
            )
            S = jnp.asarray(S, dtype=_complex_dtype)
            for col in range(n_lw + n_msl):
                n_ill = int(_ill_cond[col].sum())
                if n_ill:
                    _wf.warn(
                        f"compute_mixed_s_matrix: flux magnitude for "
                        f"driven port index {col} is ill-conditioned at "
                        f"{n_ill}/{n_freqs_used} bins "
                        f"(1 - |S_jj|^2 < 0.05 — near-total reflection; "
                        "the incident-power reconstruction divides by "
                        "almost nothing). Off-diagonals of that column "
                        "are UNRELIABLE at those bins.",
                        stacklevel=2,
                    )
                n_neg = int(_neg_power[col].sum())
                if n_neg:
                    _wf.warn(
                        f"compute_mixed_s_matrix: flux power at port "
                        f"index {col} came out NEGATIVE at "
                        f"{n_neg}/{n_freqs_used} bins — a sign or "
                        "accounting defect in that port's flux "
                        "surface (net launched power cannot flow INTO "
                        "a driven port, and arriving power cannot be "
                        "negative at a receive port). Clipped to zero, "
                        "which reports |S| = 0 rather than failing: "
                        "treat those bins as an extraction failure, "
                        "not as an absence of coupling.",
                        stacklevel=2,
                    )

        # MSL standing-wave-null reliability from each MSL port's own
        # driven run (mirrors compute_msl_s_matrix issue #337).
        reliable = None
        try:
            v_port = np.stack([
                v0_msl[n_lw + p, p, :] for p in range(n_msl)
            ])
            i_port = np.stack([
                i_msl[n_lw + p, p, :] for p in range(n_msl)
            ])
            reliable = _msl_wave_split_reliability(
                v_port, i_port, freqs_arr
            )
            _warn_msl_wave_split_unreliable(reliable, freqs_arr)
        except (ValueError, TypeError):
            pass

        port_names = tuple(
            [f"lw{k}" for k in range(n_lw)]
            + [pe.name for pe in entries]
        )
        port_families = tuple(
            [("wire" if wire_mode else "lumped")] * n_lw
            + ["msl"] * n_msl
        )
        z0_ref = np.asarray(
            [float(pe.impedance) for pe in lw_entries]
            + z0_hj_per_port
        )

        s_raw = None
        passivity_correction = None
        if enforce_passivity and not is_tracer(S):
            s_projected, correction = _project_passive(S)
            if bool(np.any(np.asarray(correction) > 0.0)):
                s_raw = S
                passivity_correction = correction
                S = s_projected

        result = MixedSMatrixResult(
            S=S,
            freqs=np.asarray(freqs_arr),
            port_names=port_names,
            port_families=port_families,
            z0_ref=z0_ref,
            settling_db=settling_db_runs,
            s21_power_witness=s21_power,
            reliable=reliable,
            S_raw=s_raw,
            passivity_correction=passivity_correction,
            S_wave=s_wave_full,
            magnitude_channel=magnitude_channel,
            beta_railed=beta_railed_msl,
            probe_clearance=probe_clearance,
        )
        _warn_if_ringdown_truncated(
            settling_db_runs, port_names, num_periods=num_periods,
        )
        if passivity_correction is not None and not is_tracer(passivity_correction):
            _warn_if_passivity_projected(passivity_correction, freqs_arr)
        import dataclasses as _dc2
        audit_result = (
            result if s_raw is None else _dc2.replace(result, S=s_raw)
        )
        _warn_if_nonpassive_smatrix(
            audit_result,
            extractor="compute_mixed_s_matrix",
            strict=strict_extractor,
            passivity_tol=0.10,
        )
        # RECIPROCITY WITNESS — the only independent runtime check on
        # the flux channel (review finding: the shared passivity audit
        # above is structurally inert here, because column power is an
        # identity under the flux normalization; and
        # validate_port_smatrix's own reciprocity option compares
        # COMPLEX S, which would misfire on this lane where
        # cross-family phase is provisional by construction).
        # S_ij and S_ji come from different drive runs with different
        # normalizations, so their MAGNITUDE agreement is real
        # evidence — and it is exactly the check that would have
        # caught a wrong diagonal feeding the flux normalization.
        #
        # Audited on the RAW (unprojected) matrix for the same reason
        # the passivity self-check above is: `_project_passive` is a
        # joint SVD clip, so it moves entries TOWARD each other and
        # would understate the disagreement actually measured. It
        # also rewrites diagonals as a side effect (measured on the
        # test fixture: a shipped MSL diagonal ~4x its raw value),
        # which is precisely what this witness exists to surface.
        _rec = _mixed_reciprocity_deviation(
            S if s_raw is None else s_raw
        )
        if _rec is not None:
            _pair, _dev_max = _rec
            if _dev_max > reciprocity_tol:
                import warnings as _wr
                _wr.warn(
                    f"compute_mixed_s_matrix: reciprocity deviation "
                    f"max {_dev_max * 100:.1f}% between |S[{_pair[0]},"
                    f"{_pair[1]}]| and |S[{_pair[1]},{_pair[0]}]| "
                    f"(tolerance {reciprocity_tol * 100:.0f}%). For a "
                    "reciprocal structure these must agree; a "
                    "disagreement means one DRIVEN-port diagonal is "
                    "wrong (the diagonals set the incident-power "
                    "normalization P_inc = P_net/(1-|S_jj|^2)), or a "
                    "flux surface is mis-signed. Note the per-column "
                    "power sum CANNOT detect this — it is an identity "
                    "on this channel. Inspect the diagonals and "
                    "settling_db before quoting any |S|.",
                    stacklevel=2,
                )
        if return_diagnostics:
            # R5 inspection surface: the raw per-run phasors behind
            # every wave, so a suspicious |S| can be traced to V/I
            # health (e.g. a broken Ampere loop shows as v0/i far
            # from the line Z0) without re-running.
            return result, {
                "v_lw": v_lw, "i_lw": i_lw,
                "v0_msl": v0_msl, "i_msl": i_msl,
                "drive_plan": drive_plan,
                "z0_hj_msl": np.asarray(z0_hj_per_port),
                # |Zc| from the N-probe fit — DIAGNOSTIC ONLY, never
                # substituted into S (sign unstable; see the block
                # above). Magnitude only: no sign claim is made.
                "z0_msl_fit_abs": z0_msl_fit,
                "box_lw_flux": box_lw,
                "plane_msl_flux": plane_msl,
            }
        return result
    finally:
        self._dft_planes = saved_dft
        self._msl_ports = saved_msl
        self._ports = saved_ports
        self._probes = saved_probes
        self._internal_probe_indices = saved_internal
        self._flux_monitors = saved_flux


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# ``compute_mixed_s_matrix`` was a ``def`` in the ``_SparamMixin`` class body,
# so its ``__qualname__`` read ``_SparamMixin.compute_mixed_s_matrix``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at
# class-composition time so that a bad keyword argument reports
# ``Simulation.compute_mixed_s_matrix() got an unexpected keyword
# argument``, and it SKIPS any function whose qualname does not match that
# pattern. Leaving the bare name here would therefore change that TypeError
# message -- a user-visible behaviour change in a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_mixed_s_matrix.__qualname__ = "_SparamMixin.compute_mixed_s_matrix"
