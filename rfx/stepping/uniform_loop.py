from __future__ import annotations

from rfx._uniform_specs import (
    DesignBoxSpec,
    DesignOccupancySpec,
    ProbeSpec,
    SimResult,
    SnapshotSpec,
    SourceSpec,
    _ORDER4_CFL_FACTOR,
    _StepContext,
    _nearest_divisor,
    resolve_periodic,
)

from rfx.backend import backend_class
from rfx.measurement import init_uniform_records, uniform_records
from dataclasses import replace
from typing import Callable, NamedTuple
from rfx import _realized, _grid_metric
import jax
import jax.numpy as jnp
from rfx.core.drives import StepDrives, drive_layout
from rfx.measurement import AUTO_END, finish_decay, make_decay_stop, one_step_xs
from rfx.model.source_coefficients import drive_table
from rfx.model import with_edge_keep
import numpy as np
from rfx.boundaries.axes import drop_periodic_axes, resolve_cpml_axes
from rfx.grid import Grid
from rfx.stepping.scan import recorded_scan
from rfx.core.yee import FDTDState, MaterialArrays, init_state, curl_h, CurlBoundary, precompute_coeffs
from rfx.stepping.yee import update_e, update_e_aniso, update_e_aniso_inv
from rfx.boundaries.pec import resolve_wall_faces, kottke_fenced_edge_masks, realized_pec_edge_masks
from rfx.progress import ProgressReporter, check_not_traced, concat_chunks, scan_with_progress, validate_report_every
from rfx.snapshots import plan_snapshot_pieces, snapshot_axes, snapshot_extractor, validate_snapshot_spec
from rfx.model import design_box_bounds, resolve_design_box

from rfx.stepping.uniform_setup import (
    _build_step_setup,
    _update_e_with_optional_dispersion,
    core_step_invariants,
    make_core_step,
)

def run(
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    boundary: str = "pec",
    cpml_axes: str | None = None,
    pec_axes: str | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    tfsf: tuple | None = None,
    _feature_tfsf: bool = False,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    dft_planes: list | None = None,
    flux_monitors: list | None = None,
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    current_moments: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    gradient: str = "autodiff",
    checkpoint_segments: int | None = None,
    aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    aniso_inv_eps_smooth: bool = False,
    pec_mask: object | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_edge_masks: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    wire_refplane_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
    mag_sources: list | None = None,
    stencil_order: int = 2,
    report_every: int | None = None,
    report_label: str = "",
    sheet_impedance: object | None = None,
    design_box: DesignBoxSpec | None = None,
    design_occupancy: DesignOccupancySpec | None = None,
    stop_fn: Callable | None = None,
    stop_interval: int = 250,
    record_dft: bool = False,
    port_record_specs: tuple | None = None,
) -> SimResult:
    """Run a compiled FDTD simulation via ``jax.lax.scan``.

    Parameters
    ----------
    grid : Grid
    materials : MaterialArrays
    n_steps : int
    gradient : {"autodiff", "adjoint"}
        Adjoint requires a uniform real-valued design box. It stores local
        DFTs instead of checkpoints; checkpoint options are unused.
        See the module docstring of ``rfx.adjoint``.
    boundary : "pec", "cpml", or "upml"
    cpml_axes : str or None: the grid's absorbing axes when omitted; subsets accepted
    pec_axes : axes string or None
        Axes on which to enforce PEC after each update. If None, uses
        all non-periodic axes.
    periodic : (bool, bool, bool) or None
        Per-axis periodic boundary flags. If None, defaults to
        ``(False, False, False)`` in 3D and ``(False, False, True)``
        in 2D modes.
    debye : (DebyeCoeffs, DebyeState) tuple, or None
    lorentz : (LorentzCoeffs, LorentzState) tuple, or None
    tfsf : (TFSFConfig, TFSFState) tuple, or None
        Optional total-field/scattered-field plane-wave source. When
        provided, the 1D auxiliary FDTD is interleaved with the 3D Yee
        updates using the Taflove leapfrog ordering.
    sources : list of SourceSpec (precomputed waveforms)
    probes : list of ProbeSpec (point time-series recorders)
    dft_planes : list of DFTPlaneProbe or None
        Optional running frequency-domain plane probes.
    waveguide_ports : list of WaveguidePortConfig or None
        Optional waveguide-port source/probe configs.
    ntff : NTFFBox or None
        If provided, accumulate near-to-far-field DFT on a Huygens box.
    snapshot : SnapshotSpec or None
        If provided, record a frame after every ``snapshot.interval``-th
        step (``n_steps // interval`` frames; see ``SnapshotSpec``), with
        their positions and times in ``SimResult.snapshot_axes``. Recording
        does not change any other output: the time series, DFT and port
        accumulators and the final state are bit-identical to the same run
        without a snapshot. ``interval=1`` records inside the step body, as
        it always has; a larger interval runs the steps in blocks of
        ``interval`` (``lax.scan`` over blocks of an inner ``lax.scan`` over
        steps, plus a scan over the remaining ``n_steps % interval``) and
        records after each block, so the step body is the one a run without
        a snapshot compiles. The remainder scan is a second compile of the
        step body; a remainder of ONE step is instead joined with the last
        step of the last block into a two-step scan (a one-step scan is
        inlined by XLA and changed the last bit), leaving the rest of that
        block as a scan of ``interval - 1`` steps. With ``report_every`` a
        chunk that does not start on a multiple of the interval adds a short
        head scan, and each distinct piece length compiles once
        (``rfx.snapshots.plan_snapshot_pieces`` lays the pieces out). With
        ``checkpoint_segments`` the interval must divide the segment length
        ``n_steps // checkpoint_segments``.
    checkpoint : bool
        If True, wrap the unsegmented step body with ``jax.checkpoint`` to
        trade compute for memory during reverse-mode AD. With segments,
        each segment is rematerialised regardless of ``checkpoint``.
    return_state : bool
        If False, do not expose the final FDTD state in the result.
        This shrinks the differentiable output surface for optimisation.
    report_every : int or None
        When set, emit one ``  [PROGRESS] ...`` line to stdout
        every *N* steps: steps done / total, wall elapsed, implied rate and
        ETA. ``None`` (default) is OFF and runs the unchanged single-scan
        code path, so existing callers are byte-identical by construction.

        When set, the same compiled scan is driven from the host in
        ``report_every``-step chunks with ``carry`` threaded through, so the
        result is a continuation and not a re-solve; the DFT accumulators
        and every port / flux / monitor state live in that carry.
        Bit-exactness against ``report_every=None`` is locked by
        ``tests/unit/runners/test_run_progress_reporting.py``.

        Costs, measured rather than hidden: each report inserts one device
        synchronisation (without it the host would dispatch every chunk at
        once and print a fabricated rate), all full chunks share one XLA
        executable while a ragged final chunk compiles once more, and the
        per-chunk outputs are joined once at the end with a bounded-arity
        grouped concatenate (a flat one takes an argument per chunk, which
        cost 232 s at 22,500 chunks against 0.99 s grouped).

        Forward-only: it reads the host wall clock, so it raises under
        ``jax.jit``/``grad``/``vmap`` and is rejected together with
        ``checkpoint_segments``. The trace check inspects the carry, the
        per-step ``xs`` and the material / geometry arrays — the routes
        ``forward()`` and ``optimize()`` actually trace through — rather
        than asking JAX whether a trace is active, which its current API
        does not expose. A tracer reaching the scan body by some other
        closure alone would print trace-time lines; the computed values
        stay byte-identical either way.
    report_label : str
        Short tag prefixed to each progress line, e.g. ``"MSL drive p1"``,
        so the per-drive solves of one ``compute_*_s_matrix`` call are
        distinguishable in a log. Ignored when ``report_every`` is None.
    design_box : DesignBoxSpec or None
        One static box whose E update is redone from its own
        (usually traced) permittivity, leaving ``materials`` constant so the
        reverse-mode tape holds box-shaped arrays instead of grid-shaped
        ones. ``None`` (default) is the unchanged path.

        ``resolve_design_box`` rejects, from what this function resolves:
        UPML, Debye/Lorentz, anisotropic eps, Kerr, ``stencil_order=4``, the
        Bloch path, a box reaching into the CPML absorber, and a box holding
        a source, magnetic source, lumped/wire S-param port or RLC cell or a
        surface-impedance sheet edge. It does NOT carry the checks that need
        the API object: the lane (non-uniform, distributed, ADI), the
        collision with ``eps_override`` / ``sigma_override`` /
        ``mu_r_override``, ``pec_occupancy_override``, a PASSIVE port (it
        leaves neither a source nor an accumulator here), and the promotion
        of ``materials.eps_r`` to the design dtype. Those are
        ``Simulation.forward``'s, in ``_resolve_design_box_override``; a
        caller reaching this function directly owns them.
    design_occupancy : DesignOccupancySpec or None
        One static box whose PEC OCCUPANCY is the traced
        quantity, for the same reason and with the same effect on the tape:
        the ``1 - M`` scaling is redone on the box's window from the field
        before the grid-wide ``apply_pec_occupancy``, so what the backward
        pass keeps per step is window-shaped. ``None`` (default) is the
        unchanged path. The box's occupancy REPLACES ``pec_occupancy`` at
        the box cells; outside it, ``pec_occupancy`` is carried.

        ``_resolve_design_occupancy`` rejects a periodic axis, an empty box
        and a box off the grid. ``Simulation.forward`` owns the lane and the
        collision with ``design_eps_override``.
    stop_fn : callable or None
        Used by ``Simulation.run(..., until_identified=True)``. When
        given, the scan runs in chunks of ``stop_interval`` steps through
        :func:`rfx.progress.scan_with_progress`, the carry threaded through,
        and ``stop_fn(steps_done, peek)`` is called after each chunk;
        ``peek()`` returns the record so far as a namespace with
        ``time_series`` (``(steps_done, n_probes)``), ``wire_port_sparams``
        (the ``(spec, accumulators)`` pairs at that step, as this function
        returns them), ``grid`` and ``dt``. A true return ends the run
        there, and every output is the one a run of ``steps_done`` steps
        returns (the chunked scan is a continuation of the single one).
        Refused with a snapshot or ``checkpoint_segments``. ``None``
        (default) is the unchanged path.
    stop_interval : int
        Chunk length of the ``stop_fn`` loop (default 250). Ignored when
        ``stop_fn`` is None.

    Returns
    -------
    SimResult with final state, time series, and optional NTFF data.
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []
    wire_port_sparams = wire_port_sparams or []
    lumped_port_sparams = lumped_port_sparams or []
    wire_refplane_sparams = wire_refplane_sparams or []
    lumped_rlc = lumped_rlc or []
    mag_sources = mag_sources or []

    if gradient not in ("autodiff", "adjoint"):
        raise ValueError("gradient must be 'autodiff' or 'adjoint'")

    # ---- shared setup (W6.2) ----
    _setup = _build_step_setup(
        grid=grid,
        materials=materials,
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        debye=debye,
        lorentz=lorentz,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        sources=sources,
        probes=probes,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_ports=waveguide_ports,
        ntff=ntff,
        current_moments=current_moments,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        wire_refplane_sparams=wire_refplane_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
        design_occupancy=design_occupancy,
    )
    carry_init = _setup.carry_init
    if record_dft and dft_planes:
        carry_init["dft_time_records"] = tuple(
            jnp.zeros((n_steps,) + plane.accumulator.shape[1:], dtype=carry_init["fdtd"].ex.dtype)
            for plane in dft_planes)
    init_uniform_records(carry_init, n_steps, wire_port_sparams, lumped_port_sparams, wire_refplane_sparams, field_dtype, port_record_specs)
    dt = _setup.dt
    dx = _setup.dx
    periodic = _setup.periodic
    waveguide_meta = _setup.waveguide_meta
    wire_sparam_meta = _setup.wire_sparam_meta
    lumped_sparam_meta = _setup.lumped_sparam_meta
    wire_refplane_meta = _setup.wire_refplane_meta
    flux_meta = _setup.flux_meta_11   # 11-field: run() uses streaming DFT window
    use_snapshot = snapshot is not None
    use_flux_monitors = len(flux_monitors) > 0
    use_wire_sparams = len(wire_port_sparams) > 0
    use_wire_refplanes = len(wire_refplane_sparams) > 0
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0

    # ---- run()-specific: fast-path pre-baked coefficients ----
    # Eligible when the scan body only needs H update + E update + PEC —
    # no CPML, no TFSF, no dispersion, no anisotropic eps, no PEC mask,
    # no conformal, no lumped RLC, no Kerr, and no periodic axes.
    #
    # The fast path bakes PEC boundary enforcement into the update
    # coefficients (zero ca/cb at boundary cells), eliminating 12
    # per-step scatter operations.  This is always beneficial on GPU
    # (scatter ops launch separate kernels) and beneficial on CPU for
    # grids above ~500K cells where the scatter ops cause cache eviction.
    _on_gpu = backend_class().accelerator
    _ctx = _setup.ctx_kwargs
    _fast_eligible = (
        not _ctx["use_cpml"]
        and not _ctx["use_upml"]
        # the baked step applies no per-face masks after its H update, so a
        # magnetic wall would be neither electric nor magnetic there (#1164)
        and not _ctx["use_pmc_faces"]
        # the baked step has no slot for a TFSF or a waveguide-port correction
        and not (_ctx["use_tfsf"] or _ctx["use_waveguide_ports"])
        and not _ctx["use_debye"]
        and not _ctx["use_lorentz"]
        and not _ctx["use_pec_edges"]
        and not _ctx["use_pec_occupancy"]
        and not _ctx["use_conformal"]
        and not _ctx["use_lumped_rlc"]
        and not _ctx["use_kerr"]
        and not mag_sources
        # #677: the GPU baked fast path has an inline H+E update with no
        # sheet-operator slot, so a surface-impedance sheet is DELIBERATELY
        # unsupported here: an f0 sheet takes the standard path, which costs
        # performance only -- no accuracy loss, nothing silently wrong.
        # Per-plane baking is a parked capability, not a promised follow-up;
        # it waits on the thin-sheet plane-BC architecture decision (#701),
        # and the parking itself is backlog #787.
        and not _ctx["use_sheet_impedance"]
        # #1179: same reason as the sheet operator above -- the inline H+E
        # update has no slot for the design-box redo, and its coefficients
        # are baked from ``materials``, which is exactly the array a design
        # box stops carrying. A design box takes the standard path; that
        # costs GPU scatter kernels, nothing else.
        and not _ctx["use_design_box"]
        # #1183: and the same for the design occupancy. The fast path has
        # no occupancy slot at all -- it would drop the design variable and
        # return an all-zero gradient, which reads like a converged design.
        and not _ctx["use_design_occupancy"]
        # Either permittivity tensor (Stage 1, or the inverse one that
        # ``subpixel_smoothing="kottke_pec"`` and the occupancy tensor build):
        # the baked coefficients come from the scalar realized permittivity,
        # so a tensor would be dropped (kottke_pec returned the unsmoothed
        # field bit for bit on this path).
        and not _ctx["use_aniso_inv"]
        and (_ctx["materials"].components.tensor is None or _ctx["materials"].components.tensor.eps is None)
        and periodic == (False, False, False)
    )
    # On GPU the baked-PEC path eliminates expensive scatter-update
    # kernel launches — always beneficial.  On CPU, XLA fuses scatter
    # ops efficiently so the extra coefficient arrays hurt at small-to-
    # medium grids; enable only when explicitly requested via GPU backend.
    # The GPU fast path has an INLINE 2nd-order stencil (update_he_fast),
    # so it must never be used for stencil_order=4 — that would silently
    # produce a 2nd-order result. Gate it on order==2.
    use_fast_he = _fast_eligible and _on_gpu and stencil_order == 2
    _fast_coeffs = (
        precompute_coeffs(_setup.ctx_kwargs["materials"], dt, dx, pec_faces=_setup.pec_faces,
                          periodic=periodic)
        if use_fast_he else None
    )

    # ---- run()-specific: snapshot setup ----
    # #1258: ``interval`` is honoured. interval == 1 keeps the pre-#1258 path
    # (the frame is an extra per-step output of the step body). interval > 1
    # records from the carry after each block of ``interval`` steps, so the
    # step body is the SAME one a run without a snapshot compiles and memory
    # scales with the frame count, not with n_steps.
    if use_snapshot:
        snap_interval = validate_snapshot_spec(snapshot)
        snap_components = tuple(snapshot.components)
        _take_snapshot = snapshot_extractor(snapshot)
    else:
        snap_interval = 1
        snap_components = ()
        _take_snapshot = None
    snap_in_body = use_snapshot and snap_interval == 1
    snap_by_block = use_snapshot and snap_interval > 1

    if _realized.ACTIVE is not None:
        _realized.sources(grid, materials, sources, "uniform.sources")
    # ---- precompute source waveform matrix (n_steps, n_sources) ----
    src_waveforms = drive_table([s.waveform for s in sources], _setup.src_meta, n_steps,
        occupancy=pec_occupancy, periodic=periodic,
        sheet_edge_masks=_ctx["pec_static_edge_masks"],
        design=design_occupancy, shape=grid.shape,
        keep=_ctx["materials"].edge_keep)

    # ---- magnetic source (H-field) waveform matrix ----
    if mag_sources:
        mag_src_waveforms = jnp.stack([s.waveform for s in mag_sources], axis=-1)
    else:
        mag_src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # ---- scan body (shared kernel; W6.1 kernel + W6.2 setup) ----
    _step_ctx = _StepContext(
        port_record_specs=port_record_specs,
        **_setup.ctx_kwargs,
        drives=StepDrives(
            drive_layout(_setup.src_meta, src_waveforms.dtype),
            drive_layout(_setup.mag_src_meta, mag_src_waveforms.dtype)),
        # run()-specific overrides
        use_fast_he=use_fast_he,
        use_snapshot=snap_in_body,
        use_monitor=False,
        use_flux_window=True,
        fast_coeffs=_fast_coeffs,
        flux_meta=flux_meta if use_flux_monitors else (),
        monitor_component="ez",
        mon_idx=None,
        snapshot_extractor=_take_snapshot if snap_in_body else None,
    )
    _core_step = make_core_step(_step_ctx)

    def step_fn(carry, xs):
        _step_idx, src_vals, mag_src_vals = xs
        new_carry, probe_out, extras = _core_step(
            carry, _step_idx, src_vals, mag_src_vals)
        if snap_in_body:
            output = (probe_out, extras["snap_fields"])
        else:
            output = (probe_out,)
        return new_carry, output

    def _make_recorder(body):
        """``record(carry, xs_seg, lo)``: scan ``xs_seg`` (global steps
        ``lo, lo+1, ...``) with ``body`` and take a frame after every step
        whose completed-step count is a multiple of ``snap_interval`` (#1258).

        The steps run as the pieces :func:`rfx.snapshots.plan_snapshot_pieces`
        lays out, in step order:

        * ``plain`` -- a scan of ``body``, with the frame read from the carry
          when the piece ends on a multiple;
        * ``blocks`` -- a scan over blocks of ``snap_interval`` steps whose
          body is an inner scan of ``body`` followed by the frame read from
          the carry;
        * ``rec`` -- a scan of ``body`` over two or three steps that also
          outputs each step's frame, of which the rows ending on a multiple
          are kept.

        A one-step plain piece never runs on its own (XLA inlines a loop that
        runs once, and the inlined step moved the last bit of Ex/Ey on a PEC
        box): the planner joins it with ONE step of a neighbour into a
        ``rec`` piece, so at most three frames exist before the unwanted ones
        are dropped, whatever the interval. Every piece calls the same
        ``body`` on the same ``xs`` rows in the same order, inside a loop of
        at least two steps, so the carry is the one a single scan produces;
        the probe rows are joined back in step order. Returns
        ``(carry, (probe_rows, [frames per component]))``, the output layout
        of the ``interval == 1`` path. A segment of one step is run as the
        one-step scan the unchunked path would also run.

        The block and recording bodies are built once per ``body`` so
        repeated calls (one per ``report_every`` chunk) reuse their traces
        and compiled executables.
        """
        m = snap_interval

        def block_body(c, block_xs):
            c, (p,) = jax.lax.scan(body, c, block_xs)
            return c, (p, _take_snapshot(c["fdtd"]))

        def rec_body(c, x):
            c, (p,) = body(c, x)
            return c, (p, _take_snapshot(c["fdtd"]))

        def record(carry, xs_seg, lo):
            n = int(jax.tree_util.tree_leaves(xs_seg)[0].shape[0])
            probe_parts, frame_parts = [], []
            for piece in plan_snapshot_pieces(n, lo, m):
                a, b = piece.start, piece.start + piece.length
                rows = jax.tree_util.tree_map(lambda x: x[a:b], xs_seg)
                if piece.kind == "plain":
                    carry, (p,) = recorded_scan(body, carry, rows)
                    probe_parts.append(p)
                    if piece.frame_rows:
                        frame_parts.append(
                            [f[None] for f in _take_snapshot(carry["fdtd"])])
                elif piece.kind == "blocks":
                    nb = piece.length // m
                    blocks = jax.tree_util.tree_map(
                        lambda x: x.reshape(nb, m, *x.shape[1:]), rows)
                    carry, (p, f) = recorded_scan(block_body, carry, blocks)
                    probe_parts.append(p.reshape(nb * m, *p.shape[2:]))
                    frame_parts.append(list(f))
                else:  # "rec"
                    carry, (p, f) = recorded_scan(rec_body, carry, rows)
                    probe_parts.append(p)
                    if piece.frame_rows:
                        idx = jnp.asarray(piece.frame_rows, dtype=jnp.int32)
                        frame_parts.append([ff[idx] for ff in f])

            if not probe_parts:
                # n == 0: the zero-length scan the other paths return.
                carry, (p,) = recorded_scan(body, carry, xs_seg)
                probe_parts.append(p)
            probes = (probe_parts[0] if len(probe_parts) == 1
                      else jnp.concatenate(probe_parts, axis=0))
            if frame_parts:
                frames = [
                    parts[0] if len(parts) == 1
                    else jnp.concatenate(parts, axis=0)
                    for parts in zip(*frame_parts)]
            else:
                frames = [
                    jnp.zeros((0,) + s.shape, s.dtype)
                    for s in jax.eval_shape(_take_snapshot, carry["fdtd"])]
            return carry, (probes, frames)

        return record

    # ---- run ----
    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms, mag_src_waveforms)

    if stop_fn is not None and (use_snapshot or checkpoint_segments is not None):
        raise NotImplementedError(
            "run(stop_fn=...) ends the record at a step chosen during the run; "
            "a snapshot's frame axes and checkpoint_segments' segment lengths "
            "are fixed from n_steps before it (issue #1254).")

    if gradient == "adjoint":
        if snapshot is not None or stop_fn is not None or report_every is not None:
            raise NotImplementedError(
                "gradient='adjoint' does not support snapshots, stopping or progress")
        from rfx.adjoint import design_adjoint_scan
        final_carry, outputs = design_adjoint_scan(_step_ctx, carry_init, xs)
    elif checkpoint_segments is None:
        # Legacy path: optional per-step rematerialisation only. The scan
        # itself still keeps every step's carry, so peak memory grows
        # linearly with n_steps.
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        if stop_fn is not None:
            # Issue #1254: the same scan in chunks, the carry threaded
            # through, and the caller's stop check after each chunk.
            from types import SimpleNamespace

            from rfx.progress import concat_chunks

            def _after_chunk(done, carry, chunk_outputs):
                def peek():
                    (series,) = concat_chunks(chunk_outputs)
                    wps = (tuple(zip(wire_sparam_meta, carry["wire_sparam_accs"]))
                           if use_wire_sparams else None)
                    return SimpleNamespace(time_series=series,
                                           wire_port_sparams=wps,
                                           grid=grid, dt=dt)
                return bool(stop_fn(int(done), peek))

            final_carry, outputs = scan_with_progress(
                body, carry_init, xs,
                n_steps=n_steps,
                report_every=report_every,
                label=report_label,
                trace_probes=(_step_ctx.materials,
                              pec_mask, pec_edge_masks, pec_occupancy,
                              kerr_chi3, debye, lorentz, tfsf),
                stop_fn=_after_chunk,
                chunk=int(stop_interval),
            )
        elif report_every is None and snap_by_block:
            final_carry, outputs = _make_recorder(body)(carry_init, xs, 0)
        elif report_every is None:
            final_carry, outputs = recorded_scan(body, carry_init, xs)
        else:
            # Issue #667: same scan, driven from the host in chunks so a
            # multi-hour solve emits progress. The carry threads through
            # untouched (DFT accumulators, port/flux/monitor state, NTFF
            # compensation), so this is a continuation and not a re-solve.
            # The ``report_every is None`` branch above is byte-identical to
            # the pre-#667 code by construction.
            final_carry, outputs = scan_with_progress(
                body, carry_init, xs,
                n_steps=n_steps,
                report_every=report_every,
                label=report_label,
                # A tracer from forward()/optimize() arrives through the
                # material and geometry arrays, which the scan body captures
                # in its closure — carry_init and xs stay concrete under bare
                # grad/vmap, so checking only those would let the request
                # through and print trace-time lines.
                trace_probes=(_step_ctx.materials,
                              pec_mask, pec_edge_masks, pec_occupancy,
                              kerr_chi3, debye, lorentz, tfsf),
                # #1258: a chunk that does not start on a multiple of the
                # snapshot interval still records at the global multiples.
                chunk_scan=_make_recorder(body) if snap_by_block else None,
            )
    else:
        if report_every is not None:
            raise NotImplementedError(
                "report_every is not supported together with "
                "checkpoint_segments: segmented checkpointing is itself a "
                "scan-of-scans whose segment boundaries pin the AD "
                "rematerialisation points, and host-side progress chunking "
                "would have to cut the same axis. Progress reporting is a "
                "forward-only feature — drop checkpoint_segments for a "
                "monitored forward solve, or drop report_every for the "
                "memory-limited gradient solve."
            )
        # Segmented checkpointing (issue #73): split the n_steps scan into
        # K segments of size s and rematerialise each segment as a unit.
        # Forward keeps only K segment-boundary carries (instead of all
        # n_steps), so peak memory drops from O(n_steps · |carry|) to
        # O((K + s) · |carry|). With s ≈ √n_steps this is the standard
        # √n_steps-memory trade-off (≈2× compute for backward).
        #
        # Implementation notes:
        #   * `prevent_cse=False` on the segment-level checkpoint is
        #     required so the inner scan is not CSE-deduplicated across
        #     the outer scan (which would defeat the memory saving).
        #   * `n_steps` must be divisible by K. We require this rather
        #     than padding because the carry holds DFT accumulators
        #     (lumped/wire/waveguide port S-params) that would integrate
        #     over the padded zero-source steps and produce numerically
        #     different results from an unsegmented run with the same
        #     n_steps. Picking K as a divisor of n_steps preserves
        #     bit-exact equivalence for forward and gradient.
        K = int(checkpoint_segments)
        if K < 1:
            raise ValueError(
                f"checkpoint_segments must be ≥ 1, got {checkpoint_segments}")
        if n_steps % K != 0:
            raise ValueError(
                f"checkpoint_segments={K} does not divide n_steps={n_steps}; "
                f"pick a divisor (e.g. K={_nearest_divisor(n_steps, K)} for "
                f"≈ √n_steps memory). Padding is intentionally rejected "
                f"because it would shift DFT accumulator integration windows."
            )
        s = n_steps // K
        if snap_by_block and s % snap_interval != 0:
            raise ValueError(
                f"snapshot interval={snap_interval} does not divide the "
                f"checkpoint segment length n_steps/checkpoint_segments="
                f"{n_steps}/{K}={s}: every segment must record the same "
                f"number of frames. Pick an interval that divides {s}, or a "
                f"checkpoint_segments whose segment length is a multiple of "
                f"{snap_interval}.")
        xs_segmented = jax.tree_util.tree_map(
            lambda a: a.reshape(K, s, *a.shape[1:]), xs)

        _record_segment = _make_recorder(step_fn) if snap_by_block else None

        def segment_body(carry, seg_xs):
            if snap_by_block:
                # Each segment starts on a multiple of s, hence of the
                # interval, so ``lo=0`` places its frames correctly.
                return _record_segment(carry, seg_xs, 0)
            new_carry, seg_outputs = jax.lax.scan(step_fn, carry, seg_xs)
            return new_carry, seg_outputs

        seg_body_ckpt = jax.checkpoint(segment_body, prevent_cse=False)
        final_carry, seg_outputs = recorded_scan(
            seg_body_ckpt, carry_init, xs_segmented)
        # seg_outputs leaves: (K, per-segment rows, ...). Flatten back to
        # (K * rows, ...): n_steps for the probe rows (and for per-step
        # frames at interval 1), n_steps // interval for block frames.
        outputs = jax.tree_util.tree_map(
            lambda a: a.reshape(a.shape[0] * a.shape[1], *a.shape[2:]),
            seg_outputs)

    if use_snapshot:
        time_series = outputs[0]
        # outputs[1] is a list of arrays, each (n_steps // interval, ...)
        snapshots = {comp: outputs[1][i]
                     for i, comp in enumerate(snap_components)}
        snap_axes = snapshot_axes(grid, snapshot, n_steps, dt=dt)
        for comp in snap_components:
            n_frames = int(snapshots[comp].shape[0])
            if n_frames != len(snap_axes[comp].steps):
                raise RuntimeError(
                    f"snapshot {comp!r} recorded {n_frames} frames, expected "
                    f"{len(snap_axes[comp].steps)} for n_steps={n_steps}, "
                    f"interval={snap_interval} (#1258)")
    else:
        time_series = outputs[0]
        snapshots = None
        snap_axes = None

    final_dft_planes = None
    if use_dft_planes:
        final_dft_planes = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final_carry["dft_planes"])
        )

    final_flux_monitors = None
    if use_flux_monitors:
        final_flux_monitors = tuple(
            fm._replace(e1_dft=accs[0], e2_dft=accs[1], h1_dft=accs[2], h2_dft=accs[3])
            for fm, accs in zip(flux_monitors, final_carry["flux_monitors"])
        )

    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            cfg_meta._replace(
                # Stamp the scan's authoritative dt so the post-scan rect-DFT
                # extractor uses the right Δt. The jit-safe in-scan probe
                # accumulator cannot run update_waveguide_port_probe's
                # ``float(dt)`` stamp, so a cfg built via init_waveguide_port
                # without dt= would otherwise keep dt=0 and silently zero every
                # extracted spectrum (S→0). The manual Python-loop path already
                # stamps dt via update_waveguide_port_probe; this makes the
                # compiled run() symmetric with it. The scan's own step, not
                # grid.dt: stencil_order=4 derates it.
                dt=float(dt),
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(waveguide_meta, final_carry["waveguide_port_accs"])
        )

    from rfx.measurement.port_records import pack_port_records
    final_wire_sparams, final_lumped_sparams = pack_port_records(
        ('wire',) * len(wire_sparam_meta) + ('lumped',) * len(lumped_sparam_meta),
        final_carry.get("wire_sparam_accs", ()) + final_carry.get("lumped_sparam_accs", ()),
        wire_sparam_meta + lumped_sparam_meta)

    final_wire_refplanes = None
    if use_wire_refplanes:
        final_wire_refplanes = tuple(
            (rp_meta, accs)
            for rp_meta, accs in zip(wire_refplane_meta,
                                     final_carry["wire_refplane_accs"])
        )

    return SimResult(
        state=final_carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=final_carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        sparam_time_records=uniform_records(final_carry, _step_ctx, int(final_carry["fdtd"].step) if stop_fn is not None else None),
        dft_time_records=tuple(r[:int(final_carry["fdtd"].step)] if stop_fn is not None else r
                                for r in final_carry.get("dft_time_records", ())),
        snapshots=snapshots,
        ntff_box=ntff,
        grid=grid,
        wire_refplane_sparams=final_wire_refplanes,
        snapshot_axes=snap_axes,
        dt=dt,
        current_moment_data=final_carry.get("current_moments"),
        current_moment_monitor=current_moments,
        adjoint_settling=final_carry.get("adjoint_settling"),
    )


def run_until_decay(
    grid: Grid,
    materials: MaterialArrays,
    *,
    decay_by: float = 1e-3,
    n_end: int | None = AUTO_END,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    monitor_component: str = "ez",
    monitor_position: tuple[float, float, float] | None = None,
    boundary: str = "pec",
    cpml_axes: str | None = None,
    pec_axes: str | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    tfsf: tuple | None = None,
    _feature_tfsf: bool = False,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    dft_planes: list | None = None,
    flux_monitors: list | None = None,
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    current_moments: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    aniso_inv_eps_smooth: bool = False,
    pec_mask: object | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_edge_masks: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
    mag_sources: list | None = None,
    checkpoint_segments: int | None = None,
    stencil_order: int = 2,
    report_every: int | None = None,
    report_label: str = "",
    sheet_impedance: object | None = None,
    record_dft: bool = False,
    port_record_specs: tuple | None = None,
    advisory: dict | None = None,
) -> SimResult:
    """Run until the field energy left after the drive has decayed.

    Compiled scan chunks end at completed steps ``k * check_interval``; every
    stop check runs on the host at those boundaries. On a high-Q structure
    whose mode reaches the absorber, the stop can move by about 170 steps
    with the check phase while the spectra stay within 1e-2 of the band peak.

    Parameters
    ----------
    decay_by : float
        Absorbing boundaries: stop once the conserved interior energy is below
        ``decay_by`` times its reference (``rfx.measurement.decay``).
    n_end : int or None
        Step at which every drive has ended (None: still on at ``max_steps``,
        no energy stop). Default: computed from this call's drive tables.
    advisory : dict or None
        ``rfx.measurement.decay.advisory_inputs`` (soft drives' DC, lossy
        materials) for the cap advisory; None: this call's tables and arrays.
    check_interval : int
        Check decay every N steps.
    min_steps : int
        Always run at least this many steps.
    max_steps : int
        Hard upper limit on steps.
    decay_energy_consecutive : int
        Absorbing boundaries: consecutive sub-threshold checks before the
        energy stop. Values below 2 are refused (a single check stops on a
        transient dip). No effect on closed/PEC boundaries.
    radiated_flux_box : tuple or None
        Opt-in RADIATED-FLUX stop for absorbing boundaries: a Huygens box
        ``((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))`` in physical coordinates
        enclosing the structure (clear of the CPML). When set, the stop uses the
        outgoing Poynting flux through the box instead of the interior energy —
        a *radiation-settling* criterion that ignores non-radiating trapped
        energy (the static soft-source charge that FLOORS the energy criterion,
        and near-Nyquist grid buzz), because neither carries net radiated power.
        Appropriate for radiation / S-parameter measurements; the flux peak is
        tracked from the first check (the radiation peaks during the source
        drive). Default ``None`` keeps the interior-energy criterion, unchanged.
    flux_env_checks : int
        Number of recent checks whose ``|flux|`` max forms the envelope used by
        the radiated-flux stop, smoothing the 2f0 Poynting oscillation. Default 4.
    monitor_component : str
        Field component to monitor ("ez", "hy", etc.). Used only by the
        closed/PEC point-field fallback stop.
    monitor_position : tuple or None
        Physical position (x, y, z) to monitor. If None, use center of
        the domain.
    checkpoint_segments : int or None
        **Not supported** (a host-driven early stop); a non-None value
        raises ``NotImplementedError``.

    Notes
    -----
    **Differences from** :func:`run`:

    * ``checkpoint_segments`` is not supported (raises ``NotImplementedError``
      when not ``None``).
    * ``snapshot`` records a frame every ``snapshot.interval`` steps, as
      :func:`run` does; axes in ``SimResult.snapshot_axes``.
    * ``checkpoint`` (``jax.checkpoint`` gradient tape) is accepted and has
      no effect on the host-driven decay path.
    * Flux-monitor DFT accumulation uses a rectangular (no-window) weight
      instead of the streaming Hann window used by :func:`run`.  Flux values
      from the two paths are therefore not numerically identical even for
      the same step count.

    .. note::

       **The stop criterion depends on the boundary.**

       * **Absorbing boundaries (``cpml`` / ``upml``)** — the stop reads the
         leapfrog-conserved interior energy
         ``W = sum((eps E^n.E^n + (mu0/eps0) mu H^(n-1/2).H^(n+1/2)) dV)``
         at checks from ``n_end`` on, against the running maximum of W over
         those checks: the drive's near field never enters the reference.
         It stops after ``decay_energy_consecutive`` consecutive checks with
         ``W < max(decay_by * reference, KAPPA * Wmax)``, ``KAPPA = 1e-12``
         and Wmax the largest W at any check (drive included): a threshold
         below float32 resolution ends on that floor and the result says so
         (``rfx.measurement.decay.KAPPA``; ``SimResult.decay_stop``). In a
         lossy dielectric the run can last longer than the field's ringing:
         charge moved onto the dielectric's surfaces during the drive relaxes
         with the time constant eps/sigma after it. That field is quasi-static
         (out of band), the record stays correct, and the stop record and the
         cap advisory say what ended the run.

       * **Closed / PEC boundaries** — domain energy does not decay in a
         lossless closed cavity, so the stop falls back to the historical
         *instantaneous* squared field at the single ``monitor_component`` cell
         (``val_sq < decay_by * peak_sq``). This fallback retains the original
         limitation: it is a valid decay witness only for lossy / radiating
         structures with a clean ring-down envelope, and is **not** suitable
         for flux / S-parameter / transmission gating on guided / low-loss
         closed geometries — for those use a fixed ``n_steps`` via
         :func:`run`.
    report_every : int or None
        When set, emit one ``  [PROGRESS] ...`` line every *N*
        steps, at the first decay-chunk boundary at or past the interval,
        plus a final line at the actual stop step. ``None`` (default) is OFF.
        Reporting never changes the chunks or the decay-check cadence.
        The denominator is ``max_steps``, i.e. a CAP:
        the line marks it ``(cap)`` and the ETA is an upper bound, because a
        decay stop can fire at any check.
    report_label : str
        Short tag prefixed to each progress line. Ignored when
        ``report_every`` is None.

    Returns
    -------
    SimResult
    """
    cpml_axes = resolve_cpml_axes(grid, cpml_axes)
    if checkpoint_segments is not None:
        raise NotImplementedError(
            "checkpoint_segments is not supported by run_until_decay: "
            "this function uses a host-driven early stop, so "
            "scan-level gradient checkpointing does not apply. "
            "Use run() with checkpoint_segments if you need scan-level "
            "checkpointing."
        )
    if check_interval < 1:
        raise ValueError(f"check_interval must be >= 1, got {check_interval}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []
    wire_port_sparams = wire_port_sparams or []
    lumped_port_sparams = lumped_port_sparams or []
    lumped_rlc = lumped_rlc or []
    mag_sources = mag_sources or []

    # ---- shared setup (W6.2) ----
    _setup = _build_step_setup(
        grid=grid,
        materials=materials,
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        debye=debye,
        lorentz=lorentz,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        sources=sources,
        probes=probes,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_ports=waveguide_ports,
        ntff=ntff,
        current_moments=current_moments,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
        sheet_impedance=sheet_impedance,
    )
    carry = _setup.carry_init
    if record_dft and dft_planes:
        carry["dft_time_records"] = tuple(
            jnp.zeros((max_steps,) + plane.accumulator.shape[1:], dtype=carry["fdtd"].ex.dtype)
            for plane in dft_planes)
    init_uniform_records(carry, max_steps, wire_port_sparams, lumped_port_sparams, (), field_dtype, port_record_specs)
    dx = _setup.dx
    waveguide_meta = _setup.waveguide_meta
    wire_sparam_meta = _setup.wire_sparam_meta
    lumped_sparam_meta = _setup.lumped_sparam_meta
    flux_meta_decay = _setup.flux_meta_8   # 8-field: no streaming DFT window
    use_flux_monitors = len(flux_monitors) > 0
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0

    # ---- monitor position ----
    if monitor_position is None:
        # Center of the physical domain
        cx = (grid.nx - 1) * dx / 2.0
        cy = (grid.ny - 1) * dx / 2.0
        cz = 0.0 if grid.is_2d else (grid.nz - 1) * dx / 2.0
        monitor_position = (cx, cy, cz)
    mon_idx = _grid_metric.field_index(grid, monitor_position, monitor_component)

    # ---- precompute source waveforms up to max_steps ----
    src_waveforms = drive_table([s.waveform for s in sources], _setup.src_meta, max_steps,
        occupancy=pec_occupancy, periodic=_setup.periodic, pad=True,
        sheet_edge_masks=_setup.ctx_kwargs["pec_static_edge_masks"], keep=_setup.ctx_kwargs["materials"].edge_keep)

    if mag_sources:
        mag_src_waveforms = jnp.stack(
            [s.waveform[:max_steps] if s.waveform.shape[0] >= max_steps
             else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
             for s in mag_sources], axis=-1)
    else:
        mag_src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)

    # ---- shared kernel (W6.1 + W6.2 setup) ----
    # run_until_decay does NOT build the GPU fast-HE coeffs and uses the
    # historical rect (no-window) flux DFT: use_flux_window=False keeps the
    # 8-field flux_meta and skips the streaming window weight so decay-path
    # flux values stay bit-identical to the pre-refactor body. Unifying the
    # decay path onto windows is a deliberate follow-up.
    _step_ctx = _StepContext(
        port_record_specs=port_record_specs,
        **_setup.ctx_kwargs,
        drives=StepDrives(
            drive_layout(_setup.src_meta, src_waveforms.dtype),
            drive_layout(_setup.mag_src_meta, mag_src_waveforms.dtype)),
        # decay-path-specific overrides
        use_fast_he=False,
        use_snapshot=False,
        use_monitor=True,
        use_flux_window=False,
        fast_coeffs=None,
        flux_meta=flux_meta_decay if use_flux_monitors else (),
        monitor_component=monitor_component,
        mon_idx=mon_idx,
        snapshot_extractor=None,
    )
    # The per-cell arrays are an argument of the jitted step, not constants
    # compiled into it (split_loop_invariants).
    from rfx.core.jax_utils import split_loop_invariants
    _inv_args, _inv_rebuild = split_loop_invariants(
        core_step_invariants(_step_ctx), tuple(grid.shape))

    @jax.jit
    def _single_step(carry_in, step_idx, src_vals, mag_src_vals, inv_args):
        _inv = _inv_rebuild(inv_args)
        _core_step = make_core_step(_inv["ctx"], invariants=_inv)
        new_carry, probe_out, extras = _core_step(
            carry_in, step_idx, src_vals, mag_src_vals)
        return new_carry, probe_out, extras["monitor_val"]

    # #1258: keep only frames at GLOBAL multiples of the interval, even
    # when an interval does not divide the decay/progress chunk. The scan
    # carries a bounded frame buffer, not a full field history.
    if snapshot is not None:
        snap_interval = validate_snapshot_spec(snapshot)
        _take_snapshot = snapshot_extractor(snapshot)
        snap_shapes = jax.eval_shape(_take_snapshot, carry["fdtd"])
        snap_chunks = []

    def step_fn(carry_in, xs, inv_args=_inv_args):
        step_idx, src_vals, mag_src_vals = xs
        new_carry, probe_out, monitor_val = _single_step(
            carry_in, step_idx, src_vals, mag_src_vals, inv_args)
        return new_carry, (probe_out, monitor_val)

    def step_with_previous(carries, xs):
        # A one-step tail is joined to the preceding scan, as in
        # scan_with_progress. Keep its penultimate carry so the check at
        # that step can still stop BEFORE the cap. No padded/masked steps.
        state, _previous = carries
        state_out, out = step_fn(state, xs)
        return (state_out, state), out

    def _make_chunk_runner(*, keep_previous=False):
        body = step_with_previous if keep_previous else step_fn

        def record(c, x):
            state, frames, first_frame = c
            state, out = body(state, x)
            done = x[0] + 1

            def take(frames):
                row = done // snap_interval - first_frame - 1
                fdtd = (state[0] if keep_previous else state)["fdtd"]
                return tuple(buf.at[row].set(frame) for buf, frame in
                             zip(frames, _take_snapshot(fdtd)))

            frames = jax.lax.cond(done % snap_interval == 0, take,
                                  lambda f: f, frames)
            return (state, frames, first_frame), out

        def run_chunk(carry_in, xs):
            if keep_previous:
                carry_in = (carry_in, carry_in)
            if snapshot is None:
                return jax.lax.scan(body, carry_in, xs), None
            # At most ceil(chunk_length / interval) frames, independent of
            # the starting phase. Keep record's identity across chunks so
            # lax.scan reuses its executable, as run()'s recorder does.
            n_frames = (xs[0].shape[0] + snap_interval - 1) // snap_interval
            frames = tuple(jnp.zeros((n_frames,) + f.shape, f.dtype)
                           for f in snap_shapes)
            first_frame = xs[0][0] // snap_interval
            (carry_out, frames, _), out = jax.lax.scan(
                record, (carry_in, frames, first_frame), xs)
            return (carry_out, out), frames

        return run_chunk

    _run_chunk = _make_chunk_runner()
    _run_tail_chunk = _make_chunk_runner(keep_previous=True)

    # ---- host chunk loop with decay check ----
    use_absorbing = boundary in ("cpml", "upml")

    # #388 opt-in RADIATED-FLUX stop: stop when the outgoing Poynting flux through a Huygens
    # box decays, instead of the interior energy. This is a *radiation-settling* criterion —
    # it ignores non-radiating trapped energy (the static soft-source charge that FLOORS the
    # energy criterion, and near-Nyquist grid buzz) because neither carries net radiated power.
    # It is the right stop for radiation / S-parameter measurements; it is opt-in (default
    # keeps the energy criterion, unchanged) because its semantics differ from energy-decay.
    use_flux_stop = radiated_flux_box is not None and use_absorbing
    from rfx.measurement import flux_box_bounds, radiated_power
    if use_flux_stop:
        _bl = flux_box_bounds(grid, radiated_flux_box)

    peak_sq = 0.0          # closed/PEC point-field running peak
    # The absorbing-boundary energy stop (rfx.measurement.decay), read at k * check_interval.
    energy_stop, energy = make_decay_stop(
        grid, _setup.ctx_kwargs["materials"], thr=decay_by, min_steps=min_steps,
        interval=check_interval, consecutive=decay_energy_consecutive, n_end=n_end,
        criterion=use_absorbing and not use_flux_stop, max_steps=max_steps, sources=sources,
        mag_sources=mag_sources, tfsf=tfsf, waveguide_ports=waveguide_ports, dx=dx,
        periodic=_setup.periodic, side_step=(lambda c, xs, inv: jax.lax.scan(lambda cc, x: step_fn(
            cc, x, inv), c, xs)[0], lambda n: one_step_xs(n, src_waveforms, mag_src_waveforms), _inv_args))
    peak_flux = 0.0        # #388 flux-stop: running peak of the |P| envelope
    flux_below = 0         # #388 flux-stop: consecutive sub-threshold checks
    flux_hist: list[float] = []   # recent |P| samples for the max-envelope
    decayed_fired = False  # #388: did the energy criterion fire (vs silently cap-hit)?
    probe_chunks = []
    actual_steps = 0

    # Match scan_with_progress(stop_fn=...): report at existing boundaries,
    # never split a scan to print. max_steps is a CAP.
    _reporter = None
    if report_every is not None:
        _report_every = validate_report_every(report_every, n_steps=max_steps)
        # Probe the material/geometry arrays too, not just the carry: under
        # a bare grad/vmap the carry stays concrete and the tracer rides in
        # through the closure instead.
        check_not_traced(carry, _step_ctx.materials,
                         pec_mask, pec_edge_masks, pec_occupancy,
                         kerr_chi3, debye, lorentz, tfsf)
        _reporter = ProgressReporter(
            max_steps, label=report_label, total_is_cap=True)

    # Like scan_with_progress, call lax.scan directly: full chunks share
    # its executable and only the ragged tail needs another length.
    pending_tail = None
    while actual_steps < max_steps:
        # All checks use completed counts k * check_interval.
        lo = actual_steps
        next_check = (lo // check_interval + 1) * check_interval
        if pending_tail is not None:
            carry, probe_out, monitor_vals, frames = pending_tail
            pending_tail = None
            hi = max_steps
        else:
            this = min(next_check - lo, max_steps - lo)
            keep_previous = max_steps - lo - this == 1
            if keep_previous:
                # Copy scan_with_progress's one-step-tail rule. Unlike its
                # generic stop_fn, decay must inspect the penultimate step.
                this += 1
            hi = lo + this
            xs = (jnp.arange(lo, hi, dtype=jnp.int32), src_waveforms[lo:hi],
                  mag_src_waveforms[lo:hi])
            chunk_scan = _run_tail_chunk if keep_previous else _run_chunk
            (carry, (probe_out, monitor_vals)), frames = chunk_scan(carry, xs)
            if keep_previous:
                final_carry, carry = carry
                tail_frames = None
                if snapshot is not None:
                    n_before = (hi - 1) // snap_interval - lo // snap_interval
                    tail_frames = tuple(f[n_before:] for f in frames)
                    frames = tuple(f[:n_before] for f in frames)
                pending_tail = (final_carry, probe_out[-1:], monitor_vals[-1:], tail_frames)
                probe_out, monitor_vals = probe_out[:-1], monitor_vals[:-1]
                hi -= 1
        probe_chunks.append(probe_out)
        actual_steps = hi
        if snapshot is not None:
            n_frames = hi // snap_interval - lo // snap_interval
            if n_frames:
                snap_chunks.append(tuple(f[:n_frames] for f in frames))

        if _reporter is not None and (
                actual_steps - _reporter.last_reported >= _report_every
                or actual_steps >= max_steps):
            # Block first: JAX dispatch is asynchronous, so an unsynchronised
            # tick would report the host's dispatch rate, not the solve rate.
            jax.block_until_ready(carry["fdtd"])
            _reporter.report(actual_steps)

        if use_absorbing and use_flux_stop:
            # #388 opt-in: RADIATED-FLUX stop. Compute the net outgoing Poynting flux at each
            # check; smooth the 2f0 oscillation with a max-envelope over the last
            # ``flux_env_checks`` checks; stop when the envelope decays below the threshold.
            # The flux PEAK is during the source drive, so track it from the FIRST check (unlike
            # the interior energy, whose peak is post-source); only the STOP check waits for
            # min_steps.
            if actual_steps % check_interval == 0:
                flux_hist.append(abs(radiated_power(carry["fdtd"], _bl, grid.shape)))
                env = max(flux_hist[-flux_env_checks:])
                if env > peak_flux:
                    peak_flux = env
                if actual_steps >= min_steps and peak_flux > 0.0 and env < decay_by * peak_flux:
                    flux_below += 1
                    if flux_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    flux_below = 0
        elif use_absorbing:
            if energy_stop.check(actual_steps, lambda: energy(carry, actual_steps)) is not None:
                decayed_fired = True
                break
        else:
            # Closed/PEC fallback with the same point-field decision.
            # Decay check
            # One transfer per chunk; square in host float64 just as
            # float(monitor_val) ** 2 did for EVERY step of the old loop.
            # Reading only the final sample loses peaks between checks.
            values_sq = np.asarray(monitor_vals, dtype=np.float64) ** 2
            chunk_peak = np.max(values_sq, where=~np.isnan(values_sq), initial=0.0)
            if chunk_peak > peak_sq:
                peak_sq = float(chunk_peak)
            val_sq = float(values_sq[-1])

            if actual_steps >= min_steps and actual_steps % check_interval == 0 and peak_sq > 0.0:
                if val_sq < decay_by * peak_sq:
                    decayed_fired = True
                    break

    if _reporter is not None and _reporter.last_reported != actual_steps:
        # The decay stop lands on an arbitrary step, so without this line the
        # log's last progress entry would understate the run by up to
        # report_every steps and never show where it actually stopped.
        jax.block_until_ready(carry["fdtd"])
        _reporter.report(actual_steps)

    # What ended the run (one Diagnostic; the static-remnant advisory on an energy cap).
    decay_stop = finish_decay(
        energy_stop, actual_steps, fired=decayed_fired, reporter=_reporter,
        flux=(flux_hist, flux_env_checks, peak_flux) if use_flux_stop else None,
        point=None if use_absorbing else (val_sq, peak_sq), state=carry["fdtd"],
        materials=materials, grid=grid, advisory=advisory, sources=sources)

    # ---- assemble result ----
    time_series = concat_chunks(probe_chunks)

    final_dft_planes = None
    if use_dft_planes:
        final_dft_planes = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, carry["dft_planes"])
        )

    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            cfg_meta._replace(
                # Stamp the scan dt (see run() above): the jit-safe in-scan
                # probe accumulator can't run update_waveguide_port_probe's
                # float(dt) stamp, so without this a cfg built without dt= keeps
                # dt=0 and the post-scan rect-DFT zeroes every spectrum.
                # The scan's own step (stencil_order=4 derates it).
                dt=float(_setup.dt),
                v_probe_t=accs[0], v_ref_t=accs[1],
                i_probe_t=accs[2], i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(waveguide_meta, carry["waveguide_port_accs"])
        )

    from rfx.measurement.port_records import pack_port_records
    final_wire_sparams, final_lumped_sparams = pack_port_records(
        ('wire',) * len(wire_sparam_meta) + ('lumped',) * len(lumped_sparam_meta),
        carry.get("wire_sparam_accs", ()) + carry.get("lumped_sparam_accs", ()),
        wire_sparam_meta + lumped_sparam_meta)

    final_flux_monitors = None
    if use_flux_monitors:
        final_flux_monitors = tuple(
            fm._replace(e1_dft=accs[0], e2_dft=accs[1], h1_dft=accs[2], h2_dft=accs[3])
            for fm, accs in zip(flux_monitors, carry["flux_monitors"])
        )

    snapshots = snap_axes = None
    if snapshot is not None:
        if snap_chunks:
            snapshots = dict(zip(snapshot.components, concat_chunks(snap_chunks)))
        else:
            snapshots = {
                comp: jnp.zeros((0,) + s.shape, s.dtype)
                for comp, s in zip(
                    snapshot.components,
                    jax.eval_shape(_take_snapshot, carry["fdtd"]))}
        snap_axes = snapshot_axes(
            grid, snapshot, actual_steps, dt=_setup.dt)

    return SimResult(
        state=carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        sparam_time_records=uniform_records(carry, _step_ctx, actual_steps),
        dft_time_records=tuple(r[:actual_steps] for r in carry.get("dft_time_records", ())),
        snapshots=snapshots,
        ntff_box=ntff,
        grid=grid,
        snapshot_axes=snap_axes,
        dt=_setup.dt,
        current_moment_data=carry.get("current_moments"),
        current_moment_monitor=current_moments,
        decay_stop=decay_stop,
    )


run_uniform_loop = run
run_uniform_until_decay = run_until_decay
