"""Single-device graded time loops."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rfx.core.jax_utils import recorded_scan
from rfx.core.yee import MaterialArrays
from rfx.measurement import (
    AUTO_END, finish_decay, init_graded_records, make_decay_stop, one_step_xs,
)
from rfx.measurement import assemble_graded_result as _assemble_nu_result
from rfx.model.graded_grid import NonUniformGrid
from rfx.stepping import segmented_scan
from rfx.stepping.graded_setup import _build_nu_scan


def run_nonuniform(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    n_steps: int,
    *,
    cpml_axes: str | None = None,
    pec_mask=None,
    pec_sheets=(),
    pec_wires=(),
    pec_edge_masks=None,
    pec_occupancy=None,
    pinned_sheet_layout=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    current_moments=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    _feature_tfsf: bool = False,
    flux_monitors: list | None = None,
    checkpoint: bool = False,
    emit_time_series: bool = True,
    checkpoint_every: int | None = None,
    n_warmup: int = 0,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
    design_box=None,
    record_dft: bool = False,
) -> dict:
    """Run non-uniform FDTD via jax.lax.scan.

    Parameters
    ----------
    cpml_axes : str or None: axes with absorber pads when omitted; subsets accepted
    sources : list of (i, j, k, component, waveform_array)
    probes : list of (i, j, k, component)
    wire_ports : list of dict with keys:
        mid_i, mid_j, mid_k, component, impedance, waveform_array
    s_param_freqs : (n_freqs,) array for S-param DFT
    debye : (DebyeCoeffs, DebyeState) or None
    lorentz : (LorentzCoeffs, LorentzState) or None
    dft_planes : list of DFTPlaneProbe or None
        Frequency-domain plane accumulators. The accumulation is
        identical to the uniform path (acc += field * exp(-j2pi f t) * dt);
        dt is scalar on both paths so no per-axis weighting is required.
    """
    setup = _build_nu_scan(
        grid, materials, n_steps,
        cpml_axes=cpml_axes,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        pinned_sheet_layout=pinned_sheet_layout,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        current_moments=current_moments,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
    )
    step_fn = setup.step_fn
    carry_init = setup.carry_init
    if record_dft and dft_planes:
        carry_init["dft_time_records"] = tuple(
            jnp.zeros((n_steps,) + p.accumulator.shape[1:], dtype=carry_init["fdtd"].ex.dtype)
            for p in dft_planes)
    init_graded_records(carry_init, n_steps, setup)
    src_waveforms = setup.src_waveforms

    xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)

    # ---- n_warmup split --------------------------------------------------
    # When n_warmup > 0, run the first n_warmup steps with the carry
    # stop_gradient'd so AD builds no tape for that transient lead-in
    # (issue #40). Only the trailing n_optimize = n_steps - n_warmup
    # steps participate in reverse-mode autodiff.
    #
    # The truncation error DEPENDS ON DISTANCE FROM THE SOURCE TO THE
    # DESIGN REGION -- it is not a blanket property of n_warmup itself
    # (corrects an earlier version of this comment that said the
    # design-relevant support "generally" extends into the warmup window;
    # see the #626 addendum below for why that was an overgeneralization
    # from a single, worst-case fixture). Mechanism: severing the carry
    # severs every gradient path from a design variable's influence during
    # steps < n_warmup back into the loss -- but only steps during which
    # the WAVEFRONT HAS ALREADY REACHED the design region carry any such
    # influence to sever. Before the wavefront arrives, the field there
    # (and hence the loss's sensitivity to that cell) is ~0, so severing
    # those steps' carry discards ~nothing and the gradient is
    # (near-)exact -- n_warmup is genuinely free compute/memory relief in
    # that regime, not merely an approximation. Define
    #     K_safe ~= floor(min_distance(source, design_region) / (C0 * dt))
    # where min_distance is the MINIMUM over EVERY active source AND over
    # each source's own spatial extent (not just its nominal position) --
    # a TFSF plane-wave source, for example, illuminates from an entire
    # box face, not a point, so "distance from the source" means distance
    # from the NEAREST point of that box to the nearest point of the
    # design region. (grid steps for the wavefront to reach the closest
    # design cell, C0 = vacuum lightspeed -- a conservative floor valid
    # for any slower, non-vacuum propagation). For n_warmup <= K_safe
    # truncation error is negligible; beyond it, error grows and can
    # reach the full gradient magnitude by the time n_warmup reaches the
    # loss window.
    #
    # Forward output is exactly n_warmup-invariant (bit-identical) in every
    # placement measured. Two measured regimes (issue #626 part 2 /
    # addendum, both vs an independent central-FD oracle):
    #   - NEAR-SOURCE placement (design cell ~3 cells from the source, so
    #     K_safe ~ 0 -- the wavefront is already present at every step):
    #     error grows smoothly and monotonically from a ~1.5% noise floor
    #     at small n_warmup up to 58% at the loss-window boundary itself,
    #     and EXACTLY zero (gradient fully vanished) once n_warmup extends
    #     far enough into the loss window (fixture:
    #     tests/unit/autodiff/test_n_warmup.py::test_warmup_truncation_error_grows_with_k,
    #     n_steps=100, loss window [80,100)).
    #   - FAR-FROM-SOURCE placement (design cell 62 cells from the source,
    #     K_safe=108 measured from the grid's own dt): AD matches the K=0
    #     FD oracle to <0.01% rel_err through K=88 (deep sub-wavefront,
    #     K_safe-20), then grows SMOOTHLY -- not a sharp cliff -- as K
    #     approaches K_safe: 0.015% at K=98, 0.098% at K=102, 0.259% at
    #     K=104, 0.601% at K=106, 1.186% AT K=108 (=K_safe itself). Every
    #     one of those K<=K_safe values sits comfortably inside this
    #     repo's own established ~1.5% AD-vs-FD noise floor. Past K_safe
    #     the curve keeps growing (1.56% at K=109, ~2.5% around K=112),
    #     though non-monotonically farther out (numerical-dispersion
    #     ripple, not noise) before the broader trend continues upward
    #     toward the loss window (75% by K=200). K_safe is therefore a
    #     BOUND on where truncation stays within the established noise
    #     floor, not a literal discontinuity -- an earlier version of
    #     this comment read the original coarse (multiples-of-20) sweep,
    #     which skipped every point in [101,119], as "exactly the
    #     wavefront-arrival prediction" and overstated how sharp the
    #     transition is (fixture:
    #     scripts/diagnostics/i626_n_warmup_wavefront_locality.py,
    #     n_steps=220, loss window [180,220), finely sampled around
    #     K_safe).
    # The near-source curve is the WORST case, not the general case --
    # use K_safe to decide whether a given source/design placement is in
    # the exact-or-noise-floor or the truncated regime; do not read
    # "n_warmup truncates the gradient" as true unconditionally.
    if n_warmup > 0:
        if n_warmup >= n_steps:
            raise ValueError(
                f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
            )
        warmup_steps = jnp.arange(n_warmup, dtype=jnp.int32)
        warmup_xs = (warmup_steps, src_waveforms[:n_warmup])
        warmup_final, warmup_ys = recorded_scan(step_fn, carry_init, warmup_xs)
        carry_init = jax.tree_util.tree_map(
            jax.lax.stop_gradient, warmup_final
        )
        warmup_ys = jax.lax.stop_gradient(warmup_ys)
        xs = (
            jnp.arange(n_warmup, n_steps, dtype=jnp.int32),
            src_waveforms[n_warmup:],
        )
        n_steps_opt = n_steps - n_warmup
    else:
        warmup_ys = None
        n_steps_opt = n_steps

    use_segmented = (
        checkpoint_every is not None
        and 0 < int(checkpoint_every) < n_steps_opt
    )
    if use_segmented:
        final, opt_ys = segmented_scan(
            step_fn, carry_init, xs, int(checkpoint_every), scan=recorded_scan
        )
    else:
        body = jax.checkpoint(step_fn) if checkpoint else step_fn
        final, opt_ys = recorded_scan(body, carry_init, xs)
    # Merge warmup + optimize outputs back into one time_series so the
    # downstream result shape stays (n_steps, n_probes).
    if warmup_ys is not None:
        time_series = jnp.concatenate([warmup_ys, opt_ys], axis=0)
    else:
        time_series = opt_ys

    return _assemble_nu_result(setup, final, time_series)


def run_nonuniform_until_decay(
    grid: NonUniformGrid,
    materials: MaterialArrays,
    *,
    cpml_axes: str | None = None,
    decay_by: float = 1e-3,
    n_end: int | None = AUTO_END,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50_000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    report_every: int | None = None,
    report_label: str = "",
    pec_mask=None,
    pec_sheets=(),
    pec_wires=(),
    pec_edge_masks=None,
    pec_occupancy=None,
    pinned_sheet_layout=None,
    sources: list = None,
    probes: list = None,
    wire_ports: list = None,
    s_param_freqs=None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    dft_planes: list | None = None,
    rlc_metas: tuple = (),
    rlc_states: tuple = (),
    ntff_box=None,
    ntff_data=None,
    current_moments=None,
    waveguide_ports: list | None = None,
    tfsf: tuple | None = None,
    _feature_tfsf: bool = False,
    flux_monitors: list | None = None,
    emit_time_series: bool = True,
    aniso_eps: tuple | None = None,
    sheet_impedance=None,
    design_box=None,
    stop_fn=None,
    record_dft: bool = False,
    advisory: dict | None = None,
) -> dict:
    """Run non-uniform FDTD until the field energy left after the drive decays.

    A Python loop drives constant-length jitted chunks (``chunk =
    check_interval`` steps) of the SAME ``step_fn`` that
    :func:`run_nonuniform` scans, threading the SAME carry (NTFF Kahan
    compensation arrays included) across chunks. Per-chunk ``xs``
    continue the GLOBAL step indices, exactly like the ``n_warmup``
    split in :func:`run_nonuniform`, so source waveforms / DFT phases
    are identical to a fixed-step run of the same length.

    Stop rule: ``rfx.measurement.decay`` (shared with the uniform loop; the
    leapfrog-conserved interior energy W after the drive end ``n_end``, per-cell
    dV, the float32 floor ``KAPPA = 1e-12``); ``result["decay_stop"]`` is the
    ``until_decay.stop`` Diagnostic of what ended the run. ``decay_by = 0`` runs
    exactly ``max_steps``. On a high-Q structure whose mode reaches the absorber,
    the stop can move by about 170 steps with the check phase while the spectra
    stay within 1e-2 of the band peak. In a lossy dielectric the run can last
    longer than the field's ringing: charge moved onto the dielectric's surfaces
    during the drive relaxes with the time constant eps/sigma after it. That
    field is quasi-static (out of band), the record stays correct, and the stop
    record and the cap advisory say what ended the run.

    On a closed domain (no absorbing pads) the interior energy does not decay
    and this runs to ``max_steps``; ``run()`` refuses ``until_decay`` there.

    Forward-only: a host loop, not reachable from ``forward()``/``optimize()``,
    no AD tape contract; ``checkpoint`` / ``checkpoint_every`` / ``n_warmup``
    are absent from the signature (the caller raises before dispatching here).

    ``stop_fn`` (issue #1254, ``run(..., until_identified=True)``), when
    given, is called at every chunk boundary after the energy check as
    ``stop_fn(steps_done, peek)``; ``peek()`` returns
    :func:`_assemble_nu_result` of the record so far, and a true return ends
    the loop there. ``None`` (default) leaves the loop unchanged.

    Returns
    -------
    dict
        Same schema as :func:`run_nonuniform` (:func:`_assemble_nu_result`),
        plus ``"decay_checks"``: ``(step, W, reference)`` per energy check from
        ``max(min_steps, n_end)`` on, on this dict only (not on ``Result``),
        for a caller verifying the fire condition (rule R5).
    """
    if check_interval < 1:
        raise ValueError(
            f"check_interval must be >= 1, got {check_interval}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    setup = _build_nu_scan(
        grid, materials, max_steps,
        cpml_axes=cpml_axes,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        pinned_sheet_layout=pinned_sheet_layout,
        sources=sources,
        probes=probes,
        wire_ports=wire_ports,
        s_param_freqs=s_param_freqs,
        debye=debye,
        lorentz=lorentz,
        pec_faces=pec_faces,
        pmc_faces=pmc_faces,
        dft_planes=dft_planes,
        rlc_metas=rlc_metas,
        rlc_states=rlc_states,
        ntff_box=ntff_box,
        ntff_data=ntff_data,
        current_moments=current_moments,
        waveguide_ports=waveguide_ports,
        tfsf=tfsf, _feature_tfsf=_feature_tfsf,
        flux_monitors=flux_monitors,
        emit_time_series=emit_time_series,
        aniso_eps=aniso_eps,
        sheet_impedance=sheet_impedance,
        design_box=design_box,
    )
    carry = setup.carry_init
    if record_dft and dft_planes:
        carry["dft_time_records"] = tuple(
            jnp.zeros((max_steps,) + p.accumulator.shape[1:], dtype=carry["fdtd"].ex.dtype)
            for p in dft_planes)
    init_graded_records(carry, max_steps, setup)

    # Source table: pad/truncate to max_steps so every chunk slice is
    # full-length (mirrors the uniform run_until_decay pad/truncate).
    src_waveforms = jnp.asarray(setup.src_waveforms)
    n_have = int(src_waveforms.shape[0])
    if n_have < max_steps:
        src_waveforms = jnp.pad(
            src_waveforms, ((0, max_steps - n_have), (0, 0)))
    elif n_have > max_steps:
        src_waveforms = src_waveforms[:max_steps]

    # One compiled program per chunk length: full chunks share one XLA
    # executable; the final partial chunk (if any) compiles once more. The
    # per-cell arrays are an argument, not constants compiled into it.
    from rfx.core.jax_utils import split_loop_invariants
    inv_args, inv_rebuild = split_loop_invariants(
        setup.invariants, (grid.nx, grid.ny, grid.nz))
    step_fn_inv = setup.step_fn_inv

    @jax.jit
    def _run_chunk(carry_in, xs, inv_args):
        step = partial(step_fn_inv, invariants=inv_rebuild(inv_args))
        return jax.lax.scan(step, carry_in, xs)

    # #388 opt-in RADIATED-FLUX stop (NU lane; mirrors the uniform run_until_decay path):
    # stop on the outgoing Poynting flux through a Huygens box instead of the interior energy,
    # so the non-radiating static soft-source charge (which FLOORS the energy criterion) and
    # near-Nyquist buzz do not stall the stop. Opt-in; default keeps the energy criterion.
    use_flux_stop = radiated_flux_box is not None
    from rfx.measurement import flux_box_bounds, radiated_power
    if use_flux_stop:
        _bl = flux_box_bounds(grid, radiated_flux_box)

    energy_stop, energy = make_decay_stop(
        grid, materials, thr=decay_by, min_steps=min_steps, interval=check_interval,
        consecutive=decay_energy_consecutive, n_end=n_end, criterion=not use_flux_stop,
        rows=True, max_steps=max_steps, sources=sources, tfsf=tfsf,
        waveguide_ports=waveguide_ports, side_step=(lambda c, xs, inv: _run_chunk(c, xs, inv)[0],
                                                    lambda n: one_step_xs(n, src_waveforms), inv_args))
    peak_flux = 0.0        # #388 flux-stop: running peak of the |P| envelope (from first check)
    flux_below = 0         # #388 flux-stop: consecutive sub-threshold checks
    flux_hist: list = []   # recent |P| samples for the max-envelope
    decayed_fired = False  # #388: did the energy criterion fire (vs cap-hit)?
    ys_chunks = []
    steps_done = 0
    reporter = None
    if report_every:
        from rfx.progress import ProgressReporter
        # decay_by == 0.0 is the documented forced-N escape: max_steps is then
        # the exact run length, not a cap, and the line should not say "(cap)".
        reporter = ProgressReporter(max_steps, label=report_label,
                                    total_is_cap=(decay_by > 0.0
                                                  or stop_fn is not None))

    while steps_done < max_steps:
        this_chunk = min(int(check_interval), max_steps - steps_done)
        if stop_fn is not None and max_steps - steps_done - this_chunk == 1:
            this_chunk += 1   # #1254: no one-step last chunk (XLA inlines it)
        xs = (
            jnp.arange(steps_done, steps_done + this_chunk, dtype=jnp.int32),
            src_waveforms[steps_done:steps_done + this_chunk],
        )
        carry, ys = _run_chunk(carry, xs, inv_args)
        ys_chunks.append(ys)
        steps_done += this_chunk
        if reporter is not None and (
                steps_done - reporter.last_reported >= int(report_every)
                or steps_done >= max_steps):
            jax.block_until_ready(carry["fdtd"])   # honest wall-clock rate
            reporter.report(steps_done)

        # Stop check at the chunk boundary (check-step-only — the whole-domain
        # reduction / surface integral is the expensive part).
        if use_flux_stop:
            # RADIATED-FLUX stop: track the flux peak from the FIRST check (radiation peaks
            # during the source drive), gate only the STOP on min_steps.
            flux_hist.append(abs(radiated_power(carry["fdtd"], _bl, grid.shape)))
            env = max(flux_hist[-flux_env_checks:])
            if env > peak_flux:
                peak_flux = env
            if steps_done >= min_steps and peak_flux > 0.0 and env < decay_by * peak_flux:
                flux_below += 1
                if flux_below >= decay_energy_consecutive:
                    decayed_fired = True
                    break
            else:
                flux_below = 0
        elif energy_stop.check(steps_done, lambda: energy(carry, steps_done)) is not None:
            decayed_fired = True
            break

        # #1254: the caller's stop check at the chunk boundary.
        if stop_fn is not None:
            from rfx.progress import concat_chunks
            if stop_fn(steps_done, lambda: _assemble_nu_result(
                    setup, carry, concat_chunks(ys_chunks), n_record=steps_done)):
                if reporter is not None and reporter.last_reported != steps_done:
                    reporter.report(steps_done)
                break

    # What ended the run (one Diagnostic; the static-remnant advisory on an energy cap).
    # decay_by == 0.0 is the forced-N escape (fixed-step progress route): none.
    decay_stop = finish_decay(
        energy_stop, steps_done, fired=decayed_fired, reporter=reporter,
        flux=(flux_hist, flux_env_checks, peak_flux) if use_flux_stop else None,
        state=carry["fdtd"], materials=materials, grid=grid, advisory=advisory,
        sources=sources)

    # Per-chunk ys concatenate to exactly steps_done rows; the
    # emit_time_series=False convention ((steps, 0)) is preserved
    # because each chunk emits (this_chunk, 0).
    time_series = jnp.concatenate(ys_chunks, axis=0)

    result = _assemble_nu_result(setup, carry, time_series, n_record=steps_done)
    result["decay_checks"] = energy_stop.decay_checks
    result["decay_stop"] = decay_stop
    return result
