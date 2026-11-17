"""Completion of field-probe channels on their physical leapfrog clocks."""
from dataclasses import replace
import math
from types import SimpleNamespace
import warnings

import numpy as np

from rfx import ringdown as rd
from rfx._warning_origin import user_stacklevel
from rfx.errors import AccuracyWarning
from rfx.ringdown._geometry import _user_geometry
from rfx.ringdown._incident_norm import normalized


RELATIVE_RULE = "max_c max_f |X_A,c - X_B,c| / max_f |X_main,c| over probe_freqs"


def probe_reference(plan):
    """The same concrete decimation reference for memory, host and trace."""
    return max(plan.freq_max, float(np.max(plan.bins)))


def probe_stamp(plan, dt):
    """Unit phase; slot spectra already include dt, as measurement.dft does."""
    from rfx.measurement.dft import host_stamp

    kinds = ["E" if comp.startswith("e") else "H" for comp, _ in plan.identification_keys]
    return host_stamp(plan.bins, dt, kinds)


def relative_difference(a, b, reference):
    """A worst-channel relative difference, normalized at each channel's peak."""
    numerator = np.max(np.abs(np.asarray(a) - np.asarray(b)), axis=0)
    denominator = np.max(np.abs(reference), axis=0)
    return float(np.max(numerator / np.maximum(denominator, 1e-300)))


def require_field(plan, window):
    """Name a declared channel that contains no field in this fit window."""
    for c, (position, component) in enumerate(plan.spec.identification_probes):
        if not np.any(window[:, c]):
            raise ValueError(
                f"identification probe {component} at {position} records no field over the window")


def _not_formed(name, reason):
    return rd.RingdownWitness(name, math.nan, math.nan, False, reason,
                              "not formed: " + reason, judged=False)


def _error_witnesses(plan, Y, dt, ref_hz, plain, w2, state):
    spec = plan.spec
    tol = float(spec.witness_tol)
    main = normalized(plan, w2.spectra)
    w2_value = relative_difference(main, normalized(plan, w2.spectra_short), main)
    rule = RELATIVE_RULE + ("" if plan.spec.probe_normalization is None
                            else "; every operand normalized by the incident spectrum")
    if plan.source_ratio_long > spec.source_off_tol:
        why = (f"a source reaches {plan.source_ratio_long:.3e} of its peak in "
               f"[{0.5 * spec.window_start:g} T, T], above source_off_tol = {spec.source_off_tol:.1e}")
        return [rd.RingdownWitness("WE", math.nan, tol, False, rule,
                                   "not formed: " + why, judged=False),
                rd.RingdownWitness("W2", w2_value, tol, w2_value <= tol, rule,
                                   "judged in place of WE: " + why)]
    try:
        require_field(plan, Y[plan.n_long:])
        we = rd.early_start_witness(
            Y, dt, plan.bins, plan.n_steps, spec.window_start,
            freq_max=ref_hz, guard=spec.guard, sv_rel=spec.sv_rel,
            unit_tol=spec.unit_tol, observable=None, plain=plain, spectra=w2.spectra)
        value = relative_difference(main, normalized(plan, we.spectra_long), main)
        note = ""
        state.update(long_rank=we.model_long.rank, long_n_kept=int(we.model_long.s.size))
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded in RingdownWitness.note
        value, note = math.nan, f"the identification on the early window failed: {exc}; a WE that cannot be read fails"
    return [rd.RingdownWitness("WE", value, tol, value <= tol, rule, note),
            rd.RingdownWitness("W2", w2_value, tol, w2_value <= tol, rule,
                               "information (WE is the judged error witness)", judged=False)]


def _complete_probe(plan, result, Y, dt, ref_hz, witnesses):
    from rfx.measurement.ringdown_tail import plain_dft

    plan._check_solver_step(result)
    spec = plan.spec
    plain = plain_dft(Y, dt, plan.bins)
    try:
        require_field(plan, Y[plan.n_start:])
        require_field(plan, Y[plan.n_start:plan.n_split])
        w2 = rd.two_window_witness(
            Y, dt, plan.bins, plan.n_steps, plan.n_start, freq_max=ref_hz,
            split=spec.split, guard=spec.guard, sv_rel=spec.sv_rel,
            unit_tol=spec.unit_tol, observable=None, plain=plain)
    except rd.IdentificationMemoryRefused:
        raise
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise rd._NotCompleted("identification", math.nan, math.nan, f"{exc}.") from None
    model = w2.model
    growth = rd.judge_growing_poles(model, plan.n_steps, plan.bins, np.abs(plain),
                                    unit_tol=spec.unit_tol)
    g = np.asarray(model.growing_amplitude, dtype=np.float64)
    if g.size:
        kg = int(np.argmax(g))
        g_max = float(g[kg])
        g_f = float(np.asarray(model.s_growing)[kg].imag / (2.0 * np.pi))
    else:
        g_max, g_f = 0.0, math.nan
    state = {}
    witnesses += _error_witnesses(plan, Y, dt, ref_hz, plain, w2, state)
    src_ratio = max((r for _l, r in plan.source_off), default=0.0)
    witnesses += [
        rd.RingdownWitness("growing_poles", float(np.max(growth.share_per_bin)),
                            1e-2, growth.ok, rd.GROWING_POLE_RULE,
                            f"{growth.status}: {growth.reason}; {growth.n_kept_growing} kept growing; "
                            f"{growth.n_exempt} zero-frequency exempt; {g.size} discarded as growing "
                            f"(largest {g_max:.3g} at {g_f / 1e9:.4g} GHz, reported, not judged)"),
        rd.RingdownWitness("source_off", src_ratio, float(spec.source_off_tol),
                            src_ratio <= spec.source_off_tol,
                            "largest drive level over the window as a fraction of its peak "
                            "(a waveform; for a plane wave its face-read level)"),
    ]
    amp = np.asarray(model.amplitude, dtype=np.float64)
    alpha = -model.s.real
    strong = amp >= 1.0e-3
    window_s = (plan.n_steps - plan.n_start) * dt
    if np.any(strong):
        a_min = float(np.min(alpha[strong]))
        slowest = math.inf if a_min <= 0 else 1.0 / a_min / window_s
    else:
        slowest = 0.0
    rms = np.where(model.window_rms > 0, model.window_rms, 1.0)
    shares = np.abs(model.c) / rms[None, :]
    shares = shares[np.argsort(-np.nan_to_num(amp, nan=-1.0), kind="stable")]
    extra = dict(
        identification_channels=plan.identification_keys,
        identification_amplitude_shares=tuple(tuple(float(v) for v in row) for row in shares),
        decimation_factors=model.factors, rank=model.rank, n_kept=int(model.s.size),
        n_growing_discarded=model.n_growing_discarded, n_invalid=model.n_invalid,
        n_guard_dropped=model.n_guard_dropped, short_rank=w2.model_short.rank,
        short_n_kept=int(w2.model_short.s.size), long_rank=state.get("long_rank"),
        long_n_kept=state.get("long_n_kept"), poles=model.poles(),
        growing_pole_rule=rd.GROWING_POLE_RULE,
        tail_share=relative_difference(normalized(plan, w2.spectra), normalized(plan, plain),
                                       normalized(plan, w2.spectra)),
        slowest_decay_over_window=slowest, discarded_growth_max=g_max,
        discarded_growth_f_hz=g_f)
    return normalized(plan, w2.spectra * probe_stamp(plan, dt)), extra


def host_probe_report(plan, result, ts_all, n_user):
    """Float64 probe completion, with no port metadata or accumulator rebuild."""
    dt = float(result.grid.dt)
    ref_hz = probe_reference(plan)
    base = dict(n_record=plan.n_steps, dt=dt, window_steps=(plan.n_start, plan.n_steps),
                window_s=(plan.n_start * dt, plan.n_steps * dt),
                short_window_steps=(plan.n_start, plan.n_split),
                long_window_steps=(plan.n_long, plan.n_steps), freq_max_hz=ref_hz,
                probe_normalization=plan.spec.probe_normalization,
                incident_spectra=getattr(plan, "incident_spectra", None))
    no_acc = "no port accumulator; a probe channel is the recorded field itself"
    witnesses = [_not_formed("W0", no_acc), _not_formed("W1", no_acc),
                 _not_formed("passivity", "a field spectrum has no passivity bound")]
    columns = [n_user + j for j in plan._identification_columns()]
    Y = np.asarray(ts_all[:, columns], dtype=np.float64)
    try:
        spectra, extra = _complete_probe(plan, result, Y, dt, ref_hz, witnesses)
    except rd._NotCompleted as nc:
        witnesses.append(rd.RingdownWitness(nc.name, nc.value, nc.bar, False, nc.message))
        return None, rd.RingdownReport(**base, witnesses=tuple(witnesses),
                                      failure=f"{nc.name}: {nc.message}"), nc
    return spectra, rd.RingdownReport(**base, witnesses=tuple(witnesses), **extra), None


def finish_probe_run(plan, result, n_user):
    full = result.time_series
    n_int = len(plan.probe_keys)
    if np.shape(full) != (plan.n_steps, n_user + n_int):
        raise RuntimeError(f"run(ringdown=...): unexpected probe series shape {np.shape(full)}")
    stripped = _user_geometry(result._replace(time_series=full[:, :n_user]), n_user, n_int)
    spectra, report, nc = plan._host_report(result, np.asarray(full), n_user)
    if nc is not None:
        warnings.warn(
            f"ring-down completion not produced -- {nc.name} failed: {nc.message} "
            "Result.ringdown.probe_spectra is None; every other output of this run is as without ringdown=.",
            AccuracyWarning, stacklevel=user_stacklevel())
    elif not report.ok:
        failed = ", ".join(f"{w.name} {w.value:.3e} (bar {w.bar:.3e})"
                           + (f" -- {w.note}" if w.note else "")
                           for w in report.witnesses if w.judged and not w.ok)
        warnings.warn(
            f"ring-down completion: witness failed -- {failed}; Result.ringdown.probe_spectra "
            "is not supported by its own check (see Result.ringdown.report).",
            AccuracyWarning, stacklevel=user_stacklevel())
    return stripped._replace(ringdown=rd.RingdownResult(
        s_params=None, freqs=plan.bins, report=report, probe_spectra=spectra))


def forward_probe_report(ctx, res):
    plan = ctx.plan
    result = SimpleNamespace(grid=ctx.grid, dt=ctx.step)
    host, report, nc = plan._host_report(result, np.asarray(res._channels), 0)
    tol = float(plan.spec.witness_tol)
    note = "; ".join(
        f"{window}: {rd._status_reason(code, kept_count=count, pole_budget=budget)}"
        for window, code, count, budget in zip(
            ("main", "early"), np.asarray(res._status),
            np.asarray(res._pole_counts), np.asarray(res._pole_budgets)) if int(code) != 0)
    value = math.nan if nc is not None else relative_difference(res.probe_spectra, host, host)
    if nc is not None:
        note += f"; no float64 completion to compare: {nc.name} failed on the host"
    traced = rd.RingdownWitness("traced", value, tol, value <= tol,
                                RELATIVE_RULE + "; traced against host", note, judged=nc is None)
    consistency = _not_formed("consistency", "no port accumulator to compare against")
    return replace(report, witnesses=report.witnesses + (traced, consistency))


def traced_probe_completion(plan, result, channels):
    """Complete the raw probe columns; failures poison values and derivatives."""
    import jax.numpy as jnp
    from rfx.ringdown import _jax as rj
    from rfx.ringdown._forward_context import report_context
    from rfx.measurement.ringdown_resolution import completion_poles
    from rfx.measurement.ringdown_tail_jax import plain_dft

    bins = plan.bins
    rdt, cdt = rj.working_dtypes()
    ctx = report_context(plan, result, bins)
    cells = tuple(getattr(result.grid, store) for _axis, store in ctx.traced_axes)
    consistency = jnp.asarray(jnp.nan, dtype=jnp.real(channels).dtype)
    try:
        plan._check_solver_step(result)
    except rd._NotCompleted:
        nan = jnp.sum(channels) * jnp.nan
        spectra = jnp.broadcast_to(nan, (len(bins), len(plan.identification_keys))).astype(cdt)
        pole_nan = jnp.reshape(nan, (1,)).astype(cdt)
        mask = jnp.zeros(1, jnp.int32)
        return rd.RingdownForwardResult(
            None, None, jnp.asarray(bins, dtype=rdt), channels, (), None,
            jnp.full(2, rj.STATUS_PRECONDITION, jnp.int32), consistency,
            jnp.full(2, -1, jnp.int32), jnp.zeros(2, jnp.int32), ctx,
            poles=pole_nan, poles_long=pole_nan, pole_mask=mask, pole_mask_long=mask,
            probe_spectra=spectra, probe_spectra_long=spectra, cells=cells)
    Y = channels[:, np.asarray(plan._identification_columns())].astype(rdt)
    dt, n, spec = float(result.grid.dt), plan.n_steps, plan.spec
    ref_hz = probe_reference(plan)

    def identify_window(window):
        require_field(plan, window)
        return completion_poles(rd.identify(
            window, dt, 0, window.shape[0], freq_max=ref_hz, guard=spec.guard,
            sv_rel=spec.sv_rel, unit_tol=spec.unit_tol), n, spec.unit_tol)

    stamp = jnp.asarray(probe_stamp(plan, dt), dtype=cdt)
    plain = plain_dft(Y, dt, bins)
    out, status, counts, budgets, poles, masks = [], [], [], [], [], []
    for n0 in (plan.n_start, plan.n_long):
        k_max = min(rd.static_pole_bound(n - n0, Y.shape[1], dt, ref_hz), rd.TRACED_POLE_BUDGET)
        s0, mask, st, tail_arg, count = rj.host_poles(
            identify_window if n0 == plan.n_start else rd._witness_identification(identify_window),
            Y[n0:], k_max, dt, bins)
        spectra, aux = rj.completion(Y, dt, bins, n0, s0, mask.astype(rdt),
                                     tail_arg, plain=plain, return_aux=True)
        factor = jnp.where(st == rj.STATUS_OK, 1.0, jnp.nan).astype(rdt)
        out.append(normalized(plan, spectra * stamp * factor, cdt))
        poles.append(aux["s"] * factor)
        masks.append(mask)
        status.append(st)
        counts.append(count)
        budgets.append(k_max)
    return rd.RingdownForwardResult(
        None, None, jnp.asarray(bins, dtype=rdt), channels, (), None,
        jnp.stack(status), consistency, jnp.stack(counts), jnp.asarray(budgets), ctx,
        poles=poles[0], poles_long=poles[1], pole_mask=masks[0], pole_mask_long=masks[1],
        probe_spectra=out[0], probe_spectra_long=out[1], cells=cells)
