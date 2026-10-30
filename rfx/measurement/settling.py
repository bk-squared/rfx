"""Settling measurements from named port records."""

from __future__ import annotations

from rfx.errors import AccuracyWarning
import math

import numpy as np


_SETTLING_RECORD_NAMES = ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t")
# #1909: only co-located records tell a field null's rounding from signal.
_LONE_PROBE_NOTE = (
    " (no other component of its field (E or H) is recorded at its cell: a probe on a field null "
    "holds only rounding and cannot be judged alone; record the co-located "
    "components, e.g. add_vector_probe(), so the check can tell rounding "
    "from signal)")


def settling_db_from_named_records(named_records, *, dt=None, freqs=None,
                                   freq_max=None, source_end_index=None,
                                   record_noun="port records", return_detail=False,
                                   rounding_groups=None, _warn_stacklevel=2,
                                   known_drives=None, n_record=None):
    """Adapt the per-bin pole-tail value witness to legacy dB report fields.

    ``rounding_groups`` maps record names to co-located same-kind groups
    compared for rounding-only peaks (#1909).
    sensitivities are judged by a separate witness
    """
    from rfx.sparams._tail_witness import tail_share_witness

    named = tuple(named_records)
    per_record = {}
    result = tail_share_witness(named, dt, source_end_index, freqs, freq_max=freq_max,
                                rounding_groups=rounding_groups,
                                known_drives=known_drives, n_record=n_record,
                                _record_results=per_record if return_detail else None)
    if not return_detail:
        return result.db
    finite = {name: value for name, value in per_record.items() if np.isfinite(value)}
    return result.db, {**result._asdict(), "per_record_db": per_record,
                       "n_witnessed": len(finite),
                       "worst_record": max(finite, key=finite.get) if finite else None,
                       "skipped_records": list(result.skipped_records)}


def settling_db_from_port_records(final_cfgs, *, freqs=None, freq_max=None,
                                   return_detail: bool = False):
    """Per-bin pole-tail value witness over the four modal records per port.

    sensitivities are judged by a separate witness
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.measurement.modal import recorded as _recorded
    from rfx.measurement.record_end import injection_end
    from rfx.model.waveguide_modes import incident_tables_uncut

    final_cfgs = tuple(final_cfgs)
    named = [
        (f"port{port_index}/{name}", _recorded(cfg, name))
        for port_index, cfg in enumerate(final_cfgs)
        for name in _SETTLING_RECORD_NAMES
    ]
    n_record = len(named[0][1]) if named else 0
    timing_fields = ("e_inc_table", "src_amp", "source_x_m", "reference_x_m", "probe_x_m", "dt")
    timing_available = all(
        hasattr(cfg, name) and not is_tracer(getattr(cfg, name))
        for cfg in final_cfgs for name in timing_fields)
    source_end = None
    axes = {getattr(cfg, "normal_axis", "x") for cfg in final_cfgs}
    if final_cfgs and timing_available and len(axes) == 1:
        # The applied incident E table with the one source-end rule, plus
        # free-space travel to every scored plane.
        axis = "xyz".index(axes.pop())
        planes = []
        for other in final_cfgs:
            for x in (other.reference_x_m, other.probe_x_m):
                point = np.zeros(3)
                point[axis] = float(x)
                planes.append(point)
        source_end = injection_end(n_record, final_cfgs[0].dt, records=planes,
                                   waveguide_ports=final_cfgs)
    try:
        known_drives = [table for cfg in final_cfgs for table in incident_tables_uncut(cfg)
                        if len(table) > n_record]
    except ValueError as exc:   # the witness reads "cannot judge" with this reason
        known_drives = exc
    bins = freqs if freqs is not None else (
        getattr(final_cfgs[0], "freqs", None) if final_cfgs else None)
    if freq_max is None and bins is not None and not is_tracer(bins):
        freq_max = float(np.max(bins)) if np.size(bins) else None
    return settling_db_from_named_records(
        named, source_end_index=source_end,
        dt=getattr(final_cfgs[0], "dt", None) if final_cfgs else None,
        freqs=bins, freq_max=freq_max, record_noun="port records",
        known_drives=known_drives if isinstance(known_drives, ValueError) else (known_drives or None),
        n_record=n_record,
        return_detail=return_detail, _warn_stacklevel=3)


def warn_postrun_energy_witness(
    sim,
    result,
    *,
    fixed_num_periods: bool,
    until_decay: float | None,
) -> None:
    """Warn about measured late energy or an exactly zero excitation.

    Tracer-safe: a no-op under a JAX transformation.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.sources.sources import GaussianPulse

    import warnings

    time_series = getattr(result, "time_series", None)
    if is_tracer(time_series):
        return
    series = None if time_series is None else np.asarray(time_series)
    # Issue #470: exclude library-internal witness probes (the MSL
    # settling witnesses) — their record is judged by the MSL driver's
    # own settling_db witness, so #332/#336 here speak for USER probes
    # only and the two differently-defined witnesses never double-fire
    # on the same columns. With only internal probes present this
    # behaves as if no probe series was recorded (the pre-#468 state).
    # Time-series columns are in probe registration order (the same
    # convention the MSL driver's witness slice relies on).
    _internal = getattr(sim, "_internal_probe_indices", None)
    if series is not None and series.ndim == 2 and _internal:
        _keep = [i for i in range(series.shape[1]) if i not in _internal]
        series = series[:, _keep] if _keep else None
    has_probe_series = series is not None and series.size > 0

    if has_probe_series:
        magnitude = np.abs(series)
        peak = float(np.max(magnitude))
        if peak == 0.0:
            warnings.warn(
                "no field energy was recorded — the source may not have "
                "coupled (position off-grid, wrong component, or zero "
                "amplitude)", AccuracyWarning,
                stacklevel=4,
            )
            return

        pulsed = any(
            isinstance(getattr(source, "waveform", None), GaussianPulse)
            for source in (*sim._ports, *sim._msl_ports)
        )
        open_boundary = sim._boundary_spec.absorber_type in ("cpml", "upml")
        if open_boundary and pulsed and fixed_num_periods and until_decay is None:
            tail_samples = max(1, math.ceil(magnitude.shape[0] * 0.05))
            tail = float(np.max(magnitude[-tail_samples:]))
            tail_db = -math.inf if tail == 0.0 else 20.0 * math.log10(tail / peak)
            from rfx.api._sparams import settling_verdict as _verdict
            if _verdict(tail_db) == "fail":
                warnings.warn(
                    f"run ended at {tail_db:.1f} dB of peak — ring-down "
                    "truncated; Harminv/DFT/NTFF quantities may carry "
                    "transient error; consider until_decay or more periods"
                    "", AccuracyWarning,
                    stacklevel=4,
                )
        return

    state = getattr(result, "state", None)
    fields = [
        np.asarray(field)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        if (field := getattr(state, name, None)) is not None
    ]
    if fields and all(field.size > 0 and float(np.max(np.abs(field))) == 0.0
                      for field in fields):
        warnings.warn(
            "no field energy was recorded — the source may not have "
            "coupled (position off-grid, wrong component, or zero "
            "amplitude)", AccuracyWarning,
            stacklevel=4,
        )



def attach_run_settling_witness(sim, result, *, n_steps=None,
                                 num_periods=None, context="run"):
    """Attach the value witness and warn for requested spectral outputs."""
    from rfx.api._spec import ForwardResult

    from rfx.measurement.coverage import accumulator_coverage, warn_uncovered, with_coverage
    forward_result = isinstance(result, ForwardResult)
    if not forward_result and (not hasattr(result, "_replace") or not hasattr(result, "settling_witness")):
        return result
    coverage = accumulator_coverage(result, sim=sim, n_steps=n_steps)
    if forward_result:
        from rfx.core.jax_utils import is_tracer
        from rfx.sparams._record_selection import s_read_records
        result = result._replace(
            settling_probe_info=sim._settling_probe_selection(result),
            settling_freq_max=sim._freq_max,
            settling_accumulator_end_index=coverage.max_end(),
            settling_coverage=coverage,
            settling_source_end_index=sim._settling_source_end(
                result, records=sim._ports if s_read_records(result) else None),
            settling_probe_source_end_index=sim._settling_source_end(result))
        if is_tracer(result.time_series):
            return result
    from rfx.measurement import decay_stop_first
    settling_db, witness = decay_stop_first(result, *sim._run_settling_witness(result))
    tail_status = witness["status"]
    settling_db, witness = with_coverage(settling_db, witness, coverage)
    if not forward_result:
        result = result._replace(settling_db=settling_db,
                                 settling_witness=witness)
    if getattr(sim, "_internal_probe_indices", None):
        return result
    # The coverage warning stands in for the absent advisory or a passing
    # record; a tail check that failed keeps its own warning below.
    if tail_status in {"pass", "absent"} and warn_uncovered(witness):
        return result

    from rfx.api._sparams import (
        _warn_if_ringdown_truncated,
        settling_verdict,
    )

    has_s = (getattr(result, 's_params', None) is not None
              or bool(getattr(result, 'waveguide_sparams', None)))
    if sim._ntff is None and not sim._dft_planes and not has_s:
        return result

    if has_s and witness['status'] in {'fail', 'undetermined'}:
        worst = witness['worst_freq_hz']
        bin_label = (f"worst bin {worst:.9g} Hz" if np.isfinite(worst)
                     else "worst bin undetermined")
        shares = np.asarray(witness["share_per_bin"])
        peak = float(np.max(shares)) if shares.size else float("nan")
        warning_witness = {**witness, "reason": witness["reason"] if "rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES" in witness["reason"] else f"share {peak:.9g}"}
        _warn_if_ringdown_truncated(
            [settling_db], (), witnesses=[warning_witness],
            n_steps=n_steps, num_periods=None if n_steps is not None else num_periods,
            drive_labels=(f"S-channel {bin_label}, share {peak:.9g}",),
            consequence="S-parameter truncation is suspect")
        return result

    operation = "forward() call" if context == "forward" else "run"
    verdict = settling_verdict(settling_db)
    if verdict == "fail":
        _warn_if_ringdown_truncated(
            np.array([float(settling_db)]),
            (),
            n_steps=None if n_steps is None else int(n_steps),
            num_periods=None if n_steps is not None else num_periods,
            drive_labels=(
                f"worst probe record {witness['worst_record']}"
                + (_LONE_PROBE_NOTE if witness['worst_record']
                   in witness.get('lone_records', ()) else ""),),
            consequence=(
                f"every DFT-derived quantity of this {operation} — NTFF far "
                "fields, field-DFT planes, Harminv modes — integrates a "
                "cut transient"),
            quoted_thing="any of them",
        )
    elif verdict == "absent":
        import warnings
        warnings.warn(
            f"no ring-down settling witness on this {operation}: "
            f"{witness['reason']}. This {operation} requests NTFF and/or "
            "field-DFT output, so its truncation is unguarded; "
            "settling_db is None, which is not a pass.", AccuracyWarning,
            stacklevel=4,
        )
    elif witness.get("source_dominated"):
        # A RUNTIME warning, not a preflight advisory: the emission
        # contract (113 sites / 74 codes) is frozen, and this is a
        # fact about the finished record, not about the declared
        # inputs. Scoped exactly like the two above -- NTFF/field-DFT
        # runs only, silent on library-internal driver runs -- so it
        # cannot flood (#470). It fires only where the verdict PASSED
        # on a dominated record, which is the #1090 vacuous pass; a
        # dominated FAIL already warns above, in the safe direction.
        import warnings
        warnings.warn(
            f"ring-down settling witness on this {operation} is "
            f"source-dominated: {witness['qualifier']} Reported "
            f"{witness['worst_record']} = {float(settling_db):+.1f} dB "
            "is a source turn-off measurement, so this "
            f"{operation}'s NTFF/field-DFT output is effectively "
            "unguarded.", AccuracyWarning,
            stacklevel=4,
        )
    return result


def tfsf_source_drive(cfg, grid):
    """TF/SF waveform and reach from its realized launch through the box."""
    import jax.numpy as jnp

    def waveform(t):
        custom = getattr(cfg, "custom_waveform", None)
        if custom is not None:
            return cfg.src_amp * custom(t)
        arg = (t - cfg.src_t0) / cfg.src_tau
        env = jnp.exp(-(arg ** 2))
        kind = getattr(cfg, "src_waveform", "analytic")
        if kind == "modulated_gaussian":
            return cfg.src_amp * env * jnp.cos(2 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
        if kind == "continuous_wave":
            ramp = .5 * (1 - jnp.cos(jnp.pi * jnp.clip(t / cfg.src_t0, 0., 1.)))
            return cfg.src_amp * ramp * jnp.sin(2 * jnp.pi * cfg.src_fcen * t)
        if kind == "analytic":
            return cfg.src_amp * env * jnp.exp(-1j * 2 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
        return cfg.src_amp * (-2 * arg) * env

    is_traced = getattr(grid, "is_traced", None)
    if is_traced is not None and any(is_traced(axis) for axis in "xyz"):
        from rfx.measurement.record_end import TRACED
        return None, ("traced", TRACED)   # the box corners sit on traced nodes
    from rfx.measurement.tfsf_reach import tfsf_reach
    return waveform, tfsf_reach(cfg, grid)
