"""Settling measurements from named port records."""

from __future__ import annotations

import numpy as np


_SETTLING_RECORD_NAMES = ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t")


def _settling_record_floor_amplitude(tiny_normal: float) -> float:
    """Smallest record peak amplitude on which the settling decision is still
    made in the storage format's NORMAL range.

    ``floor = tiny_normal * 10 ** (|bar| / 20)``, with ``bar`` the one shared
    ring-down threshold (``rfx.api._sparams._SETTLING_WITNESS_DB``, -40 dB)
    and ``tiny_normal`` the smallest normal of the format the records were
    stored in (``np.finfo(dtype).tiny``): 1.1754944e-38 for float32, so the
    float32 floor is 1.1754944e-36.

    Derivation, not taste. The witness compares a tail mean POWER against a
    peak POWER, so a record sitting exactly at the bar has
    ``end = 1e-4 * peak``, i.e. a tail whose rms AMPLITUDE is
    ``1e-2 * peak_amplitude``. Setting that rms equal to ``tiny_normal``
    gives ``peak_amplitude = 100 * tiny_normal``. Above the floor the bar
    decision is arithmetic on normal-range numbers; below it the tail the
    decision reads is subnormal (progressively fewer mantissa bits, and
    exactly zero once the format flushes).

    The coupling to the bar is deliberate: the floor exists so that the
    comparison the witness actually makes is representable. Tighten the bar
    and the floor rises with it.
    """
    from rfx.api._sparams import _SETTLING_WITNESS_DB
    return float(tiny_normal) * 10.0 ** (abs(float(_SETTLING_WITNESS_DB)) / 20.0)


def _settling_db_for_record(peak_power: float,
                            end_power: float,
                            floor_amplitude: float) -> float | None:
    """End/peak tail ratio of ONE record in dB, or ``None`` when the record
    carries no witnessable ring-down.

    ``None`` means the record's peak amplitude is below ``floor_amplitude``
    (see :func:`_settling_record_floor_amplitude`) -- including a record that
    underflowed to exactly zero. The caller drops such a record from the
    worst-over-records and reports it by name; it must never be scored.

    The arithmetic is ``10*log10((end + tiny)/(peak + tiny))``. The public
    record scorer tests the storage floor in amplitude units first, then
    supplies normalized powers and a zero floor here to avoid overflowing
    or underflowing a square. The regularizer then bounds an unrepresentable
    tail ratio independently of the record's amplitude units.
    """
    if not (peak_power > 0.0):
        return None
    if float(np.sqrt(peak_power)) < floor_amplitude:
        return None
    tiny = float(np.finfo(float).tiny)
    return float(10.0 * np.log10((end_power + tiny) / (peak_power + tiny)))


def settling_db_from_named_records(named_records, *, dt=None, freqs=None,
                                   freq_max=None, source_end_index=None,
                                   record_noun="port records", return_detail=False,
                                   _warn_stacklevel=2):
    """Adapt the per-bin pole-tail value witness to legacy dB report fields.

    sensitivities are judged by a separate witness
    """
    from rfx.sparams._tail_witness import tail_share_witness

    named = tuple(named_records)
    per_record = {}
    result = tail_share_witness(named, dt, source_end_index, freqs, freq_max=freq_max,
                                _record_results=per_record if return_detail else None)
    if not return_detail:
        return result.db
    return result.db, {**result._asdict(), "per_record_db": per_record,
                       "n_witnessed": int(sum(np.isfinite(v) for v in per_record.values())),
                       "worst_record": max(per_record, key=per_record.get) if per_record else None,
                       "skipped_records": [], "floor_amplitude": float("nan")}


def settling_db_from_port_records(final_cfgs, *, freqs=None, freq_max=None,
                                   return_detail: bool = False):
    """Per-bin pole-tail value witness over the four modal records per port.

    sensitivities are judged by a separate witness
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.measurement.modal import recorded as _recorded
    from rfx.probes.settling import source_end_step

    final_cfgs = tuple(final_cfgs)
    timing_fields = ("v_inc_t", "src_amp", "source_x_m", "reference_x_m", "probe_x_m", "dt")
    timing_available = all(
        hasattr(cfg, name) and not is_tracer(getattr(cfg, name))
        for cfg in final_cfgs for name in timing_fields)
    drives = []
    source_end = None
    if (final_cfgs and timing_available
            and len({getattr(cfg, "normal_axis", "x") for cfg in final_cfgs}) == 1):
        # Use the recorded incident pulse with the same amplitude threshold.
        # Include free-space travel to every scored plane.
        for cfg in final_cfgs:
            if not cfg.src_amp:
                continue
            distance = max(abs(float(cfg.source_x_m) - float(x))
                           for other in final_cfgs
                           for x in (other.reference_x_m, other.probe_x_m))
            drives.append((_recorded(cfg, "v_inc_t"), distance))
        if drives:
            source_end = source_end_step(drives, len(drives[0][0]), final_cfgs[0].dt)
    named = [
        (f"port{port_index}/{name}", _recorded(cfg, name))
        for port_index, cfg in enumerate(final_cfgs)
        for name in _SETTLING_RECORD_NAMES
    ]
    bins = freqs if freqs is not None else (
        getattr(final_cfgs[0], "freqs", None) if final_cfgs else None)
    if freq_max is None and bins is not None and not is_tracer(bins):
        freq_max = float(np.max(bins)) if np.size(bins) else None
    return settling_db_from_named_records(
        named, source_end_index=source_end,
        dt=getattr(final_cfgs[0], "dt", None) if final_cfgs else None,
        freqs=bins, freq_max=freq_max, record_noun="port records",
        return_detail=return_detail, _warn_stacklevel=3)
