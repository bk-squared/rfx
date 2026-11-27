"""Warnings for concrete settling verdicts and their remedies."""

from __future__ import annotations
from rfx._warning_origin import user_stacklevel

from rfx.errors import AccuracyWarning
import numpy as np


def _warn_if_ringdown_truncated(
    settling_db: np.ndarray,
    port_names: tuple,
    *,
    num_periods: float | None = None,
    n_steps: int | None = None,
    drive_labels: tuple | None = None,
    consequence: str | None = None,
    quoted_thing: str = "any S value",
    witnesses: list | tuple | None = None,
    outputs_sentence: str = "",
) -> None:
    """Warn for excessive pole tails or concrete undetermined witnesses."""
    from rfx.sparams._common import settling_verdict, _SETTLING_WITNESS_DB

    values = np.atleast_1d(np.asarray(settling_db, dtype=float))
    hot = np.array([settling_verdict(v) == "fail" for v in values], dtype=bool)
    unknown = np.zeros(len(values), dtype=bool)
    if witnesses is not None:
        for i, row in enumerate(witnesses):
            if i < len(values) and 'share_per_bin' in row and 'traced' not in row['reason']:
                unknown[i] = row['status'] in {'undetermined', 'absent'}
    hot |= unknown
    if not bool(np.any(hot)):
        return

    import warnings

    if num_periods is not None:
        knob, record = "num_periods", f"num_periods={num_periods:g}"
    elif n_steps is not None:
        knob, record = "n_steps", f"n_steps={int(n_steps):d}"
    else:  # pragma: no cover
        knob, record = "the record length", "fixed-length"
    def _label(i: int) -> str:
        if drive_labels is not None:
            return drive_labels[i] if i < len(drive_labels) else str(i)
        label = f"port {port_names[i] if i < len(port_names) else i} driven"
        if witnesses is not None and i < len(witnesses):
            row = witnesses[i]
            error = np.asarray(row.get('record_end_error_per_bin', ()))
            kappa = np.asarray(row.get('kappa_per_bin', ()))
            if error.size and kappa.size:
                label += f", e_Q {float(np.max(error)):.8g}, kappa_Q {float(np.max(kappa)):.8g}"
        return label

    per_run = ", ".join(
        (f"{_label(i)}: {witnesses[i]['status']} ({witnesses[i]['reason']})"
         if unknown[i] else f"{_label(i)}: {values[i]:+.1f} dB")
        for i in np.flatnonzero(hot)
    )
    from rfx.measurement.record_end import merge_growing_poles

    poles = merge_growing_poles(*(witnesses[i].get('growing_poles', ())
                                 for i in np.flatnonzero(hot)
                                 if witnesses is not None and i < len(witnesses)))
    all_unknown = bool(np.all(unknown[hot]))
    consequence_text = f" {consequence}." if consequence and not all_unknown else ""
    if not poles and any(not unknown[i] for i in np.flatnonzero(hot)):
        consequence_text += " A failing record-end witness means lengthen the record."
    memory_refused = witnesses is not None and any(
        unknown[i] and 'rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES' in witnesses[i]['reason']
        for i in np.flatnonzero(hot))
    verdict = "NOT PASSED"
    if all_unknown:
        verdict = "cannot judge"
        if not poles and all(
                'rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES' in witnesses[i]['reason']
                for i in np.flatnonzero(hot)):
            verdict += " (identification not run)"
    advice = ("Check identification before quoting " if memory_refused else
              f"Increase {knob} and check identification before quoting ")
    drive = max((float(np.max(row['drive_share_per_bin'])) for i, row in enumerate(witnesses or ())
                 if i < len(hot) and hot[i] and len(row.get('drive_share_per_bin', ()))), default=0.)
    if drive > 0 and not poles:
        advice = (f"Up to {drive:.3g} of that share is the port's known drive beyond the record (a waveguide "
                  "port's H near cutoff decays as t^-3/2). Run a longer fixed record: until_decay stops on field "
                  "energy and does not judge this per-bin budget, and until_identified does not yet admit "
                  "waveguide ports. " + advice)
    if poles:
        named = "; ".join(
            f"{pole['frequency_hz'] / 1e9:.6g} GHz, e-folding {pole['e_folding_steps']:,.6g} steps, "
            f"max share {pole['max_share']:.6g}"
            for pole in poles[:3])
        if len(poles) > 3:
            named += f"; and {len(poles) - 3} more"
        advice = (
            "The recorded fields contain a term growing after the source ended. "
            "A passive structure cannot gain energy after its source ends. "
            "A longer record carries a larger error from this term. "
            f"Growing poles (largest growth first): {named}. "
            "No passing value witness supports quoting ")
    warnings.warn(
        f"ring-down settling witness {verdict} (pole-tail amplitude threshold "
        f"{_SETTLING_WITNESS_DB:.0f} dB): {per_run}. The {record} "
        f"record has no passing value witness.{consequence_text} {advice}"
        f"{quoted_thing} (see the result's settling_db and settling_witness fields)."
        f"{outputs_sentence}", AccuracyWarning,
        stacklevel=user_stacklevel(),
    )
