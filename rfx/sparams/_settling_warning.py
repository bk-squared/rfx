"""Warnings for concrete settling verdicts and their remedies."""

from __future__ import annotations

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
    else:  # pragma: no cover - guarded by the call sites
        knob, record = "the record length", "fixed-length"
    def _label(i: int) -> str:
        if drive_labels is not None:
            return drive_labels[i] if i < len(drive_labels) else str(i)
        return f"port {port_names[i] if i < len(port_names) else i} driven"

    per_run = ", ".join(
        (f"{_label(i)}: {witnesses[i]['status']} ({witnesses[i]['reason']})"
         if unknown[i] else f"{_label(i)}: {values[i]:+.1f} dB")
        for i in np.flatnonzero(hot)
    )
    consequence_text = f" {consequence}." if consequence else ""
    memory_refused = witnesses is not None and any(
        unknown[i] and 'rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES' in witnesses[i]['reason']
        for i in np.flatnonzero(hot))
    advice = ("Check identification before quoting " if memory_refused else
              f"Increase {knob} and check identification before quoting ")
    # The share is one budget: pole tail plus a port's known drive beyond the record (#2030).
    drive = max((float(np.max(row['drive_share_per_bin'])) for i, row in enumerate(witnesses or ())
                 if i < len(hot) and hot[i] and len(row.get('drive_share_per_bin', ()))), default=0.)
    if drive > 0:
        advice = (f"Up to {drive:.3g} of that share is the port's known drive beyond the record (a waveguide "
                  "port's H near cutoff decays as t^-3/2). Run a longer fixed record: until_decay stops on field "
                  "energy and does not judge this per-bin budget, and until_identified does not yet admit "
                  "waveguide ports. " + advice)
    warnings.warn(
        "ring-down settling witness NOT PASSED (pole-tail amplitude threshold "
        f"{_SETTLING_WITNESS_DB:.0f} dB): {per_run}. The {record} "
        f"record has no passing value witness.{consequence_text} {advice}"
        f"{quoted_thing} (see the result's settling_db and settling_witness fields).", AccuracyWarning,
        stacklevel=2,
    )
