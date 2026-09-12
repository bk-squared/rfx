"""Host diagnostics for already-recorded user probes, shared by run/forward."""
from __future__ import annotations

import warnings

import jax
import numpy as np

from rfx.core.jax_utils import is_tracer


_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")


def probe_record_info(time_series, entries, internal_indices=()):
    """Numeric provenance survives JAX result trees without string leaves."""
    shape = getattr(time_series, "shape", ())
    count = 1 if len(shape) == 1 else shape[1] if len(shape) == 2 else 0
    internal = internal_indices or ()
    result = []
    # forward() may create a source-position fallback record when the user
    # registered no probe. Only declared probe columns have user provenance;
    # an extra backend record must not silently provide this witness.
    for col in range(min(count, len(entries))):
        if col in internal:
            continue
        component = getattr(entries[col], "component", "?") if col < len(entries) else "?"
        code = _COMPONENTS.index(component) if component in _COMPONENTS else -1
        result.append((col, code))
    return tuple(result)


def probe_record_settling_witness(time_series, probe_info=None, *, warn=True):
    """Return (dB or None, provenance) using the canonical record arithmetic.

    None provenance selects all columns with unknown component labels; an
    empty tuple selects none. Traced data are explicitly unavailable. A
    concrete result can be scored after JIT without adding this host work to
    the differentiated solver. ``warn=False`` makes repeated property reads
    quiet; the execution entry point retains the canonical coverage warning.
    """
    from rfx.sources.waveguide_port import settling_db_from_named_records

    def absent(reason, skipped=()):
        return None, {"status": "absent", "route": None,
                      "worst_record": None, "per_record_db": {},
                      "skipped_records": list(skipped), "reason": reason}

    advice = ("add a point probe (sim.add_probe(position, component)) and "
              "retain its time series, or use run(until_decay=...) to "
              "bound the ring-down through its stop criterion")
    if is_tracer(time_series) or any(is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(probe_info)):
        return absent("probe records or their selection are traced; inspect "
                      "the concrete result after JIT/AD evaluation")
    if time_series is None:
        return absent("this result returned no probe time series: " + advice)
    series = np.asarray(time_series)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2 or series.size == 0:
        return absent("this result recorded no probe time series: " + advice)
    info = (tuple((i, -1) for i in range(series.shape[1]))
            if probe_info is None else probe_info)
    if not info:
        return absent("no user probe records are selected; library-internal "
                      "witness probes are judged by their own driver: " + advice)
    named = []
    for col, component in info:
        col, component = int(col), int(component)
        if not 0 <= col < series.shape[1]:
            raise ValueError("settling probe metadata indexes a missing record column")
        label = _COMPONENTS[component] if 0 <= component < len(_COMPONENTS) else "?"
        named.append((f"probe{col}({label})", series[:, col]))
    with warnings.catch_warnings():
        if not warn:
            warnings.filterwarnings("ignore", message="ring-down settling witness has (NO COVERAGE|INVALID RECORDS)", category=UserWarning)
        worst, detail = settling_db_from_named_records(
            named, record_noun="probe records", return_detail=True,
            _warn_stacklevel=4,
        )
    skipped = list(detail["skipped_records"])
    per_record = dict(detail["per_record_db"])
    invalid = detail.get("invalid_records", {})
    if invalid:
        value, witness = absent(
            "selected probe records are invalid: "
            + "; ".join(f"{name}: {reason}" for name, reason in invalid.items())
            + "; fix the non-finite result before using a settling verdict", skipped)
        witness["invalid_records"] = dict(invalid)
        return value, witness
    if not np.isfinite(worst):
        return absent(
            "no probe record carries a witnessable ring-down (records below "
            "the underflow floor are skipped rather than scored: "
            f"{', '.join(skipped) or 'none'}; records shorter than 10 samples "
            "are also unwitnessable): " + advice, skipped,
        )
    worst_record = max(per_record, key=lambda key: per_record[key])
    return float(worst), {"status": "measured", "route": "probe_records",
                          "worst_record": worst_record,
                          "per_record_db": per_record,
                          "skipped_records": skipped, "reason": ""}
