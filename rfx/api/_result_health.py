"""Result numeric-health diagnostics."""
import numpy as np


def _nonfinite_fields(result) -> list[tuple[str, int]]:
    """Return ``[(field_name, nonfinite_count), ...]`` for the numeric array
    observables on a ``Result`` / ``ForwardResult``.

    Tracer-safe: a value under ``jax.grad`` / ``jax.jit`` tracing is an
    abstract tracer with no concrete data, so it is skipped (returns ``[]``
    rather than raising). Concrete conversion and port-record defects propagate.
    """
    from jax.errors import TracerArrayConversionError
    from rfx.core.jax_utils import is_tracer

    bad: list[tuple[str, int]] = []
    for name in ("time_series", "s_params"):
        arr = getattr(result, name, None)
        if arr is None:
            continue
        if is_tracer(arr):
            continue
        try:
            from rfx.measurement.port_records import graded_record_pairs
            graded = hasattr(getattr(result, "grid", None), "dx_arr")
            pairs = (graded_record_pairs(result) if graded else
                     (getattr(result, "lumped_port_sparams", None)
                      or getattr(result, "wire_port_sparams", None)))
            if name == "s_params" and pairs:
                from rfx.api._spec import ForwardResult
                from rfx.measurement.port_s import observed_port_s
                if isinstance(result, ForwardResult) or graded or not hasattr(pairs[0][0], "impedance"):
                    arr = observed_port_s(arr, pairs)
            a = np.asarray(arr)
        except TracerArrayConversionError:
            continue
        if a.size == 0 or not np.issubdtype(a.dtype, np.number):
            continue
        n_bad = int(a.size - np.count_nonzero(np.isfinite(a)))
        if n_bad:
            bad.append((name, n_bad))
    return bad

