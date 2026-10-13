"""Result numeric-health diagnostics."""
import numpy as np


def _nonfinite_fields(result) -> list[tuple[str, int]]:
    """Return ``[(field_name, nonfinite_count), ...]`` for the numeric array
    observables on a ``Result`` / ``ForwardResult``.

    Tracer-safe: a value under ``jax.grad`` / ``jax.jit`` tracing is an
    abstract tracer with no concrete data, so it is skipped (returns ``[]``
    rather than raising). Never raises — a divergence diagnostic must not
    itself break the return path.
    """
    import jax

    bad: list[tuple[str, int]] = []
    for name in ("time_series", "s_params"):
        arr = getattr(result, name, None)
        if arr is None:
            continue
        try:
            if isinstance(arr, jax.core.Tracer):
                continue
            pairs = (getattr(result, "lumped_port_sparams", None)
                     or getattr(result, "wire_port_sparams", None))
            if name == "s_params" and pairs:
                from rfx.api._spec import ForwardResult
                from rfx.measurement.port_s import observed_port_s
                if isinstance(result, ForwardResult) or not hasattr(pairs[0][0], "impedance"):
                    arr = observed_port_s(arr, pairs)
            a = np.asarray(arr)
        except Exception:
            continue
        if a.size == 0 or not np.issubdtype(a.dtype, np.number):
            continue
        n_bad = int(a.size - np.count_nonzero(np.isfinite(a)))
        if n_bad:
            bad.append((name, n_bad))
    return bad

