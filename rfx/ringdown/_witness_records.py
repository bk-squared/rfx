"""Concrete longer-record pair reading and checks for gradient witnesses."""

import numpy as np


def _read_witness_records(ringdown):
    """Read a pair, refusing wrong types, batched results and traced arrays."""
    from collections.abc import Sequence
    from rfx.core.jax_utils import is_tracer
    from rfx.ringdown import RingdownForwardResult

    if (not isinstance(ringdown, Sequence) or len(ringdown) != 2
            or not all(isinstance(res, RingdownForwardResult) for res in ringdown)):
        raise TypeError("ringdown must be a pair of RingdownForwardResult objects")
    for res in ringdown:
        if np.ndim(res._channels) != 2:
            raise ValueError(
                "RingdownForwardResult.pole reads one result; this one is batched "
                f"(its port channels have shape {tuple(np.shape(res._channels))}, "
                "one record is (n_steps, channels)), as jax.vmap returns it. Take "
                "one element first: jax.tree_util.tree_map(lambda a: a[i], result).")
    if any(is_tracer(a) for res in ringdown for a in (
            res._status, res.poles, res.poles_long, res.pole_mask, res.pole_mask_long)):
        raise TypeError("gradient_witness reads concrete results; call it outside jax.jit")
    return ringdown


def _check_witness_records(records, pole_target):
    """Return the cannot-judge note or None, and the judged record note."""
    from rfx.ringdown import _jax as rj, _status_reason, pole_selection_reason

    lengths = [res._ctx.plan.n_steps * res._ctx.plan.grid.dt * 1e9 for res in records]
    labels = [f"{role}'s record ({length:.4g} ns)"
              for role, length in zip(("grad", "grad_other"), lengths)]
    windows = [res._pole_window_s() for res in records]
    reasons, selected = [], []
    for res, label, window in zip(records, labels, windows):
        code = int(res._status[0])
        reason = None
        if code != rj.STATUS_OK:
            reason = "failed window: " + _status_reason(
                code, kept_count=int(res._pole_counts[0]),
                pole_budget=int(res._pole_budgets[0]))
        elif pole_target is not None:
            reason = pole_selection_reason(
                res.poles, res.pole_mask, pole_target, window, status=None)
        if reason:
            radius = f" (radius {1e-6 / window:.4g} MHz)" if pole_target is not None else ""
            reasons.append(f"{label}: {reason}{radius}")
        elif pole_target is not None:
            selected.append(tuple(map(float, res.pole(pole_target))))
    if reasons:
        return "cannot judge: " + "; ".join(reasons), ""
    ratio = lengths[1] / lengths[0]
    if pole_target is not None:
        (f1, q1), (f2, q2) = selected
        if abs(f1 - f2) > 1 / max(windows):
            return f"cannot judge (selection differs: {f1:.6g} Hz vs {f2:.6g} Hz)", ""
        record_note = (f"selected at pole_target={pole_target:.6g} Hz: "
                       f"{labels[0]}: f={f1:.6g} Hz, 1/Q={q1:.4g}; "
                       f"{labels[1]}: f={f2:.6g} Hz, 1/Q={q2:.4g}; "
                       f"record ratio {ratio:.3g}")
    else:
        record_note = f"records: {labels[0]}; {labels[1]}; record ratio {ratio:.3g}"
    return None, record_note
