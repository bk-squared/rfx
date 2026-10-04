"""The S0 cross-trace bars, including exact prerequisites."""
import numpy as np

STEP_ULPS = 9
ACCUMULATED_RELATIVE = 1e-4


def compare(a, b, *, record, kind, measurements):
    """Record the evidence before asserting; absent records never pass."""
    if a is None or b is None:
        side = 'A and B' if a is None and b is None else 'A' if a is None else 'B'
        raise AssertionError(f'{record}: record missing on {side}')
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{record}: shape differs: {a.shape} vs {b.shape}"
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{record}: nonfinite"
    peak = max(float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
    difference = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128)), initial=0))
    bar = (0.0 if kind == 'exact' else
           STEP_ULPS * float(np.spacing(np.float32(peak))) if kind == 'step' else
           ACCUMULATED_RELATIVE * peak)
    relative = difference / peak if peak else (0.0 if difference == 0 else float('inf'))
    measurements.append(dict(record=record, kind=kind, difference=difference,
                             peak=peak, relative=relative, bar=bar,
                             relative_bar=bar / peak if peak else 0.0))
    assert difference <= bar, f"{record}: relative diff {relative:.9g}; absolute diff {difference:.9g} > bar {bar:.9g}"
