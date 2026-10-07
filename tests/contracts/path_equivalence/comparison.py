"""The S0 cross-trace bars, including exact prerequisites."""
from dataclasses import fields, is_dataclass
from math import isfinite, sqrt
from numbers import Real

import numpy as np

from rfx.core.yee import MU_0, EPS_0

VACUUM_IMPEDANCE = sqrt(MU_0 / EPS_0)

STEP_ULPS = 9
ACCUMULATED_RELATIVE = 1e-4


# Far-field transforms consume the six running sums, never these Kahan carries
# (rfx/farfield.py:127). Exclude by name, not by magnitude or a c_* wildcard.
NTFF_KAHAN_RESIDUALS = frozenset((
    'c_x_lo', 'c_x_hi', 'c_y_lo', 'c_y_hi', 'c_z_lo', 'c_z_hi',
))
# Configuration and geometric weights are exact prerequisites, not amplitudes.
OBSERVER_METADATA = frozenset((
    'freqs', 'component', 'axis', 'index', 'total_steps', 'window',
    'window_alpha', 'region', 'lo1', 'hi1', 'lo2', 'hi2', 'dA', 'dA2',
))
NAMED_RECORD_COLLECTIONS = frozenset(('dft_planes', 'flux_monitors'))


def container(value):
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if hasattr(value, '_asdict'):
        return value._asdict()
    return value


def excluded(name, key):
    return name == 'ntff_data' and key in NTFF_KAHAN_RESIDUALS


def metadata(name, key):
    return name.split('.')[0] in NAMED_RECORD_COLLECTIONS and key in OBSERVER_METADATA


def array_peak(*arrays):
    """Maximum magnitude on both traces (also accepts a field's components)."""
    return max((float(np.max(np.abs(np.asarray(a)), initial=0))
                for a in arrays if a is not None), default=0.0)


COMPONENT_GROUPS = (('ex', 'ey', 'ez'), ('hx', 'hy', 'hz'),
                    ('e1_dft', 'e2_dft'), ('h1_dft', 'h2_dft'))


FIELD_PAIRS = {
    ('ex', 'ey', 'ez'): ('hx', 'hy', 'hz'),
    ('e1_dft', 'e2_dft'): ('h1_dft', 'h2_dft'),
}


def _positive_float(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('impedance must be finite and positive, and a real number')
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError('impedance must be finite and positive')
    return value


def wave_impedance_range(eps_r_max):
    """Scene impedance bounds for nonmagnetic dielectrics including vacuum."""
    eps_r_max = _positive_float(eps_r_max)
    if eps_r_max < 1:
        raise ValueError('eps_r_max must be at least 1')
    return VACUUM_IMPEDANCE / sqrt(eps_r_max), VACUUM_IMPEDANCE


def paired_field_peaks(e_peak, h_peak, paired_impedance):
    """Convert partner peaks in SI units, without propagating nonfinite bars."""
    if paired_impedance is None:
        return e_peak, h_peak
    if isinstance(paired_impedance, tuple) and len(paired_impedance) == 2:
        z_low, z_high = map(_positive_float, paired_impedance)
    else:
        z_low = z_high = _positive_float(paired_impedance)
    if z_low > z_high:
        raise ValueError('paired_impedance requires z_low <= z_high')
    paired_e, paired_h = z_low * h_peak, e_peak / z_high
    return (max(e_peak, paired_e) if isfinite(paired_e) else e_peak,
            max(h_peak, paired_h) if isfinite(paired_h) else h_peak)


def component_peaks(a, b, *, paired_impedance=None):
    """Share sibling peaks; optionally pair using caller-supplied SI impedance.

    A scalar Z uses max(E, Z * H), max(H, E / Z). A (z_low, z_high)
    range uses max(E, z_low * H), max(H, E / z_high). An absent partner
    is unchanged. No scene impedance is assumed by default.
    """
    if paired_impedance is not None:
        paired_field_peaks(0., 0., paired_impedance)  # Validate even without fields.
    peaks = {}
    for group in COMPONENT_GROUPS:
        keys = set(group) & (a.keys() | b.keys())
        if keys:
            peak = array_peak(*(tree[key] for tree in (a, b)
                                for key in keys if key in tree))
            peaks.update(dict.fromkeys(keys, peak))
    if paired_impedance is not None:
        for electric, magnetic in FIELD_PAIRS.items():
            e_keys, h_keys = set(electric) & peaks.keys(), set(magnetic) & peaks.keys()
            if e_keys and h_keys:
                e_peak = max(peaks[key] for key in e_keys)
                h_peak = max(peaks[key] for key in h_keys)
                e_peak, h_peak = paired_field_peaks(e_peak, h_peak, paired_impedance)
                peaks.update(dict.fromkeys(e_keys, e_peak))
                peaks.update(dict.fromkeys(h_keys, h_peak))
    return peaks


def compare(a, b, *, record, kind, measurements, peak=None):
    """Record the evidence before asserting; absent records never pass."""
    if a is None or b is None:
        side = 'A and B' if a is None and b is None else 'A' if a is None else 'B'
        raise AssertionError(f'{record}: record missing on {side}')
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{record}: shape differs: {a.shape} vs {b.shape}"
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{record}: nonfinite"
    leaf_peak = array_peak(a, b)
    peak = leaf_peak if peak is None else peak
    difference = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128)), initial=0))
    bar = (0.0 if kind == 'exact' else
           STEP_ULPS * float(np.spacing(np.float32(peak))) if kind == 'step' else
           ACCUMULATED_RELATIVE * peak)
    relative = difference / peak if peak else (0.0 if difference == 0 else float('inf'))
    measurements.append(dict(record=record, kind=kind, difference=difference,
                             peak=peak, leaf_peak=leaf_peak, relative=relative, bar=bar,
                             relative_bar=bar / peak if peak else 0.0))
    assert difference <= bar, f"{record}: relative diff {relative:.9g}; absolute diff {difference:.9g} > bar {bar:.9g}"
