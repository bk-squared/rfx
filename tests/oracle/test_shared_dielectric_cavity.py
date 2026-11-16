"""PREDECLARE-kottke §§1–4: lossless two-layer cavity, independent comparator."""
import json
import time

import numpy as np
from scipy.optimize import brentq

from rfx import Box, GaussianPulse, Simulation, ringdown as rd
from rfx.measurement.ringdown_tail import completion_tail, plain_dft

A, B, D, T1 = .020, .013, .007, .0029
EPS = (3.38, 10.2)
FMAX = 12e9


def closed_form(mode):
    """Lowest transverse-resonance root; no product comparator helpers.

    Multiply the tan equation by cos(k1*t1)*cos(k2*t2) to avoid its poles.
    sin(k*t)/k is t at zero and sinh(|k|*t)/|k| for imaginary k.
    """
    kt2 = (np.pi / A)**2 + ((np.pi / B)**2 if mode == 'TM' else 0.)

    def determinant(f):
        q2 = [eps * (2 * np.pi * f / 299792458.)**2 - kt2 for eps in EPS]
        terms = []
        for squared, thickness in zip(q2, (T1, D - T1)):
            k = np.sqrt(abs(squared))
            if abs(squared) * thickness**2 < 1e-14:
                terms.append((1., thickness))
            elif squared > 0:
                terms.append((np.cos(k * thickness), np.sin(k * thickness) / k))
            else:
                terms.append((np.cosh(k * thickness), np.sinh(k * thickness) / k))
        (c1, s1), (c2, s2) = terms
        if mode == 'TM':
            return q2[0] / EPS[0] * s1 * c2 + q2[1] / EPS[1] * s2 * c1
        return s1 * c2 + s2 * c1

    points = np.linspace(0., 20e9, 2001)
    for lo, hi in zip(points[:-1], points[1:]):
        if determinant(lo) * determinant(hi) <= 0:
            return brentq(determinant, lo, hi, xtol=1e-3, rtol=1e-13)
    raise AssertionError('No lowest transverse-resonance root')


def measure_cavity(dx=.0005, mode='TM', periods=60):
    started = time.monotonic()
    reference = closed_form(mode)
    sim = Simulation(domain=(A, B, D), dx=dx, freq_max=FMAX,
                     boundary='pec', precision='float32')
    sim.add_material('lower', eps_r=EPS[0])
    sim.add_material('upper', eps_r=EPS[1])
    sim.add(Box((0., 0., 0.), (A, B, T1)), material='lower')
    sim.add(Box((0., 0., T1), (A, B, D)), material='upper')
    component = 'ez' if mode == 'TM' else 'ey'
    pulse = GaussianPulse(f0=reference, bandwidth=.5)
    sim.add_source((.37*A, .31*B, .27*D), component,
                   waveform=pulse, amplitude_kind='field')
    sim.add_probe((.61*A, .43*B, .71*D), component)
    geometry = sim.realized_geometry()
    for axis, length in zip(geometry.domain, (A, B, D)):
        assert abs(axis.length_m - length) < 1e-12, 'PEC wall moved'
    offset = float(np.min(np.abs(np.asarray(geometry.nodes[2]) - T1)) / dx)
    assert offset > .05, 'interface must be off-grid'
    dt = float(geometry.dt)
    n = int(np.ceil((pulse.t0 + 8*pulse.tau + periods/reference) / dt))
    result = sim.forward(n_steps=n, subpixel_smoothing=True)
    series = np.asarray(result.time_series, dtype=float).reshape(n, -1)
    assert np.isfinite(series).all()
    assert np.max(np.abs(np.asarray(pulse(np.arange(n//4, n)*dt)))) <= 1e-6
    model = rd.identify(series, dt, n//2, n, freq_max=FMAX)
    poles = model.s.imag / (2*np.pi)
    candidates = poles[(poles > .75*reference) & (poles < 1.25*reference)]
    assert len(candidates), 'No pole near the predeclared lowest root'
    frequency = float(min(candidates, key=lambda f: abs(f-reference)))
    bins = np.array([frequency])
    plain = plain_dft(series, dt, bins)
    completed = plain + completion_tail(model, n-1, bins)
    norm = max(float(np.max(np.abs(completed))), np.finfo(float).tiny)
    witness = rd.early_start_witness(
        series, dt, bins, n, .5, freq_max=FMAX, plain=plain, spectra=completed,
        observable=lambda x: x/norm)
    row = dict(mode=mode, dx_mm=dx*1e3, periods=periods, steps=n,
               reference_hz=reference, frequency_hz=frequency,
               error_percent=100*(frequency/reference-1),
               identification_WE=float(witness.value),
               identification_tolerance=rd.RingdownSpec().witness_tol,
               wall_seconds=time.monotonic()-started)
    print('SHARED_FACE_CAVITY ' + json.dumps(row), flush=True)
    return row


def test_tm_shared_dielectric_cavity():
    row = measure_cavity()
    assert row['identification_WE'] <= row['identification_tolerance'], row
    assert abs(row['error_percent']) < 1., row
