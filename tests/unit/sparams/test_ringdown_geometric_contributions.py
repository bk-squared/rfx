"""F1b floating-point bound and F5 cancellation/phase falsifiers."""
from pathlib import Path

import numpy as np
import pytest

from rfx import ringdown as rd
from rfx.measurement.ringdown_tail import geometric_term_dft


def compare(s, coefficient, count, dt, bins, end_referenced):
    reference = count - 1 if end_referenced else 0
    time = np.arange(count) * dt
    samples = np.exp(s * (time - reference * dt))[:, None] * coefficient
    summed = rd.plain_dft(samples, dt, bins)
    closed = geometric_term_dft(s, coefficient, count, dt, bins,
                                 end_referenced=end_referenced)
    # Independent magnitude sum, from lambda and the reference sample.
    lam = abs(np.exp(s * dt))
    amplitude = dt * abs(coefficient) * np.sum(lam ** (np.arange(count) - reference))
    denominator = np.abs(rd.plain_dft(samples + .37, dt, bins))
    bound = 8 * count * 2**-53 * amplitude[None, :] / denominator
    difference = np.abs(closed - summed) / denominator
    ratio = float(np.max(difference / bound))
    print(f'F1b count={count} end={end_referenced} largest_ratio={ratio:.17g}')
    assert np.all(difference <= bound), f'F1b |v_closed-v_summed| exceeds bound: ratio={ratio}'
    return ratio


@pytest.mark.parametrize('end_referenced', [False, True])
@pytest.mark.parametrize('count', [37, 80, 210])
def test_closed_term_matches_summed_dft_within_f1b(count, end_referenced):
    dt = .1
    s = (.03 if end_referenced else .001) + 2j*np.pi*.7
    compare(s, np.array([.2+.1j, -.3+.4j]), count, dt,
            np.array([.7, .9, 1.3]), end_referenced)


def test_near_read_bin_numerator_needs_expm1():
    count, dt, bin_hz = 80, .1, .7
    spacing = 1/(count*dt)
    s = -5e-13/dt + 2j*np.pi*(bin_hz + 1e-9*spacing)
    compare(s, np.array([.2+.1j]), count, dt, np.array([bin_hz]), False)


def test_exact_read_bin_uses_geometric_limit():
    count, dt, bin_hz = 80, .1, .7
    compare(2j*np.pi*bin_hz, np.array([.2+.1j]), count, dt,
            np.array([bin_hz]), False)


def test_discarded_growing_210_sample_reference_stays_finite():
    path = Path(__file__).parents[2] / 'fixtures/settling_witness/waveguide_worst.npz'
    with np.load(path) as data:
        y = data['selected_record']
    dt, start, stop = 4.842700168608755e-12, 503, 713
    model = rd.identify(y, dt, start, stop, freq_max=11.6e9)
    assert np.max(model.s_growing.real)*dt*209 > 709
    window = np.asarray(y[start:stop]).reshape(210, -1)
    coefficients = rd._fit_residues(window, np.r_[model.s, model.s_growing], dt,
                                   column_reference=True)[len(model.s):]
    for s, c in zip(model.s_growing, coefficients):
        compare(s, c, 210, dt, np.array([8e9, 10e9, 11.6e9]), True)
