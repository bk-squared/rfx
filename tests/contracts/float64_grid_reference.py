"""Independent NumPy B-grid formulas, rfx#2613.

Signed error is actual minus reference. Both sides use the same declared
profiles, pads, constants and timestep. b and c use the realized inputs,
as specified by the revised design; no rfx numerical helper builds a reference.
"""
import numpy as np

from tests.contracts.float64_value_reference import assert_spacings

EPS0 = np.float64(8.8541878128e-12)
MU0 = np.float64(1.25663706212e-6)
U = 2.**-53
FACES = tuple(a+s for a in 'xyz' for s in ('_lo', '_hi'))
QUANTITIES = ('sigma', 'kappa', 'alpha', 'b', 'c')


def padded(profile, lo=8, hi=8):
    """The high padding includes the duplicated bounding cell."""
    return np.pad(np.asarray(profile, dtype=np.float64), (lo, hi+1), mode='edge')


def inverse_reference(widths):
    w = np.asarray(widths, dtype=np.float64)
    return np.r_[1/w[0], 2/(w[:-1]+w[1:])], np.r_[1/w[:-1], 0.]


def assert_scaled(actual, expected, count):
    """Absolute bar count*u*max(abs(reference)); signed error actual-reference."""
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.dtype('float64'), actual.dtype
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    error = actual-expected
    bar = count*U*np.max(np.abs(expected))
    assert np.all(np.abs(error) <= bar), (
        f'bar={bar}; signed error range=[{error.min()}, {error.max()}]')


def check_profile(profile, quantity, *, dt, width, n, high, magnetic):
    """Polynomial grading; revised realized-input exponential and recursion.

    The 13 u allowance for exp is an estimate for XLA exp, not a measured error.
    """
    i = np.arange(n, dtype=np.float64)
    offset = (-.5 if high else .5) if magnetic else 0.
    rho = 1-i/max(n-1, 1)
    if offset and n > 1:
        rho = np.clip((n-1-i-offset)/max(n-1, 1), 0., 1.)
    if high:
        rho = rho[::-1]
    peak = -np.log(1e-15)*4/(2*np.sqrt(MU0/EPS0)*(n*width))
    formulas = dict(sigma=peak*rho**3, kappa=np.ones(n), alpha=.05*(1-rho))
    actual = getattr(profile, quantity)
    if quantity in formulas:
        assert_spacings(actual, formulas[quantity], 8)
        return
    sigma, kappa, alpha, b = (np.asarray(getattr(profile, q), dtype=np.float64)
                              for q in ('sigma', 'kappa', 'alpha', 'b'))
    if quantity == 'b':
        argument = -(sigma/kappa+alpha)*np.float64(dt)/EPS0
        count = int(np.ceil(4*np.max(np.abs(argument))))+13+1
        assert_spacings(actual, np.exp(argument), count)
    else:
        denom = sigma*kappa+kappa**2*alpha
        expected = np.zeros_like(denom)
        np.divide(sigma*(b-1), denom, out=expected, where=denom > 1e-30)
        assert_scaled(actual, expected, 8)


def gradient_reference(profile, a, b, *, lo=8, hi=8):
    """Analytic derivative, including the leading E and zero trailing H.

    Pad and appended-cell copies accumulate into the respective endpoint.
    """
    w = padded(profile, lo, hi)
    grad = np.zeros_like(w)
    grad[0] = -a[0]/w[0]**2
    dual = -2*a[1:]/(w[:-1]+w[1:])**2
    grad[:-1] += dual
    grad[1:] += dual
    grad[:-1] -= b[:-1]/w[:-1]**2
    mapping = np.pad(np.arange(len(profile)), (lo, hi+1), mode='edge')
    result = np.zeros(len(profile), dtype=np.float64)
    np.add.at(result, mapping, grad)
    return result
