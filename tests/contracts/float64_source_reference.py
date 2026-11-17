"""B-src NumPy references, retyped from the declared expressions (rfx#2613).

Errors are actual minus reference, with identical geometry, realized inputs,
and selected nodes on both sides. u = 2**-53. Each exp allowance is an
estimate of 13 u; no reference calls an implementation numerical helper.
"""
import numpy as np

U = 2.**-53
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


def assert_value(actual, expected, units, *, scale=None):
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.dtype('float64'), actual.dtype
    expected = np.broadcast_to(expected, actual.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = np.max(np.abs(expected)) if scale is None else scale
    error = actual - expected
    bar = units * U * scale
    print(f'B_SRC_VALUE units={units} scale={scale} '
          f'error_min={error.min(initial=0)} error_max={error.max(initial=0)}')
    assert np.all(np.abs(error) <= bar), (error.min(), error.max(), bar)


def pulse(t, tau, cutoff, *, t0=None):
    t = np.asarray(t, dtype=np.float64)
    u, c = t / tau, np.float64(cutoff)
    uc = np.clip(u, -50., 50.)
    arg = np.clip((t - (c*tau if t0 is None else t0)) / tau, -50., 50.)
    onset = np.exp(-c**2 - uc**2) * (
        2*c + (4*c**2 - 2)*uc - 4*c*uc**2 - 4*c**2*uc**3)
    far = -2*arg*np.exp(-arg**2) - onset
    z = np.clip(2*c*u, 0., 3.)
    term, r1, r2 = z**2/2, 0., 0.
    for k in range(2, 22):
        r1 = r1 + term
        r2 = r2 + (term if k >= 3 else 0.)
        term = term*z/(k+1)
    near = np.exp(-c**2-uc**2)*(2*c*r1 - 2*uc*r2)
    return np.where(t >= 0, np.where(2*c*u < 3, near, far), 0.)


def gaussian(t, waveform):
    tau = 1. / (np.float64(waveform.f0)*waveform.bandwidth*np.pi)
    return waveform.amplitude*pulse(t, tau, waveform.cutoff, t0=waveform.cutoff*tau)


def profile(n, dt, dx, order, kappa_max, reflection, offset):
    eta = float(np.sqrt(MU0/EPS0))
    sigma_max = -float(np.log(reflection))*(order+1)/(2.*eta*(n*dx))
    sigma_max = sigma_max*kappa_max
    rho = 1. - np.arange(n, dtype=np.float64)/max(n-1, 1)
    if offset != 0 and n > 1:
        rho = np.clip((n-1-np.arange(n, dtype=np.float64)-offset)/max(n-1, 1), 0., 1.)
    sigma = sigma_max*rho**order
    kappa = 1. + (kappa_max-1.)*rho**order
    alpha = .05*(1.-rho)
    arg = -(sigma/kappa+alpha)*dt/EPS0
    return sigma, kappa, alpha, arg


def noisy_or(occupancy):
    """Two transverse passes, zero-padding OCCUPANCY at each low face."""
    masks = []
    for c in range(3):
        m = np.asarray(occupancy, dtype=np.float64).copy()
        for axis in range(3):
            if axis == c:
                continue
            shifted = np.zeros_like(m)
            dst, src = [slice(None)]*3, [slice(None)]*3
            dst[axis], src[axis] = slice(1, None), slice(None, -1)
            shifted[tuple(dst)] = m[tuple(src)]
            m = 1.-(1.-m)*(1.-shifted)
        masks.append(m)
    return masks


def wire_distance(points, coords, *, mutated=False):
    """The mutation narrows segment data only; coordinates remain float64."""
    points = np.asarray(points, dtype=np.float64)
    a, d = points[:-1], np.diff(points, axis=0)
    lengths = np.sum(d**2, axis=1)
    if mutated:
        a, d, lengths = (v.astype(np.float32).astype(np.float64) for v in (a, d, lengths))
    x, y, z = (np.asarray(v, dtype=np.float64).reshape(
        tuple(len(v) if b == axis else 1 for b in range(3))) for axis, v in enumerate(coords))
    distance = np.full((len(coords[0]), len(coords[1]), len(coords[2])), np.inf)
    for (ax, ay, az), (dx, dy, dz), length in zip(a, d, lengths):
        t = ((x-ax)*dx+(y-ay)*dy+(z-az)*dz)/length
        t = np.clip(t, 0., 1.)
        dist = (x-(ax+t*dx))**2+(y-(ay+t*dy))**2+(z-(az+t*dz))**2
        distance = np.minimum(distance, dist)
    return distance
