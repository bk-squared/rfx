"""B4R/B6: re-declaration and an independent float64 TE Airy oracle."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.unit.autodiff.test_forward_tfsf_angle_gradient import (
    F0, N2, declared, emit, functions, gradient, run, scene, transmission,
)

pytestmark = pytest.mark.slow


def airy(theta, thickness):
    """Float64 NumPy TE transmission, no implementation angle/optics helpers."""
    theta = np.deg2rad(np.float64(theta))
    k0 = np.float64(2*np.pi*F0/299792458.)
    kx0 = k0*np.cos(theta)
    kx1 = k0*np.sqrt(np.float64(4)-np.sin(theta)**2)
    r = (kx0-kx1)/(kx0+kx1)
    return (1-r*r)*np.exp(-1j*kx1*thickness)/(1-r*r*np.exp(-2j*kx1*thickness))


def airy_derivative(theta, thickness):
    step = np.float64(1e-4)
    return float((abs(airy(theta+step, thickness))-abs(airy(theta-step, thickness)))/(2*step))


def redeclared_value(pol, theta, h, n, *, layers=16):
    pair = []
    for slab in (True, False):
        sim, facts = scene(pol, theta, slab=slab, h=h, thickness=.024, layers=layers)
        pair.append(run(sim, n).time_series)
    return float(transmission(jnp.stack(pair), facts["dt"]))


@pytest.mark.parametrize("pol", ["ez", "ey"])
def test_b4r_redeclaration_derivative(pol):
    dt = functions(pol)[4]["dt"]
    fd = (float(transmission(jnp.asarray(declared(pol, 30.25)), dt)) -
          float(transmission(jnp.asarray(declared(pol, 29.75)), dt)))/.5
    ad = gradient(pol)
    emit("B4R", pol=pol, n=N2, ad=ad, fd=fd, relative=abs(ad-fd)/abs(fd))
    assert abs(ad-fd) <= 1e-3*abs(fd), "B4R AD versus re-declaration FD"


@pytest.mark.parametrize("pol", ["ez", "ey"])
def test_b6_airy_mesh_trend(pol):
    values, derivatives = [], []
    for h in (.002, .001, .0005):
        _, facts = scene(pol, h=h, thickness=.024)
        # The construction's first prefix: N_long / 1.75, quoted as 6.9 ns.
        n = int(math.ceil(12e-9/facts["dt"])/1.75)
        _, fn, _, compiled, _ = functions(pol, n, h, .024)
        value = float(compiled(jnp.float32(30)))
        _, tangent = jax.jvp(fn, (jnp.float32(30),), (jnp.float32(1),))
        ad = float(tangent)
        fd = (redeclared_value(pol, 30.25, h, n)-redeclared_value(pol, 29.75, h, n))/.5
        reference = float(abs(airy(30., facts["thickness"])))
        derivative = airy_derivative(30., facts["thickness"])
        padded = {.002: .7595029419210677, .001: .7519246022624299, .0005: .7500933902129858}[h]
        values.append(abs(value-reference))
        derivatives.append(abs(ad-derivative))
        emit("B6", pol=pol, h=h, n=n, record_seconds=n*facts["dt"], facts=facts,
             value=value, padded_construction=padded, difference_from_padded=value-padded, airy=reference, airy_declared=float(abs(airy(30., .024))),
             error=value-reference, ad=ad, fd=fd, airy_derivative=derivative,
             derivative_error=ad-derivative, fd_relative=abs(ad-fd)/abs(fd))
        assert abs(ad-fd) <= 1e-3*abs(fd), "B6(ii) AD versus re-declaration FD"
        jax.clear_caches()
    ratios = [values[i]/values[i+1] for i in (0, 1)]
    emit("B6_trend", pol=pol, magnitude_errors=values, ratios=ratios, derivative_errors=derivatives)
    assert all(3.5 <= ratio <= 4.6 for ratio in ratios), "B6(i) Airy convergence ratio"
    assert derivatives[-1] < derivatives[0], "B6(iii) derivative refinement"
    assert derivatives[-1] <= .005*abs(derivative), "B6(iii) fine derivative error"
