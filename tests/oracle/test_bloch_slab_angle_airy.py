"""Slab transmission and its angle derivative against re-declaration and float64 TE Airy theory."""
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
def test_angle_derivative_matches_redeclaration(pol):
    dt = functions(pol)[4]["dt"]
    fd = (float(transmission(jnp.asarray(declared(pol, 30.25)), dt)) -
          float(transmission(jnp.asarray(declared(pol, 29.75)), dt)))/.5
    ad = gradient(pol)
    emit("redeclared_transmission_derivative", pol=pol, n=N2, ad=ad, fd=fd, relative=abs(ad-fd)/abs(fd))
    assert abs(ad-fd) <= 1e-3*abs(fd), "Transmission angle derivative versus re-declaration finite difference"


@pytest.mark.parametrize("pol", ["ez", "ey"])
def test_slab_transmission_and_derivative_converge_to_airy(pol):
    """Each mesh reads a settled pulse record of about 6.9 ns."""
    values, derivatives = [], []
    for h in (.002, .001, .0005):
        _, facts = scene(pol, h=h, thickness=.024)
        n = int(math.ceil(12e-9/facts["dt"])/1.75)
        _, fn, _, compiled, _ = functions(pol, n, h, .024)
        value = float(compiled(jnp.float32(30)))
        _, tangent = jax.jvp(fn, (jnp.float32(30),), (jnp.float32(1),))
        ad = float(tangent)
        fd = (redeclared_value(pol, 30.25, h, n)-redeclared_value(pol, 29.75, h, n))/.5
        reference = float(abs(airy(30., facts["thickness"])))
        derivative = airy_derivative(30., facts["thickness"])
        values.append(abs(value-reference))
        derivatives.append(abs(ad-derivative))
        emit("slab_airy_error", pol=pol, h=h, n=n, record_seconds=n*facts["dt"], facts=facts,
             value=value, airy=reference, airy_declared=float(abs(airy(30., .024))),
             error=value-reference, ad=ad, fd=fd, airy_derivative=derivative,
             derivative_error=ad-derivative, fd_relative=abs(ad-fd)/abs(fd))
        assert abs(ad-fd) <= 1e-3*abs(fd), "Transmission angle derivative versus re-declaration finite difference"
        jax.clear_caches()
    ratios = [values[i]/values[i+1] for i in (0, 1)]
    emit("slab_airy_mesh_convergence", pol=pol, magnitude_errors=values, ratios=ratios, derivative_errors=derivatives)
    assert all(3.5 <= ratio <= 4.6 for ratio in ratios), "Slab transmission error must converge quadratically to Airy theory"
    assert derivatives[-1] < derivatives[0], "Transmission angle derivative error must decrease with mesh refinement"
    assert derivatives[-1] <= .005*abs(derivative), "Fine-mesh transmission angle derivative differs from Airy theory"


def smoothed_off_grid_record(theta, slab):
    from rfx import Box, Simulation

    h = .002
    sim = Simulation(freq_max=10e9, domain=(.30, .016, 4*h), dx=h,
                     mode="3d", cpml_layers=16,
                     boundary={"x": "cpml", "y": "periodic", "z": "periodic"})
    sim.add_tfsf_source(f0=F0, bandwidth=.15, direction="+x", angle_deg=theta,
                        waveform="modulated_gaussian", polarization="ez", method="bloch")
    grid = sim.freeze_mesh()
    front = grid.node_of(0, grid.cpml_layers+sim._tfsf.margin)+.040+.3*h
    thickness = .0246
    nodes = np.array([grid.node_of(0, i) for i in range(grid.shape[0])])
    assert np.min(abs(nodes-front)) > 1e-12
    assert np.min(abs(nodes-front-thickness)) > 1e-12
    if slab:
        sim.add_material("slab", eps_r=4.)
        sim.add(Box((front, 0, 0), (front+thickness, .016, 4*h)), material="slab")
    sim.add_probe((front+.045, .3*.016, 2*h), component="ez")
    n = math.ceil(12e-9/grid.dt/1.75)
    result = sim.run(n_steps=n, subpixel_smoothing=True, skip_preflight=False)
    freqs = np.arange(50e6, 10e9+1, 50e6, dtype=np.float64)
    weights = np.exp(-2j*np.pi*freqs[:, None]*np.arange(n, dtype=np.float64)*grid.dt)
    return freqs, weights @ np.asarray(result.time_series)[:, 0]


def test_smoothed_off_grid_oblique_slab_adds_no_transmission_error():
    errors = []
    for theta0 in (0., 30.):
        freqs, slab = smoothed_off_grid_record(theta0, True)
        _, vacuum = smoothed_off_grid_record(theta0, False)
        band = abs(vacuum) >= .1*np.max(abs(vacuum))
        f = freqs[band]
        # Both source polarizations are TE; transverse wavenumber is fixed at f0.
        k0 = 2*np.pi*f/np.float64(299792458.)
        kt = 2*np.pi*F0/np.float64(299792458.)*np.sin(np.deg2rad(theta0))
        kx0, kx1 = np.sqrt(k0*k0-kt*kt), np.sqrt(4*k0*k0-kt*kt)
        r = (kx0-kx1)/(kx0+kx1)
        propagation = np.exp(-1j*kx1*.0246)
        reference = abs((1-r*r)*propagation/(1-r*r*propagation*propagation))
        value = abs(slab[band]/vacuum[band])
        errors.append(float(np.max(abs(value-reference))))
        emit("smoothed_off_grid_transmission", theta=theta0, f=f.tolist(),
             value=value.tolist(), airy=reference.tolist(), max_error=errors[-1])
    assert errors[1] <= max(3*errors[0], .02)
