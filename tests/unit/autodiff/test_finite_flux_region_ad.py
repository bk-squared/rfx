"""Material AD through the raw port hook that actually consumes finite flux.

This short numerical referee checks differentiation and run/forward parity,
not RF accuracy or settling of an antenna/network.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation, flux_spectrum
from rfx.sources.sources import GaussianPulse
from tests._x64_compat import enable_x64


def test_clamped_flux_in_raw_port_hook_preserves_material_ad_and_primal():
    with enable_x64():
        sim = Simulation(freq_max=30e9, domain=(.008,) * 3, dx=.001,
                         cpml_layers=2, boundary="cpml", precision="float64")
        sim.add_port(position=(.002, .004, .004), component="ez", impedance=50.,
                     waveform=GaussianPulse(f0=20e9, bandwidth=.8))
        freqs = np.array([20e9])
        sim.add_flux_monitor(axis="x", coordinate=.004, freqs=freqs,
                             size=(.016, .016), name="finite")
        grid = sim._build_grid()
        mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
        mats = mats._replace(eps_r=mats.eps_r.astype(jnp.float64),
                              sigma=mats.sigma.astype(jnp.float64),
                              mu_r=mats.mu_r.astype(jnp.float64))

        def observable(alpha):
            changed = mats._replace(eps_r=mats.eps_r * alpha)
            result = sim._forward_from_materials(
                grid, changed, None, None, n_steps=80,
                port_s11_freqs=freqs, _return_raw_port_sparams=True,
            )
            monitor = result["flux_monitors"]["finite"]
            assert (monitor.lo1, monitor.hi1, monitor.lo2, monitor.hi2) == (2, 10, 2, 10)
            return flux_spectrum(monitor)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value, gradient = jax.jit(jax.value_and_grad(observable))(jnp.float64(1.))
            h = 1e-3
            fd = (float(observable(jnp.float64(1. + h)))
                  - float(observable(jnp.float64(1. - h)))) / (2 * h)
            run = sim.run(n_steps=80, compute_s_params=False, skip_preflight=True)
            imperative = float(flux_spectrum(run.flux_monitors["finite"])[0])
        assert np.isfinite([value, gradient, fd, imperative]).all()
        assert abs(fd) > 1e-5 * abs(float(value)) > 0
        np.testing.assert_allclose(gradient, fd, rtol=3e-4, atol=0)
        # The imperative material assembly stores float32 constants even on
        # a float64-field sim. Here they are exactly 1/0/1 on both paths.
        np.testing.assert_allclose(value, imperative, rtol=1e-6, atol=0)
