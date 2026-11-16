"""Graded power quadrature against an analytic mode integral, not its weights."""
import numpy as np

from rfx import Simulation, GaussianPulse
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
from rfx.probes.probes import flux_spectrum
from tests.unit.boundaries.test_pmc_mirror_judges import record


def test_graded_te10_flux_matches_fitted_mode_integral():
    width = .020
    dy = np.r_[1., np.linspace(.57, 1.83, 15), 1.] * .001
    sim = Simulation(freq_max=15e9, domain=(.060, width, .004), dx=.001,
                     dy_profile=dy, cpml_layers=16,
                     boundary=BoundarySpec(x='cpml', y='pec', z='pec'))
    sim.add_source((.008, .007, .002), 'ez', amplitude_kind='field',
                   waveform=GaussianPulse(f0=10.5e9, bandwidth=.5))
    freqs = np.array([9e9, 10e9, 11e9, 12e9])
    sim.add_flux_monitor(axis='x', coordinate=.036, freqs=freqs, name='power')
    result = sim.run(n_steps=3000, skip_preflight=True, compute_s_params=False)
    mon = result.flux_monitors['power']
    grid = sim._build_realized_grid()
    y = np.asarray(coords_from_nonuniform_grid(grid).y)
    # Ez*Hy is nodal in y, edge-centered in z. Sum over z at the
    # physical uniform height; the PEC high ghost contributes zero.
    product = -np.real(np.asarray(mon.e2_dft, dtype=np.complex128)
                       * np.conj(np.asarray(mon.h1_dft, dtype=np.complex128)))
    profile = product.sum(axis=2) * .001
    shape = np.sin(np.pi*y/width)**2
    inside = (y > 0) & (y < width)
    coefficient = (profile[:, inside] @ shape[inside]) / np.sum(shape[inside]**2)
    exact = coefficient * width/2
    fit = coefficient[:, None]*shape
    fit_residual = np.linalg.norm(profile-fit, axis=1)/np.linalg.norm(profile, axis=1)
    actual = np.asarray(flux_spectrum(mon, exact_f64=True))
    old = profile @ np.asarray(grid.dy_arr, dtype=np.float64)
    ratio = actual/exact
    record('te10-integral', dict(frequencies_Hz=freqs.tolist(), fit_residual=fit_residual.tolist(),
           flux=actual.tolist(), exact=exact.tolist(), ratio=ratio.tolist(),
           primal_ratio=(old/exact).tolist(), after_before=(actual/old).tolist()))
    assert np.all(exact > 0), 'propagating TE10 must carry positive x power'
    assert np.all(fit_residual < .02), fit_residual
    # Predeclared from the independent review: <0.3% with dual widths,
    # >3% with primal widths; 1% separates them without pinning a formula.
    np.testing.assert_allclose(ratio, 1., rtol=.01, atol=0.)
