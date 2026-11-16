"""#1826: short-record open-ended WR-90 on constant-cell twin meshes.

The scene's 12/24-layer sensitivity is measured separately before pinning
the predeclared 0.01 dB / 0.1 degree parity bar. No normalization run.
"""
import warnings

import jax
import numpy as np

from rfx import Box, Simulation


def scene(graded, layers=12, margin=6):
    dx = .00254
    ny, nz = 11 + 2 * margin, 6 + 2 * margin
    kwargs = {"dz_profile": np.full(nz, dx)} if graded else {}
    sim = Simulation(11e9, (32 * dx, ny * dx, nz * dx), dx=dx,
                     cpml_layers=layers, boundary="cpml", **kwargs)
    y0, y1 = (margin + 1) * dx, (margin + 10) * dx
    z0, z1 = (margin + 1) * dx, (margin + 5) * dx
    for lo, hi in (
        ((0, y0 - dx, z0 - dx), (16 * dx, y0, z1 + dx)),
        ((0, y1, z0 - dx), (16 * dx, y1 + dx, z1 + dx)),
        ((0, y0 - dx, z0 - dx), (16 * dx, y1 + dx, z0)),
        ((0, y0 - dx, z1), (16 * dx, y1 + dx, z1 + dx)),
    ):
        sim.add(Box(lo, hi), material="pec")
    sim.add_waveguide_port(4 * dx, direction="+x", y_range=(y0, y1), z_range=(z0, z1),
                           f0=10e9, bandwidth=.6, freqs=np.array([9e9, 10e9, 11e9]),
                           ref_offset=1, probe_offset=2, name="p1")
    return sim


def test_partial_aperture_uniform_graded_parity():
    values = []
    try:
        for graded in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = scene(graded).run(n_steps=240, skip_preflight=True)
                values.append(np.asarray(result.waveguide_sparams["p1"].s11))
        assert all(np.all(np.isfinite(value)) for value in values)
        ratio = values[0] / values[1]
        db = np.abs(20 * np.log10(np.abs(ratio)))
        degrees = np.abs(np.angle(ratio, deg=True))
        assert np.all(db <= .01), db
        assert np.all(degrees <= .1), degrees
    finally:
        jax.clear_caches()


def test_partial_aperture_uniform_margin_invariance():
    """An open guide's reflection is independent of the lateral domain margin."""
    values = []
    try:
        for margin in (6, 10):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = scene(False, margin=margin).run(n_steps=240, skip_preflight=True)
                values.append(np.asarray(result.waveguide_sparams["p1"].s11))
        assert all(np.all(np.isfinite(value)) for value in values)
        ratio = values[0] / values[1]
        db = np.abs(20 * np.log10(np.abs(ratio)))
        degrees = np.abs(np.angle(ratio, deg=True))
        assert np.all(db <= .01), db
        assert np.all(degrees <= .1), degrees
    finally:
        jax.clear_caches()
