"""A 380.6-cell guide must continue the same plates through its rounded end."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests.oracle import test_leontovich_alpha_oracle as oracle
from tests.unit.materials.test_sheet_continuation import _arrays
from tests.unit.materials.test_sheet_impedance import PlanarSheet


def _rounded_guide(nonuniform, nonbox):
    dx, lx, ly = oracle.DX, .1903, .0023  # x=380.6 and y=4.6 cells, both round up
    sim = Simulation(freq_max=10e9, domain=(lx, ly, .006), dx=dx,
                     boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"), y="pmc", z="pec"),
                     cpml_layers=10,
                     **({"dz_profile": np.full(12, dx)} if nonuniform else {}))
    for i in range(oracle.ABSORBER_N):
        sigma = oracle.ABSORBER_SIGMA_MAX * ((i + .5) / oracle.ABSORBER_N) ** 2
        sim.add_material(f"abs{i}", eps_r=1., sigma=sigma)
        sim.add(Box(((oracle.ABSORBER_N - i - 1) * dx, 0., 0.),
                    ((oracle.ABSORBER_N - i) * dx, ly, .006)), material=f"abs{i}")
    for z in (oracle.Z_SHEET_LO, oracle.Z_SHEET_HI):
        shape = (PlanarSheet(2, z, (0., 0.), (lx, ly)) if nonbox
                 else Box((0., 0., z), (lx, ly, z)))
        sim.add_thin_conductor(shape, sigma_bulk=oracle.SIGMA_BULK,
                               thickness=oracle.THICKNESS, surface_impedance_f0=oracle.F0)
    for k in range(9):
        sim.add_source((.180, .001, .001 + k * dx), "ez",
                       waveform=GaussianPulse(f0=oracle.F0, bandwidth=.5), amplitude_kind="field")
    sim.add_dft_plane_probe(axis="z", coordinate=.003, component="ez",
                            freqs=jnp.asarray([oracle.F0]), name="midplane")
    sim.add_probe((.120, .001, .003), "ez")
    return sim


@pytest.mark.parametrize("nonuniform", [False, True], ids=["uniform", "nu"])
def test_noninteger_guide_has_identical_arrays_and_attenuation(nonuniform, record_property):
    arrays, alphas = [], []
    try:
        for nonbox in (False, True):
            sim = _rounded_guide(nonuniform, nonbox)
            with warnings.catch_warnings(record=True) as caught:
                grid, coords, actual = _arrays(sim, nonuniform)
                result = sim.run(n_steps=oracle.N_STEPS, compute_s_params=False)
            assert grid.shape == (392, 6, 13)
            assert not any("no geometry continuation" in str(w.message) for w in caught)
            arrays.append(actual)
            acc = np.asarray(result.dft_planes["midplane"].accumulator)
            start = int(np.argmin(abs(coords.x - .065)))
            end = int(np.argmin(abs(coords.x - .165)))
            j = int(np.argmin(abs(coords.y - .001)))
            alpha, residual = oracle._fit_alpha(-coords.x[start:end+1], np.abs(acc[0, start:end+1, j]))
            assert np.isfinite(alpha) and alpha > 0
            assert residual < .02
            alphas.append(alpha)
            record_property("nonbox_alpha_np_m" if nonbox else "box_alpha_np_m", alpha)
        for name in arrays[0]:
            np.testing.assert_array_equal(arrays[0][name], arrays[1][name], err_msg=name)
        np.testing.assert_allclose(alphas[0], alphas[1], rtol=2e-6, atol=1e-9)
    finally:
        jax.clear_caches()
