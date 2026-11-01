"""Normalized transmission and reflection share the declared port planes."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box


pytestmark = pytest.mark.slow


FREQS = np.linspace(4.5e9, 8.0e9, 20)
F0 = float(FREQS.mean())
BW = max(0.2, min(0.8, (FREQS[-1] - FREQS[0]) / F0))
C0 = 299_792_458.0
_COARSE_DX = C0 / FREQS[-1] / 20.0
_LAMBDA_G_LOW = C0 / FREQS[0] / np.sqrt(1 - (3.476e9 / FREQS[0]) ** 2)
CPML_LAYERS = int(np.ceil(0.75 * _LAMBDA_G_LOW / _COARSE_DX))


def build_simulation(pair, *, graded=False, slab=True):
    """Copy of the two-port slab builder, with equal cells on the graded lane."""
    dx = 0.75e-3
    sim = Simulation(
        freq_max=float(FREQS[-1]), domain=(0.12, 54 * dx, 27 * dx), dx=dx,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=CPML_LAYERS,
        dz_profile=np.full(27, dx) if graded else None,
    )
    if slab:
        sim.add_material("diel", eps_r=4.0, sigma=0.0)
        sim.add(Box((0.05, 0.0, 0.0), (0.07, 0.04, 0.02)), material="diel")
    for position, direction, name, reference in (
        (0.01, "+x", "left", pair[0]), (0.09, "-x", "right", pair[1]),
    ):
        sim.add_waveguide_port(
            position, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(FREQS), f0=F0, bandwidth=BW,
            ref_offset=3, probe_offset=15, name=name,
            reference_plane=reference / 1000,
        )
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_normalized_one_plane_set(graded):
    matrices = []
    for pair in ((10, 90), (20, 80)):
        result = build_simulation(pair, graded=graded).compute_waveguide_s_matrix(
            num_periods=40, normalize=True,
        )
        s = np.asarray(result.s_params, dtype=np.complex128)
        matrices.append(s)
        # A passive lossless slab cannot amplify any coherent incident vector.
        sigma_max = np.linalg.svd(np.moveaxis(s, 2, 0), compute_uv=False)[:, 0]
        assert np.all(sigma_max - 1 <= 1e-2), (pair, sigma_max)
        assert result.s21_phase_residual_deg_rms is not None
    rotations = np.angle(matrices[1] * np.conj(matrices[0]))
    # Wrap the doubled residual before dividing, never a reflection angle.
    doubled = 2 * rotations[1, 0] - rotations[0, 0] - rotations[1, 1]
    residual_deg = np.degrees(np.angle(np.exp(1j * doubled))) / 2
    assert np.all(np.abs(residual_deg) <= 2), residual_deg


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_empty_guide_preserves_single_run_transmission(graded):
    sim = build_simulation((10, 90), graded=graded, slab=False)
    normalized = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
    # The graded lane refuses False; equal cells admit the uniform single run.
    single_sim = build_simulation((10, 90), slab=False) if graded else sim
    single = single_sim.compute_waveguide_s_matrix(num_periods=40, normalize=False)
    s_true = np.asarray(normalized.s_params, dtype=np.complex128)
    s_false = np.asarray(single.s_params, dtype=np.complex128)
    for i, j in ((1, 0), (0, 1)):
        assert np.max(np.abs(s_true[i, j] - s_false[i, j])) <= (
            1e-5 * np.max(np.abs(s_false[i, j]))
        )
    assert np.array_equal(s_true[[0, 1], [0, 1]], np.zeros((2, len(FREQS))))
