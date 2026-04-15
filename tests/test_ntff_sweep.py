"""Issue #43: streaming NTFF multi-frequency sweep.

Batched sweep must produce identical NTFFData to the single-run
sweep (same frequencies accumulated from the same source). Side
effect: sim._ntff.freqs restored on return.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.ntff_sweep import ntff_sweep


def _small_sim(freqs):
    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5, dtype=np.float64)
    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.01),
        dx=0.5e-3, dz_profile=dz, boundary="upml",
    )
    sim.add_source((0.01, 0.01, 0.004), "ez")
    sim.add_ntff_box(
        corner_lo=(0.004, 0.004, 0.002),
        corner_hi=(0.016, 0.016, 0.006),
        freqs=freqs,
    )
    return sim


def _close(a, b, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=1e-5)


def test_batched_equals_full():
    full_freqs = np.array([2.0e9, 2.4e9, 2.8e9, 3.2e9], dtype=np.float64)
    sim_ref = _small_sim(full_freqs)
    ref_res = sim_ref.run(n_steps=120, compute_s_params=False)
    ref_data = ref_res.ntff_data

    sim_batch = _small_sim(np.array([full_freqs[0]]))  # seed with one
    batched, freqs_out = ntff_sweep(
        sim_batch, full_freqs, batch_size=2,
        run_kwargs=dict(n_steps=120, compute_s_params=False),
    )
    np.testing.assert_array_equal(freqs_out, full_freqs)
    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
        ref_face = np.asarray(getattr(ref_data, face))
        batch_face = np.asarray(getattr(batched, face))
        assert ref_face.shape == batch_face.shape, (
            f"{face}: shape {ref_face.shape} != {batch_face.shape}")
        assert _close(ref_face, batch_face), (
            f"{face}: batched differs from full-run "
            f"max abs diff {np.max(np.abs(ref_face-batch_face))}")


def test_requires_ntff_box():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
                     cpml_layers=4)
    with pytest.raises(ValueError, match="add_ntff_box"):
        ntff_sweep(sim, [2.4e9])


def test_restores_freqs_on_success():
    original = np.array([2.4e9, 3.0e9])
    sim = _small_sim(original)
    ntff_sweep(sim, [2.0e9, 2.4e9], batch_size=1,
               run_kwargs=dict(n_steps=40, compute_s_params=False))
    restored = sim._ntff[2]
    np.testing.assert_array_equal(restored, original)


def test_restores_freqs_on_exception():
    original = np.array([2.4e9, 3.0e9])
    sim = _small_sim(original)
    # Force failure by passing an invalid run kwarg.
    with pytest.raises(TypeError):
        ntff_sweep(sim, [2.0e9], batch_size=1,
                   run_kwargs=dict(not_a_real_kwarg=True))
    restored = sim._ntff[2]
    np.testing.assert_array_equal(restored, original)
