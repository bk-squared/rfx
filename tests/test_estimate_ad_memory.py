"""Issue #39 pin: estimate_ad_memory predictions must match observed
memory on the segmented scan-of-scan path within a tolerance.

Reference observations from VESSL job 369367233490 on RTX 4090:
  geometry: 2.4 GHz FR4 patch, dx=0.5mm NU (~608k cells)
  n_steps = 10000, emit_time_series=False

    checkpoint_every | peak GB
    ---------------- | -------
    50               | 4.82
    100              | 2.45
    200              | 1.26
    500              | 0.59
    1000             | 0.33

The formula `2 × n_segments × field_bytes + forward_bytes` fits these
points within ~25% (factor-of-2 accounts for carry + cotangent stacks
during reverse-mode).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rfx import Simulation


def _patch_like_sim():
    """Mirror the VESSL 369367233490 geometry: ext=40mm cube, dx=0.5mm,
    graded dz (0.3mm × 20 + 0.6mm × 30). Grid ends up 96 × 96 × 66 ≈ 608k.
    """
    ext = 40e-3
    dx = 0.5e-3
    dz = np.concatenate([np.full(20, 0.3e-3), np.full(30, 0.6e-3)])
    sim = Simulation(freq_max=10e9, domain=(ext, ext, float(np.sum(dz))),
                     dx=dx, dz_profile=dz, boundary="cpml", cpml_layers=8)
    sim.add_source((ext / 2, ext / 2, 1e-3), "ez")
    sim.add_probe((ext / 2, ext / 2, 2e-3), "ez")
    return sim


@pytest.mark.parametrize("chunk,observed_gb", [
    (50, 4.82),
    (100, 2.45),
    (200, 1.26),
    (500, 0.59),
    (1000, 0.33),
])
def test_segmented_estimate_within_tolerance(chunk, observed_gb):
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=10000, checkpoint_every=chunk)
    assert est.ad_segmented_gb is not None
    pred = est.ad_segmented_gb
    # Predictions should be within 2x (very loose to tolerate XLA
    # allocator slack); typical is ~1.2x.
    assert 0.5 * observed_gb <= pred <= 2.0 * observed_gb, (
        f"chunk={chunk}: predicted {pred:.3f} GB vs observed {observed_gb} GB"
    )


def test_checkpoint_every_none_leaves_segmented_null():
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=1000)
    assert est.ad_segmented_gb is None
    assert est.checkpoint_every is None


def test_monotone_in_chunk():
    """Bigger chunk → smaller segmented memory (fewer segment boundaries)."""
    sim = _patch_like_sim()
    gbs = [
        sim.estimate_ad_memory(n_steps=10000, checkpoint_every=c).ad_segmented_gb
        for c in [50, 100, 500, 1000]
    ]
    assert gbs == sorted(gbs, reverse=True), gbs
