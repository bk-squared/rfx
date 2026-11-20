"""Pin for smooth_grading(preserve_regions=...) — issue #48 / Meep/OpenEMS
thin-PEC convention.

Without preserve_regions, smooth_grading inflates a thin fine-cell block
(e.g. a substrate of 6 × 0.25mm) into a graded sequence that destroys
the user's intended geometry. The preserve_regions kwarg keeps the
protected block intact and only smooths the transitions outside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.auto_config import smooth_grading


def _raw_profile():
    """The profile represents a 0.25 mm substrate between 1 mm air cells."""
    return np.concatenate([
        np.full(12, 1e-3), np.full(6, 0.25e-3), np.full(25, 1e-3)
    ])


def test_default_smoothing_inflates_total_length():
    """Without preserve_regions, smooth_grading inserts transition cells that
    expand the total profile length. This is the real failure mode observed
    in issue #48 — geometry placed at absolute z coords no longer lines up
    with the intended fine cells."""
    dz = _raw_profile()
    total_raw = float(np.sum(dz))
    sm = smooth_grading(dz)
    total_sm = float(np.sum(sm))
    assert total_sm > total_raw + 1e-6, (
        f"raw total {total_raw*1e3:.3f} mm vs smoothed {total_sm*1e3:.3f} mm "
        "— smoothing did not expand total length; test is moot"
    )


def test_preserve_keeps_substrate_intact():
    """Cells on the interior side of each metal boundary follow the Meep/OpenEMS symmetric-cell convention."""
    dz = _raw_profile()
    sm = smooth_grading(dz, preserve_regions=[(12, 18)])

    fine_run = sm[np.isclose(sm, 0.25e-3, rtol=1e-6)]
    assert len(fine_run) == 6, (
        f"expected 6 fine cells preserved, got {len(fine_run)}")

    idx_first_fine = int(np.argmax(np.isclose(sm, 0.25e-3, rtol=1e-6)))
    idx_last_fine = idx_first_fine + 5
    assert np.allclose(
        sm[idx_first_fine:idx_last_fine + 1], 0.25e-3, rtol=1e-6
    )

    before = sm[:idx_first_fine]
    after = sm[idx_last_fine + 1:]
    assert before[-1] > 0.25e-3 and before[-1] < 1e-3
    assert after[0] > 0.25e-3 and after[0] < 1e-3


def test_preserve_respects_max_ratio_outside_block():
    dz = _raw_profile()
    sm = smooth_grading(dz, preserve_regions=[(12, 18)], max_ratio=1.3)
    ratios = sm[1:] / sm[:-1]
    assert ratios.max() <= 1.301, f"max ratio {ratios.max()} exceeded 1.30"
    assert (1 / ratios).max() <= 1.301


def test_preserve_invalid_region_raises():
    dz = _raw_profile()
    with pytest.raises(ValueError, match="outside"):
        smooth_grading(dz, preserve_regions=[(0, 1000)])
    with pytest.raises(ValueError, match="lo>=hi"):
        smooth_grading(dz, preserve_regions=[(5, 5)])


def test_preserve_none_matches_default():
    dz = _raw_profile()
    default = smooth_grading(dz)
    none = smooth_grading(dz, preserve_regions=None)
    empty = smooth_grading(dz, preserve_regions=[])
    np.testing.assert_array_equal(default, none)
    np.testing.assert_array_equal(default, empty)
