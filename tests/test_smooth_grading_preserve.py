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
    # 12 × 1mm air / 6 × 0.25mm substrate / 25 × 1mm air
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
    dz = _raw_profile()
    sm = smooth_grading(dz, preserve_regions=[(12, 18)])

    # 1) The 6 protected cells appear verbatim.
    fine_run = sm[np.isclose(sm, 0.25e-3, rtol=1e-6)]
    assert len(fine_run) == 6, (
        f"expected 6 fine cells preserved, got {len(fine_run)}")

    # 2) Adjacent-cell ratio on the interior side of the boundary is 1.0
    #    (symmetric cells across the metal plane — Meep/OpenEMS convention).
    idx_first_fine = int(np.argmax(np.isclose(sm, 0.25e-3, rtol=1e-6)))
    idx_last_fine = idx_first_fine + 5
    # Sub_start: cells inside the block are all 0.25mm.
    assert np.allclose(
        sm[idx_first_fine:idx_last_fine + 1], 0.25e-3, rtol=1e-6
    )

    # 3) Transitions exist on BOTH sides (outside the block).
    before = sm[:idx_first_fine]
    after = sm[idx_last_fine + 1:]
    # Descending transition coming into the block
    assert before[-1] > 0.25e-3 and before[-1] < 1e-3
    # Ascending transition leaving the block
    assert after[0] > 0.25e-3 and after[0] < 1e-3


def test_preserve_respects_max_ratio_outside_block():
    dz = _raw_profile()
    sm = smooth_grading(dz, preserve_regions=[(12, 18)], max_ratio=1.3)
    ratios = sm[1:] / sm[:-1]
    # Outside the block, max ratio must be <= 1.3 (transitions).
    # INSIDE the block ratios are 1.0 by construction.
    # The first-contact step from transition into preserved block is
    # allowed (that's the whole point — user says "keep my fine cells").
    # So we only check global ratio ≤ 1.301 with tolerance.
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
