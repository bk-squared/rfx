"""Regression pins for issue #763 — `_make_dz_profile` must realize the
mesh it declares.

Pre-fix production (main @ b29f9de) returned
``smooth_grading(cells, max_ratio=1.3)`` with no ``preserve_regions``
(rfx/auto_config.py), so transition cells inflated the declared z-column
and broke the substrate-top snap. Measured pre-fix on the demo fixture
below (reproduced in the fix worktree BEFORE the fix was applied, matching
issue #763's adversarial-reviewer numbers exactly):

  nz = 24, sum(dz) = 2.3800 mm (declared 1.754 mm),
  dz_min = 21.167 um, substrate-top interface mid-cell at fraction 0.3462
  (inside preflight's own [0.10, 0.40] mixed-cell danger zone).

These tests pin the pre-declared fix-validation falsifiers (a)-(c) from
docs/design_notes/issue763_dz_profile_preserve_regions.md and are
revert-proof: reverting the fix reproduces the numbers above and fails
every assertion in the demo-fixture tests.

Profile level only — no FDTD.
"""

from __future__ import annotations

import numpy as np

from rfx.auto_config import _make_dz_profile

# Snap tolerance pre-declared in the design note (never widened): 1e-12 m.
ATOL = 1e-12

# Demo fixture from the graded-z closure note / issue #763: W = 6*h_sub
# low-Z MSL board, h_sub = 254 um, eps_r 3.38, dx = W/8, margin 1.5 mm.
H_SUB = 254e-6
DX = 6 * H_SUB / 8            # 190.5 um
PHYS_Z = H_SUB + 1.5e-3       # 1.754 mm declared column


def _demo_profile() -> np.ndarray:
    return np.asarray(
        _make_dz_profile([(0.0, H_SUB, 3.38)], PHYS_Z, DX), dtype=float
    )


def test_demo_fixture_interface_on_node():
    """Falsifier (a): substrate-top edge within 1e-12 m of declared 254 um.

    Pre-fix: interface mid-cell at fraction 0.3462 (nearest edge tens of
    microns away)."""
    dz = _demo_profile()
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    dist = float(np.min(np.abs(edges - H_SUB)))
    assert dist <= ATOL, (
        f"substrate-top edge missed by {dist*1e6:.3f} um "
        f"(declared {H_SUB*1e6:.1f} um)"
    )


def test_demo_fixture_protected_block_bit_identical():
    """Falsifier (b): post-thirds substrate cells appear verbatim.

    The builder constructs the block as 4 x 63.5 um, then the thirds rule
    splits the top cell into [2/3, 1/3]. Those exact float64 values must
    survive smoothing bit-identically (np.array_equal, no tolerance)."""
    dz = _demo_profile()
    dz_feat = H_SUB / 4  # 63.5 um, exactly as the builder computes it
    expected_block = np.array(
        [dz_feat, dz_feat, dz_feat, dz_feat * 2 / 3, dz_feat / 3]
    )
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    j = int(np.argmin(np.abs(edges - H_SUB)))
    assert np.array_equal(dz[:j], expected_block), (
        f"protected block not bit-identical: {dz[:j]} vs {expected_block}"
    )
    # dt anchor: the profile minimum is the block's thirds cell, unchanged
    # from pre-fix (21.167 um) — the fix does not shrink the timestep here.
    assert float(np.min(dz)) == float(np.min(expected_block))


def test_demo_fixture_column_length_matches_declared():
    """Falsifier (c): |sum(dz) - 1.754 mm| <= 1e-12 m.

    Pre-fix: realized 2.3800 mm, 0.626 mm over the declaration."""
    dz = _demo_profile()
    err = abs(float(np.sum(dz)) - PHYS_Z)
    assert err <= ATOL, f"column length off by {err*1e3:.6f} mm"


def test_demo_fixture_air_run_ratio_discipline():
    """Outside the protected block the smoothing contract still holds:
    adjacent-cell ratios <= 1.3 within the air run (first-contact step at
    the block edge exempt, per the preserve_regions convention)."""
    dz = _demo_profile()
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    j = int(np.argmin(np.abs(edges - H_SUB)))
    air = dz[j:]
    ratios = air[1:] / air[:-1]
    max_r = float(np.max(np.maximum(ratios, 1.0 / ratios)))
    assert max_r <= 1.301, f"air-run ratio {max_r:.3f} exceeds 1.3"


def test_generic_two_layer_all_interfaces_on_nodes():
    """Generic fixture from the design note: two-layer stack with an
    interior air gap. ALL four declared interfaces land on cell edges and
    the total column equals the declaration, within 1e-12 m."""
    feats = [(0.2e-3, 0.5e-3, 4.0), (1.1e-3, 1.35e-3, 2.2)]
    domain_z = 3.0e-3
    dz = np.asarray(_make_dz_profile(feats, domain_z, 0.3e-3), dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    for coord in (0.2e-3, 0.5e-3, 1.1e-3, 1.35e-3):
        dist = float(np.min(np.abs(edges - coord)))
        assert dist <= ATOL, (
            f"interface {coord*1e3} mm missed by {dist*1e6:.3f} um"
        )
    assert abs(float(np.sum(dz)) - domain_z) <= ATOL


def test_generic_two_layer_blocks_bit_identical():
    """Falsifier (b) on the generic fixture: each feature block's
    post-thirds cells survive verbatim."""
    feats = [(0.2e-3, 0.5e-3, 4.0), (1.1e-3, 1.35e-3, 2.2)]
    dz = np.asarray(_make_dz_profile(feats, 3.0e-3, 0.3e-3), dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    for z_lo, z_hi, _eps in feats:
        thickness = z_hi - z_lo
        n_feat = max(4, int(np.ceil(thickness / 0.3e-3)))
        d = thickness / n_feat
        # thirds rule splits BOTH edge cells (air below and above):
        # [1/3, 2/3, interior..., 2/3, 1/3]
        expected = np.array(
            [d / 3, d * 2 / 3] + [d] * (n_feat - 2) + [d * 2 / 3, d / 3]
        )
        i = int(np.argmin(np.abs(edges - z_lo)))
        j = int(np.argmin(np.abs(edges - z_hi)))
        assert np.array_equal(dz[i:j], expected), (
            f"block ({z_lo}, {z_hi}) not bit-identical:\n"
            f"got {dz[i:j]}\nexpected {expected}"
        )


def test_no_features_path_unchanged():
    """With no z-features the uniform early-return path is untouched."""
    dz = np.asarray(_make_dz_profile([], 3.0e-3, 0.3e-3), dtype=float)
    assert np.array_equal(dz, np.full(10, 0.3e-3))
