"""Preflight advisory for Boxes displaced from a graded-mesh fine band."""

from __future__ import annotations

import numpy as np

from rfx import Box, Simulation


def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in issue for issue in issues)


def _graded_sim(z_lo: float, z_hi: float) -> Simulation:
    dz = np.array([1.0e-3, 1.0e-3] + [0.25e-3] * 6 + [1.0e-3] * 2)
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, float(np.sum(dz))),
        dx=1e-3,
        dz_profile=dz,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, z_lo), (15e-3, 15e-3, z_hi)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")
    return sim


def test_shifted_box_warns_with_actual_and_implied_counts():
    sim = _graded_sim(0.6e-3, 2.1e-3)

    issues = _issues(sim)

    # 2, not 1: the advisory reports what the RUN realizes, and the run puts
    # this box on the nodes at z = 1.0 and 2.0 mm (measured on the production
    # rasterize path). The former "1" came from the validator modelling the
    # rasterizer with cell centres — of which exactly one, z = 1.5 mm, fell in
    # the span. #562 made coordinates nodes and this validator now calls
    # Box.mask_on_coords instead of imitating it, so the number is the true
    # one. The ADVISORY still fires, which is what this test is about: 2 is
    # still below ceil(0.5 * 6.0) = 3.
    assert _has(issues, "rasterizes to 2 z cells (implied 6.0)"), issues
    assert _has(issues, "smooth_grading transition cells may have shifted"), issues


def test_box_pinned_to_actual_fine_band_is_silent():
    sim = _graded_sim(2.0e-3, 3.5e-3)

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")


def test_uniform_dz_simulation_skips_check():
    sim = Simulation(
        freq_max=10e9,
        domain=(20e-3, 20e-3, 6e-3),
        dx=1e-3,
        cpml_layers=2,
    )
    sim.add_material("substrate", eps_r=3.5)
    sim.add(Box((5e-3, 5e-3, 0.5e-3), (15e-3, 15e-3, 2.0e-3)),
            material="substrate")
    sim.add_source((10e-3, 10e-3, 1e-3), "ez")

    assert not _has(_issues(sim), "smooth_grading transition cells may have shifted")


# --------------------------------------------------------------------------- #
# The validator MODELS the rasterizer, so it has to agree with it (#562 F2).
# --------------------------------------------------------------------------- #
_AGREEMENT_CASES = [
    # (z_lo_mm, z_hi_mm) — the 4.50-5.50 case is the reviewer's: it straddles a
    # grading transition, is classified THIN by Box.mask_on_coords (extent ==
    # one local cell) and so realizes ONE plane, and the hand-rolled
    # centre-model counted three and stayed silent where it should warn.
    (4.50, 5.50),
    (5.00, 6.00),
    (5.00, 7.00),
    (4.00, 5.00),
    (5.25, 6.75),
    (0.00, 5.00),
]


def _real_rasterized_z_count(sim) -> int:
    """The z-cell count the RUN actually produces, from the production path."""
    import numpy as _np
    from rfx.geometry.rasterize_grid import (rasterize_geometry,
                                             coords_from_nonuniform_grid)
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    out = rasterize_geometry(sim._geometry, sim._resolve_material, coords,
                             pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    eps = _np.asarray(getattr(out[0], "eps_r", out[0]))
    # the substrate is the only non-background material in these fixtures
    return int(_np.count_nonzero((eps > 1.0 + 1e-6).max(axis=(0, 1))))


def _validator_counts(sim):
    """(reported cell count, reported `implied`) or (None, None) when silent."""
    for issue in _issues(sim):
        if "rasterizes to" in issue:
            count = int(issue.split("rasterizes to")[1].split()[0])
            implied = float(issue.split("(implied")[1].split(")")[0])
            return count, implied
    return None, None


def _implied_cells(dz, z_lo, z_hi):
    """The validator's own `implied` figure: span thickness over the finest
    cell in a +-5-cell neighbourhood of the span (the #325 shifted-band
    recipe). Duplicated here on purpose so the SILENT direction can be
    justified rather than skipped; it is cross-checked against the
    validator's reported value on every case that fires.
    """
    edges = np.concatenate(([0.0], np.cumsum(dz)))
    local = (edges[:-1] < z_hi) & (edges[1:] > z_lo)
    idx = np.flatnonzero(local)
    lo_i = max(0, int(idx[0]) - 5)
    hi_i = min(dz.size, int(idx[-1]) + 6)
    return (z_hi - z_lo) / float(np.min(dz[lo_i:hi_i]))


import math  # noqa: E402
import pytest  # noqa: E402


@pytest.mark.parametrize("z_lo_mm,z_hi_mm", _AGREEMENT_CASES,
                         ids=[f"{a:.2f}-{b:.2f}mm" for a, b in _AGREEMENT_CASES])
def test_validator_count_matches_the_real_rasterizer(z_lo_mm, z_hi_mm):
    """This advisory predicts what the rasterizer will do, and a prediction
    that disagrees with the thing it predicts is worse than no advisory: it
    reads as a clean bill of health.

    Two separate ways the hand-rolled model was wrong before #562's review:
    it sampled cell CENTRES where the rasterizer samples E-NODES, and it knew
    nothing of the THIN-SHEET branch that snaps a box no thicker than its
    local cell onto a single nearest node. The validator now calls
    ``Box.mask_on_coords`` on node positions built by the same composition the
    grid builder uses, so agreement is by construction rather than by a copy
    that can drift — but only a test that runs both can say it stayed that way.
    """
    dz = np.array([1.0e-3] * 5 + [0.25e-3] * 8 + [1.0e-3] * 5)
    sim = Simulation(freq_max=30e9, domain=(4e-3, 4e-3, float(np.sum(dz))),
                     dx=1e-3, boundary="pec", dz_profile=dz)
    sim.add_material("substrate", eps_r=4.0)
    sim.add(Box((0.0, 0.0, z_lo_mm * 1e-3), (4e-3, 4e-3, z_hi_mm * 1e-3)),
            material="substrate")
    sim.add_source((2e-3, 2e-3, 1e-3), "ez")

    real = _real_rasterized_z_count(sim)
    predicted, implied = _validator_counts(sim)

    # Both directions, because the SILENT one is the F2 failure mode. The
    # first version of this test only asserted when the advisory fired, so
    # four of six cases skipped the assertion entirely — and "validator quiet
    # where it should warn" is exactly what F2 was (#568 item 1).
    should_warn = real < math.ceil(0.5 * _implied_cells(dz, z_lo_mm * 1e-3,
                                                       z_hi_mm * 1e-3)) and real <= 4
    if should_warn:
        assert predicted is not None, (
            f"advisory SILENT for z-span [{z_lo_mm}, {z_hi_mm}) mm, but the "
            f"rasterizer realizes {real} cells against "
            f"{_implied_cells(dz, z_lo_mm * 1e-3, z_hi_mm * 1e-3):.1f} implied "
            f"— that silence reads as a clean bill of health")
        assert predicted == real, (
            f"validator predicts {predicted} z cells, rasterizer realizes "
            f"{real} for z-span [{z_lo_mm}, {z_hi_mm}) mm")
        # cross-check this test's own `implied` formula against the validator's
        assert implied == pytest.approx(
            _implied_cells(dz, z_lo_mm * 1e-3, z_hi_mm * 1e-3), abs=0.05)
    else:
        assert predicted is None, (
            f"advisory fired for z-span [{z_lo_mm}, {z_hi_mm}) mm where the "
            f"realized count {real} does not meet its own criterion")

    # and the reviewer's case must actually fire: 1 realized against 4 implied
    if (z_lo_mm, z_hi_mm) == (4.50, 5.50):
        assert real == 1, real
        assert predicted == 1, predicted
