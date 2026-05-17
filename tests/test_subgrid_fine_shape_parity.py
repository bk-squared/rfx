"""Fine-grid shape conventions for the two subgrid region builders.

``build_subgrid_region`` describes the overlap z-slab runner
(``rfx.runners.subgridded``), which is **node-aligned**: a fine slab spanning
``N`` coarse nodes owns ``(N - 1) * ratio + 1`` fine nodes.  ``rfx/runners/
subgridded.py`` builds the run ``SubgridConfig3D`` with exactly that formula
(via the shared ``overlap_fine_extent`` helper), so the validation region must
report the same shape.

``build_stage2_disjoint_region`` describes the Stage-2 disjoint runner, whose
contract (``build_disjoint_runner_contract``) defaults to the ``cell_extent``
convention ``N * ratio``.  The two builders therefore use different, per-
topology conventions on purpose -- a single shared convention would force one
builder to disagree with its own runner.
"""

from __future__ import annotations

import pytest

from rfx.api import Simulation
from rfx.subgridding.disjoint_runner_contract import build_disjoint_runner_contract
from rfx.subgridding.validation import (
    build_stage2_disjoint_region,
    build_subgrid_region,
)


@pytest.fixture
def overlap_fixture():
    """Return ``(sim, grid, ratio)`` for an overlap z-slab refinement."""
    ratio = 4
    sim = Simulation(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.02),
        boundary="pec",
        cpml_layers=0,
        dx=0.002,
    )
    sim.add_refinement((0.004, 0.012), ratio=ratio)
    return sim, sim._build_grid(), ratio


@pytest.fixture
def disjoint_fixture():
    """Return ``(sim, grid, ratio)`` for a Stage-2 disjoint refinement."""
    ratio = 2
    sim = Simulation(
        freq_max=8e9,
        domain=(0.04, 0.04, 0.024),
        boundary="pec",
        dx=0.002,
    )
    sim.add_refinement(
        z_range=(0.006, 0.018),
        ratio=ratio,
        validation="research",
        topology="stage2_disjoint_3d",
    )
    return sim, sim._build_grid(), ratio


def test_overlap_region_uses_node_aligned_fine_shape(overlap_fixture):
    """build_subgrid_region must match the node-aligned overlap runner.

    ``rfx/runners/subgridded.py`` builds the run config with
    ``nx_f = (fi_hi - fi_lo - 1) * ratio + 1``.  A cell-extent
    ``(fi_hi - fi_lo) * ratio`` region over-reports the fine shape by
    ``ratio - 1`` nodes per axis relative to the grid the runner builds.
    """
    sim, grid, ratio = overlap_fixture
    region = build_subgrid_region(sim, grid)
    assert region is not None
    assert region.nx_f == (region.fi_hi - region.fi_lo - 1) * ratio + 1
    assert region.ny_f == (region.fj_hi - region.fj_lo - 1) * ratio + 1
    assert region.nz_f == (region.fk_hi - region.fk_lo - 1) * ratio + 1


def test_overlap_fine_extent_helper_is_node_aligned():
    """overlap_fine_extent is the single source of truth for the overlap shape."""
    from rfx.subgridding.validation import overlap_fine_extent

    assert overlap_fine_extent(1, 4) == 1
    assert overlap_fine_extent(6, 4) == 21
    assert overlap_fine_extent(11, 2) == 21


def test_disjoint_region_uses_cell_extent_fine_shape(disjoint_fixture):
    """build_stage2_disjoint_region matches the disjoint contract default."""
    sim, grid, ratio = disjoint_fixture
    region = build_stage2_disjoint_region(sim, grid)
    assert region.nx_f == (region.fi_hi - region.fi_lo) * ratio
    assert region.ny_f == (region.fj_hi - region.fj_lo) * ratio
    assert region.nz_f == (region.fk_hi - region.fk_lo) * ratio

    # The disjoint runner contract consumes this region; its default
    # cell_extent convention must reproduce the same fine shape.
    contract = build_disjoint_runner_contract(sim, grid)
    assert contract.shape_f == (region.nx_f, region.ny_f, region.nz_f)


def test_region_builders_use_distinct_per_topology_conventions(
    overlap_fixture, disjoint_fixture
):
    """The two builders intentionally differ: each matches its own runner."""
    overlap_sim, overlap_grid, overlap_ratio = overlap_fixture
    disjoint_sim, disjoint_grid, disjoint_ratio = disjoint_fixture

    overlap_region = build_subgrid_region(overlap_sim, overlap_grid)
    disjoint_region = build_stage2_disjoint_region(disjoint_sim, disjoint_grid)

    overlap_extent = overlap_region.fk_hi - overlap_region.fk_lo
    disjoint_extent = disjoint_region.fk_hi - disjoint_region.fk_lo

    # overlap is node-aligned (not cell-extent); disjoint is cell-extent.
    assert overlap_region.nz_f != overlap_extent * overlap_ratio
    assert overlap_region.nz_f == (overlap_extent - 1) * overlap_ratio + 1
    assert disjoint_region.nz_f == disjoint_extent * disjoint_ratio
