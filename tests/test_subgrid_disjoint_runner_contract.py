from __future__ import annotations

import pytest

from rfx import Simulation
from rfx.subgridding.disjoint_runner_contract import build_disjoint_runner_contract


def _disjoint_contract_sim() -> Simulation:
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(
        z_range=(0.006, 0.018),
        ratio=2,
        validation="research",
        topology="stage2_disjoint_3d",
    )
    sim.add_source((0.04 / 3.0, 0.04 / 3.0, 0.0114), "ez")
    sim.add_probe((2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.0126), "ez")
    return sim


def test_disjoint_runner_contract_maps_sources_and_probes_into_fine_block():
    sim = _disjoint_contract_sim()
    grid = sim._build_grid()

    contract = build_disjoint_runner_contract(sim, grid)

    assert contract.topology == "stage2_disjoint_3d"
    assert contract.ratio == 2
    assert contract.shape_c == (grid.nx, grid.ny, grid.nz)
    assert contract.fine_region[0:4] == (
        grid.pad_x_lo,
        grid.nx - grid.pad_x_hi,
        grid.pad_y_lo,
        grid.ny - grid.pad_y_hi,
    )
    assert contract.shape_f == (
        (contract.fine_region[1] - contract.fine_region[0]) * contract.ratio,
        (contract.fine_region[3] - contract.fine_region[2]) * contract.ratio,
        (contract.fine_region[5] - contract.fine_region[4]) * contract.ratio,
    )
    assert len(contract.source_mappings) == 1
    assert len(contract.probe_mappings) == 1
    for mapping in (*contract.source_mappings, *contract.probe_mappings):
        assert mapping.component == "ez"
        assert all(0 <= idx < hi for idx, hi in zip(mapping.fine_index, contract.shape_f))
    assert contract.cell_savings_factor > 1.0
    assert contract.allocated_cell_savings_factor > 1.0


def test_disjoint_runner_contract_supports_endpoint_node_shape():
    sim = _disjoint_contract_sim()
    sim._refinement["disjoint_shape_convention"] = "endpoint_node"
    grid = sim._build_grid()

    contract = build_disjoint_runner_contract(sim, grid)

    assert contract.shape_convention == "endpoint_node"
    assert contract.shape_f == (
        (contract.fine_region[1] - contract.fine_region[0] - 1)
        * contract.ratio
        + 1,
        (contract.fine_region[3] - contract.fine_region[2] - 1)
        * contract.ratio
        + 1,
        (contract.fine_region[5] - contract.fine_region[4] - 1)
        * contract.ratio
        + 1,
    )
    assert len(contract.source_mappings) == 1
    assert len(contract.probe_mappings) == 1
    assert contract.cell_savings_factor > 1.0


def test_disjoint_runner_contract_requires_disjoint_topology():
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.04, 0.024), boundary="pec", dx=0.002)
    sim.add_refinement(
        z_range=(0.006, 0.018),
        ratio=2,
        validation="research",
    )
    grid = sim._build_grid()

    with pytest.raises(ValueError, match="stage2_disjoint_3d"):
        build_disjoint_runner_contract(sim, grid)
