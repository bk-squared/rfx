"""Compact records retain the owner at forward and distributed handoffs."""
import pytest

from rfx import Box, Simulation


@pytest.mark.parametrize('lane', ['fwd_uniform', 'run_distributed'])
def test_compact_record_retains_kernel_owner(lane, monkeypatch):
    from rfx.model import conductors, realization_reference

    sim = Simulation(1e9, (.010,) * 3, dx=.001, boundary='pec')
    sim.add(Box((.002,) * 3, (.004,) * 3), material='pec')
    seen = []
    remember = realization_reference.remember_kernel

    def observe(sim, root, record):
        seen.append(root)
        remember(sim, root, record)

    monkeypatch.setattr(realization_reference, 'remember_kernel', observe)
    if lane == 'fwd_uniform':
        record = sim.forward(n_steps=1, skip_preflight=True).realized_geometry
    else:
        # Exercise the full-domain ownership transfer without requiring GPUs.
        root = conductors.solve_conductors(sim, sim._build_grid())
        transferred, record = conductors.at_kernel(
            sim, root, lane=lane, pec_edges=root.pec_edges)
        assert transferred is not root and transferred._prepared_owner is root
    assert len(seen) == 1
    assert record.conductors is seen[0]
    assert record.entities and all(e.mask is None and not e.edge_masks for e in record.entities)
    assert record.pec_mask is None and record.materials is None
    if lane == 'run_distributed':
        assert sim.realized_geometry().conductors is seen[0]
