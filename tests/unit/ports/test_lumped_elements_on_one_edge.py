"""Two lumped elements with their own solve on one edge are refused (#1245).

The physics
-----------
A parallel inductor, and a series element with two or more components, is
solved together with its edge field in one implicit step. Two of them on the
same edge and component each solve against a field the other one changes in
the same step, so the pair is not the parallel circuit it declares. MEASURED
in a closed, lossless 8 mm PEC box (1 mm cells, float64 fields, 6400 steps):
a parallel 4 nH + 1 pF with a second 4 nH on its edge gains energy x2.03 by
step 3200 and x4.13 by 6400; 2 nH + 1 pF with 20 nH x1.67; two series
4 nH + 10 pF beside a folded 1 pF x3.99 -- the last identically on main
before #1245, where the two parallel pairs instead lost 97.6 % of their
energy (the old inductor's w^2 L dt loss hid the same defect). The same box
with the folded 1 pF and ONE 2 nH declared as two elements keeps its energy
to 3.3e-9, exactly as the single 2 nH + 1 pF element does in that harness,
and so does a folded 1 pF beside one series element: folded R and C have no
solve of their own and add into the edge material that the one solved
element reads through D0.

So the refusal counts elements with their own solve per REALIZED edge (two
declarations 0.3 mm apart on a 1 mm mesh land on one edge), on every lane
that builds them: run() on a uniform and on a graded mesh, and forward().
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation

L4 = 4e-9


def _box(pairs, dz_profile=None):
    kw = {} if dz_profile is None else {"dz_profile": dz_profile}
    sim = Simulation(freq_max=10e9, domain=(8e-3, 8e-3, 8e-3), dx=1e-3,
                     boundary="pec", **kw)
    for position, component, values in pairs:
        sim.add_lumped_rlc(position=position, component=component, **values)
    sim.add_source((3e-3, 3e-3, 4e-3), "ez",
                   waveform=GaussianPulse(f0=3.5e9, bandwidth=1.0),
                   amplitude_kind="current")
    sim.add_probe((4e-3, 4e-3, 4e-3), "ez")
    return sim


#: The review's case: 4 nH + 1 pF declared at x = 4.0 mm and 4 nH at
#: x = 4.3 mm; on 1 mm cells both snap to the ez edge of cell (4, 4, 4).
TWO_INDUCTORS = [((4e-3, 4e-3, 4e-3), "ez", dict(L=L4, C=1e-12, topology="parallel")),
                 ((4.3e-3, 4e-3, 4e-3), "ez", dict(L=L4, topology="parallel"))]
TWO_SERIES = [((4e-3, 4e-3, 4e-3), "ez", dict(L=L4, C=10e-12, topology="series")),
              ((4e-3, 4e-3, 4e-3), "ez", dict(R=5.0, L=L4, topology="series"))]


def _realized_cells(sim, graded=False):
    if graded:
        from rfx.nonuniform import position_to_index
        grid = sim._build_nonuniform_grid()
        return [tuple(int(v) for v in position_to_index(grid, e.position))
                for e in sim._lumped_rlc]
    grid = sim._build_grid()
    return [tuple(int(v) for v in grid.position_to_index(e.position))
            for e in sim._lumped_rlc]


def _assert_refused(call, sim, graded=False):
    idx = _realized_cells(sim, graded)
    assert len(set(idx)) == 1, f"fixture: the elements must share one cell, got {idx}"
    with pytest.raises(NotImplementedError) as info:
        call(sim)
    msg = str(info.value)
    assert "add_lumped_rlc #0" in msg and "add_lumped_rlc #1" in msg, msg
    assert f"ez edge of cell {idx[0]}" in msg, msg
    assert "ONE add_lumped_rlc with the combined value" in msg, msg


@pytest.mark.parametrize("pairs", [TWO_INDUCTORS, TWO_SERIES],
                         ids=["two-parallel-inductors", "two-series-elements"])
def test_run_refuses_two_solved_elements_on_one_edge(pairs):
    _assert_refused(lambda s: s.run(n_steps=10, skip_preflight=True), _box(pairs))


def test_run_on_a_graded_mesh_refuses_them_too():
    dz = np.array([1e-3] * 3 + [0.5e-3] * 4 + [1e-3] * 3)
    sim = _box(TWO_INDUCTORS, dz_profile=dz)
    _assert_refused(lambda s: s.run(n_steps=10, skip_preflight=True), sim,
                    graded=True)


def test_forward_refuses_them_too():
    _assert_refused(lambda s: s.forward(n_steps=10, skip_preflight=True),
                    _box(TWO_INDUCTORS))


def _probe(sim, n=300):
    return np.asarray(sim.run(n_steps=n, skip_preflight=True).time_series)


def test_folded_elements_share_an_edge_with_one_solved_element():
    """A folded 1 pF declared beside a 2 nH is the 2 nH + 1 pF element, bit
    for bit; a folded R and a folded C may share an edge with ONE solved
    element (here the series 50 ohm + 1 F)."""
    here = (4e-3, 4e-3, 4e-3)
    split = _probe(_box([(here, "ez", dict(C=1e-12, topology="parallel")),
                         (here, "ez", dict(L=2e-9, topology="parallel"))]))
    joined = _probe(_box([(here, "ez", dict(L=2e-9, C=1e-12, topology="parallel"))]))
    assert np.all(np.isfinite(split)) and np.array_equal(split, joined)

    folded = _probe(_box([(here, "ez", dict(R=100.0, topology="parallel")),
                          (here, "ez", dict(C=1e-12, topology="parallel")),
                          (here, "ez", dict(R=50.0, C=1.0, topology="series"))]))
    assert np.all(np.isfinite(folded))


def test_solved_elements_on_different_edges_of_one_cell_are_allowed():
    here = (4e-3, 4e-3, 4e-3)
    ts = _probe(_box([(here, "ez", dict(L=L4, topology="parallel")),
                      (here, "ex", dict(L=L4, topology="parallel"))]), n=50)
    assert np.all(np.isfinite(ts))
