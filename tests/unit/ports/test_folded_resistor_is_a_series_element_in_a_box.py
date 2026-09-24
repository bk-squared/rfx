"""A folded resistor and a series R + C (C -> infinity) are one device (#1236).

Both declare a 300 ohm resistor across the same Ez edge, one cell from a 50 ohm
port, inside a 5-cell PEC box: ``add_lumped_rlc(R=300, topology="parallel")``
folds it into the material as a conductivity on that edge, and
``add_lumped_rlc(R=300, C=1 F, topology="series")`` solves it as a two-terminal
element together with the edge field (#1163). A 1 F capacitor is a short at
every frequency the box carries, so the two must give the same fields.

They did not while the folded conductivity was added to all three E
components at its node: the fold was then also a 300 ohm resistor on the Ex
and Ey edges leaving the node, and the two runs differed by 14 % of the peak
field here (found by #1163's gate). On a one-cell parallel-plate line, where
those two edges carry no field, they already agreed to 3.5e-7.

Needs the series update of #1163 (``rfx.lumped.edge_update_denominator``);
skipped until that lands. The threshold is the one pre-declared for this gate
(G2): 1e-5 of the peak field.
"""
from __future__ import annotations

import numpy as np
import pytest

import rfx.lumped
from rfx import GaussianPulse, Simulation

_DX = 1e-3
_STEPS = 1200
_R = 300.0

pytestmark = pytest.mark.skipif(
    not hasattr(rfx.lumped, "edge_update_denominator"),
    reason="needs #1163's coupled series-element update "
           "(rfx.lumped.edge_update_denominator)")


def _box(element):
    port = (2 * _DX, 2 * _DX, 2 * _DX)
    elem = (3 * _DX, 2 * _DX, 2 * _DX)
    sim = Simulation(freq_max=10e9, domain=(5 * _DX,) * 3, dx=_DX,
                     boundary="pec")
    sim.add_port(position=port, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    element(sim, elem)
    sim.add_vector_probe(elem)
    sim.add_vector_probe(port)
    grid = sim._build_grid()
    assert tuple(int(v) for v in grid.position_to_index(elem)) == (3, 2, 2)
    return sim


def test_a_folded_resistor_is_a_series_resistor_with_a_shorted_capacitor():
    series = _box(lambda s, p: s.add_lumped_rlc(
        position=p, component="ez", R=_R, C=1.0, topology="series"))
    folded = _box(lambda s, p: s.add_lumped_rlc(
        position=p, component="ez", R=_R, topology="parallel"))
    assert series._lumped_rlc[0].topology == "series"
    assert folded._lumped_rlc[0].topology == "parallel"

    a = np.asarray(series.run(n_steps=_STEPS, skip_preflight=True).time_series)
    b = np.asarray(folded.run(n_steps=_STEPS, skip_preflight=True).time_series)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    scale = float(np.abs(b).max())
    assert scale > 0.0
    rel = float(np.abs(a - b).max()) / scale
    assert rel <= 1e-5, (
        f"series {_R:.0f} ohm + 1 F and the folded {_R:.0f} ohm on the same "
        f"Ez edge differ by {rel:.3e} of the peak field. The folded "
        "conductivity is loading an edge the series element does not -- the "
        "Ex / Ey edges at its node (#1236: 14 % before).")
