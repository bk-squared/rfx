"""A series RLC element is solved together with the field on its edge (#1163).

The physics
-----------
``add_lumped_rlc(..., topology="series")`` with two or more of R, L, C puts a
two-terminal device across ONE Yee edge. Its current and the edge field are
one system: Ampere at the edge, ``E^{n+1} = e_std - I/(D0*A)``, and the
element law, ``d*E = R*I + L*dI/dt + Q/C``. The update this file guards
solves the two together (trapezoidal in time), so the element realizes the R,
L and C that were declared.

The update it replaced took the current from ``e_std`` -- the field before the
element's own current had acted -- and subtracted it afterwards. That
subtracts the edge's own impedance ``d/(D0*A)`` (215 ohm in a vacuum cubic
cell, independent of the cell size) from R: a series 300 ohm + C read 85 ohm,
and anything below ~215 ohm was a negative resistance that grew to NaN.

What is checked here (fast, always on)
--------------------------------------
1. With C -> infinity and no inductor, a series R + C must BE the folded
   resistor (``topology="parallel"``, R only, which stamps sigma = d/(R*A)) on
   the same edge, to float32 rounding, through the public API. The two sides
   share no helper: one is the series ADE, the other is the Yee update's own
   conductivity term. The fixture is a one-cell-wide parallel-plate line (PEC
   plates, magnetic side walls) because there the Ex and Ey edges of the
   element's cell carry no field; in a 3-D cell the folded stamp also loads
   those two edges (``component_e_materials`` adds the stamp to all three
   components), which a series element does not, so the two are different
   devices there.
2. The 5-cell PEC box of ``scripts/diagnostics/lumped_rlc_adjacent_to_port_nan.py``
   (a 50 ohm port one cell from the element) decays for series R + C, R + L
   and R + L + C at R down to 0.1 ohm. The full sweep (R x L/C x cell size)
   is evidence for #1163, not this test.

The closed-form check of the realized impedance is
``tests/oracle/test_series_rlc_load_on_line.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

ETA0 = 376.730313668

# ---------------------------------------------------------------------------
# 1. C -> inf series R + C == the folded resistor on the same edge
# ---------------------------------------------------------------------------

_LINE_DX = 1e-3
_LINE_NODES = 20
_LINE_STEPS = 600
_LOAD_NODE = _LINE_NODES - 3

#: Float32 storage rounding, accumulated over the run. MEASURED on this
#: fixture (600 steps, all six components at three probes, R = 0.1 ... 1e4
#: ohm): max |series - folded| / max |folded| = 2.5e-7 ... 3.5e-7, i.e. about
#: 3 float32 epsilons. The gate is 32 epsilons. The replaced update misses by
#: the edge impedance (R - 215 ohm), which moves the fields by tens of percent.
_F32_REL = 32.0 * float(np.finfo(np.float32).eps)


def _line(load):
    """One-cell-wide parallel-plate line, lumped port on node 1, load on
    node ``_LOAD_NODE``; probes at the port, mid-line and the load."""
    sim = Simulation(
        freq_max=10e9, domain=((_LINE_NODES - 1) * _LINE_DX, _LINE_DX, _LINE_DX),
        dx=_LINE_DX,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")))
    sim.add_port(position=(_LINE_DX, 0.0, 0.0), component="ez", impedance=ETA0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    load(sim, (_LOAD_NODE * _LINE_DX, 0.0, 0.0))
    for node in (1, 8, _LOAD_NODE):
        sim.add_vector_probe((node * _LINE_DX, 0.0, 0.0))
    return sim


@pytest.mark.parametrize("r_ohm", [0.1, 50.0, 300.0, 1e4])
def test_series_r_with_infinite_c_is_the_folded_resistor(r_ohm):
    series = _line(lambda s, p: s.add_lumped_rlc(
        position=p, component="ez", R=r_ohm, C=1.0, topology="series"))
    folded = _line(lambda s, p: s.add_lumped_rlc(
        position=p, component="ez", R=r_ohm, topology="parallel"))

    # Realized, not declared: both elements on the same edge, and the series
    # one really is the series ADE (two components) while the folded one is a
    # single-component parallel element.
    for sim in (series, folded):
        grid = sim._build_grid()
        assert tuple(int(v) for v in grid.position_to_index(
            sim._lumped_rlc[0].position)) == (_LOAD_NODE, 0, 0)
    assert series._lumped_rlc[0].topology == "series"
    assert series._lumped_rlc[0].C == 1.0 and series._lumped_rlc[0].R == r_ohm

    a = np.asarray(series.run(n_steps=_LINE_STEPS, skip_preflight=True).time_series)
    b = np.asarray(folded.run(n_steps=_LINE_STEPS, skip_preflight=True).time_series)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    scale = float(np.abs(b).max())
    assert scale > 1.0, "the line carried no wave; the comparison would be empty"
    rel = float(np.abs(a - b).max()) / scale
    assert rel <= _F32_REL, (
        f"series R={r_ohm} ohm + C=1 F differs from the folded {r_ohm} ohm "
        f"resistor on the same edge by {rel:.3e} of the peak field "
        f"(float32 rounding bound {_F32_REL:.1e}); the series element is not "
        "realizing its declared resistance")


# ---------------------------------------------------------------------------
# 2. The adjacent-to-port box decays
# ---------------------------------------------------------------------------

_BOX_DX = 1e-3
_BOX_STEPS = 2400

#: MEASURED with this update (2400 steps, dx = 1 mm): the largest field over
#: the last 400 steps is 8e-6 ... 3.4e-5 of the run's peak on these rows (the
#: 87-case sweep over R 0.1-300 ohm x RC/RL/RLC/LC x dx 0.5/1/2 mm at 4800
#: steps: at most 3.3e-5). Every row here fails with the replaced update:
#: non-finite or past 1e33 by step 2400 at R = 0.1 and 50 ohm (RC, RL, RLC),
#: 4.8e24 at R = 150 ohm (RC).
_DECAY_FRACTION = 1e-3

_L = 1e-9
_C = 0.2e-12


@pytest.mark.parametrize("r_ohm,l_h,c_f", [
    (0.1, 0.0, _C), (50.0, 0.0, _C), (150.0, 0.0, _C),
    (0.1, _L, 0.0), (50.0, _L, 0.0),
    (0.1, _L, _C), (50.0, _L, _C),
], ids=["RC-0.1", "RC-50", "RC-150", "RL-0.1", "RL-50", "RLC-0.1", "RLC-50"])
def test_series_element_next_to_a_port_decays(r_ohm, l_h, c_f):
    dx = _BOX_DX
    port = (2 * dx, 2 * dx, 2 * dx)
    elem = (3 * dx, 2 * dx, 2 * dx)
    sim = Simulation(freq_max=10e9, domain=(5 * dx,) * 3, dx=dx, boundary="pec")
    sim.add_port(position=port, component="ez", impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    sim.add_lumped_rlc(position=elem, component="ez", R=r_ohm, L=l_h, C=c_f,
                       topology="series")
    sim.add_vector_probe(elem)
    grid = sim._build_grid()
    assert tuple(int(v) for v in grid.position_to_index(elem)) == (3, 2, 2)

    ts = np.abs(np.asarray(sim.run(n_steps=_BOX_STEPS, skip_preflight=True).time_series))
    assert np.all(np.isfinite(ts)), (
        f"series R={r_ohm} L={l_h} C={c_f}: non-finite field at step "
        f"{int(np.argmax(~np.isfinite(ts).all(axis=1)))}")
    peak = float(ts.max())
    tail = float(ts[-400:].max())
    assert tail <= _DECAY_FRACTION * peak, (
        f"series R={r_ohm} L={l_h} C={c_f}: the last 400 steps reach "
        f"{tail:.3e} against a peak of {peak:.3e}; the element is feeding "
        "energy into the box")
