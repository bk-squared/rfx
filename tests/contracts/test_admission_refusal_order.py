"""Unsupported PEC declarations refuse before material gates assemble them."""

import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.runners import _admission as admission


@pytest.mark.parametrize("lane", (
    "run_distributed", "fwd_distributed_nu", "run_adi", "fwd_adi", "run_subgridded",
))
@pytest.mark.parametrize("kind,feature", (
    ("sheet", "a PEC sheet (a zero-thickness PEC Box)"),
    ("wire", "a sub-cell PEC wire (PolylineWire)"),
))
def test_pec_refusal_names_input_before_material_gate(lane, kind, feature):
    sim = Simulation(freq_max=15e9, domain=(.01, .008, .006),
                     dx=.001, boundary="pec")
    sim.add_material("dielectric", eps_r=2.5)
    sim.add(Box((.002, .002, .002), (.004, .004, .004)), material="dielectric")
    conductor = (Box((.005, .002, .001), (.005, .004, .004)) if kind == "sheet"
                 else PolylineWire(((.005, .003, .001), (.005, .003, .004)), radius=0))
    sim.add(conductor, material="pec")

    with pytest.raises(NotImplementedError) as exc:
        admission.admit(sim, lane)
    assert feature in str(exc.value)
    assert "refused before the first time step" in str(exc.value)
