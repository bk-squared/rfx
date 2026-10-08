"""Unsupported PEC declarations refuse before material gates assemble them."""

import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.runners import _admission as admission


@pytest.mark.parametrize("lane", (
    "run_adi", "fwd_adi", "run_subgridded",
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


def test_distributed_nu_entries_share_runner_carriers():
    assert admission.ADMITS['run_distributed_nu'] == admission.ADMITS['fwd_distributed_nu']
    assert admission.ADMITS['run_distributed_nu'] == admission._DISTRIBUTED_NU_ROWS


@pytest.mark.parametrize('row', (
    ('_ports', 'lumped_port'), ('_ports', 'wire_port'),
    ('_ntff', 'ntff_box'), ('_cpml_kappa_max', 'kappa'),
))
def test_uniform_distributed_carriers_do_not_leak_to_graded(row):
    assert row in admission.ADMITS['run_distributed']
    assert row not in admission.ADMITS['run_distributed_nu']


@pytest.mark.parametrize('lane', ('run_distributed', 'run_distributed_nu', 'fwd_distributed_nu'))
def test_distributed_mode2d_is_not_admitted(lane):
    assert ('_mode', '') not in admission.ADMITS[lane]
