"""Unsupported PEC declarations refuse before material gates assemble them."""

import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.runners import _admission as admission


@pytest.mark.parametrize("lane", (
      "run_subgridded",
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


@pytest.mark.parametrize('row,graded', (
    (('_ntff', 'ntff_box'), True),
    (('_dft_planes', 'dft_plane'), True),
    (('_flux_monitors', 'flux'), True),
    (('_current_moments', 'block_moments'), True),
    (('_cpml_kappa_max', 'kappa'), False),
))
def test_distributed_carriers_match_graded_support(row, graded):
    assert row in admission.ADMITS['run_distributed']
    assert (row in admission.ADMITS['run_distributed_nu']) is graded


@pytest.mark.parametrize('lane', ('run_distributed', 'run_distributed_nu', 'fwd_distributed_nu'))
def test_distributed_mode2d_is_not_admitted(lane):
    assert ('_mode', '') not in admission.ADMITS[lane]


@pytest.mark.parametrize('lane', [
    'coaxial_line_reflection', 'coaxial_two_port', 'coax_msl_transition',
    'mixed_s_matrix', 's_matrix_scan', 'run_subgridded',
])
def test_conformal_pec_refusal_rows(lane):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = Simulation(20e9, (.012, .012, .012), dx=.001,
                     boundary=BoundarySpec(x=Boundary('pec', 'pec', conformal=True), y='pec', z='pec'))
    sim.add(Box((.004, .004, .004), (.006, .006, .006)), material='pec')
    if lane == 's_matrix_scan':
        admission.admit(sim, lane, run_args={'conformal_pec': True})
        with pytest.raises(NotImplementedError, match='cut_cell_floor'):
            admission.admit(sim, lane, run_args={'conformal_pec': 'cut_cell_floor'})
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        with pytest.raises(NotImplementedError, match=r'devices=.*conformal PEC'):
            compute_lumped_wire_s_matrix_via_scan(sim, [10e9], n_steps=1, devices=[object(), object()])
    else:
        with pytest.raises(NotImplementedError, match='conformal'):
            admission.admit(sim, lane, run_args={'conformal_pec': True})


def test_2d_box_face_on_slice_refuses_conformal_by_name():
    sim = Simulation(20e9, (.012, .012, .001), dx=.001, boundary='pec',
                     mode='2d_tmz', snap='declared')
    sim.add(Box((.004, .004, 0.), (.008, .008, .004)), material='pec')
    with pytest.raises(NotImplementedError, match=r'conformal PEC.*Box.*2-D.*/issues/1373'):
        sim.run(n_steps=4, conformal_pec=True, skip_preflight=True, compute_s_params=False)
