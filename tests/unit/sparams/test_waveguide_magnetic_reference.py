"""Empty-guide references start from vacuum, independent of device physics."""
import numpy as np
import pytest
from rfx import Simulation, Box, DebyePole
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sparams.waveguide import _empty_waveguide_reference


def _guide(kind, graded):
    h = .002
    sim = Simulation(freq_max=11e9, domain=(.024,.004,.096), dx=h,
        dz_profile=np.full(48,h) if graded else None, cpml_layers=12,
        boundary=BoundarySpec(x=Boundary('pec','pec'), y=Boundary('pec','pec'),
                              z=Boundary('cpml','cpml')))
    if kind in ('mu', 'debye'):
        props = {'mu_r':4.} if kind == 'mu' else {
            'debye_poles':[DebyePole(delta_eps=3., tau=1e-11)]}
        sim.add_material('slab', eps_r=1., **props)
        sim.add(Box((0.,0.,.036),(.024,.004,.046)), material='slab')
    elif kind == 'rlc':
        for y in (0., .002):
            sim.add_lumped_rlc((.012,y,.040), 'ey', R=25., topology='parallel')
    for z, direction in ((.016,'+z'),(.080,'-z')):
        sim.add_waveguide_port(z, direction=direction, mode=(1,0),
            freqs=np.linspace(8e9,10e9,5), f0=9e9, bandwidth=.4)
    return sim


@pytest.mark.parametrize('kind', ['mu', 'debye', 'rlc'])
def test_empty_reference_scattering_contract(kind):
    if kind == 'rlc':
        # Both public calculators refuse RLC (#1263). Keep that boundary;
        # exercise the internal NU reference without claiming public support.
        for graded in (False, True):
            with pytest.raises(NotImplementedError, match='lumped RLC'):
                _guide(kind, graded).compute_waveguide_s_matrix(
                    num_periods=40, normalize=True)
        result = _guide(kind, True)._compute_waveguide_s_matrix_nu(
            n_steps=None, num_periods=40, normalize=True)
        assert np.max(np.abs(np.asarray(result.s_params)[0,0])) > .01
        return
    # Same equal-cell mesh. V/I normalization compares the summed modal
    # quantities directly; no power-ratio square root at a reflection zero.
    values = [np.asarray(_guide(kind, graded).compute_waveguide_s_matrix(
        num_periods=40, normalize=True).s_params)[0,0] for graded in (False,True)]
    peak = np.max(np.abs(values[0]))
    assert peak > .01
    assert np.max(np.abs(values[1])) > .01
    assert np.max(np.abs(values[1]-values[0])) <= 1e-4 * peak


@pytest.mark.parametrize('kind', ['mu', 'debye', 'rlc'])
def test_reference_has_vacuum_physics_and_same_mesh(kind):
    from rfx.runners.nonuniform import assemble_materials_nu
    from rfx.core.yee import init_materials
    device = _guide(kind, True)
    ref = _empty_waveguide_reference(device)
    assert ref._geometry == []
    assert ref._lumped_rlc == []
    assert ref._thin_conductors == []
    assert ref._boundary_spec == device._boundary_spec
    grid = device._build_nonuniform_grid()
    ref_grid = ref._build_nonuniform_grid()
    for name in ('dx_arr', 'dy_arr', 'dz'):
        np.testing.assert_array_equal(getattr(grid,name), getattr(ref_grid,name))
    assert grid.dt == ref_grid.dt
    mats, debye, lorentz, pec = assemble_materials_nu(ref, ref_grid,
        pec_sheets=[], pec_wires=[])
    vacuum = init_materials(ref_grid.shape)
    for name in ('eps_r','sigma','mu_r'):
        np.testing.assert_array_equal(getattr(mats,name),getattr(vacuum,name))
    assert debye is None and lorentz is None
    assert pec is None or not np.any(pec)
