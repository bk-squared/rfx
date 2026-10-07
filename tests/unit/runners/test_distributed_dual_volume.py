"""Current drive on a graded dielectric interface at a distributed seam."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rfx import Simulation, Box, DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.nonuniform import position_to_index

pytestmark = pytest.mark.distributed


def model(kind):
    widths = np.tile([.0009,.0011],8)
    widths[[0,-1]] = .001
    face = float(widths[:9].sum())
    sim = Simulation(freq_max=15e9,domain=(.016,.006,.006),dx=.001,
                     dx_profile=widths,boundary='pec',cpml_layers=0)
    poles = ({'debye_poles':[DebyePole(2.,1e-11)]} if kind=='debye' else
             {'lorentz_poles':[lorentz_pole(2.,2*np.pi*15e9,2*np.pi*.75e9)]}
             if kind=='lorentz' else {})
    sim.add_material('slab',eps_r=4.,sigma=.2 if kind=='lossy' else 0.,**poles)
    sim.add(Box((face,0,0),(.014,.006,.006)),material='slab')
    sim.add_source((face,.003,.003),'ez',waveform=jnp.ones_like,amplitude_kind='current')
    for x in (face,.010,.012):
        for component in ('ex','ey','ez','hx','hy','hz'):
            sim.add_probe((x,.003,.003),component)
    grid=sim._build_realized_grid()
    assert not grid.is_constant(0)
    assert grid.cells(0)[8] != grid.cells(0)[9]
    assert position_to_index(grid,(face,.003,.003))[0] == 9
    drawn=sim._assemble_materials_nu(grid)[0]
    assert drawn.eps_r[8,3,3] != drawn.eps_r[9,3,3]
    return sim


@pytest.mark.parametrize('kind',['plain','lossy','debye','lorentz'])
@pytest.mark.parametrize('entry',['run','forward'])
def test_unequal_transverse_interface_matches_single(kind,entry,record_property):
    devices=jax.devices('cpu')[:2]
    if len(devices)<2:
        pytest.skip('needs two virtual CPU devices')
    kwargs=dict(n_steps=24,skip_preflight=True)
    if entry=='run': kwargs['compute_s_params']=False
    single=getattr(model(kind),entry)(**kwargs)
    extra=dict(devices=devices)
    if entry=='forward':extra['distributed']=True
    distributed=getattr(model(kind),entry)(**kwargs,**extra)
    scale=np.tile([1.,1.,1.,376.730313668,376.730313668,376.730313668],3)
    a,b=np.asarray(single.time_series),np.asarray(distributed.time_series)
    peak=float(np.max(abs(a)*scale))
    assert peak>0 and np.isfinite(b).all()
    relative=float(np.max(abs(a-b)*scale)/peak)
    record_property('trace_relative',relative)
    assert relative<=1e-4,(kind,entry,relative)
    step_peak=np.max(abs(a)*scale,axis=1)
    step_error=np.max(abs(a-b)*scale,axis=1)
    step_ulps=float(np.max(step_error/np.spacing(step_peak.astype(np.float32))))
    record_property('trace_step_ulps',step_ulps)
    assert step_ulps<=9,(kind,entry,step_ulps)
    if entry=='run' and single.state is not None and distributed.state is not None:
        errors=[];peaks=[]
        for name in ('ex','ey','ez','hx','hy','hz'):
            a,b=np.asarray(getattr(single.state,name)),np.asarray(getattr(distributed.state,name))
            factor=376.730313668 if name.startswith('h') else 1.
            errors.append(float(np.max(abs(a-b)))*factor)
            peaks.append(float(np.max(abs(a)))*factor)
        ulps=max(errors)/float(np.spacing(np.float32(max(peaks))))
        record_property('field_peak_ulps',ulps)
        assert ulps<=9,(kind,entry,ulps)
