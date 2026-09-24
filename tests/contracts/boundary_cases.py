"""B0 declarations, copied from the two-step measurement builder."""
import numpy as np
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

CASES = ('pec', 'cpml', 'upml', 'pmc-pec', 'pmc-cpml', 'pec-zlo', 'periodic-xy', 'tfsf', 'waveguide-cpml', 'waveguide-pmc', 'waveguide-pec', 'floquet')
ENTRIES = ('run', 'wire-fast', 'forward', 'sweep', 'nonuniform', 'subgridded', 'distributed', 'adi', 'gpu-query')

def build(case,entry):
    dx=.001; domain=(.024,.020,.016)
    b=BoundarySpec.uniform('cpml')
    if case in ('pec','cpml','upml'): b=BoundarySpec.uniform(case)
    if case=='pmc-pec': b=BoundarySpec(x='pmc',y='pec',z='pec')
    if case=='pmc-cpml': b=BoundarySpec(x='pmc',y='cpml',z='cpml')
    if case=='pec-zlo': b=BoundarySpec(x='cpml',y='cpml',z=Boundary(lo='pec',hi='cpml'))
    if case=='periodic-xy': b=BoundarySpec(x='periodic',y='periodic',z='cpml')
    if case.startswith('waveguide-'): b=BoundarySpec(x='cpml',y=case.split('-')[1],z=case.split('-')[1])
    kw={}
    if entry=='nonuniform': kw['dz_profile']=np.full(16,dx)
    if entry=='adi': kw['solver']='adi'
    sim=Simulation(freq_max=20e9,domain=domain,dx=dx,boundary=b,cpml_layers=8,**kw)
    if entry=='subgridded': sim.add_refinement((.006,.010),ratio=2)
    if case=='tfsf': sim.add_tfsf_source(f0=10e9,margin=3)
    elif case.startswith('waveguide-'):
        sim.add_waveguide_port(.004,direction='+x',y_range=(0,.020),z_range=(0,.016),freqs=np.array([12e9]),f0=12e9,probe_offset=3,ref_offset=2)
    elif case=='floquet':
        sim.add_floquet_port(.003,freqs=np.array([10e9]),f0=10e9)
        sim.add_floquet_port(.013,freqs=np.array([10e9]),f0=10e9,amplitude=0)
    elif entry!='wire-fast': sim.add_source((.010,.010,.004),'ez')
    if entry=='wire-fast': sim.add_port((.010,.010,.004),'ez',extent=.002)
    sim.add_probe((.014,.011,.005),'ez')
    declared={a+'_'+s:getattr(getattr(b,a),s) for a in 'xyz' for s in ('lo','hi')}
    return sim,declared
