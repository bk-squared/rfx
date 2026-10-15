import numpy as np
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry import Box
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.sources.sources import GaussianPulse

def build(lane):
    domain=(.012,.016,.018);dx=.001
    boundary=BoundarySpec(x=Boundary(lo='cpml',hi='cpml',lo_thickness=3,hi_thickness=5),y=Boundary(lo='cpml',hi='cpml',lo_thickness=4,hi_thickness=6),z=Boundary(lo='cpml',hi='pec',lo_thickness=7))
    kw={}
    if lane=='graded':
        dz=np.concatenate((np.full(7,.001),np.linspace(.0009,.0011,11)));kw=dict(dz_profile=dz)
    sim=Simulation(freq_max=10e9,domain=domain,dx=dx,boundary=boundary,cpml_layers=7,precision='float32',**kw)
    sim.add_material('dielectric',eps_r=2.25)
    sim.add(Box((.005,.002,.002),(.007,.012,.015)),material='dielectric')
    w=2*np.pi*8e9
    sim.add_material('debye',eps_r=2.0,debye_poles=[DebyePole(.7,2e-11)])
    sim.add_material('lorentz',eps_r=1.8,lorentz_poles=[LorentzPole(w,.15*w,.4*w*w)])
    sim.add(Box((.008,.003,.004),(.010,.007,.011)),material='debye')
    sim.add(Box((.003,.009,.005),(.005,.012,.014)),material='lorentz')
    sim.add_source((.004,.006,.008),'ez',waveform=GaussianPulse(f0=5e9,bandwidth=.8),amplitude_kind='current')
    sim.add_probe((.008,.010,.012),'ez')
    return sim,dict(domain=domain,dx=dx,requested_faces=dict(x_lo=3,x_hi=5,y_lo=4,y_hi=6,z_lo=7,z_hi=0),pec=['z_hi'],source=[.004,.006,.008],probe=[.008,.010,.012],material_regions=['dielectric','Debye','Lorentz'])
