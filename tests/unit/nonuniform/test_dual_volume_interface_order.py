"""Two-mesh forms of the S1 M3 graded-slab continuum oracles.

Geometry, band and record lengths match the three-mesh Q1 declaration.
The references are continuous transfer matrices, independent of Yee averaging.
"""
import jax
import numpy as np
from scipy.optimize import brentq
from rfx import Simulation, Box, GaussianPulse
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.ringdown import identify

C0 = 299792458.
A, B = .024, .004
FREQS = np.linspace(8e9, 10e9, 21)


def cascade(frequencies, faces, end, eps, start=0.):
    omega = 2*np.pi*np.asarray(frequencies)
    k0 = omega/C0
    matrix = np.broadcast_to(np.eye(2, dtype=complex), omega.shape+(2,2)).copy()
    for epsilon, length in ((1., faces[0]-start), (eps, faces[1]-faces[0]),
                            (1., end-faces[1])):
        beta = np.sqrt(k0*k0*epsilon-(np.pi/A)**2+0j)
        impedance = omega*1.25663706212e-6/beta
        layer = np.zeros_like(matrix)
        layer[...,0,0] = layer[...,1,1] = np.cos(beta*length)
        layer[...,0,1] = 1j*impedance*np.sin(beta*length)
        layer[...,1,0] = 1j*np.sin(beta*length)/impedance
        matrix = matrix @ layer
    return matrix


def test_te10_slab_second_order_on_unequal_face_cells():
    matrix = cascade(FREQS, (.036,.046), .076, 4., .020)
    omega = 2*np.pi*FREQS
    z0 = omega*1.25663706212e-6/np.sqrt((omega/C0)**2-(np.pi/A)**2)
    aa, bb, cc, dd = matrix[:,0,0], matrix[:,0,1], matrix[:,1,0], matrix[:,1,1]
    denominator = aa+bb/z0+cc*z0+dd
    exact = np.array([(aa+bb/z0-cc*z0-dd)/denominator, 2/denominator])
    errors = []
    # The prototype enables x64 for modal/DFT arithmetic, while the solver's
    # declared storage remains float32. Restore the ambient setting on exit.
    with jax.enable_x64(True):
        for level in (0,1):
            h = .001/2**level
            cells = np.full(96,.001)
            cells[[34,35,36,37,44,45,46,47]] *= [.9,1.1,.9,1.1,1.1,.9,1.1,.9]
            sim = Simulation(freq_max=1.1*FREQS[-1], domain=(A,B,.096), dx=h,
                dz_profile=np.repeat(cells/2**level,2**level), cpml_layers=round(.048/h),
                boundary=BoundarySpec(x=Boundary('pec','pec'), y=Boundary('pec','pec'),
                    z=Boundary('cpml','cpml',lo_thickness=round(.040/h),hi_thickness=round(.048/h))))
            sim.add_material('slab',eps_r=4.)
            sim.add(Box((0.,0.,.036),(A,B,.046)),material='slab')
            for pos, ref, direction, name in zip((.016,.080),(.020,.076),('+z','-z'),('left','right')):
                sim.add_waveguide_port(pos,direction=direction,mode=(1,0),mode_type='TE',freqs=FREQS,
                    f0=float(FREQS.mean()),bandwidth=.4,waveform='modulated_gaussian',
                    ref_offset=round(.002/h),probe_offset=round(.004/h),reference_plane=ref,name=name)
            result = sim.compute_waveguide_s_matrix(num_periods=60,normalize='flux')
            errors.append(float(np.max(np.abs(np.asarray(result.s_params)[[0,1],0,:]-exact))))
    slope = float(np.log2(errors[0]/errors[1]))
    print({'oracle':'TE10 slab','errors':errors,'slope':slope})
    assert 1.7 <= slope <= 2.4, (errors,slope)


def test_pec_slab_cavity_second_order_on_unequal_face_cells():
    def equation(frequency):
        return cascade(frequency,(.014,.024),.048,4.)[0,1].imag
    scan = np.linspace(.1e9,20e9,3001)
    brackets = [(lo,hi) for lo,hi in zip(scan[:-1],scan[1:]) if equation(lo)*equation(hi)<0]
    exact = brentq(equation,*brackets[0],xtol=1e-4)
    errors = []
    with jax.enable_x64(False):
        for level in (0,1):
            h = .002/2**level
            cells = np.full(24,.002)
            cells[[5,6,7,8,10,11,12,13]] *= [.9,1.1,.9,1.1,1.1,.9,1.1,.9]
            sim = Simulation(freq_max=2*exact,domain=(A,B,.048),dx=h,boundary='pec',cpml_layers=0,
                             dz_profile=np.repeat(cells/2**level,2**level))
            sim.add_material('slab',eps_r=4.)
            sim.add(Box((0,0,.014),(A,B,.024)),material='slab')
            sim.add_source((.012,.002,.010),'ey',waveform=GaussianPulse(f0=exact,bandwidth=.6),amplitude_kind='field')
            sim.add_probe((.010,.002,.030),'ey')
            sim.add_probe((.016,.002,.020),'ey')
            grid = sim._build_realized_grid()
            result = sim.run(n_steps=int(np.ceil(8e-9/grid.dt)),compute_s_params=False,skip_preflight=True)
            series = np.asarray(result.time_series)
            frequencies = []
            for t0,t1 in ((2e-9,8e-9),(3e-9,7e-9)):
                fit = identify(series,grid.dt,int(t0/grid.dt),min(len(series),int(t1/grid.dt)),freq_max=2*exact)
                poles = [p for p in fit.poles() if .8*exact<p.f_hz<1.2*exact]
                assert poles, 'no fundamental pole in source band'
                frequencies.append(max(poles,key=lambda p:p.amplitude).f_hz)
            assert abs(frequencies[1]/frequencies[0]-1) <= 1e-6, frequencies
            errors.append(abs(frequencies[0]/exact-1))
    slope = float(np.log2(errors[0]/errors[1]))
    print({'oracle':'PEC slab cavity','exact_hz':exact,'errors':errors,'slope':slope})
    assert 1.7 <= slope <= 2.4, (errors,slope)
