"""Causal source tables: record prefixes and independent spectral references."""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.model import waveguide_modes as wm


def parameters(case="step", mode_type="TE"):
    # Realized step vacuum port (a/45), and ports_and_sparams_101 guide.
    fc, dt, dx, f0, bw = {
        "step": (6555808844.538607, 9.68540033721751e-13, .02286/45, 7.45e9, .6),
        "tutorial": (7487108531.983443, 1.9065748695310057e-12, .001, 8e9, .5),
        # Step guide, source off cutoff: the TM admittance diverges at f_c.
        "off_cutoff": (6555808844.538607, 9.68540033721751e-13, .02286/45, 10e9, .3),
    }[case]
    return dict(f_c=fc, dt=dt, dx=dx, f0=f0, t0=5/(f0*bw),
                tau=np.sqrt(2)/(f0*bw), medium=wm.PortMedium(),
                waveform="modulated_gaussian", mode_profile="discrete", mode_type=mode_type)


def numpy_reference(p, length=2**23):
    """Maxwell/Yee admittance, written without production spectral helpers."""
    mu = 4e-7*np.pi
    eps = 8.854187817e-12
    speed = 1/np.sqrt(mu*eps*p['medium'].mu_r*p['medium'].eps_r)
    tt = np.arange(length)*p['dt']-p['t0']
    pulse = np.cos(2*np.pi*p['f0']*tt)*np.exp(-(tt/p['tau'])**2)
    pulse[abs(tt)>p['t0']] = 0
    spectrum = np.fft.rfft(pulse)
    omega = 2*np.pi*np.fft.rfftfreq(length,p['dt'])
    sw = 2/p['dt']*np.sin(omega*p['dt']/2)
    q = (sw/speed)**2-(2*np.pi*p['f_c']/speed)**2
    beta = 2/p['dx']*np.arcsin(np.sqrt(np.clip(q*(p['dx']/2)**2,0,1))).astype(complex)
    ev = q<0
    beta[ev] = -2j/p['dx']*np.arcsinh(np.sqrt(-q[ev])*p['dx']/2)
    sb = 2/p['dx']*np.sin(beta*p['dx']/2)
    admittance = np.zeros_like(beta)
    if p['mode_type']=='TE':
        np.divide(sb,mu*p['medium'].mu_r*sw,out=admittance,where=sw!=0)
    else:
        np.divide(eps*p['medium'].eps_r*sw,sb,out=admittance,where=sb!=0)
    admittance[0]=0
    h = np.fft.irfft(spectrum*admittance*np.exp(.5j*(beta*p['dx']+omega*p['dt'])),length)
    return pulse,h


@pytest.mark.parametrize('n',[4096,2**20+1])
@pytest.mark.parametrize('mode_type',['TE','TM'])
def test_table_accuracy(n,mode_type):
    p=parameters('step' if mode_type=='TE' else 'off_cutoff',mode_type)
    e,h=wm._incident_tables(**p,dft_total_steps=n,field_dtype=jnp.float32)
    er,hr=numpy_reference(p)
    if mode_type=='TM':
        # Spectrum at f_c over its peak, estimate (not yet measured): Gaussian part exp(-(pi*tau*(f0-f_c))^2) = 5e-12;
        # the +-t0 truncation (envelope 3.7e-6 there) adds about 8e-7.
        spectrum=abs(np.fft.rfft(er))
        at_cutoff=spectrum[round(p['f_c']*len(er)*p['dt'])]
        assert at_cutoff<=1e-4*spectrum.max(), 'T2 TM source off cutoff'
    peak=abs(hr).max()
    assert np.max(abs(np.asarray(h)-hr[:n])) <= 1e-5*peak, 'T2 causal H accuracy'
    assert np.max(abs(np.asarray(e)-er[:n])) <= 1e-5*abs(er).max(), 'T2 E accuracy'
    if mode_type=='TE' and n>2**19:
        assert np.all(np.asarray(h)[2**19:]==0), 'T2 zero beyond K'
        assert np.all(np.asarray(e)[2**19:]==0), 'T2 E zero beyond K'


def test_tm_source_at_cutoff_refused():
    # The TM admittance diverges at cutoff, so the H tail falls only as t^(-1/2).
    with pytest.raises(ValueError) as refusal:
        wm._incident_tables(**parameters('step','TM'),dft_total_steps=4096,field_dtype=jnp.float32)
    message=str(refusal.value)
    assert 'f0=7450000000.0' in message and 'f_c=6555808844.538607' in message, message
    assert abs(float(message.split('bandwidth=')[1].split(',')[0])-.6)<=1e-12, message


def test_tm_bin_exactly_on_cutoff_refused(monkeypatch):
    # A bin with b = 0 gives Z_TM = 0; without the guard e/Z makes every table entry NaN.
    real=wm._compute_mode_impedance
    def zero_at_cutoff(freqs,f_cutoff,mode_type,**kw):
        z=real(freqs,f_cutoff,mode_type,**kw)
        z[np.argmin(abs(freqs-f_cutoff))]=0
        return z
    monkeypatch.setattr(wm,'_compute_mode_impedance',zero_at_cutoff)
    with pytest.raises(ValueError,match='exactly on the TM cutoff'):
        wm._incident_tables(**parameters('off_cutoff','TM'),dft_total_steps=4096,field_dtype=jnp.float32)


@pytest.mark.parametrize('case',['step','tutorial'])
def test_negative_time_causality(case):
    _,h=wm._incident_table_build(**parameters(case),length=2**20)
    assert abs(h[len(h)//2:]).max() <= 2e-5*abs(h).max(), 'T2b negative-time precursor'


def test_direct_run_prefix():
    from unittest.mock import patch
    from rfx import Simulation
    import rfx.stepping.uniform_loop as stepping
    from rfx.measurement.waveguide_s import extract_waveguide_port_waves
    n=600
    series=('v_probe_t','v_ref_t','i_probe_t','i_ref_t','v_inc_t')
    def run(steps):
        dx=.002
        fc=299792458/np.pi/dx*np.sin(np.pi/20)
        sim=Simulation(freq_max=10e9,domain=(.05,.02,.01),dx=dx,
                       boundary=dict(x='cpml',y='pec',z='pec'),cpml_layers=4)
        sim.add_waveguide_port(.014,direction='+x',name='p',
                              f0=fc,bandwidth=.6,freqs=jnp.asarray([8e9]))
        sim.add_waveguide_port(.036,direction='-x',name='q',
                              f0=fc,bandwidth=.6,freqs=jnp.asarray([8e9]))
        captured=[]
        direct=stepping.run
        def capture(*args,**kwargs):
            captured.append(direct(*args,**kwargs).waveguide_ports[0])
            raise StopIteration
        with patch.object(stepping,'run',capture), pytest.raises(StopIteration):
            sim.compute_waveguide_s_matrix(n_steps=steps,normalize=False,skip_preflight=True)
        return captured[0]
    short,long=run(n),run(3*n//2)
    # An n-step run records n-1 entries; compare exactly the recorded ones.
    m=int(short.n_steps_recorded)
    assert 0<m<=int(long.n_steps_recorded), 'T1 recorded counts'
    for name in series:
        np.testing.assert_array_equal(np.asarray(getattr(short,name))[:m],
                                      np.asarray(getattr(long,name))[:m],
                                      err_msg=f'T1 prefix {name}')
    prefix=long._replace(**{name:getattr(long,name)[:len(getattr(short,name))] for name in series},
                         dft_total_steps=short.dft_total_steps,n_steps_recorded=short.n_steps_recorded)
    a,b=extract_waveguide_port_waves(short)
    ap,bp=extract_waveguide_port_waves(prefix)
    np.testing.assert_allclose(b/a,bp/ap,atol=1e-6,rtol=0,err_msg='T1 S prefix')
