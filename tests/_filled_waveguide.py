"""WR-90 filled-guide and dielectric-step oracles with independent modal formulas."""
import warnings
import numpy as np
from rfx import Simulation, Box

A, B, L, EPS = .02286, .01016, .120, 2.2
C0 = 299792458.0


def build(case, cells=45, *, layers=12, shift=False, one_port=False):
    dx = A/cells
    interface = round(.060/dx)*dx
    step_bins = np.arange(70,80,dtype=float)*1e8
    # O1 includes O2's exact bins for the leader's filled-side comparator.
    freqs = (np.unique(np.r_[np.linspace(5e9,7.9e9,9),step_bins])
             if case in ('O1','W2','W2_empty','empty_filled_band') else step_bins)
    if case in ('empty_filled_band','W2_empty'):
        freqs = freqs*np.sqrt(EPS)
    sim = Simulation(freq_max=14e9, domain=(L,A,B), dx=dx,
                     boundary=dict(x='cpml',y='pec',z='pec'),cpml_layers=layers,
                     snap='declared')
    if case in ('O1','O2','W2'):
        sim.add_material('fill',eps_r=EPS)
        sim.add(Box((interface if case=='O2' else -.05,0,0),(L+.05,A,B)),material='fill')
    kw=dict(mode=(1,0),mode_type='TE',freqs=freqs,f0=float(np.mean(freqs)),bandwidth=.6)
    sim.add_waveguide_port(.020,direction='+x',name='p1',
                          reference_plane=.050 if shift else (interface-.040 if case=='O2' else .020),**kw)
    if one_port:
        # The public probe_plane is a reporting-plane shift, not a recorder
        # placement. W2 instruments the existing modal recorder upstream before
        # setup/scan; it leaves the injection and reference recorder untouched.
        builder = sim._build_waveguide_port_config
        def upstream_recorder(entry, grid, freqs, n_steps):
            cfg = builder(entry, grid, freqs, n_steps)
            index = grid.index_of('x', .010)
            return cfg._replace(probe_x=index, probe_x_m=grid.node_of('x', index))
        sim._build_waveguide_port_config = upstream_recorder
    if not one_port:
        sim.add_waveguide_port(.100,direction='-x',name='p2',reference_plane=.070 if shift else (interface+.040 if case=='O2' else .100),**kw)
    return sim,freqs


def preflight_findings(sim, freqs, n_steps):
    """Capture general and actual-record setup checks, without launching FDTD."""
    issues = list(sim.preflight())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim._preflight_waveguide_setup(warnings, freqs=freqs, n_steps=n_steps)
    rows = [dict(code=i.code, severity=i.severity, message=str(i)) for i in issues]
    rows += [dict(code=getattr(w.message,'code','uncoded'),
                  severity=getattr(w.message,'severity','warning'),message=str(w.message)) for w in caught]
    return rows


def assert_preflight_clean(rows):
    propagation = {'port_source_below_cutoff','port_freqs_below_cutoff','port_evanescent',
                   'record_far_boundary_band_below_cutoff','waveguide_setup_audit_skipped'}
    bad = [r for r in rows if r['severity']=='error' or r['code'] in propagation]
    assert not bad, f'filled-guide preflight cutoff/propagation clean: {bad}'


def beta(freqs, eps, grid, *, discrete):
    """Closed form written here, independent of port implementation helpers."""
    omega = 2*np.pi*freqs
    if not discrete:
        return np.sqrt(eps*(omega/C0)**2-(np.pi/A)**2)
    dx,dt=float(grid.dx),float(grid.dt)
    kc=2/dx*np.sin(np.pi/(2*round(A/dx)))
    return 2/dx*np.arcsin(dx/2*np.sqrt((np.sin(omega*dt/2)/(C0/np.sqrt(eps)*dt/2))**2-kc**2))


def measure(case,cells=45,*,layers=12,shift=False):
    sim,f=build(case,cells,layers=layers,shift=shift,one_port=case.startswith('W2'))
    grid=sim._build_realized_grid()
    n_steps=round(8000*cells/45)
    preflight = preflight_findings(sim,f,n_steps) if case in ('O1','O2') else []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if case.startswith('W2'):
            from rfx.measurement.waveguide_s import (
                extract_waveguide_port_waves, _extract_port_waves_from_time_series,
            )
            result=sim.run(n_steps=n_steps,skip_preflight=True)
            cfg=result.waveguide_ports['p1']
            assert cfg.probe_x >= grid.pad_x_lo + 3, 'W2 probe clearance from CPML'
            assert cfg.probe_x <= cfg.x_index - 1 - 3, 'W2 probe clearance from H correction'
            a,_=extract_waveguide_port_waves(cfg)
            _,away=_extract_port_waves_from_time_series(cfg,cfg.v_probe_t,cfg.i_probe_t)
            ratio=np.asarray(away/a)
            return dict(case=case,cells=cells,layers=layers,dx=float(grid.dx),dt=float(grid.dt),
                        n_steps=n_steps,source_index=cfg.x_index,probe_index=cfg.probe_x,
                        cpml_end_index=grid.pad_x_lo,probe_plane=cfg.probe_x_m,
                        freqs=f.tolist(),reflection=np.abs(ratio).tolist(),
                        reflection_complex=[[float(z.real),float(z.imag)] for z in ratio],
                        warnings=[str(w.message) for w in caught])
        result=sim.compute_waveguide_s_matrix(n_steps=n_steps,normalize=False,skip_preflight=True)
    s=np.asarray(result.s_params)
    eps=EPS if case=='O1' else 1.
    distance=abs(np.diff(result.reference_planes)[0])
    bd,bc=beta(f,eps,grid,discrete=True),beta(f,eps,grid,discrete=False)
    phase=-np.angle(s[1,0]*np.exp(1j*bd*distance))
    phase_cont=-np.angle(s[1,0]*np.exp(1j*bc*distance))
    b0,bf=(beta(f,1.,grid,discrete=False),beta(f,EPS,grid,discrete=False)) if case=='O2' else (np.ones_like(f),np.ones_like(f))
    expected_r=(b0-bf)/(b0+bf) if case=='O2' else np.zeros_like(f)
    expected_t=2*np.sqrt(b0*bf)/(b0+bf) if case=='O2' else np.ones_like(f)
    b0d = beta(f,1.,grid,discrete=True) if case=='O2' else np.zeros_like(f)
    interface_phase=np.angle(s[0,0]*np.exp(2j*b0d*.04)/(-1))
    return dict(case=case,cells=cells,layers=layers,shift=shift,dx=float(grid.dx),dt=float(grid.dt),
                n_steps=n_steps,shape=grid.shape,interface=round(.060/float(grid.dx))*float(grid.dx),freqs=f.tolist(),planes=np.asarray(result.reference_planes).tolist(),
                s11_abs=np.abs(s[0,0]).tolist(),s21_abs=np.abs(s[1,0]).tolist(),
                s11_complex=[[float(z.real),float(z.imag)] for z in s[0,0]],
                s21_complex=[[float(z.real),float(z.imag)] for z in s[1,0]],
                sum=(np.abs(s[0,0])**2+np.abs(s[1,0])**2).tolist(),
                phase_error_discrete=(phase/(bd*distance)).tolist(),
                phase_error_continuous=(phase_cont/(bc*distance)).tolist(),
                s11_closed=expected_r.tolist(),s21_closed=expected_t.tolist(),
                interface_phase_error=interface_phase.tolist(),
                half_cell_phase=(b0d*float(grid.dx)/2).tolist(),
                preflight=preflight,settling_db=np.asarray(result.settling_db).tolist(),warnings=[str(w.message) for w in caught])
