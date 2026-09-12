"""One-step same-Yee mode diagnostic. No eigensolver or transient campaign.

Input cross-section arrays contain physical phasors at common reference x=0,
with exp(+i*omega*t-i*beta*x). Each component already occupies its OWN transverse
Yee coordinates. Ex/Hy/Hz are shifted by +dx/2 along x; Ey/Ez/Hx are not.
Raw same-index H records therefore need their spatial phase removed before
being presented as common-reference input. They also need DFT time alignment.

Three physical x planes (-dx,0,+dx) are sufficient for the middle plane's
second-order H-then-E update. No Bloch seam or finite-length periodic assumption
is used; complex beta is allowed. Only the middle plane is scored.

CPML is deliberately refused. For each actual recursion psi_new=b*psi_old+c*D,
harmonic initialization requires psi_old=c*D/(exp(i*omega*dt)-b), where D is the
derivative at THAT E/H substep's actual time level. H-psi uses E^n; E-psi uses
H^(n+1/2). All auxiliary buffers, kappa terms, active faces and canonical PEC
must participate in the update; zero psi is a startup transient, not a mode.
Other ADE/anisotropic/conformal/source/load operators are not modeled here.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
import rfx.core.yee as yee
from rfx.boundaries.pec import apply_pec_faces, apply_pec_edges

assert Path(yee.__file__).resolve().is_relative_to(ROOT)
NAMES = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
X_OFFSETS = dict(ex=.5, ey=0., ez=0., hx=0., hy=.5, hz=.5)


def evaluate(reference_fields, *, beta, frequency_hz, dx, dt,
             eps_r=1., mu_r=1., sigma=0., periodic_yz=(True, True),
             pec_faces=(), pec_edge_masks=None, arithmetic='real_quadratures',
             native_storage='complex64', cpml=False):
    """One-step residual of the actual scalar uniform order-2 Yee+PEC operator.

    real_quadratures applies the same real-linear production operator once to
    each quadrature; it does not modify core code or claim native complex128.
    Stored phasors represent E^n and H^(n-1/2), not simultaneous E/H.
    omega_dt_scaled_step_residual is a Maxwell-scale INDICATOR, not a literal
    continuous curl-equation residual. Use omega_discrete when comparing a
    frequency-domain discretization. PEC compatibility is scored BEFORE clearing.
    """
    if cpml:
        raise NotImplementedError('CPML needs harmonic auxiliary-state initialization; zero psi is not valid.')
    if arithmetic not in ('real_quadratures', 'native_complex'):
        raise ValueError(arithmetic)
    if not np.isfinite([dx,dt,frequency_hz]).all() or min(dx,dt,frequency_hz)<=0:
        raise ValueError('dx, dt and frequency must be positive finite values')
    theta = 2*np.pi*frequency_hz*dt
    if not 0 < theta < np.pi:
        raise ValueError('Require a positive frequency below the temporal Nyquist limit')
    if any(face.startswith('x_') for face in pec_faces):
        raise ValueError('Propagation-face PEC is incompatible with the invariant cross-section slab')
    transverse = {name: np.asarray(reference_fields[name], dtype=complex) for name in NAMES}
    shape = transverse['ex'].shape
    if len(shape)!=2 or any(value.shape!=shape for value in transverse.values()):
        raise ValueError('All six component arrays must have the same [y,z] index shape')
    arrays = {}
    for name, profile in transverse.items():
        x = (np.arange(3)-1+X_OFFSETS[name])*dx
        phase = np.exp(-1j*complex(beta)*x)
        if name.startswith('h'):
            phase *= np.exp(-.5j*theta)
        arrays[name] = phase[:,None,None]*profile[None,:,:]
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite lifted phasor')
    material = {}
    for name, value in (('eps_r',eps_r), ('mu_r',mu_r), ('sigma',sigma)):
        a=np.broadcast_to(np.asarray(value,dtype=float),shape)
        if not np.isfinite(a).all() or np.any(a < 0) or (name!='sigma' and np.any(a==0)):
            raise ValueError('Require positive eps/mu and nonnegative scalar sigma')
        material[name]=np.broadcast_to(a,(3,)+shape).copy()
    if dt/dx/np.sqrt(yee.EPS_0*yee.MU_0*np.min(material['eps_r'])*np.min(material['mu_r'])) >= 1/np.sqrt(3):
        raise ValueError('Diagnostic requires the conservative cubic-grid CFL bound')
    masks=None
    if pec_edge_masks is not None:
        if len(pec_edge_masks)!=3 or any(np.shape(m)!=shape for m in pec_edge_masks):
            raise ValueError('Pass canonical per-component [y,z] PEC masks')
        masks=tuple(jnp.asarray(np.broadcast_to(np.asarray(m,dtype=bool),(3,)+shape)) for m in pec_edge_masks)
    faces=set(pec_faces)
    periodic=(False,)+tuple(periodic_yz)
    if any(axis in face and periodic[k] for k,axis in ((1,'y'),(2,'z')) for face in faces):
        raise ValueError('A transverse axis cannot be both periodic and PEC')

    def constrain(state):
        state=apply_pec_faces(state,faces)
        return apply_pec_edges(state,masks) if masks is not None else state

    def update(input_arrays, dtype):
        state=yee.FDTDState(**{k:jnp.asarray(v,dtype=dtype) for k,v in input_arrays.items()},step=jnp.int32(0))
        mats=yee.MaterialArrays(**{k:jnp.asarray(v,dtype=jnp.float64) for k,v in material.items()})
        st=yee.update_h(state,mats,dt,dx,periodic=periodic,stencil_order=2)
        st=yee.update_e(st,mats,dt,dx,periodic=periodic,stencil_order=2)
        st=constrain(st)
        assert int(st.step)==1
        return {k:np.asarray(getattr(st,k)) for k in NAMES}

    with enable_x64():
        # Report incompatible candidate fields; never silently repair them first.
        before=yee.FDTDState(**{k:jnp.asarray(v) for k,v in arrays.items()},step=jnp.int32(0))
        constrained=constrain(before)
        e_peak=max(np.max(abs(arrays[k][1])) for k in NAMES[:3])
        pec_mismatch=max(np.max(abs(np.asarray(getattr(constrained,k))[1]-arrays[k][1])) for k in NAMES[:3])
        if arithmetic=='real_quadratures':
            re=update({k:v.real for k,v in arrays.items()},jnp.float64)
            im=update({k:v.imag for k,v in arrays.items()},jnp.float64)
            after={k:re[k]+1j*im[k] for k in NAMES}
        else:
            after=update(arrays,{'complex64':jnp.complex64,'complex128':jnp.complex128}[native_storage])
    target={k:np.exp(1j*theta)*arrays[k][1] for k in NAMES}
    errors={k:after[k][1]-target[k] for k in NAMES}
    def energy(values):
        return sum(np.sum((yee.EPS_0*material['eps_r'][1] if k[0]=='e' else yee.MU_0*material['mu_r'][1])*abs(v)**2) for k,v in values.items())
    reference_energy=energy(target)
    if reference_energy<=0:
        raise ValueError('All-zero candidate')
    rho=float(np.sqrt(energy(errors)/reference_energy))
    return dict(arithmetic=arithmetic,native_storage=(native_storage if arithmetic=='native_complex' else None),
                native_complex_core_field_cast='complex64 regardless of requested complex storage',
                beta_per_m=[float(np.real(beta)),float(np.imag(beta))], frequency_hz=float(frequency_hz),
                omega_dt=float(theta),omega_discrete_per_s=float(2*np.sin(theta/2)/dt),
                timestep_relative_energy_norm_residual=rho,omega_dt_scaled_step_residual=rho/theta,
                exact_phase_increment_scaled_step_residual=rho/abs(np.expm1(1j*theta)),
                pec_input_relative_mismatch=float(pec_mismatch/max(e_peak,np.finfo(float).tiny)),
                max_absolute_component_residuals={k:float(np.max(abs(v))) for k,v in errors.items()},
                scored_x_node=0., slab_x_nodes_m=[-dx,0.,dx], transverse_shape=list(shape),cpml_supported=False)


def self_test():
    dx=2.**-10
    c=1/np.sqrt(yee.EPS_0*yee.MU_0)
    eta=np.sqrt(yee.MU_0/yee.EPS_0)
    dt=.5*dx/c
    kdx=np.pi/8
    omega=2*np.arcsin(c*dt/dx*np.sin(kdx/2))/dt
    frequency=omega/(2*np.pi)
    records=[]
    for plate in (False,True):
        for direction in (1,-1):
            fields={k:np.zeros((4,6),dtype=complex) for k in NAMES}
            fields['ez'][:]=1; fields['hy'][:]=-direction/eta
            if plate:
                fields['ez'][:,-1]=0;fields['hy'][:,-1]=0
            kw=dict(beta=direction*kdx/dx,frequency_hz=frequency,dx=dx,dt=dt,
                    periodic_yz=(True,not plate),pec_faces=('z_lo','z_hi') if plate else ())
            result=evaluate(fields,**kw)
            assert result['omega_dt_scaled_step_residual']<2e-13,result
            assert result['pec_input_relative_mismatch']==0
            records.append(dict(case=('parallel_plate' if plate else 'vacuum')+f'_direction_{direction}',**result))
    # A known parallel-plate TE1 mode exercises nonconstant transverse
    # staggering: Ey/Hz at z nodes and Hx at z half-cells.
    nz=9; beta=kdx/dx; kz=np.pi/((nz-1)*dx)
    kx_discrete=2*np.sin(beta*dx/2)/dx
    kz_discrete=2*np.sin(kz*dx/2)/dx
    omega_te=2*np.arcsin(c*dt/2*np.hypot(kx_discrete,kz_discrete))/dt
    omega_d=2*np.sin(omega_te*dt/2)/dt
    te={k:np.zeros((4,nz),dtype=complex) for k in NAMES}
    te['ey'][:]=np.sin(kz*np.arange(nz)*dx);te['ey'][:,[0,-1]]=0
    te['hz'][:]=kx_discrete/(omega_d*yee.MU_0)*te['ey']
    te['hx'][:]=-1j*kz_discrete/(omega_d*yee.MU_0)*np.cos(kz*(np.arange(nz)+.5)*dx)
    te['hx'][:,-1]=0
    te_kw=dict(beta=beta,frequency_hz=omega_te/(2*np.pi),dx=dx,dt=dt,
               periodic_yz=(True,False),pec_faces=('z_lo','z_hi'))
    result=evaluate(te,**te_kw)
    assert result['omega_dt_scaled_step_residual']<2e-13,result
    assert result['pec_input_relative_mismatch']==0
    records.append(dict(case='parallel_plate_TE1',**result))
    bad_te={k:v.copy() for k,v in te.items()}
    bad_te['hx'][:]=-1j*kz_discrete/(omega_d*yee.MU_0)*np.cos(kz*np.arange(nz)*dx)
    bad_te['hx'][:,-1]=0
    result=evaluate(bad_te,**te_kw)
    assert result['omega_dt_scaled_step_residual']>.01,result
    records.append(dict(case='TE1_wrong_Hx_transverse_z_nodes',**result))
    fields={k:np.zeros((4,6),dtype=complex) for k in NAMES}
    fields['ez'][:]=1;fields['hy'][:]=-1/eta
    kw=dict(beta=kdx/dx,frequency_hz=frequency,dx=dx,dt=dt)
    for storage in ('complex64','complex128'):
        result=evaluate(fields,**kw,arithmetic='native_complex',native_storage=storage)
        assert result['omega_dt_scaled_step_residual']<2e-5
        records.append(dict(case='native_'+storage,**result))
    for kind,phase in (('missing_H_half_space',np.exp(1j*kdx/2)),('missing_H_half_time',np.exp(1j*omega*dt/2))):
        bad={k:v.copy() for k,v in fields.items()};bad['hy']*=phase
        result=evaluate(bad,**kw)
        assert result['omega_dt_scaled_step_residual']>.01,result
        records.append(dict(case=kind,**result))
    for scale in (1.,.01):
        result=evaluate(fields,beta=kdx/dx,frequency_hz=1.1*frequency,dx=dx,dt=dt*scale)
        assert result['omega_dt_scaled_step_residual']>.05,result
        records.append(dict(case=f'detuned_frequency_dt_scale_{scale}',**result))
    assert records[-1]['timestep_relative_energy_norm_residual']<.03*records[-2]['timestep_relative_energy_norm_residual']
    try:
        evaluate(fields,**kw,cpml=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('CPML without harmonic psi must not be accepted')
    report=dict(scope='Bounded independent one-step checks only: vacuum TEM and PEC parallel-plate TEM/TE1. No eigenmode solver, transient, CPML claim, production change or AD change.',
                rfx_yee_path=yee.__file__,yee_sha256=hashlib.sha256(Path(yee.__file__).read_bytes()).hexdigest(),
                jax_version=jax.__version__,records=records,
                limits=['Physical input fields must use own transverse Yee positions and a common x-reference phasor convention.',
                        'Native complex update demotes fields to complex64 even for complex128 storage.',
                        'Real-quadrature path evaluates real float64 production update twice, once per quadrature.',
                        'Scaled step residual is not a direct continuous Maxwell curl residual.',
                        'CPML/ADE/conformal/anisotropic/source/load extensions require their full actual update state.'])
    out=ROOT/'.git/issue726-yee-mode-residual.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    self_test()
