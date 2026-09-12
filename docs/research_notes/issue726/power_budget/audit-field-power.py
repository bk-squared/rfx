"""Independent NumPy face/VI power Grams from frozen raw field records.

Unnormalized Fourier phasor products: report dimensionless ratios, never W.
No S fitting, passivity projection, inferred field samples or calibration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

COMPONENTS = {0: ('ey','ez','hy','hz'), 1: ('ez','ex','hz','hx'), 2: ('ex','ey','hx','hy')}


def weights(face, component, dx):
    """Second-order Yee SBP face rule: cell along E, nodal across E."""
    lo1, hi1, lo2, hi2 = face['region']
    counts = (hi1-lo1, hi2-lo2)
    tangent = face['tangential_axes']
    result = []
    for count, axis in zip(counts, tangent):
        w = np.ones(count)
        if axis == 'xyz'.index(component[1]):
            w[-1] = 0  # E-axis edge at the upper node lies outside the face.
        else:
            w[[0,-1]] = .5
        result.append(w)
    return result[0][:,None]*result[1][None,:]*dx**2


def hermitian(k):
    return (k + k.swapaxes(-1,-2).conj())/2


def gram(e, h, w):
    # Input [drive,freq,u,v]. Output [freq,conjugated drive,other drive].
    return np.einsum('dfuv,efuv,uv->fed', e, h.conj(), w, optimize=True)


def face_gram(face, fields, dx):
    e1,e2,h1,h2 = COMPONENTS[face['axis']]
    if face.get('zero_flux_pec_boundary'):
        assert np.max(abs(fields[e1])) == 0 and np.max(abs(fields[e2])) == 0
        return np.zeros((fields[e1].shape[1],fields[e1].shape[0],fields[e1].shape[0]), dtype=complex)
    return hermitian(gram(fields[e1], fields[h2], weights(face,e1,dx))
                     - gram(fields[e2], fields[h1], weights(face,e2,dx)))


def prepare_fields(face, raw, freqs, dt):
    fields = {}
    for component, names in face['probes'].items():
        arrays = [np.stack([np.asarray(d[name], dtype=np.complex128) for d in raw]) for name in names]
        if component[0] == 'h':
            assert len(arrays) == 2
            fields[component] = .5*(arrays[0]+arrays[1])*np.exp(1j*np.pi*freqs*dt)[None,:,None,None]
        else:
            assert len(arrays) == 1
            fields[component] = arrays[0]
    return fields


def to_input_basis(q, inverse_a):
    return inverse_a.swapaxes(-1,-2).conj() @ q @ inverse_a


def profile_coherence(face, fields, pair, dx):
    # A one-mode E or H profile is proportional across both drive records.
    e1,e2,h1,h2 = COMPONENTS[face['axis']]
    comps = (e1,e2) if pair == 'e' else (h2,h1)
    vec = np.concatenate([(fields[c]*np.sqrt(weights(face,e,dx))[None,None]).reshape(2,len(fields[c][0]),-1)
                          for c,e in zip(comps,(e1,e2))], axis=-1)
    norms = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        coherence = abs(np.sum(vec[0].conj()*vec[1],axis=-1))/(norms[0]*norms[1])
    return coherence, norms


def station_vi(face, fields, span, trace_k, dx):
    assert face['axis'] == 0
    ylo,_,zlo,_ = face['region']
    j0,j1,jc = (span[k]-ylo for k in ('w_lo','w_hi','w_centre'))
    ground = span['n_lo']-zlo
    k = trace_k-zlo
    v_paths = fields['ez'][:,:,j0:j1+1,ground:k].sum(axis=-1)*dx
    v = fields['ez'][:,:,jc,ground:k].sum(axis=-1)*dx
    hy,hz = fields['hy'],fields['hz']
    # Native +x MSL current, paired with native +sum(Ez) voltage.
    # Both are opposite the trace-to-ground / physical +x conductor pair.
    i = -dx*(hy[:,:,j0:j1+1,k-1].sum(axis=-1) - hy[:,:,j0:j1+1,k].sum(axis=-1)
             + hz[:,:,j1,k] - hz[:,:,j0-1,k])
    return v,i,v_paths


def self_test():
    rng=np.random.default_rng(726)
    ecoeff=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    hcoeff=rng.normal(size=(2,3))+1j*rng.normal(size=(2,3))
    # H has a prescribed affine global profile; the exact surface average
    # equals its centre value. E is constant. No production helper is used.
    dx=.25
    for axis in range(3):
        tangent=[a for a in range(3) if a != axis]
        face=dict(axis=axis,region=[2,7,3,10],tangential_axes=tangent)
        fields={}
        for comp in COMPONENTS[axis]:
            c='xyz'.index(comp[1]); coords=[]
            for a,idx in zip(tangent,(np.arange(2,7),np.arange(3,10))):
                # Co-located H shares the paired E's transverse lattice.
                shifted=(a==c) if comp[0]=='e' else (a!=c)
                coords.append((idx+.5*shifted)*dx)
            profile=1+coords[0][:,None]+2*coords[1][None,:]
            if comp[0]=='e':
                fields[comp]=np.broadcast_to(ecoeff[:,c,None,None,None],(2,1,5,7)).copy()
            else:
                fields[comp]=hcoeff[:,c,None,None,None]*profile[None,None]
        e1,e2,h1,h2=COMPONENTS[axis]
        ix=lambda c:'xyz'.index(c[1])
        centre_profile=1+(2+6)*dx/2+2*(3+9)*dx/2
        area=(6-2)*(9-3)*dx**2
        k=np.outer(hcoeff[:,ix(h2)].conj(),ecoeff[:,ix(e1)])-np.outer(hcoeff[:,ix(h1)].conj(),ecoeff[:,ix(e2)])
        expected=hermitian(k*centre_profile*area)
        np.testing.assert_allclose(face_gram(face,fields,dx)[0],expected,atol=1e-13)
        c=np.array([1,1j])
        combined={n:np.einsum('d,dfuv->fuv',c,a)[None] for n,a in fields.items()}
        direct=face_gram(face,combined,dx)[0,0,0]
        np.testing.assert_allclose(c.conj()@expected@c,direct,atol=1e-13)
    print('PASS: x/y/z component-specific Yee quadrature and complex coherent cross terms')


def audit(root, out):
    plan=json.loads((root/'plan.json').read_text())
    freqs=np.asarray(plan['field_freqs_hz']); dx=plan['dx_m']; dt=plan['dt_s']
    raw=[]
    for d in range(2):
        with np.load(root/f'drive-{d}-fields.npz',allow_pickle=False) as f:
            raw.append(dict(f))
    with np.load(root/'raw-vi.npz',allow_pickle=False) as f:
        meta=json.loads(str(f['metadata_json'])); selected=plan['selected_indices']
        v=np.asarray(f['raw_v'][:,:,0,selected],dtype=complex)
        i=np.asarray(f['raw_i1'][:,:,selected],dtype=complex)
        r=np.asarray(meta['s_reference_impedances_ohm'])
        s=np.asarray(f['production_smatrix'][:,:,selected]).transpose(2,0,1)
    a=((v+r[None,:,None]*i)/(2*np.sqrt(r)[None,:,None])).transpose(2,1,0)
    inverse_a=np.linalg.inv(a)
    q_vi=hermitian(np.einsum('dpf,epf->fed',v,i.conj()))
    q_vi_normal=to_input_basis(q_vi,inverse_a)
    np.testing.assert_allclose(q_vi_normal,np.eye(2)[None]-s.swapaxes(-1,-2).conj()@s,atol=2e-6)
    arrays=dict(freqs_hz=freqs,vi_inward_gram=q_vi,vi_inward_in_input_basis=q_vi_normal)
    report=dict(scope='diagnostic only; raw Fourier products are not watts; no calibrated accuracy verdict',
                frequencies_hz=freqs.tolist(),faces=[],stations=[],finite_window_bound=None,
                face_flux_sign='positive coordinate axis; outward applied only in closed sum',
                normalization='A from native measured VI; invertible congruence preserves inertia, but absolute incident power is not independently calibrated',
                profile_coherence_limit='one is necessary for proportional fields, not proof of a single mode or accurate power')
    closed=np.zeros_like(q_vi)
    for face in plan['faces']:
        fields=prepare_fields(face,raw,freqs,dt)
        q=face_gram(face,fields,dx)
        qn=to_input_basis(q,inverse_a)
        arrays[face['name']+'_gram']=q
        report['faces'].append(dict(name=face['name'],outward=face['outward'],
                                    eigs_in_input_basis=np.linalg.eigvalsh(qn).tolist()))
        if face['outward']:
            closed += face['outward']*q
        if face['axis'] == 0:
            vf,ip,vpaths=station_vi(face,fields,plan['port_spans'][0],plan['inputs']['realized_metal']['plane_k'],dx)
            qlocal=hermitian(np.einsum('df,ef->fed',vf,ip.conj()))
            difference=to_input_basis(qlocal-q,inverse_a)
            coh_e,norm_e=profile_coherence(face,fields,'e',dx)
            coh_h,norm_h=profile_coherence(face,fields,'h',dx)
            entry=dict(name=face['name'],x_m=face['coordinate_m'],
                       vi_minus_flux_eigs_in_input_basis=np.linalg.eigvalsh(difference).tolist(),
                       electric_profile_coherence=coh_e.tolist(),magnetic_profile_coherence=coh_h.tolist(),
                       electric_profile_norms=norm_e.tolist(),magnetic_profile_norms=norm_h.tolist())
            if face['name'] in ('x_station_0','x_station_5'):
                p=0 if face['name']=='x_station_0' else 1
                sign=1 if p==0 else -1
                entry['vs_production_v_relative_error']=float(np.max(abs(vf-v[:,p]))/np.max(abs(v[:,p])))
                entry['vs_production_i_relative_error']=float(np.max(abs(sign*ip-i[:,p]))/np.max(abs(i[:,p])))
            report['stations'].append(entry)
            arrays[face['name']+'_v']=vf; arrays[face['name']+'_i_native_plus_x']=ip
            arrays[face['name']+'_v_paths']=vpaths
    cn=to_input_basis(closed,inverse_a)
    report['closed_outward_eigs_in_input_basis']=np.linalg.eigvalsh(cn).tolist()
    report['vi_inward_eigs_in_input_basis']=np.linalg.eigvalsh(q_vi_normal).tolist()
    arrays['closed_outward_gram']=closed; arrays['closed_outward_in_input_basis']=cn
    assert not out.exists() and not out.with_suffix('.npz').exists()
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path); parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.root and args.out
        audit(args.root,args.out)
