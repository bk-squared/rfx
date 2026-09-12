"""Compare the frozen absorbing-lead experiment without rescaling its fields.

Each S uses its own measured incident-wave system. Field changes also use ONE
common original input basis and original worst-gain drive direction, so changed
feed response cannot masquerade as changed lateral flux through normalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(root):
    plan=json.loads((root/'plan.json').read_text())
    with np.load(root/'raw-vi.npz',allow_pickle=False) as f:
        meta=json.loads(str(f['metadata_json']))
        v=np.asarray(f['raw_v'][:,:,0,:],dtype=np.complex128)
        i=np.asarray(f['raw_i1'],dtype=np.complex128)
        s=np.asarray(f['production_smatrix'],dtype=np.complex128)
        freqs=np.asarray(f['freqs_hz'])
    assert meta['s_wave_convention']=='power'
    assert meta['production_smatrix_assembly']=='multi_drive_solve'
    refs=np.asarray(meta['s_reference_impedances_ohm'])
    a=((v+refs[None,:,None]*i)/(2*np.sqrt(refs)[None,:,None])).transpose(2,1,0)
    q=np.einsum('dpf,epf->fed',v,i.conj())
    q=(q+q.conj().transpose(0,2,1))/2
    outcome=json.loads((root/'outcome.json').read_text())
    return dict(plan=plan,v=v,i=i,s=s,f=freqs,refs=refs,a=a,q=q,outcome=outcome)


def spectral_readout(record):
    s=record['s']; power=np.linalg.svd(s.transpose(2,0,1),compute_uv=False)[:,0]**2
    k=int(np.argmax(power));abs_s=abs(s)
    return dict(max_coherent_gain=float(power[k]),frequency_hz=float(record['f'][k]),
                max_column_power=float(np.max(np.sum(abs_s**2,axis=0))),
                max_entry_magnitude=float(np.max(abs_s)),
                settling_db=record['outcome']['settling_db'])


def compare(base_root,new_root,base_audit,new_audit,out):
    old,new=load(base_root),load(new_root)
    for key in ('inputs','box_node_lo','box_node_hi','box_coordinates_m','grid_shape','dt_s','dx_m',
                'all_freqs_hz','selected_indices','field_freqs_hz','faces','probes','port_spans',
                'num_periods','passivity_projection'):
        assert old['plan'][key]==new['plan'][key],key
    assert new['plan']['lead_change']['all_material_arrays_identical']
    assert new['plan']['lead_change']['source_load_declarations_identical']
    np.testing.assert_array_equal(old['f'],new['f'])
    np.testing.assert_array_equal(old['refs'],new['refs'])
    selected=old['plan']['selected_indices']
    inv=np.linalg.inv(old['a'][selected])
    def original_basis(q):
        return inv.conj().transpose(0,2,1)@q@inv
    records=[]
    for rec,path in ((old,base_audit),(new,new_audit)):
        with np.load(path,allow_pickle=False) as a:
            xq=-a['x_station_0_gram']+a['x_station_5_gram']
            records.append(dict(vi_out=original_basis(-rec['q'][selected]),
                                x_out=original_basis(xq),
                                side_out=original_basis(a['closed_outward_gram']-xq),
                                closed_out=original_basis(a['closed_outward_gram'])))
    rows=[]
    for k,full_k in enumerate(selected):
        _eig,vec=np.linalg.eigh(records[0]['vi_out'][k]);c=vec[:,-1]
        row=dict(frequency_hz=float(old['f'][full_k]),fixed_original_input_vector_complex=[[float(z.real),float(z.imag)] for z in c])
        for label,rec in zip(('declared_ends','continued_lead'),records):
            row[label]={name:float(np.real(c.conj()@q[k]@c)) for name,q in rec.items()}
        rows.append(row)
    report=dict(scope='one explicit exterior lead change; causal diagnostic, no calibrated accuracy verdict',
                field_comparison_basis='same old VI A basis and same old worst-VI-gain drive combination for both records',
                normalization_limit='original VI A does not independently calibrate incident electromagnetic power',
                lead_change=new['plan']['lead_change'],
                original=spectral_readout(old),continued=spectral_readout(new),
                selected_fixed_drive_power=rows,
                max_raw_s_change=float(np.max(abs(new['s']-old['s']))),
                settled=all(np.all(np.asarray(r['outcome']['settling_db'])<-40) for r in (old,new)),
                physical_accuracy_verdict=None,
                inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (base_root/'raw-vi.npz',new_root/'raw-vi.npz',base_audit,new_audit)})
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--baseline-audit',type=Path,required=True);ap.add_argument('--candidate-audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    compare(a.baseline,a.candidate,a.baseline_audit,a.candidate_audit,a.out)
