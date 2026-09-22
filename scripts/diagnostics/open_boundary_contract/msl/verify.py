import ast
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path('/root/workspace/bk-workspace/.801-measure/msl')
SRC=OUT.parent/'src-main'

def read(path):
    return json.loads(Path(path).read_text())

def main():
    spec=ast.parse((SRC/'rfx/api/_spec.py').read_text())
    result_class=next(n for n in spec.body if isinstance(n,ast.ClassDef) and n.name=='MSLSMatrixResult')
    expected={n.target.id for n in result_class.body if isinstance(n,ast.AnnAssign)}
    records={}
    for fixture,nfreq in (('cv06b',100),('cv20',30)):
        reduced=read(OUT/(fixture+'_reduced.json'))
        prov=read(OUT/fixture/'provenance.json')
        hashes={k:int(hashlib.sha256((OUT.parent/k).read_bytes()).hexdigest()==v) for k,v in prov['sha256'].items()}
        assert all(hashes.values()),hashes
        dry=read(OUT/(fixture+'_dry')/'baseline/status.json')
        assert dry['timestepping_calls']==0
        per={}
        for variant in ('baseline','continued'):
            folder=OUT/fixture/variant
            diagnostics=read(folder/'diagnostics.json')
            assert set(diagnostics)==expected,(set(diagnostics),expected)
            status=read(folder/'status.json')
            assert status['completed']==1 and status['solve_calls']==2 and status['timestepping_calls']==2
            with np.load(folder/'s.npz') as z:
                s=z['S'];freqs=z['freqs']
            assert s.shape==(2,2,nfreq) and freqs.shape==(nfreq,)
            assert np.isfinite(s).all() and np.all(np.diff(freqs)>0)
            with np.load(folder/'diagnostics.npz') as z:
                assert np.array_equal(z['S'],s) and np.array_equal(z['freqs'],freqs)
            drive_hashes=[]
            source_indices=[]
            for drive in range(2):
                rec=reduced['edges'][variant][drive]
                source_indices.append(rec['source_x_indices_received'])
                with np.load(folder/f'assembly_received_{drive:02d}.npz') as z:
                    checks={k:int(hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()==v) for k,v in rec['sha256'].items()}
                    assert all(checks.values()),checks
                    for plane in rec['planes']:
                        j=plane['y_index'];k=plane['z_index']
                        for w in plane['windows'].values():
                            ix=w['x_indices']
                            for field in ('pec_edge_x','pec_edge_y','pec_mask'):
                                assert np.array_equal(z[field][ix,j,k],w[field])
                            assert np.array_equal(z['eps_r'][ix,j,k-1],w['eps_r_below'])
                drive_hashes.append(checks)
            d=reduced['variants'][variant]
            assert all(abs(w['result_minus_recomputed_db'])<1e-10 for w in d['ringdown_recomputed'])
            per[variant]=dict(S_shape=list(s.shape),S_finite_count=int(np.isfinite(s).sum()),
                frequency_count=len(freqs),frequency_first_hz=float(freqs[0]),frequency_last_hz=float(freqs[-1]),
                diagnostic_field_count=len(diagnostics),expected_diagnostic_field_count=len(expected),
                solve_calls=status['solve_calls'],mask_hash_equal=drive_hashes,
                source_x_indices_received=source_indices,
                ringdown_max_abs_recompute_difference_db=max(abs(w['result_minus_recomputed_db']) for w in d['ringdown_recomputed']))
        assert reduced['checks']['frequency_vectors_equal']==1
        assert all(reduced['checks'][a+'_changed_entries']==0 for a in ('eps_r','mu_r','sigma'))
        records[fixture]=dict(source_hash_equal=hashes,dry_time_stepping_calls=dry['timestepping_calls'],variants=per)
    with (OUT/'verification.json').open('x') as f:
        json.dump(records,f,indent=2)
        f.write('\n')
    print(json.dumps({k:{v:d['diagnostic_field_count'] for v,d in r['variants'].items()} for k,r in records.items()}))

if __name__=='__main__':
    main()
