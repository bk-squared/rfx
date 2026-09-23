from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path('/root/workspace/bk-workspace/.801-measure/msl_inset')
BASE = OUT.parent
OLD = BASE / 'msl'

def read(p):
    return json.loads(Path(p).read_text())

def main():
    selection = read(OUT/'selection.json')
    results = {}
    for fixture, bins in [('cv06b',100), ('cv20',30)]:
        result = read(OUT/(fixture+'_reduced.json'))
        checks = {}
        for variant in ('baseline','continued','inset1'):
            root = OUT if variant == 'inset1' else OLD
            folder = root/fixture/variant
            status = read(folder/'status.json')
            assert status['completed'] == 1 and status['solve_calls'] == 2 and status['timestepping_calls'] == 2
            with np.load(folder/'s.npz') as z:
                s, f = z['S'], z['freqs']
            assert s.shape == (2,2,bins) and f.shape == (bins,)
            assert np.isfinite(s).all() and np.all(np.diff(f)>0)
            with np.load(folder/'diagnostics.npz') as z:
                assert np.array_equal(z['S'],s) and np.array_equal(z['freqs'],f)
            hashes = []
            for drive in (0,1):
                rec = read(folder/f'assembly_received_{drive:02d}.json')
                with np.load(folder/f'assembly_received_{drive:02d}.npz') as z:
                    verified = {k:int(hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()==v) for k,v in rec['sha256'].items()}
                    assert all(verified.values())
                    for plane in rec['planes']:
                        j,k = plane['y_index'],plane['z_index']
                        for window in plane['windows'].values():
                            ix=window['x_indices']
                            for field in ('pec_edge_x','pec_edge_y','pec_mask'):
                                assert np.array_equal(z[field][ix,j,k],window[field])
                        if variant == 'inset1':
                            ex,ey=z['pec_edge_x'][:,j,k],z['pec_edge_y'][:,j,k]
                            p,q=rec['face_pads']['x_lo'],rec['face_pads']['x_hi']
                            union=ex|ey
                            assert not union[0] and not union[-1]
                            assert np.all(union[1:p]) and np.all(union[-q:-1])
                            dry=read(OUT/selection[fixture]['record_file'])
                            assert rec['sha256']==dry['sha256']
                hashes.append(verified)
            reduced = result['variants'][variant]
            recompute_error=max(abs(w['result_minus_recomputed_db']) for w in reduced['ringdown_recomputed'])
            assert recompute_error < 1e-10
            checks[variant]=dict(S_shape=list(s.shape),S_finite_count=int(np.isfinite(s).sum()),
                bins=bins,frequency_first_hz=float(f[0]),frequency_last_hz=float(f[-1]),
                diagnostic_fields=len(read(folder/'diagnostics.json')),
                array_hash_equal=hashes,ringdown_recompute_max_abs_difference_db=recompute_error)
            if variant != 'inset1':
                prior=read(OLD/(fixture+'_reduced.json'))['variants'][variant]
                for field in ('max_reciprocity','max_column_power','max_column_power_raw','max_passivity_correction','settling_db'):
                    assert np.array_equal(reduced[field],prior[field]),(fixture,variant,field)
        for c in result['comparisons'].values():
            assert c['frequency_vectors_equal']==1
            assert all(c['arrays'][a+'_changed_entries']==0 for a in ('eps_r','mu_r','sigma'))
        prov=read(OUT/fixture/'provenance.json')
        source_hashes={name:int(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest) for name,digest in prov['sha256'].items()}
        assert all(source_hashes.values())
        dry_statuses=[read(folder/'inset1/status.json') for folder in sorted(OUT.glob(fixture+'_dry_*'))]
        assert all(s['completed']==1 and s['timestepping_calls']==0 for s in dry_statuses)
        settings=read(OUT/fixture/'inset1/settings.json')
        original=read(OLD/fixture/'baseline/settings.json')
        assert settings['kwargs']==original['kwargs'] and settings['x64']==original['x64']
        assert settings['ports']==original['ports'] and settings['boundary']==original['boundary']
        fact={}
        er=read(OLD/fixture/'baseline/assembly_received_00.json')
        fact['grid_shape']=er['grid_shape']
        fact['dx_um']=er['dx_m']*1e6
        fact['absorber_indices']=[[0,er['face_pads']['x_lo']-1],[er['grid_shape'][0]-er['face_pads']['x_hi'],er['grid_shape'][0]-1]]
        fact['baseline_edge_extents_xy']=[[p['edge_counts'][a]['first_x_index'],p['edge_counts'][a]['last_x_index']] for p in er['planes'] for a in 'xy']
        with np.load(OLD/fixture/'baseline/assembly_received_00.npz') as b, np.load(OLD/fixture/'continued/assembly_received_00.npz') as c:
            fact['baseline_continued_material_changed_entries']={key:int(np.count_nonzero(b[key]!=c[key])) for key in ('eps_r','mu_r','sigma')}
        assert all(v==0 for v in fact['baseline_continued_material_changed_entries'].values())
        if fixture=='cv06b':
            assert fact['grid_shape']==[553,280,37] and abs(fact['dx_um']-63.5)<1e-10
            assert fact['absorber_indices']==[[0,7],[545,552]]
            assert fact['baseline_edge_extents_xy']==[[8,542],[8,543]]
        else:
            assert fact['grid_shape']==[297,66,45] and abs(fact['dx_um']-50)<1e-10
            assert all(len(result['variants'][v]['analytic_beta']['gated_bin_indices'])==9 for v in result['variants'])
        results[fixture]=dict(variants=checks,source_hash_equal=source_hashes,
            dry_readbacks=len(dry_statuses),dry_timestepping_calls=0,
            initial_A1=dry_statuses[0]['A1'],final_A1=dry_statuses[-1]['A1'],
            settings_equal=1,fact_readback=fact,FACT_mismatches=0)
    with (OUT/'verification.json').open('x') as f:
        json.dump(results,f,indent=2)
        f.write('\n')
    print(json.dumps({k:{'FACT_mismatches':v['FACT_mismatches'],'initial_A1':v['initial_A1'],'final_A1':v['final_A1']} for k,v in results.items()}))

if __name__=='__main__':
    main()

