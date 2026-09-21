import json
from pathlib import Path
import numpy as np
import measure_base as m
from readback_report_v2 import FACTORS

OUT=m.OUT

def read(p):
    return json.loads(p.read_text())

def main():
    data=read(OUT/'reduced.json')
    rows=[]
    for lane in ('cv20','patch'):
        for label,factor in FACTORS:
            folder=OUT/f'{lane}_{label}'
            prov=read(folder/'provenance.json')
            hashes={p:int(m.sha(m.BASE/p)==v) for p,v in prov['sha256'].items()}
            assert all(hashes.values())
            expected_calls=2 if lane=='cv20' else 1
            captures=sorted(folder.glob('cpml_received_*.json'))
            assert len(captures)==expected_calls
            identity=[]
            for path in captures:
                d=read(path)
                assert d['factor']==factor and all(d['finite'].values())
                assert all(d['dry_bit_identity'].values())
                identity.append(sum(d['dry_bit_identity'].values()))
            row=dict(lane=lane,factor=factor,source_hash_equal=hashes,core_calls=len(captures),
                     dry_bit_identical_array_counts=identity)
            if lane=='cv20':
                status=read(folder/'status.json')
                assert status['completed']==1 and status['solve_calls']==2 and status['timestepping_calls']==2
                recs=[read(folder/f'assembly_received_{d:02d}.json') for d in (0,1)]
                baseline=read(OUT/'cv20_f1/assembly_received_00.json')
                row['assembly_hash_equal_to_factor1']=[int(d['sha256']==baseline['sha256']) for d in recs]
                assert all(row['assembly_hash_equal_to_factor1'])
                old=read(m.BASE/'msl/cv20/continued/assembly_received_00.json')
                row['assembly_hash_equal_to_previous_continued']=[int(d['sha256']==old['sha256']) for d in recs]
                assert all(row['assembly_hash_equal_to_previous_continued'])
                row['witness_shapes']=[]
                for drive in (0,1):
                    with np.load(folder/f'witness_series_{drive:02d}.npz') as z:
                        ts=z['time_series']
                        assert ts.shape==(25177,10) and np.isfinite(ts).all()
                        row['witness_shapes'].append(list(ts.shape))
                dr=data['cv20'][label]
                assert dr['analytic_beta']['statistics_pct']['count']==9
                row['ringdown_recomputed_abs_diff_db']=[abs(x['difference_db']) for x in dr['ringdown_recomputed']]
                assert max(row['ringdown_recomputed_abs_diff_db'])<1e-10
                for t in dr['tail']:
                    blocks=t['blocks']
                    assert len(blocks)==12
                    assert blocks[0]['start_index']==int(.4*25177) and blocks[-1]['stop_exclusive']==25177
                    assert all(a['stop_exclusive']==b['start_index'] for a,b in zip(blocks,blocks[1:]))
                    assert max(b['steps'] for b in blocks)-min(b['steps'] for b in blocks)<=1
                    assert abs(t['rate_per_ns']*t['reciprocal_ns']-1)<1e-14
                if label=='f1':
                    for entry,delta in dr['deltas_against_1'].items():
                        assert delta['dB']['max_abs']==0 and delta['phase_deg']['max_abs']<1e-12
            else:
                dr=read(folder/'result.json')
                assert dr['completed']==1 and dr['num_periods']==150 and len(dr['rates_per_step'])==4
                with np.load(folder/'time_series.npz') as z:
                    ts=z['time_series']
                    assert ts.shape==(dr['steps'],4) and np.isfinite(ts).all()
                    row['witness_shape']=list(ts.shape)
                assert dr['worst_rate_per_step']==max(dr['rates_per_step'])
            rows.append(row)
    facts={}
    for variant in ('baseline','continued'):
        with np.load(m.BASE/f'msl/cv20/{variant}/witness_series_00.npz') as z:
            facts[variant+'_shape']=list(z['time_series'].shape)
        facts[variant+'_dt_s']=read(m.BASE/f'msl/cv20/{variant}/assembly_received_00.json')['dt_s']
    facts['profile_alpha_literal_count']=(m.SRC/'rfx/boundaries/cpml.py').read_text().count('alpha = 0.05 * (1.0 - rho)')
    facts['source_tree_count']=1
    m.write_json(OUT/'verification.json',dict(rows=rows,fact_checks=facts,FACT_discrepancy_count=0))
    print(json.dumps(dict(verified_rows=len(rows),FACT_discrepancy_count=0)))

if __name__=='__main__':
    main()
