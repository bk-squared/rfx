from __future__ import annotations
import json
from pathlib import Path
from render_edges import table

OUT = Path('/root/workspace/bk-workspace/.801-measure/msl_inset')

def read(p):
    return json.loads(Path(p).read_text())

def main():
    verification=read(OUT/'verification.json')
    selection=read(OUT/'selection.json')
    current_ids=dict(line.split() for line in (OUT/'run_id.txt').read_text().splitlines())
    prior_ids=dict(line.split() for line in (OUT.parent/'msl/run_id.txt').read_text().splitlines())
    lines=[
        'cv06b: 30 mm line; 12 mm stub; 63.5 µm cells; 7 GHz maximum excitation frequency; 2 drives; 100 frequency bins; 20 periods.',
        'cv20: 10 mm line; 50 µm cells; 5 GHz maximum excitation frequency; 2 drives; 30 frequency bins; 12 periods.',
        '',
        table(['fixture','baseline/continued run ID','inset1 run ID','inset1 launches','inset1 solve calls'],
              [[fixture,prior_ids[fixture],current_ids[fixture],1,2] for fixture in ('cv06b','cv20')]),
        'Commands: [COMMANDS.md](COMMANDS.md). GPU commands: [vessl_cv06b.yaml](vessl_cv06b.yaml), [vessl_cv20.yaml](vessl_cv20.yaml).',
        '',
        'A1: trace-edge occupancy = Ex PEC OR Ey PEC; outermost occupancy 0; occupancy 1 in each remaining x absorber cell.',
        '',
        table(['fixture','initial A1 (0/1)','final A1 (0/1)','dry read-backs','dry time steps','final x lo (µm)','final x hi (µm)','lo adjustment (cells)','hi adjustment (cells)'],
              [[fixture,verification[fixture]['initial_A1'],verification[fixture]['final_A1'],
                verification[fixture]['dry_readbacks'],0,selection[fixture]['trace_bounds_m'][0][0]*1e6,
                selection[fixture]['trace_bounds_m'][1][0]*1e6,selection[fixture]['lo_shift_cells'],
                selection[fixture]['hi_shift_cells']] for fixture in ('cv06b','cv20')]),
    ]
    rows=[]
    for fixture in ('cv06b','cv20'):
        d=read(OUT/selection[fixture]['record_file'])
        for plane in d['planes']:
            for name,ix in [('lo outer',0),('lo adjacent',1),('hi adjacent',d['grid_shape'][0]-2),('hi outer',d['grid_shape'][0]-1)]:
                w=next(w for w in plane['windows'].values() if ix in w['x_indices'])
                n=w['x_indices'].index(ix)
                rows.append([fixture,plane['z_index'],name,ix,w['pec_edge_x'][n],w['pec_edge_y'][n]])
    lines += [table(['fixture','z index','position','x index','Ex PEC','Ey PEC'],rows)]
    lines += ['FACT read-back; numerical comparisons: verification.json.', '']
    rows=[]
    for fixture in ('cv06b','cv20'):
        v=verification[fixture]
        f=v['fact_readback']
        rows.append([fixture,f['grid_shape'],f['dx_um'],f['absorber_indices'],
                     f['baseline_edge_extents_xy'],
                     [f['baseline_continued_material_changed_entries'][a] for a in ('eps_r','mu_r','sigma')],
                     v['FACT_mismatches']])
    lines += [table(['fixture','grid shape','dx (µm)','x absorbers','baseline edge extents x/y','eps_r/mu_r/sigma changed entries','FACT mismatches'],rows)]
    lines += ['**EDGES_inset1.md**','',(OUT/'EDGES_inset1.md').read_text(),
              '**TABLE.md**','',(OUT/'TABLE.md').read_text()]
    lines += ['**Recorded diagnostics**','']
    for e in read(OUT/'events_import.json'):
        lines += [e['item'], '', '~~~text', e['message'], '~~~', '',
                  table(['quantity','value'],[['completed delete operations',e['delete_operations_completed']],['lock file size (bytes)',e['lock_file_bytes']]])]
    r=read(OUT/'cv20_reduced.json')
    lines += [table(['cv20 arm','analytic witness RuntimeError count','exact diagnostic field'],
                   [[arm,int(d['analytic_beta']['exception'] is not None),f'cv20_reduced.json: variants.{arm}.analytic_beta.exception']
                    for arm,d in r['variants'].items()])]
    lines += [
        table(['quantity','value'],[
            ['new GPU runs',2],['YAML relaunches',0],['run_id.txt append commands',1],
            ['GitHub posts',0],['git commands',0],['pre-existing file edits',0],
            ['completed delete operations',0],['FACT mismatches',sum(v['FACT_mismatches'] for v in verification.values())]]),
        '**Files; sizes in bytes**','',
    ]
    files={str(p.relative_to(OUT)):p.stat().st_size for p in OUT.rglob('*') if p.is_file()}
    files['REPORT.md']=0
    for _ in range(10):
        content='\n'.join(lines+[table(['file','bytes'],sorted(files.items()))])
        size=len(content.encode())
        if size==files['REPORT.md']:
            break
        files['REPORT.md']=size
    else:
        raise RuntimeError('report byte-count fixed point not reached')
    with (OUT/'REPORT.md').open('x') as f:
        f.write(content)
    assert (OUT/'REPORT.md').stat().st_size==files['REPORT.md']
    print(json.dumps({'report_bytes':files['REPORT.md'],'files':len(files),'sum_file_bytes':sum(files.values())}))

if __name__=='__main__':
    main()

