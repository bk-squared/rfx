from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from reduce_ports import table,fmt

ROOT=Path('/root/workspace/bk-workspace/.801-measure/ports')
BASE=ROOT.parent

def verify():
    report=(ROOT/'REPORT.md').read_text()
    assert (ROOT/'CENSUS.md').read_text() in report
    assert (ROOT/'TABLE.md').read_text() in report
    rows=re.findall(r'^\| (ports/[^|]+) \| ([0-9]+) \|$',report,re.M)
    assert rows
    for path,size in rows:
        assert (BASE/path.strip()).stat().st_size==int(size),(path,size)
    existing={str(p.relative_to(BASE)) for p in ROOT.rglob('*') if p.is_file()}
    listed={p.strip() for p,s in rows}
    assert existing==listed,(existing-listed,listed-existing)
    for run in (369367262622,369367262623):
        text=re.sub(r'\x1b\[[0-9;]*m','',(ROOT/f'vessl_{run}_status.txt').read_text())
        assert re.search(r'Status\s+completed',text)
        logs=(ROOT/f'vessl_{run}.log').read_text()
        for driver in ('port_run.py','measure_common.py','census_driver.py','census_v2.py'):
            assert hashlib.sha256((ROOT/driver).read_bytes()).hexdigest() in logs
        assert hashlib.sha256((BASE/'src-main/rfx/boundaries/cpml.py').read_bytes()).hexdigest() in logs
    print(json.dumps(dict(report_bytes=(ROOT/'REPORT.md').stat().st_size,listed_files=len(rows),file_sizes_equal=len(rows),census_inline=1,table_inline=1,completed_runs=2,provenance_hash_matches=10)))

def main():
    if '--verify' in sys.argv:
        return verify()
    records=json.loads((ROOT/'census.json').read_text())['variants']
    verification=json.loads((ROOT/'verification.json').read_text())
    counts=verification['counts']
    eligible=[r for r in records if r.get('affected') and r.get('settings') and not r.get('manufactured')]
    msl=sorted([r for r in eligible if r['lane']=='msl_two_port'],key=lambda r:(r['cells_times_steps'],r['index']))
    mixed=sorted([r for r in eligible if r['lane'] in ('coax_msl_transition','mixed')],key=lambda r:(r['cells_times_steps'],r['index']))
    assert msl[0]['index']==62 and mixed[0]['index']==81
    ranking=[]
    for r in msl+mixed:
        ranking.append([r['lane'],r['index'],r['cells'],r['steps_for_ranking'],r['cells_times_steps'],int(r['index'] in (62,81))])
    sheen=next(r for r in records if r['index']==29)
    sheen_steps=__import__('math').ceil(60.0/(20e9*sheen['dt_s']))
    ranking.append(['msl_two_port',29,sheen['cells'],sheen_steps,sheen['cells']*sheen_steps,0])
    ranking.append(['coax_two_port',77,586850,6000,3521100000,1])
    text='Audited variants: 62. Additional fixture variants: 24. Generated port-geometry records: 1.\n\n'
    text+=table(['quantity','count'],[['census records completed',87],['records meeting the brief affected criterion',counts['affected']],['baseline/continued S-matrix pairs',2],['time-stepped internal drives',8],['solver edge-mask equality checks',counts['readback_component_equal']],['coax time-stepped internal drives',0],['coax continuation assertion exceptions',1],['waveguide affected records',0],['lumped/wire affected records',0],['VESSL submissions',2],['VESSL deletions',0],['git commands',0],['exported source edits',0],['existing-file overwrites',0]])
    text+='\n## Commands and runs\n\n'
    text+='[commands.md](commands.md). Local Python: `/root/workspace/bk-workspace/rfx/.venv/bin/python`; `JAX_PLATFORMS=cpu`; `-B`.\n\n'
    text+=table(['run id','yaml','fixture','returned S matrices','log'],[[369367262622,'vessl_ports_msl.yaml',62,2,'vessl_369367262622.log'],[369367262623,'vessl_ports_mixed.yaml',81,2,'vessl_369367262623.log']])
    text+='\n## Fixture selection\n\n'
    text+='Rank quantity: cells × committed steps. Equal products: ascending census index. Coax-to-MSL/mixed selection: one shared slot.\n\n'
    text+=table(['lane','fixture','cells','steps','cells × steps','selected'],ranking)
    text+='\n'+table(['fixture','committed calculation / record','FDTD calculation eligible'],[[64,'forward(num_periods=15); no public S-matrix call',0],[82,'manufactured plane accumulators',0],[83,'manufactured plane accumulators',0],[84,'manufactured backend; n_steps=1',0],[85,'manufactured backend; n_steps=1',0],[77,'generated geometry: census record 86; stopped during continuation build',1]])
    text+='\n## A1–A3\n\n'
    audited=[r for r in records if r['kind']=='audited']
    nu=[r for r in audited if r.get('nonuniform')]
    uniform=[r for r in audited if not r.get('nonuniform')]
    walls=[r.get('measurement_wall_s',r['wall_s']) for r in audited]
    text+=table(['assumption / measured quantity','numerator / value','denominator / unit'],[['A1: uniform construction',len(uniform),62],['A1: non-uniform fallback completed',len(nu),len(nu)],['A1: completed CPU assemblies',len(audited),62],['A1: audited assembly maximum wall time',max(walls),'s'],['A1: audited assembly summed wall time',sum(walls),'s'],['A2: zero-thickness Box trace continuation fixtures with solver read-back',2,2],['A3: selected MSL/mixed public calls returning an S matrix',4,4],['A3: selected MSL/mixed public-call refusals',0,4],['A3: coax continuation solver calls',0,1],['A3: coax-MSL transition selected',0,1]])
    text+='\nA1 fallback: `Simulation._build_nonuniform_grid()` and `Simulation._assemble_materials_nu()`. Initial records: `census_records/`; final records: `census_records_v2/`.\n\n'
    text+=table(['non-uniform audited index','uniform refusal','fallback completed'],[[r['index'],r.get('uniform_grid_refusal'),r['completed']] for r in nu])
    a2=[]
    for lane,index,entry_index in [('msl_two_port',62,2),('mixed',81,1)]:
        for variant in ('baseline','continued'):
            p=ROOT/lane/f'fixture_{index:03d}'/variant
            mm=json.loads((p/'assembly_before_solve.json').read_text())
            e=next(e for e in mm['entities'] if e['collection']=='_geometry' and e['index']==entry_index)
            for face in ('x_lo','x_hi'):
                a2.append([index,variant,entry_index,e['sheet_count'],face,e['volume_cells'][face]['absorber'],*[e['edges'][a][face]['absorber'] for a in 'xyz']])
    text+='\n'+table(['fixture','variant','entry','sheets','face','absorber volume cells','absorber Ex','absorber Ey','absorber Ez'],a2)
    text+='\nCoax continuation stop, verbatim (`dry_coax/fixture_077/continued/status.json`):\n\n```text\n'
    text+=json.loads((ROOT/'dry_coax/fixture_077/continued/status.json').read_text())['traceback']+'```\n'
    text+='\nCoax continuation follow-up solves: 0. Coax YAML submissions: 0.\n\n'
    text+='## FACT record checks\n\n'
    uniform_lengths=collections.Counter(r['assembly_tuple_length'] for r in records if 'assembly_tuple_length' in r and not r.get('nonuniform'))
    nu_lengths=collections.Counter(r['assembly_tuple_length'] for r in records if 'assembly_tuple_length' in r and r.get('nonuniform'))
    text+=table(['record','measured value','source / record'],[['uniform assembly tuple length and count',dict(uniform_lengths),'census.json'],['non-uniform assembly tuple length and count',dict(nu_lengths),'census.json'],['declared PEC/PMC faces with pad > 0',0,'census.json'],['audited variant triples',62,'tests/_example_fidelity_lib.py'],['capture script direct iter_audited_variants definition',0,'scripts/capture_example_fidelity_snapshot.py'],['capture script lib.iter_audited_variants call',1,'scripts/capture_example_fidelity_snapshot.py'],['source tree','src-main; df08175c','../PROVENANCE.txt']])
    text+='\n`iter_audited_variants` path: `src-main/tests/_example_fidelity_lib.py`; call site: `src-main/scripts/capture_example_fidelity_snapshot.py`, `lib.iter_audited_variants()`.\n\n'
    text+='## Additional recorded messages\n\n'
    text+='Initial local import, verbatim:\n\n```text\nCould not save font_manager cache NO_MUTATION: os.remove (\'/root/workspace/bk-workspace/.801-measure/ports/mpl_config/fontlist-v3.11.0.json.matplotlib-lock\', -1)\n```\n\n'
    text+='VESSL 369367262623 initialization, verbatim excerpt; full text: `vessl_369367262623.log`.\n\n```text\nValueError: bad marshal data (unknown type code)\nVESSL CLI not installed.\n```\n\n'
    text+='Returned resonance-frequency fields: 0 in each of the 4 S-matrix result objects. Resonance estimates added: 0.\n\n'
    text+='## CENSUS.md\n\n'+(ROOT/'CENSUS.md').read_text()
    text+='\n## TABLE.md\n\n'+(ROOT/'TABLE.md').read_text()
    text+='\n## Files and sizes\n\n'
    files=[(str(p.relative_to(BASE)),p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file()]
    assert not (ROOT/'REPORT.md').exists()
    size=0
    for _ in range(20):
        result=text+table(['file','bytes'],sorted(files+[('ports/REPORT.md',size)]))
        new_size=len(result.encode())
        if new_size==size:
            break
        size=new_size
    else:
        raise RuntimeError('report size fixed point not reached')
    with (ROOT/'REPORT.md').open('x') as f:
        f.write(result)
    print(json.dumps(dict(report_bytes=size,files=len(files)+1)))

if __name__=='__main__':
    main()
