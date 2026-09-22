import json
from pathlib import Path
from readback_report_v2 import table

OUT=Path('/root/workspace/bk-workspace/.801-measure/alpha')

def read(p):
    return json.loads(p.read_text())

def main():
    verification=read(OUT/'verification.json')
    reduced=read(OUT/'reduced.json')
    readback=read(OUT/'readback_verification.json')
    events=read(OUT/'events.json')
    run_ids=[line.split() for line in (OUT/'run_id.txt').read_text().splitlines()]
    lines=['Alpha factors: 0.01, 0.25, 1, 4.',
           'cv20: continued trace; 30 frequency bins; 12 periods; 2 drives; 25177 steps × 10 probes per drive; dt = 9.53287434765503e-14 s.',
           'Patch: n = 2; pad_h = 10; CPML layers = 4; 150 periods; 4 probes.', '',
           'Commands: [COMMANDS.md](COMMANDS.md). GPU commands: [vessl_cv20.yaml](vessl_cv20.yaml), [vessl_patch.yaml](vessl_patch.yaml).', '',
           table(['lane','run ID','YAML launches','completed (0/1)'],
                 [[lane,rid,1,int('completed' in (OUT/f'vessl_{lane}_{rid}.txt').read_text())] for lane,rid in run_ids]), '',
           table(['item','value'],[['A1 (0/1)',readback['A1']],['A2 (0/1)',readback['A2']],
                 ['requested zero factor',0],['replacement factor',.01],['factor-1 / unpatched bit-identical arrays',60],
                 ['readback time steps',0],['FACT discrepancy count',verification['FACT_discrepancy_count']],
                 ['GPU factor runs',len(verification['rows'])],['YAML relaunches',0],['GitHub posts',0],
                 ['repository commands',0],['exported-source edits',0],['deletions',0],['existing-file append commands: run_id.txt',1]]),'',
           'A2 evaluation: `numpy.errstate(divide="raise", invalid="raise")`.',
           '```text',readback['zero_error']['exception_type']+': '+readback['zero_error']['exception'],'```','',
           'readback.md', '',(OUT/'readback.md').read_text(),'TABLE.md','',(OUT/'TABLE.md').read_text(),
           'Verification','',table(['lane','factor','core calls','GPU/dry bit-identical arrays'],
                  [[r['lane'],r['factor'],r['core_calls'],sum(r['dry_bit_identical_array_counts'])] for r in verification['rows']]),'',
           'FACT read-backs','',table(['quantity','value'],list(verification['fact_checks'].items())),'']
    for event in events:
        lines += [event['item'],'','```text',event['exception'],'```',
                  table(['item','value'],[[k,v] for k,v in event.items() if k not in ('item','exception')]),'']
    lines += ['Recorded exceptions','',table(['item','count'],[['factor 0 coefficient readback',1],
              ['AST validation',len(events)],['reduction',len(reduced['exceptions'])]]),'']
    for exc in reduced['exceptions']:
        record=exc['record']
        lines += [exc['item'],'','```text',record.get('type',record.get('exception_type',''))+': '+record.get('text',record.get('exception','')),'```','']
    body='\n'.join(lines)
    files=[(str(p.relative_to(OUT)),p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]
    assert not (OUT/'REPORT.md').exists()
    size=0
    for _ in range(20):
        rows=sorted(files+[('REPORT.md',size)])
        text=body+'\nFile sizes (bytes)\n\n'+table(['file','bytes'],rows)
        newsize=len(text.encode('utf-8'))
        if newsize==size:
            break
        size=newsize
    else:
        raise RuntimeError('report size iteration count 20')
    with (OUT/'REPORT.md').open('x') as f:
        f.write(text)
    print(json.dumps(dict(report_bytes=(OUT/'REPORT.md').stat().st_size,files=len(files)+1)))

if __name__=='__main__':
    main()
