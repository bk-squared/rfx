"""Prepare only the new local measurement artifacts; no repository operations."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import yaml

ROOT=Path(__file__).resolve().parent
names=['oracle_3.py','diagnostic_3.py','job_runner_3.py','sweep_driver.py','extra_rigs.py']
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}
(ROOT/'job_source_hashes_3.json').write_text(json.dumps(hashes,indent=2)+'\n')
for p in names:
    ast.parse((ROOT/p).read_text())
template=yaml.safe_load((ROOT/'vessl_plane.yaml').read_text())
for lane in ['oracle','msl','waveguide','frequency_1','frequency_2','frequency_4']:
    out=ROOT/'jobs_3'/lane
    out.mkdir()
    (out/'tmp').mkdir()
    spec=dict(template)
    spec['name']='rfx-801-B-addendum-2-'+lane.replace('_','-')
    spec['description']='Addendum 2 measurement only: '+lane
    spec['tags']=['rfx','issue-801','option-B-addendum-2']
    spec['env']=dict(template['env'],TMPDIR=str(out/'tmp'),PIP_CACHE_DIR=str(ROOT/'cache/pip'))
    script=['set -eu',f'base={ROOT}',f'out="$base/jobs_3/{lane}"','src="$base/src"','cd "$base"',
            'cp "$src/PROVENANCE.txt" "$out/PROVENANCE.txt"',
            'test "$(cat "$out/PROVENANCE.txt")" = e7f7e02704fd46ea7e21f127b19fc81cb66d6148']
    for name,digest in hashes.items():
        script.append(f'printf "%s  %s\\n" "{digest}" "$base/{name}" | sha256sum -c -')
    script += ['python -m pip install -q --target "$out/deps" "scipy>=1.11" pytest "numpy<2"',
               'export PYTHONPATH="$out/deps:$src"','i=0','while [ ! -s "$out/run_id.txt" ]; do','  test "$i" -lt 12',
               '  sleep 5','  i=$((i + 1))','done',
               "trap 'test -f \"$out/summary.json\" && cat \"$out/summary.json\"' EXIT",
               f'python -B "$base/job_runner_3.py" {lane}']
    shell='\n'.join(script)+'\n'
    subprocess.run(['sh','-n'],input=shell,text=True,check=True)
    assert '<<' not in shell and 'git ' not in shell
    spec.pop('run')
    dest=ROOT/f'vessl_3_{lane}.yaml'
    dest.write_text(yaml.safe_dump(spec,sort_keys=False)+'run: |-\n'+''.join('  '+line+'\n' for line in script))
    assert yaml.safe_load(dest.read_text())['run']==shell.rstrip('\n')
    print(dest.name,'AST and sh -n passed; no Git commands or heredocs')
