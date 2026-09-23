"""Run each selected Addendum 2 measurement once, in a fresh GPU process."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
lane=sys.argv[1]
out=ROOT/'jobs_3'/lane
rows=[]
summary=dict(lane=lane,status='running',arms=rows,run_id=(out/'run_id.txt').read_text().strip(),
             commit=(ROOT/'src/PROVENANCE.txt').read_text().strip())


def save():
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def interrupted(signum,frame):
    summary.update(status='interrupted',signal=signum)
    save()
    raise SystemExit(128+signum)


for sig in (signal.SIGTERM,signal.SIGINT):
    signal.signal(sig,interrupted)
save()
manifest=json.loads((ROOT/'job_source_hashes_3.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print('DRIVER_SHA256',name,digest,flush=True)
assert summary['commit']=='e7f7e02704fd46ea7e21f127b19fc81cb66d6148'
assert summary['run_id'].isdigit()
if lane=='oracle':
    cases=[['oracle_3.py','exact','--scale',str(s),'--layers',str(n)] for n in (4,8,16) for s in (0,1,3)]
    cases += [['oracle_3.py','patch_source','--scale','1','--layers','8']]
elif lane.startswith('frequency_'):
    f=lane.split('_')[1]
    cases=[['oracle_3.py','frequency','--f0-ghz',f,'--scale',str(s),'--layers','8'] for s in (0,1,3)]
else:
    cases=[['diagnostic_3.py',lane]]
for case in cases:
    print('START',case,flush=True)
    start=time.monotonic()
    try:
        result=subprocess.run([sys.executable,'-B',str(ROOT/case[0]),*case[1:]],cwd=ROOT,timeout=7200)
        rc=result.returncode
    except subprocess.TimeoutExpired:
        rc=124
    rows.append(dict(command=case,returncode=rc,wall_s=time.monotonic()-start))
    save()
summary.update(status='complete' if all(r['returncode']==0 for r in rows) else 'STOP',returncode=int(any(r['returncode'] for r in rows)))
save()
print('JOB_FINAL',json.dumps(summary),flush=True)
raise SystemExit(summary['returncode'])
