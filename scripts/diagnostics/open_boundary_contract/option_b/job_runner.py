"""One submission per structure; each arm runs in a fresh GPU process once."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
rig = sys.argv[1]
out = ROOT/'jobs_2'/rig
rows = []
summary = dict(rig=rig, status='running', arms=rows)


def save():
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def interrupted(signum, frame):
    summary.update(status='interrupted', signal=signum)
    save()
    raise SystemExit(128+signum)


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)

assert (out/'run_id.txt').read_text().strip().isdigit()
summary['run_id'] = (out/'run_id.txt').read_text().strip()
save()
check = subprocess.run([sys.executable, '-B', str(ROOT/'instrument_check.py'), rig], cwd=ROOT)
summary['instrument_check_returncode'] = check.returncode
save()
if check.returncode:
    summary['status'] = 'STOP: energy instrument check failed'
    save()
    raise SystemExit(check.returncode)
try:
    for lane in (('msl','msl_low') if rig == 'msl' else (rig,)):
        for n in ((4,8,16,6) if rig == 'patch' else (4,8,16)):
            for s in (0,.01,.03,.1,.3,1,3):
                cmd = [sys.executable, '-B', str(ROOT/'sweep_driver.py'), lane, '--scale',str(s),'--layers',str(n)]
                print('ARM_COMMAND '+json.dumps(cmd), flush=True)
                start = time.monotonic()
                try:
                    result = subprocess.run(cmd, cwd=ROOT, timeout=1800)
                    rc = result.returncode
                except subprocess.TimeoutExpired:
                    rc = 124
                rows.append(dict(rig=lane,scale=s,layers=n,returncode=rc,wall_s=time.monotonic()-start))
                save()
                print('ARM_RETURN '+json.dumps(rows[-1]), flush=True)
    summary['status'] = 'complete' if all(r['returncode']==0 for r in rows) else 'completed with STOPs'
finally:
    save()
    print('JOB_FINAL '+json.dumps(summary), flush=True)
