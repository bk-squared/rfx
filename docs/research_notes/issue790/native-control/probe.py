import json, subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
cmd = [sys.executable, str(root/'child.py'), str(root/'busy.so')]
first = subprocess.run(cmd, text=True, capture_output=True, timeout=5, check=True)
lines = first.stdout.splitlines()
assert lines[0] == 'ENTER_NATIVE'
internal = json.loads(lines[-1])
assert internal['handled_after_s'] > 1.8
start = time.monotonic()
child = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    child.communicate(timeout=0.3)
    raise AssertionError('native child finished before external deadline')
except subprocess.TimeoutExpired:
    child.kill()
    stdout, stderr = child.communicate(timeout=5)
elapsed = time.monotonic()-start
assert 'ENTER_NATIVE' in stdout, (stdout, stderr)
assert child.poll() is not None and elapsed < 1.5, (child.returncode, elapsed)
result = {'signal_only':internal,'external_deadline':{'budget_s':0.3,'reaped_after_s':elapsed,'returncode':child.returncode,'entered_native':True},'scope':'mechanism control, not an electromagnetic solve or a new internal JAX stack capture'}
(root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
