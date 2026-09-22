"""Write the four job specifications and validate their exact shell blocks."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parent
src = root/'src'
sha = (src/'PROVENANCE.txt').read_text().strip()
assert sha == 'e7f7e02704fd46ea7e21f127b19fc81cb66d6148', sha
hashes = {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
          ('sweep_driver.py','extra_rigs.py','instrument_check.py','job_runner.py','verify_job_sources.py')}
for rig in ('msl','waveguide','patch','plane'):
    out = root/'jobs_2'/rig
    out.mkdir(parents=True, exist_ok=True)
    checks = '\n'.join(f'  printf "%s  %s\\n" "{sha}" "$base/{name}" | sha256sum -c -' for name,sha in hashes.items())
    script = f'''  set -eu
  base={root}
  src={src}
  out="$base/jobs_2/{rig}"
  cp "$src/PROVENANCE.txt" "$out/PROVENANCE.txt"
  test "$(cat "$out/PROVENANCE.txt")" = {sha}
{checks}
  python -B "$base/verify_job_sources.py" > "$out/source_verification.txt"
  python -m pip install -q "scipy>=1.11" pytest "numpy<2"
  i=0
  while [ ! -s "$out/run_id.txt" ]; do
    test "$i" -lt 12
    sleep 5
    i=$((i + 1))
  done
  trap 'test -f "$out/summary.json" && cat "$out/summary.json"' EXIT
  python -B "$base/job_runner.py" {rig}
'''
    yml = f'''name: rfx-801-B-sweep-2-{rig}
description: "Option B: {rig}, alpha scale and absorber depth; measurement only"
tags: [rfx, issue-801, option-B-sweep]
resources:
  cluster: remilab-c0
  preset: gpu-rtx4090
image: nvcr.io/nvidia/jax:24.10-py3
env:
  PYTHONUNBUFFERED: "1"
  PYTHONDONTWRITEBYTECODE: "1"
  JAX_ENABLE_X64: "0"
  XLA_PYTHON_CLIENT_PREALLOCATE: "false"
  HDF5_USE_FILE_LOCKING: "FALSE"
  LANG: "C.UTF-8"
  OPENBLAS_NUM_THREADS: "1"
  OMP_NUM_THREADS: "1"
  MPLCONFIGDIR: "{root}/mpl_config"
  XDG_CACHE_HOME: "{root}/cache"
mount:
  /root/workspace/: volume://remilab-fs/personal-workspaces/
run: |-
{script}'''
    (root/f'vessl_{rig}.yaml').write_text(yml)
    sh = '\n'.join(line[2:] for line in script.splitlines())+'\n'
    subprocess.run(['sh','-n'], input=sh, text=True, check=True)
    print(f'vessl_{rig}.yaml: sh -n passed; no heredocs; one structure')
(root/'job_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
