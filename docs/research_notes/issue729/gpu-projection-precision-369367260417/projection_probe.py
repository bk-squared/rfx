"""Attribute the rotation-gate error using retained S, without FDTD."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from rfx.api._sparams import _project_passive

assert jax.default_backend() == 'gpu'
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert sha == os.environ['RFX_SHA']
out = Path(os.environ['RFX_OUT'])
source = Path(os.environ['RFX_CANDIDATE_OUT']) / 'axis'
records = []
all_arrays = {}

for i in range(2):
    filename = source / f'call-{i}-result.npz'
    with np.load(filename) as data:
        raw = data['S_raw']
        observed = data['S']
    a = raw.transpose(2, 0, 1)
    ah = a.astype(np.complex128)
    radius32 = 1. - 64. * np.finfo(np.float32).eps
    uh, sh, vhh = np.linalg.svd(ah, full_matrices=False)
    ref = np.einsum('fij,fj,fjk->fik', uh, np.minimum(sh, radius32), vhh)
    u, s, vh = (np.asarray(v) for v in jnp.linalg.svd(jnp.asarray(a), full_matrices=False))
    reconstructed = np.einsum('fij,fj,fjk->fik', u, s, vh)
    p32, correction = _project_passive(jnp.asarray(raw))
    p32 = np.asarray(p32).transpose(2, 0, 1)
    with jax.experimental.enable_x64():
        p64, _ = _project_passive(jnp.asarray(raw, dtype=jnp.complex128))
        p64 = np.asarray(p64).transpose(2, 0, 1)
    uj, sj, vhj = jnp.asarray(u), jnp.asarray(s), jnp.asarray(vh)
    sj_clip = jnp.minimum(sj, radius32).astype(uj.dtype)
    p_high = np.asarray(jnp.einsum('fij,fj,fjk->fik', uj, sj_clip, vhj,
                                  precision=jax.lax.Precision.HIGHEST))
    reconstruct_default = np.asarray(jnp.einsum('fij,fj,fjk->fik', uj, sj.astype(uj.dtype), vhj))
    reconstruct_highest = np.asarray(jnp.einsum('fij,fj,fjk->fik', uj, sj.astype(uj.dtype), vhj,
                                                precision=jax.lax.Precision.HIGHEST))
    record = dict(call=i, source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                  highest_vs_host64_max_abs=float(np.max(np.abs(p_high-ref))),
                  highest_sigma_max=float(np.linalg.svd(p_high.astype(np.complex128), compute_uv=False).max()),
                  default_reconstruction_max_abs=float(np.max(np.abs(reconstruct_default-ah))),
                  highest_reconstruction_max_abs=float(np.max(np.abs(reconstruct_highest-ah))),
                  raw_sigma_max=float(sh.max()),
                  svd_reconstruction_max_abs=float(np.max(np.abs(reconstructed-ah))),
                  u_orthogonality_max_abs=float(np.max(np.abs(u.conj().transpose(0, 2, 1) @ u-np.eye(2)))),
                  vh_orthogonality_max_abs=float(np.max(np.abs(vh @ vh.conj().transpose(0, 2, 1)-np.eye(2)))),
                  gpu32_vs_saved_max_abs=float(np.max(np.abs(p32-observed.transpose(2, 0, 1)))),
                  gpu32_vs_host64_same_radius_max_abs=float(np.max(np.abs(p32-ref))),
                  gpu32_sigma_max=float(np.linalg.svd(p32.astype(np.complex128), compute_uv=False).max()),
                  gpu64_sigma_max=float(np.linalg.svd(p64, compute_uv=False).max()),
                  declared_correction_max=float(np.asarray(correction).max()))
    records.append(record)
    all_arrays.update({f'raw_{i}': raw, f'gpu32_{i}': p32,
                       f'gpu64_{i}': p64, f'host64_radius32_{i}': ref, f'gpu32_highest_{i}': p_high})

differences = {}
for name in ['raw', 'gpu32', 'gpu64', 'host64_radius32', 'gpu32_highest']:
    differences[name + '_rotation_max_abs'] = float(np.max(np.abs(
        all_arrays[name+'_0']-all_arrays[name+'_1'])))
report = dict(source_sha=sha, jax=jax.__version__, numpy=np.__version__,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              records=records, comparisons=differences)
(out/'projection-probe.json').write_text(json.dumps(report, indent=2)+'\n')
np.savez_compressed(out/'projection-probe.npz', **all_arrays)
print('PROJECTION_PROBE', json.dumps(report, indent=2), flush=True)
