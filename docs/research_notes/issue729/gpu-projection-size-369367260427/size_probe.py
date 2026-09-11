"""Diagnose near-unitary SVD roundoff without running an FDTD solver."""
import hashlib
import inspect
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

assert jax.default_backend() == 'gpu'
out = Path(os.environ['RFX_OUT'])
records = []

for kind in ['near', 'wide']:
    for n in [2, 8, 32]:
        rng = np.random.default_rng(729 if kind == 'near' else 7)
        if kind == 'near':
            values = []
            for _ in range(12):
                u, _ = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
                v, _ = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
                values.append((u*(1.+np.linspace(-1e-3, 1e-3, n))) @ v.conj().T)
            original = np.stack(values)
        else:
            original = 2.*(rng.normal(size=(64, n, n))+1j*rng.normal(size=(64, n, n)))
        for dtype in [np.complex64, np.complex128]:
            raw = original.astype(dtype)
            real_dtype = np.empty((), dtype=dtype).real.dtype
            radius = 1.-64.*np.finfo(real_dtype).eps
            hu, hs, hvh = np.linalg.svd(raw.astype(np.complex128), full_matrices=False)
            reference = (hu*np.minimum(hs, radius)[:, None, :]) @ hvh
            with jax.experimental.enable_x64(), jax.default_matmul_precision('tensorfloat32'):
                a = jnp.asarray(raw)
                u, s, vh = jnp.linalg.svd(a, full_matrices=False)
                sc = jnp.minimum(s, radius)
                contraction = lambda scale: jnp.einsum('fij,fj,fjk->fik', u, scale.astype(u.dtype), vh,
                                                       precision=jax.lax.Precision.HIGHEST)
                direct = np.asarray(contraction(sc))
                residual = np.asarray(a+contraction(sc-s))
                reconstruction = np.asarray(contraction(s))
                wu, ws, wvh = jnp.linalg.svd(a.astype(jnp.complex128), full_matrices=False)
                promoted = np.asarray(jnp.einsum('fij,fj,fjk->fik', wu, jnp.minimum(ws, radius).astype(wu.dtype), wvh,
                                                precision=jax.lax.Precision.HIGHEST).astype(a.dtype))
            un, sn, vhn = np.asarray(u), np.asarray(s), np.asarray(vh)
            rec = dict(kind=kind, n=n, dtype=str(np.dtype(dtype)),
                       reconstruction_error=float(np.max(np.abs(reconstruction-raw))),
                       singular_error=float(np.max(np.abs(sn-hs))),
                       u_orthogonality=float(np.max(np.abs(un.conj().transpose(0, 2, 1) @ un-np.eye(n)))),
                       vh_orthogonality=float(np.max(np.abs(vhn @ vhn.conj().transpose(0, 2, 1)-np.eye(n)))))
            for name, result in [('direct', direct), ('residual', residual), ('promoted', promoted)]:
                high = result.astype(np.complex128)
                rec[name] = dict(max_sigma=float(np.linalg.svd(high, compute_uv=False).max()),
                                 max_error=float(np.max(np.abs(high-reference))))
            records.append(rec)
            print(json.dumps(rec), flush=True)

report = dict(jax=jax.__version__, numpy=np.__version__, devices=str(jax.devices()),
              lax_svd_signature=str(inspect.signature(jax.lax.linalg.svd)),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=records)
(out/'size-probe.json').write_text(json.dumps(report, indent=2)+'\n')
