"""One f64 AD solve, compared with retained matching f64-field FD legs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import rfx
import rfx.simulation as core
from tests.unit.autodiff.test_msl_ad_fd_converged import (
    _build_msl_f64_referee, _NUM_PERIODS, _N_FREQS, _FD_H,
    _closest_divisor, _REL_ERR_THRESHOLD,
)
from tests._msl_ad_objective import msl_band_mean_s21_sq

out = Path(os.environ['RFX_OUT'])
assert not (out/'outcome.json').exists()
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert sha == os.environ['RFX_SHA']
assert jax.default_backend() == 'gpu'
assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd())
receipt = dict(source_sha=sha, jax=jax.__version__, numpy=np.__version__,
               devices=str(jax.devices()), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'environment.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('ENVIRONMENT', json.dumps(receipt), flush=True)
init_records = []
original_init = core.init_state


def capture_init(*args, **kwargs):
    state = original_init(*args, **kwargs)
    init_records.append({key: str(getattr(state, key).dtype) for key in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')})
    (out/'field-initializations.json').write_text(json.dumps(init_records, indent=2)+'\n')
    return state


core.init_state = capture_init
start = time.monotonic()
with jax.experimental.enable_x64():
    sim = _build_msl_f64_referee()
    grid = sim._build_grid()
    steps = int(grid.num_timesteps(num_periods=_NUM_PERIODS))
    segments = _closest_divisor(steps, int(np.sqrt(steps)))
    eps = jnp.ones(grid.shape, dtype=jnp.float64)

    def objective(alpha):
        result = sim.compute_msl_s_matrix(n_freqs=_N_FREQS, num_periods=_NUM_PERIODS,
                                          eps_override=eps*alpha, checkpoint_segments=segments)
        return msl_band_mean_s21_sq(result.S)

    loss, gradient = jax.value_and_grad(objective)(jnp.float64(1.))
    loss, gradient = float(loss), float(gradient)
    source = Path(os.environ['RFX_FD_OUT'])/'ad'
    fd_values, fd_files = [], []
    for name in ['call-2-result.npz', 'call-3-result.npz']:
        path = source/name
        with np.load(path) as data:
            assert data['S'].dtype == np.complex128
            fd_values.append(float(msl_band_mean_s21_sq(jnp.asarray(data['S']))))
        fd_files.append({'name': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    g_fd = (fd_values[0]-fd_values[1])/(2.*_FD_H)
    relative = abs(gradient-g_fd)/abs(g_fd)
    report = dict(source_sha=sha, field_precision='float64', n_steps=steps,
                  checkpoint_segments=segments, loss=loss, g_ad=gradient,
                  g_fd_retained=g_fd, fd_losses=fd_values, fd_files=fd_files,
                  relative_error=relative, unchanged_gate=_REL_ERR_THRESHOLD,
                  passes_existing_gate=relative <= _REL_ERR_THRESHOLD,
                  wall_s=time.monotonic()-start)
    (out/'outcome.json').write_text(json.dumps(report, indent=2)+'\n')
    print('F64_AD_CONTROL', json.dumps(report, indent=2), flush=True)
    # This is an attribution control, not a replacement of the original f32
    # acceptance claim. Keep its own gate result explicit in the process exit.
    raise SystemExit(0 if report['passes_existing_gate'] else 1)
