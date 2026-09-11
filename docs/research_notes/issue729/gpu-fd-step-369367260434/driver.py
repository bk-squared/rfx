"""Predeclared FD step contraction, no gate change and no repeated AD."""
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
    _build_msl_f64_referee, _NUM_PERIODS, _N_FREQS, _closest_divisor,
)
from tests._msl_ad_objective import msl_band_mean_s21_sq

out = Path(os.environ['RFX_OUT'])
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert sha == os.environ['RFX_SHA']
assert jax.default_backend() == 'gpu'
assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd())
ad_path = Path(os.environ['RFX_AD_OUT'])/'outcome.json'
ad_record = json.loads(ad_path.read_text())
assert ad_record['source_sha'] == sha
receipt = dict(source_sha=sha, jax=jax.__version__, numpy=np.__version__,
               devices=str(jax.devices()), steps=[1e-4, 1e-5, 1e-6],
               retained_ad_sha256=hashlib.sha256(ad_path.read_bytes()).hexdigest(),
               retained_g_ad=ad_record['g_ad'],
               retained_h1e3_g_fd=ad_record['g_fd_retained'],
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'environment.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('PREDECLARATION', json.dumps(receipt), flush=True)
original_init = core.init_state
states = []


def capture_init(*args, **kwargs):
    state = original_init(*args, **kwargs)
    dtypes = [str(getattr(state, k).dtype) for k in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')]
    assert set(dtypes) == {'float64'}, dtypes
    states.append(dtypes)
    (out/'field-initializations.json').write_text(json.dumps(states)+'\n')
    return state


core.init_state = capture_init
rows = []
with jax.experimental.enable_x64():
    sim = _build_msl_f64_referee()
    grid = sim._build_grid()
    steps = int(grid.num_timesteps(num_periods=_NUM_PERIODS))
    segments = _closest_divisor(steps, int(np.sqrt(steps)))
    eps = jnp.ones(grid.shape, dtype=jnp.float64)
    for h in receipt['steps']:
        values = []
        start = time.monotonic()
        for sign in [1, -1]:
            print('START_FD', h, sign, flush=True)
            result = sim.compute_msl_s_matrix(n_freqs=_N_FREQS, num_periods=_NUM_PERIODS,
                                               eps_override=eps*jnp.float64(1.+sign*h),
                                               checkpoint_segments=segments)
            value = msl_band_mean_s21_sq(result.S)
            assert value.dtype == jnp.float64
            value = float(value)
            values.append(value)
            np.savez_compressed(out/f'h-{h:g}-sign-{sign}-result.npz', S=np.asarray(result.S),
                                freqs=np.asarray(result.freqs), loss=np.asarray(value))
        gradient = (values[0]-values[1])/(2.*h)
        row = dict(h=h, f_plus=values[0], f_minus=values[1], g_fd=gradient,
                   relative_to_retained_ad=abs(gradient-ad_record['g_ad'])/abs(gradient),
                   final_loss_ulp_span=abs(values[0]-values[1])/abs(np.spacing(np.float64(np.mean(values)))),
                   wall_s=time.monotonic()-start)
        rows.append(row)
        (out/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('FD_STEP_RESULT', json.dumps(row), flush=True)
print('REPORT_ONLY_COMPLETE', flush=True)
