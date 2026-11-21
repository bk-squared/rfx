import sys, time
sys.path.insert(0, '/Users/byungkwankim/Documents/rfx-worktrees/absorber--2874-blocked')
import numpy as np
from rfx.measurement import matrix_pencil as mp
n, r, C, dt = 20000, 600, 8, 1.0
rng = np.random.default_rng(0)
s = -rng.uniform(1e-5, 1e-3, r) + 1j*rng.uniform(0.05, 3.0, r)
Y = rng.normal(size=(n, C))
def base(Y, s):
    m = np.arange(n, dtype=np.float64) * dt
    basis = np.exp(np.outer(m, s)); norms = np.linalg.norm(basis, axis=0)
    return np.linalg.lstsq(basis / norms, Y.astype(np.complex128), rcond=None)[0] / norms[:, None]
t0 = time.perf_counter(); ref = base(Y, s); tb = time.perf_counter() - t0
print(f'base dense lstsq: {tb:.6f} s')
round3_size = mp._residue_block_size
def round2_size(length, rank, channels):
    return min(length, max(1, int(mp._RESIDUE_BLOCK_TARGET_BYTES) // (16 * (rank + channels))))
for label, block_size in (("round 2", round2_size), ("round 3", round3_size)):
    mp._residue_block_size = block_size
    t0 = time.perf_counter(); got = mp._fit_residues(Y, s, dt); tt = time.perf_counter() - t0
    print(f'{label} blocked, target 1 MiB (Mb={mp._residue_block_size(n, r, C)}): {tt:.6f} s, '
          f'max|diff|/max|ref| {np.max(abs(got-ref))/np.max(abs(ref)):.2e}')
