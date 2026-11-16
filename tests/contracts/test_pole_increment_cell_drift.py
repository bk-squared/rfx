"""B7: per-cell drift using the production applier and exactly promoted coefficients."""
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx.materials.lorentz import LorentzCoeffs, lorentz_p_component, pole_update_coeffs
from tests._x64_compat import enable_x64

EPS0 = 8.8541878128e-12
DT = 4.766437173827515e-14
N = 335680
RATES = np.array([1.000e7, 1.063e7, 1.126e7])


def cell_drift():
    """Mirror M0s windows and start-modal normalization without importing its harness."""
    with enable_x64(False):
        terms = tuple(np.asarray(x).reshape(1, 3, 1) for x in pole_update_coeffs(
            np.float32(0), (RATES / 2).astype(np.float32),
            np.float32((2 * np.pi * 1e9)**2), np.float32(DT), field_dtype=None))
    cc32 = np.float32(1 / EPS0)
    e0 = np.random.default_rng(1).uniform(.5, 2., 24).astype(np.float32)

    def run(dtype):
        damping, restoring, c = (jnp.asarray(x, dtype=dtype) for x in terms)
        coeffs = LorentzCoeffs(None, None, damping, restoring, (c, c, c), None)
        cc = jnp.asarray(cc32, dtype=dtype)
        e = jnp.broadcast_to(jnp.asarray(e0, dtype=dtype), (3, 24))
        p = jnp.zeros((1, 3, 24), dtype=dtype)

        def step(carry, unused):
            e, p, v = carry
            p_new, v_new = lorentz_p_component(coeffs, 2, e, p, v)
            return (e - cc * v_new[0], p_new, v_new), e

        return np.asarray(jax.jit(lambda: jax.lax.scan(step, (e, p, p), None, length=N)[1])())

    with enable_x64(False):
        samples32 = run(jnp.float32)
    with enable_x64():
        samples64 = run(jnp.float64)
    rows = []
    for i, nu in enumerate(RATES):
        ddt = nu / 2 * DT
        a, b = 2 / (1 + ddt), -(1 - ddt) / (1 + ddt)
        c = EPS0 * (2 * np.pi * 1e9)**2 * DT**2 / (1 + ddt)
        lam = np.roots([1., -(a - c / EPS0), -b])[0]
        f_cf = abs(np.angle(lam)) / (2 * np.pi * DT)
        edges = np.rint(np.arange(17) / (f_cf * DT)).astype(int)
        edges[-1] = N
        stats = []
        for samples in (samples32, samples64):
            windows = [samples[lo:hi, i].astype(np.float64) for lo, hi in zip(edges[:-1], edges[1:])]
            stats.append((np.array([abs(w).max(axis=0) for w in windows]),
                          np.array([w.mean(axis=0) for w in windows])))
        (peak32, mean32), (peak64, mean64) = stats
        damping, restoring, c = (float(x[0, i, 0]) for x in terms)
        keep, strength = 1 - damping, restoring + c * float(cc32)
        lam = np.roots([1., -(1 + keep - strength), keep])[0]
        beta = (lam - 1 + strength) / strength
        amplitude = abs(e0.astype(np.float64)) * np.hypot(1, beta.real / beta.imag)
        g = (peak32[-1] / peak32[0]) / (peak64[-1] / peak64[0]) - 1
        s = abs((mean32[-1] - mean32[0]) - (mean64[-1] - mean64[0])) / amplitude
        residual = float(np.float64(np.float32(a)) + np.float64(np.float32(b)) - 1)
        rows.append(dict(nu=float(nu), r=residual, f_cf=f_cf,
                         max_abs_g=float(abs(g).max()), max_s=float(s.max())))
    return rows


def test_pole_increment_cell_drift():
    start = time.monotonic()
    rows = cell_drift()
    wall = time.monotonic() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(json.dumps(dict(bar="B7", rows=rows, wall_s=wall, peak_rss_bytes=peak)), flush=True)
    for row in rows:
        assert row["max_abs_g"] <= 3.45e-4, row
        assert row["max_s"] <= 3.45e-4, row
