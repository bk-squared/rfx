"""G2 NaN-grad triage probe (2026-06-12, VESSL run 369367242390).

Reproduces ONLY iter-0 value_and_grad of msl_stub_notch_tuning at
L=9.5mm, with toggles to localize the NaN:

  RFX_PEC_OCC_KOTTKE=0|1   (set outside)  Kottke vs legacy occupancy
  PROBE_CKPT=seg|plain     segmented checkpointing vs checkpoint=True
  PROBE_PERIODS=<float>    NUM_PERIODS (default 4 for cheap CPU triage)
  PROBE_GRAD=1|0           0 = forward value only

Observed on GPU (full config): value finite (8.59e-3), grad nan.
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EX = os.path.join(REPO, "examples", "inverse_design", "msl_stub_notch_tuning.py")
spec = importlib.util.spec_from_file_location("msl_demo", EX)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from rfx.probes.msl_wave_decomp import _v_from_plane, _i_from_plane, extract_msl_nprobe  # noqa: E402

PERIODS = float(os.environ.get("PROBE_PERIODS", 4.0))
CKPT = os.environ.get("PROBE_CKPT", "seg")
DO_GRAD = os.environ.get("PROBE_GRAD", "1") not in ("0", "false")

f_target_arr = jnp.asarray([m.F_TARGET], dtype=jnp.float32)
sim, y_trace, trace_y_hi, d_set, p_set = m.build_sim(f_target_arr)
grid = sim._build_grid()

period = 1.0 / float(sim._freq_max)
n_raw = int(math.ceil(PERIODS * period / float(grid.dt)))
K = max(8, int(math.isqrt(n_raw)))
n_steps = ((n_raw + K - 1) // K) * K
ck_kwargs = {"checkpoint_segments": K} if CKPT == "seg" else {}
print(f"[probe] kottke={os.environ.get('RFX_PEC_OCC_KOTTKE','0')} ckpt={CKPT} "
      f"periods={PERIODS} n_steps={n_steps} (K={K}) grad={DO_GRAD}", flush=True)


def s21_at_f_target(L_stub):
    occ = m.build_stub_occ(grid, trace_y_hi, L_stub)
    fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                     skip_preflight=True, **ck_kwargs)
    freqs_arr = f_target_arr
    beta0 = (2.0 * jnp.pi * freqs_arr * jnp.sqrt(jnp.asarray(m.EPS_EFF, dtype=jnp.float32))
             / jnp.asarray(m.C0, dtype=jnp.float32))
    x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], dtype=jnp.float32)
    v_d = jnp.stack([_v_from_plane(fr, d_set.ez1_name, d_set),
                     _v_from_plane(fr, d_set.ez2_name, d_set),
                     _v_from_plane(fr, d_set.ez3_name, d_set)], axis=-1)
    i1_d = _i_from_plane(fr, d_set.hy_name, d_set)
    v_p = jnp.stack([_v_from_plane(fr, p_set.ez1_name, p_set),
                     _v_from_plane(fr, p_set.ez2_name, p_set),
                     _v_from_plane(fr, p_set.ez3_name, p_set)], axis=-1)
    i1_p = _i_from_plane(fr, p_set.hy_name, p_set)
    res_d = extract_msl_nprobe(v_d, x_probes, i1_d, beta0)
    res_p = extract_msl_nprobe(v_p, x_probes, i1_p, beta0)
    s21 = res_p["alpha"] / (res_d["alpha"] + 1e-30)
    return s21[0]


TARGET = os.environ.get("PROBE_TARGET", "cost")


def _intermediates(L_stub):
    occ = m.build_stub_occ(grid, trace_y_hi, L_stub)
    fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps,
                     skip_preflight=True, **ck_kwargs)
    freqs_arr = f_target_arr
    beta0 = (2.0 * jnp.pi * freqs_arr * jnp.sqrt(jnp.asarray(m.EPS_EFF, dtype=jnp.float32))
             / jnp.asarray(m.C0, dtype=jnp.float32))
    x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], dtype=jnp.float32)
    v_d = jnp.stack([_v_from_plane(fr, d_set.ez1_name, d_set),
                     _v_from_plane(fr, d_set.ez2_name, d_set),
                     _v_from_plane(fr, d_set.ez3_name, d_set)], axis=-1)
    i1_d = _i_from_plane(fr, d_set.hy_name, d_set)
    return v_d, i1_d, x_probes, beta0


def cost(L_stub):
    if TARGET == "cost":
        return jnp.abs(s21_at_f_target(L_stub)) ** 2
    v_d, i1_d, x_probes, beta0 = _intermediates(L_stub)
    if TARGET == "v_plane":
        # Chain cut BEFORE the extractor: plane-integrated V magnitude.
        return jnp.sum(jnp.abs(v_d) ** 2)
    if TARGET == "alpha_fixedbeta":
        # Extractor WITHOUT the beta scan: single lstsq at analytic beta0.
        from rfx.probes.msl_wave_decomp import _lstsq_alpha_gamma
        a, _, _ = _lstsq_alpha_gamma(v_d[0], x_probes,
                                     beta0[0].astype(jnp.complex64))
        return jnp.abs(a) ** 2
    if TARGET == "alpha_d":
        # Full extractor (beta scan + refine + lstsq), driven port only.
        res_d = extract_msl_nprobe(v_d, x_probes, i1_d, beta0)
        return jnp.abs(res_d["alpha"][0]) ** 2
    raise SystemExit(f"unknown PROBE_TARGET={TARGET}")


L0 = jnp.asarray(9.5e-3, dtype=jnp.float32)
t0 = time.time()
if DO_GRAD:
    val, g = jax.value_and_grad(cost)(L0)
    print(f"[probe] value={float(val):.6e}  grad={float(g):+.6e}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    sys.exit(2 if not np.isfinite(float(g)) else 0)
else:
    val = cost(L0)
    print(f"[probe] value={float(val):.6e}  ({time.time()-t0:.0f}s)", flush=True)
    sys.exit(2 if not np.isfinite(float(val)) else 0)
