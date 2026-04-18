"""V173-A physics-equivalence gate for T7 Phase 2.

Runs a downsized FR4 patch antenna in rfx, extracts (f_res via Harminv,
|S11| dip depth via lumped port), and prints a JSON line to stdout:

    {"sha": "<git HEAD>", "f_res_hz": <float>, "s11_dip_db": <float>,
     "s11_dip_f_hz": <float>}

Intended run pattern:

    git checkout bebdd57 && python scripts/v173a_physics_equivalence.py > v173a_pre.json
    git checkout main    && python scripts/v173a_physics_equivalence.py > v173a_post.json
    python -c "import json; pre=json.loads(open('v173a_pre.json').read()); post=json.loads(open('v173a_post.json').read()); df = abs(pre['f_res_hz']-post['f_res_hz'])/pre['f_res_hz']*100; ds = abs(pre['s11_dip_db']-post['s11_dip_db']); print(f'df={df:.3f}%, ds={ds:.3f} dB')"

Acceptance thresholds (T7 Phase 2 refactor target):
 - |Δf_res / f_res| < 0.1 % (floating-point reordering + JIT cache
   invariance)
 - |Δ|S11|_min dB| < 0.5 dB (same)

If both hold, T7 Phase 2 preserved antenna physics at the real-case
scale. If either fails, the refactor silently perturbed a working
antenna sim and the next session must bisect which PR is responsible.
"""

from __future__ import annotations

import json
import sys
import subprocess

import numpy as np

from rfx import Simulation, Box
from rfx.harminv import harminv


# --- Downsized patch antenna (smaller than crossval 05) -----------------------

_DX = 1.0e-3                    # 1 mm cells (crossval 05 uses 1 mm; matches)
_F_MAX = 4e9
_DOMAIN = (0.05, 0.05, 0.02)    # 50×50×20 mm — half of crossval 05 Lx/Ly
_FR4_EPS = 4.3
_SUB_T = 0.0015                 # 1.5 mm substrate
_PATCH_X = 0.028                # patch 28×36 mm → f_res ≈ 2.4 GHz on FR4
_PATCH_Y = 0.036
_SRC_POS = (0.025, 0.018, 0.0005)   # inside substrate, probe-feed point
_PROBE_POS = (0.025, 0.018, 0.0018) # just above the substrate


def _build_sim() -> Simulation:
    sim = Simulation(
        freq_max=_F_MAX, domain=_DOMAIN, dx=_DX,
        boundary="cpml", cpml_layers=8, pec_faces={"z_lo"},
    )
    # FR4 substrate
    sim.add_material("fr4", eps_r=_FR4_EPS)
    sim.add(
        Box((0.0, 0.0, 0.0), (_DOMAIN[0], _DOMAIN[1], _SUB_T)),
        material="fr4",
    )
    # Ground plane — z_lo is PEC via pec_faces. Patch on top of substrate.
    sim.add_material("metal", sigma=1e7)
    patch_lo = ((_DOMAIN[0] - _PATCH_X) / 2.0,
                (_DOMAIN[1] - _PATCH_Y) / 2.0,
                _SUB_T)
    patch_hi = (patch_lo[0] + _PATCH_X,
                patch_lo[1] + _PATCH_Y,
                _SUB_T + _DX)
    sim.add(Box(patch_lo, patch_hi), material="metal")
    # Probe-fed source: Ez across the substrate gap.
    sim.add_source(_SRC_POS, "ez")
    sim.add_probe(_PROBE_POS, "ez")
    return sim


def _extract_harminv_f_res(sim, n_steps: int = 3000) -> float:
    res = sim.run(n_steps=n_steps)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    # Harminv in the 1.5–3.5 GHz band (covers FR4 patch fundamental mode).
    modes = harminv(ts, dt, f_min=1.5e9, f_max=3.5e9, min_Q=5.0)
    if len(modes) == 0:
        return float("nan")
    # Pick the strongest mode by amplitude.
    amps = np.asarray([m.amplitude for m in modes])
    return float(modes[int(np.argmax(amps))].freq)


def _extract_s11_dip_from_probe(sim, n_steps: int = 3000) -> tuple[float, float]:
    """Compute a rough |S11| surrogate from the probe trace (FFT-based).

    This is NOT calibrated against a reference lane — it's a delta
    metric that needs to be identical pre- vs post-refactor for the
    equivalence check to pass.
    """
    res = sim.run(n_steps=n_steps)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    n_fft = len(ts)
    freqs = np.fft.rfftfreq(n_fft, dt)
    spec = np.abs(np.fft.rfft(ts * np.hanning(n_fft)))
    band = (freqs >= 1.5e9) & (freqs <= 3.5e9)
    f_band = freqs[band]
    s_band = spec[band]
    # Use log-amplitude minimum as a proxy for the S11 dip.
    eps = float(np.max(s_band)) * 1e-6
    db = 20.0 * np.log10(s_band + eps)
    dip_idx = int(np.argmin(db))
    return float(db[dip_idx]), float(f_band[dip_idx])


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sim_a = _build_sim()
    f_res = _extract_harminv_f_res(sim_a, n_steps=3000)
    sim_b = _build_sim()
    s11_dip_db, s11_dip_f = _extract_s11_dip_from_probe(sim_b, n_steps=3000)

    payload = {
        "sha": _git_sha(),
        "f_res_hz": f_res,
        "s11_dip_db": s11_dip_db,
        "s11_dip_f_hz": s11_dip_f,
    }
    print(json.dumps(payload))
