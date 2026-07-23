"""#388 radiated-flux stop criterion — GPU validation on the LARGE uniform patch (build_cubic).

CPU is too slow for a patch-scale uniform fixture (~15M cells). On GPU we validate that on a
real radiating antenna where the soft feed deposits a static charge (flooring the interior-energy
criterion), the opt-in radiated-flux criterion STOPS (radiation settles), while the energy
criterion FLOORS. Also dumps an upper-plane flux trajectory (deciles, dB) as an independent
diagnostic of the radiation ring-down.

Run on VESSL gpu-rtx4090 via scripts/vessl_gpu_validate_flux_stop.yaml.
"""
import os
import numpy as np

os.environ.setdefault("PT_SKIP_NTFF", "1")
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, os.path.dirname(__file__))
from patch_tutorial_rfx import build_cubic  # noqa: E402

DMAX = int(os.environ.get("FLUX_DMAX", "40000"))
DECAY_BY = 1e-3
MIN_STEPS = 4000


def _fixture():
    sim, meta, _ = build_cubic()
    dom_x = meta["dom_x_mm"] * 1e-3
    dom_z = meta["z_total_mm"] * 1e-3
    # Huygens box enclosing the antenna, clear of the CPML (8 layers ~6mm)
    box_lo = (0.015, 0.015, 0.008)
    box_hi = (dom_x - 0.015, dom_x - 0.015, dom_z - 0.015)
    return sim, meta, dom_x, dom_z, box_lo, box_hi


def _add_flux_plane(sim, dom_x, dom_z):
    """Upper flux plane (broadside radiation) vector probes -> offline P_z trajectory."""
    z = 0.6 * dom_z
    cx = cy = dom_x / 2.0
    pts = [(cx + ox, cy + oy) for ox in (-0.03, -0.01, 0.01, 0.03)
           for oy in (-0.03, -0.01, 0.01, 0.03)]
    off = len(sim._probes)
    for (x, y) in pts:
        sim.add_vector_probe((x, y, z))
    return off, len(pts)


print("=== #388 radiated-flux stop — GPU validation (uniform patch, build_cubic) ===")
import jax
print("jax devices:", jax.devices())

# ---- run F: FLUX criterion + upper-plane probes for the trajectory ----
simF, meta, dom_x, dom_z, box_lo, box_hi = _fixture()
offF, nptsF = _add_flux_plane(simF, dom_x, dom_z)
print(f"domain {meta['dom_x_mm']}x{meta['z_total_mm']}mm dx={meta['dx_mm']}mm | box {box_lo}->{box_hi} | DMAX={DMAX}")
resF = simF.run(until_decay=DECAY_BY, decay_min_steps=MIN_STEPS, decay_max_steps=DMAX,
                radiated_flux_box=(box_lo, box_hi), skip_preflight=True)
tsF = np.asarray(resF.time_series)
nF = tsF.shape[0]

# offline upper-plane P_z(t) from the vector probes (independent trajectory)
P = np.zeros(nF)
for k in range(nptsF):
    c = offF + 6 * k
    P += tsF[:, c + 0] * tsF[:, c + 4] - tsF[:, c + 1] * tsF[:, c + 3]


def env(x, w=300):
    ax = np.abs(x)
    return np.array([np.max(ax[max(0, i - w):i + 1]) for i in range(len(ax))])


Penv = env(P)
w0 = int(0.2 * nF)
Ppk = np.max(Penv[w0:]) if nF > w0 else np.max(Penv)
print(f"\n[flux criterion] stopped at {nF}/{DMAX} steps  {'(FLOORED)' if nF >= DMAX - 100 else '(STOPPED)'}")
print("  upper-plane flux envelope (dB vs post-source peak), deciles:")
for d in range(1, 11):
    i = min(int(d / 10 * nF), nF - 1)
    print(f"    {d*10:3d}%: {10*np.log10(max(Penv[i]/Ppk, 1e-30)):7.1f} dB")

# ---- run E: ENERGY criterion (default) — does it floor? ----
simE, *_ = _fixture()
resE = simE.run(until_decay=DECAY_BY, decay_min_steps=MIN_STEPS, decay_max_steps=DMAX,
                skip_preflight=True)
nE = np.asarray(resE.time_series).shape[0]
print(f"\n[energy criterion] stopped at {nE}/{DMAX} steps  {'(FLOORED — the #388 bug)' if nE >= DMAX - 100 else '(STOPPED)'}")

print(f"\n=== SUMMARY: flux={nF}  energy={nE}  ===")
print(f"  flux stops while energy floors: {nF < nE and nE >= DMAX - 100}")
print(f"  flux earlier than energy by {nE - nF} steps")
