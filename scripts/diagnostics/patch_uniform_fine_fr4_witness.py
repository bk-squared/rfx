"""#325 §1-B — the physics-decisive falsifier: does the UNIFORM-FINE substrate
build give a single clean fundamental at the cv05 FR4 operating point (vs the
mode-split the FIRED STOP-1 graded 6-cell re-registration produced)?

The grid-build lock (tests/test_issue325_uniform_fine_substrate.py) proved the
geometry rasterizes to 6 fine cells with the grading transition held clear. This
runs the ACTUAL FDTD harminv at the cv05 FR4 point (eps_r=4.3, 38x29.5mm patch,
60x55mm GP, 1.5mm/6-cell substrate) at two buffer widths (N_BUF=8, 16); the
substrate is identical, only the transition-z moves. A physical mode is
buffer-invariant; a transition artifact moves/splits.

PRE-DECLARED FALSIFIER (one R2 attempt, terminates either way):
  CLEAN  -> adopt uniform-fine: each buffer yields ONE dominant fundamental
           (amp >= 3x any peak OUTSIDE the known TM-family band 2.9-3.2 GHz),
           buffer-invariant |df| <= 1.5% across N_BUF 8<->16, in the predeclared
           band ~2.35-2.48 GHz (analytic minus the ~2.9% dx=1mm staircase, since
           the committed coarse-clean +2.65% is a z-under-res/staircase
           cancellation and fixing z-under-res should drop f_res below 2.5528).
  SPLIT  -> mechanism DEAD, STOP (R2): comparable-amplitude peaks near
           2.14/2.65/3.45 (pairwise < 3x, no single dominant) OR the dominant
           moves > 2% across buffers. No third mesh variant; escalate to a
           uniform-cubic stack (architecture), OR document coarse-clean as the
           supported config (both are note-sanctioned #325 resolutions).

Settling: until_decay + a -40 dB energy witness inline (a Q~55 patch needs it).
CPU-only, no openEMS. Run:
  PYTHONPATH=$(git rev-parse --show-toplevel) \
  PT_FINE_BUFFER_CELLS=8  python scripts/diagnostics/patch_uniform_fine_fr4_witness.py
"""
import os
import json

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from rfx import Simulation, Box  # noqa: E402
from rfx.boundaries.spec import BoundarySpec  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402
from rfx.auto_config import smooth_grading  # noqa: E402
from rfx.harminv import harminv  # noqa: E402

C0 = 2.998e8
# cv05 FR4 operating point
EPS_R = 4.3
H_SUB = 1.5e-3
N_SUB = 6
DZ_SUB = H_SUB / N_SUB
DX = 1.0e-3
W, L = 38.0e-3, 29.5e-3          # patch
GX, GY = 60.0e-3, 55.0e-3        # ground plane
PROBE_INSET = 8.0e-3
N_CPML = 8
N_BUF = int(os.environ.get("PT_FINE_BUFFER_CELLS", "8"))
OUTDIR = "scripts/diagnostics/cv05_investigation_results"


def build():
    n_below, n_above = 12, 25
    raw = np.concatenate([
        np.full(n_below, DX),
        np.full(N_BUF + 1 + N_SUB + 1 + N_BUF, DZ_SUB),
        np.full(n_above, DX),
    ])
    dz = smooth_grading(raw, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz), 0, 0.0)
    fi = np.where(np.isclose(dz, DZ_SUB, rtol=1e-6))[0]
    f0 = int(fi[0]) + N_BUF
    z_gnd_lo, z_gnd_hi = float(edges[f0]), float(edges[f0 + 1])
    z_sub_lo, z_sub_hi = float(edges[f0 + 1]), float(edges[f0 + 1 + N_SUB])
    z_patch_lo, z_patch_hi = z_sub_hi, float(edges[f0 + 1 + N_SUB + 1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    assert sub_cells == N_SUB, f"substrate {sub_cells} != {N_SUB} cells"

    dom_x, dom_y = GX + 20e-3, GY + 20e-3
    gx_lo, gy_lo = (dom_x - GX) / 2, (dom_y - GY) / 2
    gx_hi, gy_hi = gx_lo + GX, gy_lo + GY
    px_lo, px_hi = dom_x / 2 - L / 2, dom_x / 2 + L / 2
    py_lo, py_hi = dom_y / 2 - W / 2, dom_y / 2 + W / 2
    feed_x, feed_y = px_lo + PROBE_INSET, dom_y / 2

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, 0), dx=DX,
                     dz_profile=dz, boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=N_CPML)
    sim.add_material("fr4", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
    sim.add(Box((px_lo, py_lo, z_patch_lo), (px_hi, py_hi, z_patch_hi)), material="pec")
    src_z = z_sub_lo + DZ_SUB * 2.5
    sim.add_source(position=(feed_x, feed_y, src_z), component="ez",
                   waveform=GaussianPulse(f0=2.4e9, bandwidth=1.2))
    sim.add_probe(position=(feed_x + 4e-3, feed_y + 4e-3, src_z), component="ez")
    return sim, z_sub_lo, z_sub_hi


def main():
    eps_eff = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 * (1 + 12 * H_SUB / W) ** -0.5
    dl = 0.412 * H_SUB * ((eps_eff + 0.3) * (W / H_SUB + 0.264)) / \
        ((eps_eff - 0.258) * (W / H_SUB + 0.8))
    f_an = C0 / (2 * (L + 2 * dl) * np.sqrt(eps_eff))
    print(f"#325 §1-B uniform-fine FR4 witness | N_BUF={N_BUF} | "
          f"analytic f_res={f_an/1e9:.4f} GHz")

    sim, z_sub_lo, z_sub_hi = build()
    res = sim.run(num_periods=200, skip_preflight=True)
    ts = np.asarray(res.time_series).ravel()
    dt = float(res.dt)
    amp = np.abs(ts)
    peak = float(amp[int(len(amp) * 0.15):].max())
    end = float(amp[-max(1, len(amp) // 20):].mean())
    settle_db = 20 * np.log10(end / peak) if peak > 0 and end > 0 else float("-inf")
    print(f"  settling: end/peak = {settle_db:.1f} dB "
          f"({'OK' if settle_db < -40 else 'UNDERSETTLED'})")

    sig = ts[int(len(ts) * 0.3):]
    sig = sig - sig.mean()
    modes = [m for m in harminv(sig, dt, 1.5e9, 3.6e9)
             if m.Q > 3 and m.amplitude > 1e-8]
    modes.sort(key=lambda m: -m.amplitude)
    print(f"  harminv modes (by amp): "
          f"{[(round(m.freq/1e9, 4), round(m.Q, 1), float(f'{m.amplitude:.2e}')) for m in modes[:6]]}")
    dom = modes[0] if modes else None
    if dom is not None:
        outside = [m for m in modes if not (2.9e9 <= m.freq <= 3.2e9)
                   and m is not dom]
        ratio = dom.amplitude / max((m.amplitude for m in outside), default=1e-30)
        print(f"  DOMINANT: f={dom.freq/1e9:.4f} GHz Q={dom.Q:.1f} "
              f"amp-ratio-vs-outside-TM-band={ratio:.1f}x")
    out = dict(n_buf=N_BUF, analytic_ghz=round(f_an / 1e9, 4),
               settle_db=round(settle_db, 1),
               dominant_ghz=round(dom.freq / 1e9, 4) if dom else None,
               dominant_Q=round(dom.Q, 1) if dom else None,
               modes=[[round(m.freq / 1e9, 4), round(m.Q, 1), m.amplitude]
                      for m in modes[:8]])
    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/uniform_fine_fr4_nbuf{N_BUF}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"  WROTE {path}")


if __name__ == "__main__":
    main()
