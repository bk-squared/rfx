"""Measure rfx waveguide-port TFSF source directionality.

Setup: empty WR-90 guide, drive LEFT port (TFSF at x=40mm). The TFSF source
should emit a +x-only forward wave. If the e_inc/h_inc table relationship has
even ~4% error, ~4% of injected energy emits backward (-x) and is absorbed by
the LEFT CPML (at 0–20 mm).

We measure (V, I) at TWO probe planes:
  - DOWNSTREAM probe at x_fwd = 60 mm (between source 40 mm and right CPML 180 mm)
  - UPSTREAM probe   at x_bwd = 28 mm (between left CPML 20 mm and source 40 mm)

For a perfectly directional source:
  fwd at downstream plane:   |V_fwd| = source amplitude, |V_bwd| = 0
  fwd at upstream plane:     |V_fwd| = 0, |V_bwd| = 0   (no backward emission)

If the source leaks backward, the upstream plane sees |V_bwd| > 0, equal in
magnitude to whatever fraction was emitted backward.

We construct a custom Simulation that adds two extra probe planes via
auxiliary `WaveguidePortConfig`s with src_amp=0 (passive probes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Patch _extract_global_waves to capture V/I/Z + ref_x for every call
from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg._extract_global_waves


def _capture(cfg, V, I):
    _CAPS.append({
        "V": np.asarray(V).copy(),
        "I": np.asarray(wg._co_located_current_spectrum(cfg, I)).copy(),
        "Z": np.asarray(wg._compute_mode_impedance(cfg.freqs, cfg.f_cutoff,
                        cfg.mode_type, dt=cfg.dt, dx=cfg.dx)).copy(),
        "f": np.asarray(cfg.freqs).copy(),
        "ref_x_m": float(cfg.reference_x_m),
        "dir": cfg.direction,
    })
    return _orig(cfg, V, I)


wg._extract_global_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402

# crossval/11 constants
A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.200, A_WG, B_WG
PORT_LEFT_X, PORT_RIGHT_X = 0.040, 0.160
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20


def _build_empty():
    """Empty WR-90 with one driven port at LEFT and a passive 'right' probe."""
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML_LAYERS,
        dx=DX_M,
    )
    port_freqs = jnp.asarray(FREQS_HZ)
    # LEFT port — driven (-x direction not used; +x is the active drive)
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.020, name="left",
    )
    # RIGHT port — passive observer at downstream of source.
    # reference_plane=0.150 → measures wave between source and right CPML
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=BW,
        waveform="modulated_gaussian", reference_plane=0.180, name="right",
    )
    return sim


print("[tfsf-dir] running EMPTY guide, drive LEFT only ...", flush=True)
_CAPS.clear()
sim = _build_empty()
result = sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize=True)
print(f"[tfsf-dir] captured {len(_CAPS)} _extract_global_waves calls",
      flush=True)
for i, c in enumerate(_CAPS):
    print(f"  [{i}] dir={c['dir']:>2s} ref_x={c['ref_x_m']*1000:6.1f}mm "
          f"|V|max={np.max(np.abs(c['V'])):.3e} "
          f"|I|max={np.max(np.abs(c['I'])):.3e}")

# In the normalize loop with empty=device:
# drive=LEFT (drive_idx=0):
#   [0] a_inc_ref @ LEFT (ref_x=43mm, dir=+x)
#   [1] b_ref @ LEFT (ref_x=43mm, dir=+x)
#   [2] b_ref @ RIGHT (ref_x=157mm, dir=-x)
#   [3] b_dev @ LEFT
#   [4] b_dev @ RIGHT
# drive=RIGHT (drive_idx=1):
#   [5..9] mirror

# We care about LEFT-driven only. For directionality:
#   capture[0] (LEFT, dir=+x): forward = source pulse going +x; backward = 0 if
#                              source is directional.
#   capture[2] (RIGHT, dir=-x): "backward" in port-local convention is +x.
#                               So a wave going +x AT RIGHT plane appears as
#                               port-local "backward" or "forward" depending
#                               on the +/- convention. For dir=-x: incident
#                               (port-local forward) is -x in global; outgoing
#                               (port-local backward) is +x in global.
#                               At RIGHT_PLANE downstream of source, the forward
#                               source wave IS going +x → seen as port-local
#                               OUTGOING.

print("\n[tfsf-dir] LEFT port a_inc capture[0] (ref_x=43mm, dir=+x):")
c = _CAPS[0]
V0, I0, Z0, f = c["V"], c["I"], c["Z"], c["f"]
fwd = 0.5 * (V0 + Z0 * I0)
bwd = 0.5 * (V0 - Z0 * I0)
ratio = np.abs(bwd) / np.abs(fwd)
print(f"  |bwd|/|fwd| at LEFT (downstream of source by 3 mm):")
for k in (0, 5, 10, 15, 20):
    print(f"    f={f[k]/1e9:5.2f} GHz   |bwd|/|fwd| = {ratio[k]:.4f}")
print(f"  mean={ratio.mean():.4f}")

# At RIGHT (downstream of source by ~117 mm, well-resolved forward wave only):
print("\n[tfsf-dir] RIGHT port b_ref capture[2] (ref_x=157mm, dir=-x):")
c = _CAPS[2]
V2, I2, Z2 = c["V"], c["I"], c["Z"]
# For dir=-x port: port-local forward is -x global. _extract_port_waves swaps
# fwd/bwd accordingly. extract_waveguide_port_waves returns (a, b) in
# port-local convention. For dir=-x at downstream-of-source, the +x forward
# source wave will appear as port-local outgoing (b).
fwd = 0.5 * (V2 + Z2 * I2)   # global +x wave
bwd = 0.5 * (V2 - Z2 * I2)   # global -x wave
print(f"  At RIGHT plane (downstream of source by 117 mm):")
print(f"    |GLOBAL +x wave| (= source forward through here) is the "
      f"larger amplitude — should match a_inc magnitude")
print(f"    |GLOBAL -x wave| should be near zero (no reflector)")
ratio_right = np.abs(bwd) / np.abs(fwd)
for k in (0, 5, 10, 15, 20):
    print(f"    f={f[k]/1e9:5.2f}  |+x|={abs(fwd[k]):.3e}  "
          f"|-x|={abs(bwd[k]):.3e}  ratio={ratio_right[k]:.4f}")
print(f"  mean |-x|/|+x| at RIGHT = {ratio_right.mean():.4f}")

# Compare: a_inc magnitude at LEFT (capture[0] forward) vs forward at RIGHT
fwd_L = 0.5 * (V0 + Z0 * I0)
fwd_R = 0.5 * (V2 + Z2 * I2)
amp_R_over_L = np.abs(fwd_R) / np.abs(fwd_L)
print("\n[tfsf-dir] Forward wave amplitude propagation:")
print(f"  |fwd at RIGHT| / |fwd at LEFT|  (should be ~1 if no losses)")
for k in (0, 5, 10, 15, 20):
    print(f"    f={f[k]/1e9:5.2f}  ratio = {amp_R_over_L[k]:.4f}")
print(f"  mean = {amp_R_over_L.mean():.4f}")

# CRITICAL TEST: in the LEFT capture[0], the bwd component in EMPTY MUST be
# entirely from extractor noise (no reflector). If extractor noise is ~4%
# AND there's also TFSF backward leakage, we'd need a separate probe upstream
# of the source to see it. capture[0] at ref_x=43mm is downstream of source
# (40mm), so it can't see upstream-emitted leakage.
#
# Capture[1] (also LEFT) is identical to capture[0] (both ref_run, drive=L).
# We don't have an upstream probe in the standard setup.
print("\n=== Interpretation ===")
print(
    "  At LEFT ref_x=43mm (downstream of source 40mm):\n"
    f"    |bwd|/|fwd| = {ratio.mean():.4f} (= V/I phase-mismatch leak, NOT TFSF leakage)\n"
    "  At RIGHT ref_x=157mm (downstream of source by 117mm):\n"
    f"    |fwd_R|/|fwd_L| = {amp_R_over_L.mean():.4f} (any deficit < 1 = real propagation loss\n"
    "                                                between 43mm and 157mm)\n"
    "  A deficit here would indicate energy absorbed by something between the\n"
    "  two planes (e.g. by the right-port TFSF apparatus at x=160mm even\n"
    "  with src_amp=0).\n"
    "\n"
    "  To detect TFSF backward leakage we need a probe UPSTREAM of source:\n"
    "  add a third manual port at x ~ 28mm. Not done in this script.\n"
)
