"""Compare measured V(f)/I(f) at the empty-run ref plane against Z_used.

In an EMPTY waveguide with a +x forward-traveling source pulse only:
    V(f) / I(f) = Z_mode_disc(f)
identically (from Maxwell + chosen normalisation).  The wave-decomposition
formula uses some Z_used(f); if Z_used != V/I, the empty-run yields
nonzero |b|/|a|, and any device run sees a systematic bias.

We dump V(f), I(f), Z_used at ref_x in the EMPTY run for the LEFT port,
then print the ratio (V/I)/Z_used.  A 1.04 ratio (4 %) at empty implies
ε ~ 8 % in the wave decomposition — exactly matches the |b_ref|/|a_inc|
~ 0.043 we already measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rfx.sources import waveguide_port as wg  # noqa: E402

# Patch _extract_global_waves to capture V_dft, I_dft, Z_mode
_CAPS: list[dict] = []
_orig = wg._extract_global_waves


def _capture_extract(cfg, voltage_dft, current_dft):
    Z = wg._compute_mode_impedance(
        cfg.freqs, cfg.f_cutoff, cfg.mode_type, dt=cfg.dt, dx=cfg.dx
    )
    I_corr = wg._co_located_current_spectrum(cfg, current_dft)
    _CAPS.append({
        "ref_x_m": float(cfg.reference_x_m),
        "direction": cfg.direction,
        "freqs": np.asarray(cfg.freqs).copy(),
        "V": np.asarray(voltage_dft).copy(),
        "I_raw": np.asarray(current_dft).copy(),
        "I_corr": np.asarray(I_corr).copy(),
        "Z_used": np.asarray(Z).copy(),
    })
    return _orig(cfg, voltage_dft, current_dft)


wg._extract_global_waves = _capture_extract

sys.path.insert(0, str(_ROOT / "examples" / "crossval"))
from importlib import import_module  # noqa: E402

cv11 = import_module("11_waveguide_port_wr90")

print("[VI] running empty run via crossval/11 ...", flush=True)
freqs, s11, _s21 = cv11.run_rfx_empty()
print(f"[VI] captured {len(_CAPS)} _extract_global_waves calls", flush=True)

# Reset and now run pec_short for comparison
_CAPS_EMPTY = list(_CAPS)
_CAPS.clear()

print("[VI] running PEC-short ...", flush=True)
_, s11_pec, _ = cv11.run_rfx_pec_short()
_CAPS_PEC = list(_CAPS)


def _print_VI(caps, label, *, only_first_per_run=4):
    """Print V/I/Z table for the LEFT-port reference-plane captures.

    Capture sequence (per drive_idx in extract_waveguide_s_params_normalized):
      ref_run -> ref_cfgs[0..n_ports-1] each call extract_waveguide_port_waves
      Each extract_waveguide_port_waves invokes _extract_port_waves_from_time_series
      which invokes _extract_global_waves_from_time_series ONCE per call.
    Per drive_idx of LEFT (=0):
      [0] _extract_global_waves at LEFT (ref run, drive=L)  <- a_inc_ref
      [1] _extract_global_waves at LEFT (ref run, drive=L)  <- b_ref @ LEFT
      [2] _extract_global_waves at RIGHT (ref run, drive=L) <- b_ref @ RIGHT
      [3] _extract_global_waves at LEFT (dev run, drive=L)  <- b_dev @ LEFT
      [4] _extract_global_waves at RIGHT (dev run, drive=L) <- b_dev @ RIGHT
    """
    # Take the first LEFT capture (a_inc_ref of LEFT-drive)
    pick = caps[0]
    V = pick["V"]
    I = pick["I_corr"]
    Z = pick["Z_used"]
    f = pick["freqs"]

    print(f"\n[{label}] ref_x={pick['ref_x_m']*1000:.1f}mm dir={pick['direction']}")
    print(f"{'f_GHz':>7s} {'|V|':>10s} {'|I|':>10s} {'|V/I|':>10s} "
          f"{'Z_used':>10s} {'(V/I)/Z':>9s} {'∠V/I':>7s} {'∠Z':>7s}")
    for k in range(0, len(f), 2):
        if abs(I[k]) < 1e-30:
            continue
        VI = V[k] / I[k]
        ratio = VI / Z[k]
        print(
            f"{f[k]/1e9:7.2f} {abs(V[k]):10.4e} {abs(I[k]):10.4e} "
            f"{abs(VI):10.2f} {abs(Z[k]):10.2f} "
            f"{abs(ratio):9.4f} "
            f"{np.degrees(np.angle(VI)):7.2f} "
            f"{np.degrees(np.angle(Z[k])):7.2f}"
        )

    ratios = np.abs(V / I) / np.abs(Z)
    phases = np.degrees(np.angle((V / I) / Z))
    print(f"  |V/I|/|Z_used| mean={ratios.mean():.4f}  "
          f"min={ratios.min():.4f}  max={ratios.max():.4f}")
    print(f"  ∠((V/I)/Z_used) mean={phases.mean():.2f}°  range=["
          f"{phases.min():.1f}, {phases.max():.1f}]°")


_print_VI(_CAPS_EMPTY, "EMPTY-only sim @ LEFT (capture[0])")

# In PEC-short normalize loop: capture[0,1,2] = empty-ref run (drive=L)
#                              capture[3,4]   = device run    (drive=L)
print("\n=== capture index map for PEC-short normalize loop ===")
for i, c in enumerate(_CAPS_PEC[:5]):
    print(f"  [{i}] dir={c['direction']:>2s} ref_x={c['ref_x_m']*1000:.1f}mm "
          f"|V|max={np.max(np.abs(c['V'])):.3e}")

# Print device-run @ LEFT (capture[3])
def _print_VI_idx(caps, idx, label):
    pick = caps[idx]
    V = pick["V"]; I = pick["I_corr"]; Z = pick["Z_used"]; f = pick["freqs"]
    ratios = np.abs(V / I) / np.abs(Z)
    phases = np.degrees(np.angle((V / I) / Z))
    print(f"\n[{label}] capture[{idx}] ref_x={pick['ref_x_m']*1000:.1f}mm "
          f"dir={pick['direction']}")
    print(f"  |V/I|/|Z_used| mean={ratios.mean():.4f}  "
          f"min={ratios.min():.4f}  max={ratios.max():.4f}")
    print(f"  ∠((V/I)/Z_used) mean={phases.mean():.2f}°  range=["
          f"{phases.min():.1f}, {phases.max():.1f}]°")
    # If V≈0 and I≠0 (current antinode) or vice versa, ratio explodes harmlessly
    return V, I, Z, f


V0, I0, Z0, _ = _print_VI_idx(_CAPS_PEC, 0, "PEC-short normalize: empty-ref @ L")
V3, I3, Z3, f = _print_VI_idx(_CAPS_PEC, 3, "PEC-short normalize: device   @ L")

# Forward/backward at LEFT in device run
fwd_dev = 0.5 * (V3 + Z3 * I3)
bwd_dev = 0.5 * (V3 - Z3 * I3)
fwd_emp = 0.5 * (V0 + Z0 * I0)
bwd_emp = 0.5 * (V0 - Z0 * I0)
print("\n[device run @ LEFT] |fwd_dev| / |fwd_emp|, |bwd_dev| / |fwd_emp|:")
print(f"{'f_GHz':>7s} {'|V_dev|':>10s} {'|I_dev|':>10s} "
      f"{'|fwd_d|':>10s} {'|bwd_d|':>10s} {'|fwd_e|':>10s} "
      f"{'|bwd|/|fwd_e|':>13s}")
for k in range(0, len(f), 2):
    print(f"{f[k]/1e9:7.2f} {abs(V3[k]):10.3e} {abs(I3[k]):10.3e} "
          f"{abs(fwd_dev[k]):10.3e} {abs(bwd_dev[k]):10.3e} "
          f"{abs(fwd_emp[k]):10.3e} {abs(bwd_dev[k])/abs(fwd_emp[k]):13.4f}")

# Also: in device run, fwd_dev should ALSO carry the source pulse (still
# being injected) plus part of the reflected wave that re-passes the TFSF.
# A good standing-wave PEC short should give |bwd_dev| ≈ |fwd_emp|
# IF the reflection is perfect AND the source is still emitting (so there
# is also significant fwd_dev). Compare the magnitudes.
print("\n[direct |V|/|V_empty| and |I|/|I_empty| at LEFT ref_x]:")
print(f"{'f_GHz':>7s} {'|V_d/V_e|':>10s} {'|I_d/I_e|':>10s}")
for k in range(0, len(f), 2):
    print(f"{f[k]/1e9:7.2f} {abs(V3[k])/abs(V0[k]):10.4f} "
          f"{abs(I3[k])/abs(I0[k]):10.4f}")
