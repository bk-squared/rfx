"""Pin down which spectrum carries the 8% deficit in PEC-short |S11|.

Hypothesis space (post-architecture-fix, 2026-04-25):
  H1: 2-plane reference shift bias in `_shift_modal_waves` for high-reflection
  H2: Reference-pulse extraction (a_inc_ref) underestimate
  H3: CPML residual reflection at port_left double-counted

The diagnostic monkey-patches `extract_waveguide_port_waves` to capture every
(a, b) it returns during the two-run normalize. We then look at:
  - |a_inc_ref(f)|: forward incident at LEFT in the EMPTY run
  - |b_ref(f)|:     backward at LEFT in the EMPTY run (CPML/disc residual)
  - |b_dev(f)|:     backward at LEFT in the PEC-SHORT run
  - |b_dev - b_ref| / |a_inc_ref| -> S11_diag, should be 1.0 at PEC short

Comparisons:
  * |b_dev|/|a_inc_ref| ~ 1.0 -> raw reflection unbiased; subtraction step is fine
  * |b_dev - b_ref|/|a_inc_ref| ~ 0.92 -> bias is in subtraction (b_ref injected)
  * Magnitude vs phase split tells whether bias is normalisation or shift error
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make rfx importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Import sim before crossval to avoid module reload weirdness
from rfx.sources import waveguide_port as _wg_mod  # noqa: E402

# Capture intercept ---------------------------------------------------------
_CAPTURED: list[dict] = []
_orig = _wg_mod.extract_waveguide_port_waves


def _capturing_extract(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPTURED.append({
        "direction": cfg.direction,
        "ref_x_m": float(cfg.reference_x_m),
        "ref_shift_m": float(ref_shift),
        "a": np.asarray(a).copy(),
        "b": np.asarray(b).copy(),
        "freqs": np.asarray(cfg.freqs).copy(),
    })
    return a, b


_wg_mod.extract_waveguide_port_waves = _capturing_extract
# Patch the api.py imported name too — it resolves at module-load time.
import rfx.api as _api_mod  # noqa: E402

_api_mod.extract_waveguide_port_waves = _capturing_extract


# Now run crossval/11 PEC-short -------------------------------------------
sys.path.insert(0, str(_ROOT / "examples" / "crossval"))
from importlib import import_module  # noqa: E402

cv11 = import_module("11_waveguide_port_wr90")

print("[diag] running PEC-short via crossval/11 path ...", flush=True)
freqs, s11, _s21 = cv11.run_rfx_pec_short()
print(f"[diag] captured {len(_CAPTURED)} extract_waveguide_port_waves calls",
      flush=True)

# Layout: 2 ports, each driven once, two runs (ref, dev) per drive.
# For each drive: 1 a_inc_ref call + 2 b_ref + 2 b_dev = 5 calls.
# Total expected = 10 calls. We care about drive=LEFT (drive_idx=0):
#   call[0] -> a_inc_ref @ LEFT (ref, drive=L)  (we keep .a)
#   call[1] -> b_ref @ LEFT      (ref, drive=L) (we keep .b)
#   call[2] -> b_ref @ RIGHT     (ref, drive=L)
#   call[3] -> b_dev @ LEFT      (dev, drive=L) (we keep .b)
#   call[4] -> b_dev @ RIGHT     (dev, drive=L)

print("\n[diag] capture sequence:")
for i, c in enumerate(_CAPTURED):
    print(f"  [{i}] dir={c['direction']:>2s} ref_x={c['ref_x_m']*1000:.1f}mm "
          f"shift={c['ref_shift_m']*1000:+.1f}mm "
          f"|a|max={np.max(np.abs(c['a'])):.4e} "
          f"|b|max={np.max(np.abs(c['b'])):.4e}")

if len(_CAPTURED) < 5:
    print("[diag] FATAL: fewer than 5 captures — abort", file=sys.stderr)
    sys.exit(1)

# Pick the LEFT-drive captures
a_inc = _CAPTURED[0]["a"]
b_ref_L = _CAPTURED[1]["b"]
b_dev_L = _CAPTURED[3]["b"]
freqs_hz = _CAPTURED[0]["freqs"]

# Sanity: same shape
assert a_inc.shape == b_ref_L.shape == b_dev_L.shape

# Reconstruct S11 the way api.py does
s11_diag = (b_dev_L - b_ref_L) / a_inc
s11_raw = b_dev_L / a_inc                    # without ref subtraction

print("\n[diag] PEC-short |S11| breakdown (drive=LEFT, recv=LEFT):")
print(f"{'f_GHz':>7s} {'|a_inc|':>10s} {'|b_ref|':>10s} {'|b_dev|':>10s} "
      f"{'|b_dev|/|a|':>10s} {'|S11_diag|':>10s} "
      f"{'∠a':>7s} {'∠b_dev':>7s} {'∠b_ref':>7s}")
for k in range(0, len(freqs_hz), 2):
    print(
        f"{freqs_hz[k]/1e9:7.2f} "
        f"{abs(a_inc[k]):10.4e} "
        f"{abs(b_ref_L[k]):10.4e} "
        f"{abs(b_dev_L[k]):10.4e} "
        f"{abs(s11_raw[k]):10.4f} "
        f"{abs(s11_diag[k]):10.4f} "
        f"{np.degrees(np.angle(a_inc[k])):7.1f} "
        f"{np.degrees(np.angle(b_dev_L[k])):7.1f} "
        f"{np.degrees(np.angle(b_ref_L[k])):7.1f}"
    )

# Summary --------------------------------------------------------------------
ratio_b_over_a = np.abs(b_dev_L) / np.abs(a_inc)
ratio_after_sub = np.abs(s11_diag)
print("\n[diag] summary:")
print(f"  |b_dev|/|a_inc|         mean={ratio_b_over_a.mean():.4f} "
      f"min={ratio_b_over_a.min():.4f} max={ratio_b_over_a.max():.4f}")
print(f"  |S11_after_sub|         mean={ratio_after_sub.mean():.4f} "
      f"min={ratio_after_sub.min():.4f} max={ratio_after_sub.max():.4f}")
print(f"  mean ref leakage |b_ref|/|a_inc| = "
      f"{(np.abs(b_ref_L)/np.abs(a_inc)).mean():.4f}")

# Save
out = {
    "freqs_hz": freqs_hz.tolist(),
    "a_inc_real": a_inc.real.tolist(), "a_inc_imag": a_inc.imag.tolist(),
    "b_ref_real": b_ref_L.real.tolist(), "b_ref_imag": b_ref_L.imag.tolist(),
    "b_dev_real": b_dev_L.real.tolist(), "b_dev_imag": b_dev_L.imag.tolist(),
    "s11_raw_real": s11_raw.real.tolist(), "s11_raw_imag": s11_raw.imag.tolist(),
    "s11_diag_real": s11_diag.real.tolist(),
    "s11_diag_imag": s11_diag.imag.tolist(),
}
out_path = Path("/tmp/pec_short_8pct_diag.json")
out_path.write_text(json.dumps(out))
print(f"\n[diag] full spectra saved to {out_path}")
