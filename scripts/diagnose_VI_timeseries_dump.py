"""Dump V(t), I(t) full time series at LEFT ref_x (43mm) for PEC-short device run.

We monkey-patch extract_waveguide_port_waves to grab cfg.v_ref_t / cfg.i_ref_t
on every call. The 4th call (index 3) is the device run @ LEFT (drive=L).
We save its time series to npz for offline analysis.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg.extract_waveguide_port_waves


def _capture_extract(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_ref_t": np.asarray(cfg.v_ref_t).copy(),
        "i_ref_t": np.asarray(cfg.i_ref_t).copy(),
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "v_inc_t": np.asarray(cfg.v_inc_t).copy(),
        "n_steps_recorded": int(cfg.n_steps_recorded),
        "dt": float(cfg.dt),
        "dx": float(cfg.dx),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "ref_x_m": float(cfg.reference_x_m),
        "probe_x_m": float(cfg.probe_x_m),
        "src_x_m": float(cfg.source_x_m),
        "direction": cfg.direction,
        "ref_shift_m": float(ref_shift),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture_extract
import rfx.api as _api_mod  # noqa: E402
_api_mod.extract_waveguide_port_waves = _capture_extract

sys.path.insert(0, str(_ROOT / "examples" / "crossval"))
from importlib import import_module  # noqa: E402
cv11 = import_module("11_waveguide_port_wr90")

print("[VI-dump] running PEC-short via crossval/11 ...", flush=True)
freqs, s11, _ = cv11.run_rfx_pec_short()
print(f"[VI-dump] captured {len(_CAPS)} extract_waveguide_port_waves calls",
      flush=True)

# Per the normalize-loop layout (drive=LEFT, drive_idx=0):
#   [0] a_inc_ref @ LEFT (empty-ref run, drive=L)
#   [1] b_ref @ LEFT (empty-ref run, drive=L)  — same time series as [0]
#   [2] b_ref @ RIGHT (empty-ref run, drive=L)
#   [3] b_dev @ LEFT (PEC-short device run, drive=L)  ← this is what we want
#   [4] b_dev @ RIGHT (PEC-short device run, drive=L)
# (then drive=R captures [5..9] which are mirrored)
empty_L = _CAPS[0]
dev_L = _CAPS[3]

assert empty_L["direction"] == "+x" and empty_L["ref_x_m"] == dev_L["ref_x_m"], \
    "first and 4th captures should both be LEFT port"

print(f"\n[VI-dump] device run @ LEFT (capture[3]):")
print(f"  ref_x = {dev_L['ref_x_m']*1000:.1f} mm, src_x = {dev_L['src_x_m']*1000:.1f} mm")
print(f"  dt = {dev_L['dt']*1e12:.3f} ps,  dx = {dev_L['dx']*1000:.3f} mm")
print(f"  n_steps_recorded = {dev_L['n_steps_recorded']}, "
      f"buffer length = {dev_L['v_ref_t'].shape[0]}")
print(f"  |v_ref_t|max = {np.max(np.abs(dev_L['v_ref_t'])):.4e}")
print(f"  |i_ref_t|max = {np.max(np.abs(dev_L['i_ref_t'])):.4e}")
print(f"  |v_probe_t|max = {np.max(np.abs(dev_L['v_probe_t'])):.4e}")
print(f"  |i_probe_t|max = {np.max(np.abs(dev_L['i_probe_t'])):.4e}")

out_path = Path("/tmp/pec_short_VI_timeseries.npz")
np.savez(
    out_path,
    # Empty-ref run @ LEFT:
    empty_v_ref_t=empty_L["v_ref_t"], empty_i_ref_t=empty_L["i_ref_t"],
    empty_v_probe_t=empty_L["v_probe_t"], empty_i_probe_t=empty_L["i_probe_t"],
    empty_v_inc_t=empty_L["v_inc_t"],
    # Device run @ LEFT:
    dev_v_ref_t=dev_L["v_ref_t"], dev_i_ref_t=dev_L["i_ref_t"],
    dev_v_probe_t=dev_L["v_probe_t"], dev_i_probe_t=dev_L["i_probe_t"],
    dev_v_inc_t=dev_L["v_inc_t"],
    # Metadata:
    dt=dev_L["dt"], dx=dev_L["dx"],
    freqs=dev_L["freqs"], f_cutoff=dev_L["f_cutoff"],
    ref_x_m=dev_L["ref_x_m"], probe_x_m=dev_L["probe_x_m"],
    src_x_m=dev_L["src_x_m"],
    n_steps_recorded=dev_L["n_steps_recorded"],
    direction=dev_L["direction"],
)
print(f"\n[VI-dump] saved to {out_path}")
print(f"  Use np.load('{out_path}') to inspect.")
