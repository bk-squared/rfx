"""Inspect V(t), I(t) traces from /tmp/pec_short_VI_timeseries.npz.

Identify:
  (a) source-pulse arrival window at ref_x (incident wave)
  (b) reflected-pulse arrival window at ref_x (round-trip from PEC)
  (c) any third-bounce / CPML re-reflection

For a perfect PEC short with reflection coeff r=-1, the time-domain V(t) and
I(t) at ref_x should each have a 2-pulse structure (incident + reflected).
If the second pulse magnitude is ~92% of the first, that's the missing 8%
showing up directly in time domain.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

dat = np.load("/tmp/pec_short_VI_timeseries.npz")
dt = float(dat["dt"])
dx = float(dat["dx"])
fc = float(dat["f_cutoff"])
n_valid = int(dat["n_steps_recorded"])
ref_x = float(dat["ref_x_m"])
src_x = float(dat["src_x_m"])
freqs = dat["freqs"]
f0 = float(freqs.mean())   # ~10.3 GHz

V_emp = dat["empty_v_ref_t"][:n_valid]
I_emp_raw = dat["empty_i_ref_t"][:n_valid]
V_dev = dat["dev_v_ref_t"][:n_valid]
I_dev_raw = dat["dev_i_ref_t"][:n_valid]
v_inc = dat["dev_v_inc_t"][:n_valid]   # analytic injected source pulse

t = np.arange(n_valid) * dt

# Estimate Z_TE at f0 (analytic continuous form, just for time-domain inspection)
omega0 = 2 * np.pi * f0
beta0 = np.sqrt((omega0 / C0)**2 - (2*np.pi*fc/C0)**2)
Z0 = omega0 * MU_0 / beta0
print(f"\n[viz] dt = {dt*1e12:.3f} ps, n_valid = {n_valid}, "
      f"sim_time = {n_valid*dt*1e9:.2f} ns")
print(f"[viz] f0 = {f0/1e9:.2f} GHz, fc = {fc/1e9:.2f} GHz, "
      f"beta(f0) = {beta0:.1f} rad/m, Z(f0) = {Z0:.1f} Ω")

# Group velocity in WR-90 at f0
vg = C0 * np.sqrt(1 - (fc/f0)**2)
print(f"[viz] group velocity at f0 = {vg/C0:.4f} c = {vg/1e8:.3f} × 10^8 m/s")

# Distance from ref_x to PEC wall (at PORT_RIGHT_X - 5mm = 155mm)
PEC_X = 0.155
d_to_wall = PEC_X - ref_x
T_round = 2 * d_to_wall / vg
print(f"[viz] ref_x = {ref_x*1000:.1f} mm, PEC at {PEC_X*1000:.1f} mm")
print(f"[viz] one-way distance = {d_to_wall*1000:.1f} mm, "
      f"round-trip time at f0 = {T_round*1e9:.3f} ns")

# Time-shift I by +dt/2 (the _co_located correction in time domain).
# Equivalently, in frequency domain: multiply by exp(+jω·dt/2).
# Easiest: just use I as-is for time-domain inspection — the dt/2 shift
# is small (one half timestep) and doesn't change pulse arrival times.
I_dev = I_dev_raw
I_emp = I_emp_raw

ZI_dev = Z0 * I_dev
ZI_emp = Z0 * I_emp

# Wave decomposition in time domain (band-limited approximation using Z(f0))
fwd_dev = 0.5 * (V_dev + ZI_dev)
bwd_dev = 0.5 * (V_dev - ZI_dev)
fwd_emp = 0.5 * (V_emp + ZI_emp)
bwd_emp = 0.5 * (V_emp - ZI_emp)

# Find the peak times of fwd and bwd in the device run
def _peak_loc(x, t):
    abs_x = np.abs(x)
    i_peak = int(np.argmax(abs_x))
    return i_peak, t[i_peak], abs_x[i_peak]


# Identify peaks of incident (early forward pulse) and reflected (later
# backward pulse).
# Strategy: split time domain into "early" and "late" halves and find peaks.
# A more robust approach: use the analytic source pulse v_inc to identify
# the source emission time window.
v_inc_abs = np.abs(v_inc)
i_src_peak = int(np.argmax(v_inc_abs))
t_src_peak = t[i_src_peak]
print(f"\n[viz] source pulse peak at t = {t_src_peak*1e9:.3f} ns "
      f"(idx {i_src_peak})")

# Expected pulse arrivals at ref_x
T_src_to_ref = (ref_x - src_x) / vg   # forward travel time
T_src_to_wall = (PEC_X - src_x) / vg
T_back_from_wall = (PEC_X - ref_x) / vg
t_inc_arrive = t_src_peak + T_src_to_ref
t_refl_arrive = t_src_peak + T_src_to_wall + T_back_from_wall
print(f"[viz] expected incident-pulse arrival at ref_x: t = {t_inc_arrive*1e9:.3f} ns")
print(f"[viz] expected reflected-pulse arrival at ref_x: t = {t_refl_arrive*1e9:.3f} ns")
print(f"[viz] separation = {(t_refl_arrive - t_inc_arrive)*1e9:.3f} ns")

# Peak amplitudes in each window
gate_half_width = 0.3e-9   # 0.3 ns half-width

def _peak_in_window(x, t, t_center, half_width):
    mask = (t > t_center - half_width) & (t < t_center + half_width)
    if not mask.any():
        return None, None
    i_local = int(np.argmax(np.abs(x) * mask))
    return i_local, np.abs(x)[i_local]


print("\n=== EMPTY run @ LEFT (device = empty, just source pulse) ===")
i_pi, p_V_inc_emp = _peak_in_window(V_emp, t, t_inc_arrive, gate_half_width)
i_pr, p_V_refl_emp = _peak_in_window(V_emp, t, t_refl_arrive, gate_half_width)
print(f"  peak |V| in incident window:    {p_V_inc_emp:.4e}")
print(f"  peak |V| in reflected window:   {p_V_refl_emp:.4e}  "
      f"(should be near 0 — no reflector in empty)")

print("\n=== DEVICE run @ LEFT (PEC short at 155mm) ===")
i_pi, p_V_inc = _peak_in_window(V_dev, t, t_inc_arrive, gate_half_width)
i_pr, p_V_refl = _peak_in_window(V_dev, t, t_refl_arrive, gate_half_width)
print(f"  peak |V| in incident window:    {p_V_inc:.4e}")
print(f"  peak |V| in reflected window:   {p_V_refl:.4e}")
print(f"  ratio  reflected / incident:    {p_V_refl / p_V_inc:.4f}")

i_pi, p_I_inc = _peak_in_window(I_dev, t, t_inc_arrive, gate_half_width)
i_pr, p_I_refl = _peak_in_window(I_dev, t, t_refl_arrive, gate_half_width)
print(f"  peak |I| in incident window:    {p_I_inc:.4e}")
print(f"  peak |I| in reflected window:   {p_I_refl:.4e}")
print(f"  ratio  reflected / incident:    {p_I_refl / p_I_inc:.4f}")

# In wave-decomposition language:
i_pi, p_fwd_inc = _peak_in_window(fwd_dev, t, t_inc_arrive, gate_half_width)
i_pi, p_fwd_refl = _peak_in_window(fwd_dev, t, t_refl_arrive, gate_half_width)
i_pi, p_bwd_inc = _peak_in_window(bwd_dev, t, t_inc_arrive, gate_half_width)
i_pi, p_bwd_refl = _peak_in_window(bwd_dev, t, t_refl_arrive, gate_half_width)
print("\n  Wave decomposition (using Z(f0) only — narrowband approx):")
print(f"    fwd peak in incident window:  {p_fwd_inc:.4e}")
print(f"    fwd peak in reflected window: {p_fwd_refl:.4e}  "
      f"(should be ~0; reflected appears in bwd)")
print(f"    bwd peak in incident window:  {p_bwd_inc:.4e}  (should be ~0)")
print(f"    bwd peak in reflected window: {p_bwd_refl:.4e}")
print(f"    bwd_refl / fwd_inc ratio:     {p_bwd_refl / p_fwd_inc:.4f}  "
      f"<- THIS is the time-domain analogue of |bwd|/|fwd|")

# Look for any third-bounce structure
# Third bounce would arrive at t = t_inc_arrive + 2*T_round + 2*(src→ref)
# But the reflected wave passes the source plane and continues into LEFT CPML.
# Some fraction (~CPML reflection) bounces back. Look for this.
late_window_start = t_refl_arrive + T_round
late_mask = t > late_window_start
if late_mask.any():
    late_peak_V = np.max(np.abs(V_dev[late_mask]))
    print(f"\n  Late-time |V| peak (after t > {late_window_start*1e9:.3f} ns): "
          f"{late_peak_V:.4e}  ({100*late_peak_V/p_V_inc:.2f}% of incident peak)")

# Save a downsampled trace for offline plot inspection
out_path = Path("/tmp/pec_short_VI_traces.npz")
ds = max(1, n_valid // 4000)
np.savez(
    out_path,
    t=t[::ds],
    V_dev=V_dev[::ds], I_dev=I_dev[::ds],
    V_emp=V_emp[::ds], I_emp=I_emp[::ds],
    fwd_dev=fwd_dev[::ds], bwd_dev=bwd_dev[::ds],
    v_inc_table=v_inc[::ds],
    Z0=Z0,
    t_src_peak=t_src_peak,
    t_inc_arrive=t_inc_arrive,
    t_refl_arrive=t_refl_arrive,
)
print(f"\n[viz] downsampled traces saved to {out_path}")
