"""Gated DFT: compare incident-only vs reflected-only spectra at ref_x.

From the V(t), I(t) traces dumped earlier, time-gate around:
  - incident pulse window  (centered at t_inc_arrive)
  - reflected pulse window (centered at t_refl_arrive)
DFT each window separately. The ratio |V_refl(f)| / |V_inc(f)| is the
true frequency-domain reflection coefficient observed at ref_x — it
factors out any DFT artifacts from mixing both pulses in one window.

If the ratio = 1 across the band, reflection is perfect and the 8 %
deficit must be a downstream extraction artifact.
If the ratio = 0.92 across the band, the deficit is a real round-trip
amplitude loss.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

dat = np.load("/tmp/pec_short_VI_timeseries.npz")
dt = float(dat["dt"])
fc = float(dat["f_cutoff"])
n_valid = int(dat["n_steps_recorded"])
ref_x = float(dat["ref_x_m"])
src_x = float(dat["src_x_m"])
freqs = dat["freqs"]
f0 = float(freqs.mean())

V_dev = dat["dev_v_ref_t"][:n_valid]
I_dev_raw = dat["dev_i_ref_t"][:n_valid]
V_emp = dat["empty_v_ref_t"][:n_valid]
I_emp_raw = dat["empty_i_ref_t"][:n_valid]
v_inc = dat["dev_v_inc_t"][:n_valid]

t = np.arange(n_valid) * dt
omega0 = 2 * np.pi * f0
beta0 = np.sqrt((omega0 / C0)**2 - (2*np.pi*fc/C0)**2)

vg = C0 * np.sqrt(1 - (fc/f0)**2)
PEC_X = 0.155
t_src_peak = t[int(np.argmax(np.abs(v_inc)))]
t_inc_arrive  = t_src_peak + (ref_x - src_x) / vg
t_refl_arrive = t_src_peak + (PEC_X - src_x)/vg + (PEC_X - ref_x)/vg
T_round = 2 * (PEC_X - ref_x) / vg
print(f"[gated] t_src_peak = {t_src_peak*1e9:.3f} ns")
print(f"[gated] t_inc_arrive = {t_inc_arrive*1e9:.3f} ns")
print(f"[gated] t_refl_arrive = {t_refl_arrive*1e9:.3f} ns "
      f"(separation {T_round*1e9:.3f} ns)")

# Gate half-width: leave room for the band-limited pulse envelope but stay
# clear of the next bounce. 4 envelope widths ≈ 4 * 50ps = 200ps gives a
# clean separation between incident and reflected.
gate_hw = 0.4e-9   # ±400 ps

# Build masks for incident-only and reflected-only windows
def _mask(t_center, hw):
    return (t > t_center - hw) & (t < t_center + hw)


m_inc = _mask(t_inc_arrive, gate_hw)
m_ref = _mask(t_refl_arrive, gate_hw)
print(f"[gated] incident window: {m_inc.sum()} samples covering "
      f"t = [{(t_inc_arrive-gate_hw)*1e9:.3f}, "
      f"{(t_inc_arrive+gate_hw)*1e9:.3f}] ns")
print(f"[gated] reflected window: {m_ref.sum()} samples covering "
      f"t = [{(t_refl_arrive-gate_hw)*1e9:.3f}, "
      f"{(t_refl_arrive+gate_hw)*1e9:.3f}] ns")
overlap = (m_inc & m_ref).sum()
if overlap > 0:
    print(f"[gated] WARNING: windows overlap by {overlap} samples — "
          f"reduce gate_hw")

# Apply the same dt/2 H time-correction in frequency domain (to be consistent
# with rfx's _co_located_current_spectrum)
def _rect_dft(x_masked, n_valid_actual):
    """OpenEMS-style rect-window DFT at the freqs grid."""
    n_idx = np.arange(n_valid_actual)
    tt = n_idx * dt
    omega = 2 * np.pi * freqs.astype(np.float64)
    phase = np.exp(-1j * omega[None, :] * tt[:, None])
    return 2.0 * dt * (x_masked.astype(np.float64) @ phase)


def _gated_VI_to_fwd_bwd(V, I_raw, mask, label):
    Vg = np.where(mask, V, 0.0)
    Ig = np.where(mask, I_raw, 0.0)
    Vdft = _rect_dft(Vg, n_valid)
    Idft = _rect_dft(Ig, n_valid)
    omega = 2 * np.pi * freqs.astype(np.float64)
    Idft_corr = Idft * np.exp(+1j * omega * 0.5 * dt)
    # Z at each freq (analytic continuous, narrowband enough for this test)
    beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
    Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
    fwd = 0.5 * (Vdft + Z * Idft_corr)
    bwd = 0.5 * (Vdft - Z * Idft_corr)
    return Vdft, Idft_corr, Z, fwd, bwd


print("\n=== EMPTY run (sanity: only forward incident, no reflector) ===")
Ve_inc, Ie_inc, Ze, fwde_inc, bwde_inc = \
    _gated_VI_to_fwd_bwd(V_emp, I_emp_raw, m_inc, "EMPTY incident")
Ve_ref, Ie_ref, _, fwde_ref, bwde_ref = \
    _gated_VI_to_fwd_bwd(V_emp, I_emp_raw, m_ref, "EMPTY reflected")
print(f"{'f_GHz':>7s} {'|fwd_inc|':>11s} {'|fwd_ref|':>11s} "
      f"{'|fwd_R/inc|':>11s}")
ratios_emp_fwd = np.abs(fwde_ref) / np.maximum(np.abs(fwde_inc), 1e-30)
for k in range(0, len(freqs), 4):
    print(f"{freqs[k]/1e9:7.2f} {abs(fwde_inc[k]):11.3e} "
          f"{abs(fwde_ref[k]):11.3e} {ratios_emp_fwd[k]:11.4f}")
print(f"  mean |fwd_ref|/|fwd_inc| = {ratios_emp_fwd.mean():.4f}  "
      f"(should be ~0; no reflector)")

print("\n=== DEVICE run (PEC short — gated DFT separates incident and reflected) ===")
Vd_inc, Id_inc, Zd, fwdd_inc, bwdd_inc = \
    _gated_VI_to_fwd_bwd(V_dev, I_dev_raw, m_inc, "DEVICE incident")
Vd_ref, Id_ref, _, fwdd_ref, bwdd_ref = \
    _gated_VI_to_fwd_bwd(V_dev, I_dev_raw, m_ref, "DEVICE reflected")

print(f"\n{'f_GHz':>7s} {'|fwd_inc|':>11s} {'|bwd_inc|':>11s} "
      f"{'|fwd_ref|':>11s} {'|bwd_ref|':>11s} "
      f"{'|bwd_R/fwd_I|':>13s}")
ratios_dev = np.abs(bwdd_ref) / np.maximum(np.abs(fwdd_inc), 1e-30)
for k in range(0, len(freqs), 2):
    print(f"{freqs[k]/1e9:7.2f} "
          f"{abs(fwdd_inc[k]):11.3e} {abs(bwdd_inc[k]):11.3e} "
          f"{abs(fwdd_ref[k]):11.3e} {abs(bwdd_ref[k]):11.3e} "
          f"{ratios_dev[k]:13.4f}")
print(f"\n  mean |bwd_REFLECTED| / |fwd_INCIDENT| = {ratios_dev.mean():.4f}  "
      f"(true reflection coeff at ref_x)")
print(f"  range = [{ratios_dev.min():.4f}, {ratios_dev.max():.4f}]")

# Compare to the ungated (full-record) DFT result we already had: 0.918 mean
print("\n=== INTERPRETATION ===")
print(
    "If gated reflection coefficient ≈ 0.92 (matches ungated 0.918):\n"
    "  → The 8% deficit is a REAL round-trip amplitude loss between\n"
    "    incident and reflected pulses. The bug is in propagation or\n"
    "    the discrete PEC reflection, NOT in DFT extraction.\n"
    "If gated ≈ 1.0:\n"
    "  → The deficit is a DFT/extraction artifact arising only when\n"
    "    both pulses are in the same time window. Time gating\n"
    "    eliminates it.\n"
)
