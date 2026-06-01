"""Solve the 3-term 1-port error model from the SOL standards and apply it to the
far-port T-junction device reflection (drive 0, S11). Reports SOL-corrected |S11|
vs the matched MEEP reference, full band and per sub-band.

Model: Gamma_m = e00 + T*Gamma_a/(1 - e11*Gamma_a).
Standards: load (Gamma_a=0 -> e00), short1 (Gamma_a=-1), short2 (Gamma_a=-e^{-2jβΔ}).
Cal reference plane = short1 plane (error terms absorb the rfx reference-plane phase);
β analytic, needed only over the small offset Δ.
"""
import numpy as np
A = "scripts/diagnostics/_artifacts"; band = np.linspace(5e9, 7e9, 11); C0 = 299792458.0; W = 0.04
DX = "2.0"
e00 = np.load(f"{A}/tj_sol_load_dx{DX}.npz")["S11"]
Gm1 = np.load(f"{A}/tj_sol_short1_dx{DX}.npz")["S11"]
Gm2 = np.load(f"{A}/tj_sol_short2_dx{DX}.npz")["S11"]
DELTA = float(np.load(f"{A}/tj_sol_short2_dx{DX}.npz")["delta"])
beta = np.sqrt((2*np.pi*band/C0)**2 - (np.pi/W)**2)
Ga1 = -np.ones_like(band, dtype=complex); Ga2 = -np.exp(-2j*beta*DELTA)
D1, D2 = Gm1 - e00, Gm2 - e00
e11 = (D1/Ga1 - D2/Ga2)/(D1 - D2)
T = D1/Ga1 - D1*e11
Gm_dut = np.load(f"{A}/tj_cal_device_dx{DX}.npz")["S"][0, 0]
D = Gm_dut - e00; S11c = np.abs(D/(T + e11*D)); raw = np.abs(Gm_dut)
z = np.load(f"{A}/meep_tjunction_farport_r500_drive0.npz"); M = np.interp(band, z["freqs_hz"], np.abs(z["col"])[0])
print(f"error terms band-mean |e00|={np.abs(e00).mean():.3f} |e11|={np.abs(e11).mean():.3f} |T|={np.abs(T).mean():.3f}")
print(f"|S11| raw std={raw.std():.3f} -> SOL-corrected std={S11c.std():.3f} (MEEP {M.std():.3f})")
for lo, hi, lbl in [(0,11,"5.0-7.0"),(0,8,"5.0-6.5"),(0,7,"5.0-6.4"),(0,6,"5.0-6.2")]:
    sl = slice(lo, hi); print(f"  {lbl}GHz: SOL cross-FDTD max={np.abs(S11c-M)[sl].max():.3f} mean={np.abs(S11c-M)[sl].mean():.3f}")
