"""Over-determined SOL solve + held-out validation. Distinguishes (a) imperfect
calibration standards (fixable) from (b) a deeper rfx reflection floor.

- S11 per standard = (Vr - Z*Ir)/(Vr + Z*Ir)  [V/I at ref plane, +x port].
- NUMERICAL beta from the load two-plane phase: beta = -angle(Vp/Vr)/(xp-xr).
- Cal plane = X_CAL (short1). Gamma_a(short@delta) = -exp(-2j*beta*delta), delta known.
- Over-determined LS solve of (T, e11) from all shorts (e00 = load). Leave-one-out
  held-out prediction error = calibration self-consistency.
- Apply to device; compare |S11| to matched MEEP, full band + sub-bands.
Run for analytic AND numerical beta to isolate the beta contribution.
"""
import numpy as np
A = "scripts/diagnostics/_artifacts"; band = np.linspace(5e9, 7e9, 11); C0 = 299792458.0; W = 0.04
DELTAS = [0, 3, 6, 9, 12, 15]

def s11(tag):
    z = np.load(f"{A}/tj_od_{tag}_dx2.0.npz")
    return (z["Vr"] - z["Z"]*z["Ir"])/(z["Vr"] + z["Z"]*z["Ir"]), z
G_load, zl = s11("load")
xr, xp = float(zl["ref_x_m"]), float(zl["probe_x_m"])
Vr, Vp = zl["Vr"], zl["Vp"]
beta_num = -np.angle(Vp/Vr)/(xp - xr)                    # empirical numerical beta
beta_ana = np.sqrt((2*np.pi*band/C0)**2 - (np.pi/W)**2)  # analytic TE10
print(f"beta num/ana band-mean: {beta_num.mean():.2f} / {beta_ana.mean():.2f} rad/m (ratio {beta_num.mean()/beta_ana.mean():.4f})")
Gm_short = {d: s11(f"s{d}")[0] for d in DELTAS}
Gm_dut = s11("device")[0]
z = np.load(f"{A}/meep_tjunction_farport_r500_drive0.npz"); M = np.interp(band, z["freqs_hz"], np.abs(z["col"])[0])

def solve(betas, idx):  # idx = which deltas to use; returns (e00,e11,T)
    e00 = G_load
    rows_T, rows_e, rhs = [], [], []
    for d in idx:
        Ga = -np.exp(-2j*betas*(d*1e-3)); D = Gm_short[d] - e00
        rows_T.append(Ga); rows_e.append(D*Ga); rhs.append(D)
    # per-freq LS solve of [T,e11]: D = T*Ga + e11*(D*Ga)
    T = np.zeros(len(band), complex); e11 = np.zeros(len(band), complex)
    for k in range(len(band)):
        Amat = np.array([[rows_T[i][k], rows_e[i][k]] for i in range(len(idx))])
        b = np.array([rhs[i][k] for i in range(len(idx))])
        sol, *_ = np.linalg.lstsq(Amat, b, rcond=None); T[k], e11[k] = sol
    return e00, e11, T

def apply(e00, e11, T, Gm):
    D = Gm - e00; return D/(T + e11*D)

for name, betas in [("analytic", beta_ana), ("numerical", beta_num)]:
    # leave-one-out held-out: predict each short from the others
    hoerr = []
    for d in DELTAS:
        others = [x for x in DELTAS if x != d]
        e00, e11, T = solve(betas, others)
        Ga_true = -np.exp(-2j*betas*(d*1e-3))
        Gm_pred = e00 + T*Ga_true/(1 - e11*Ga_true)
        hoerr.append(np.abs(Gm_pred - Gm_short[d]).max())
    e00, e11, T = solve(betas, DELTAS)
    S11c = np.abs(apply(e00, e11, T, Gm_dut))
    print(f"\n=== beta={name} ===  held-out max |pred-meas| over shorts: {max(hoerr):.3f} (mean {np.mean(hoerr):.3f})")
    print(f"  |e11| bm={np.abs(e11).mean():.3f} |T| bm={np.abs(T).mean():.3f}")
    print(f"  device |S11| SOL-corr std={S11c.std():.3f} (MEEP {M.std():.3f})")
    for lo,hi,l in [(0,11,'5.0-7.0'),(0,8,'5.0-6.5'),(8,11,'6.6-7.0')]:
        sl=slice(lo,hi); print(f"    {l}GHz cross-FDTD max={np.abs(S11c-M)[sl].max():.3f} mean={np.abs(S11c-M)[sl].mean():.3f}")
    print(f"  per-freq |S11|corr: "+" ".join(f"{v:.3f}" for v in S11c))
print(f"\nMEEP |S11|: "+" ".join(f"{v:.3f}" for v in M))