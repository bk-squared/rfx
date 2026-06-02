"""T-junction finer-mesh convergence analysis.
rfx self-convergence (dx 2.0->1.0 mm), MEEP self-convergence (res 400->1000),
converged cross-FDTD agreement, and the widest all-gates-passing band at the
finest resolution."""
import os, numpy as np
A = "/tmp/rfx-tj/scripts/diagnostics/_artifacts"
BAND = np.linspace(5.0e9, 7.0e9, 11)

# --- rfx |S| per dx (2.0, 1.6 from the saved arrays; 1.5, 1.0 from per-dx npz) ---
rfx = {}
z = np.load(f"{A}/tjunction_S_arrays.npz")
if z["band"].shape[0] == 11:
    rfx[2.0] = z["S2"]; rfx[1.6] = z["S16"]
for dxm in (1.5, 1.0):
    p = f"{A}/tj_rfx_dx{dxm:.1f}.npz"
    if os.path.exists(p): rfx[dxm] = np.load(p)["S"]

def meep_mat(res):
    M = np.zeros((3, 3, len(BAND)))
    for d in range(3):
        cand = [f"{A}/meep_tjunction_r{res}_drive{d}.npz"]
        if res == 400: cand.append(f"{A}/meep_tjunction_drive{d}.npz")
        f = next((c for c in cand if os.path.exists(c)), None)
        if f is None: return None
        zz = np.load(f); col = np.abs(zz["col"]); fm = zz["freqs_hz"]
        for j in range(3): M[j, d] = np.interp(BAND, fm, col[j])
    return M
meep = {r: meep_mat(r) for r in (400, 600, 1000)}
meep = {r: m for r, m in meep.items() if m is not None}

def passiv(S, sl=slice(None)): return float(np.sum(S[:,:,sl]**2, axis=0).max())
def recip(S, sl=slice(None)): return float(max(np.mean(np.abs(S[i,j,sl]-S[j,i,sl])) for (i,j) in ((1,0),(2,0),(2,1))))

print("=== rfx self-convergence (full 5-7 GHz) ===")
dxs = sorted(rfx, reverse=True)
for dxm in dxs:
    print(f" dx={dxm}mm: passivity_max={passiv(rfx[dxm]):.3f} recip={recip(rfx[dxm]):.3f}")
for i in range(len(dxs)-1):
    d = float(np.abs(rfx[dxs[i]] - rfx[dxs[i+1]]).max())
    print(f"  |S(dx={dxs[i]})-S(dx={dxs[i+1]})| max = {d:.3f}")

print("\n=== MEEP self-convergence (full band) ===")
ress = sorted(meep)
for r in ress: print(f" res={r}: passivity_max={passiv(meep[r]):.3f} recip={recip(meep[r]):.3f}")
for i in range(len(ress)-1):
    d = float(np.abs(meep[ress[i]] - meep[ress[i+1]]).max())
    print(f"  |MEEP(res={ress[i]})-MEEP(res={ress[i+1]})| max = {d:.3f}")

fine_dx = min(rfx); fine_res = max(meep)
R, M = rfx[fine_dx], meep[fine_res]
print(f"\n=== converged cross-FDTD: rfx(dx={fine_dx}) vs MEEP(res={fine_res}) ===")
print(" per-freq: " + " ".join(f"{BAND[k]/1e9:.1f}:{np.abs(R[:,:,k]-M[:,:,k]).max():.3f}" for k in range(len(BAND))))
print(f" full-band max||S|| = {np.abs(R-M).max():.3f}  band-mean = {np.mean(np.abs(R-M)):.3f}")

print("\n=== widest band passing all gates at FINEST (pass<=1.10, recip<=0.05, xdev<=0.05/0.08/0.11) ===")
for tol in (0.05, 0.08, 0.11):
    best = None
    for lo in range(len(BAND)):
        for hi in range(lo+2, len(BAND)):
            sl = slice(lo, hi+1)
            if passiv(R, sl) <= 1.10 and recip(R, sl) <= 0.05 and np.abs(R[:,:,sl]-M[:,:,sl]).max() <= tol:
                bw = (BAND[hi]-BAND[lo])/((BAND[hi]+BAND[lo])/2)*100
                if best is None or bw > best[0]: best = (bw, BAND[lo]/1e9, BAND[hi]/1e9)
    print(f" xdev<={tol}: " + (f"{best[1]:.1f}-{best[2]:.1f} GHz ({best[0]:.0f}% BW)" if best else "none"))
