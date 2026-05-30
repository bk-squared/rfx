"""Issue #80 — does fitting ONLY clean-region patch probes give a passive,
correctly-resonant |S11|? Offline re-fit of the GPU dump (no FDTD).

The GPU raw-dump diag showed the single-mode fit residual is high (18-23%) for
probes <~9 cells from the MSL source and <0.3% for probes >=~13 cells (x>=8mm).
The full 18-probe fit (includes corrupted near-source probes) gives |S11|=5.04.
Here we re-fit using only clean-region probes across ALL 81 freqs and report
max|S11| (passivity) and the resonance dip frequency.
"""
from __future__ import annotations
import numpy as np
from rfx.api import Simulation
from rfx.probes.msl_wave_decomp import extract_msl_nprobe
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

EPS_R=3.38; H_SUB=0.787e-3; W_MSL=1.8e-3; L_MSL=8.0e-3; PORT_MARGIN=5.0e-3
DX=0.197e-3; DOM_X=29.747e-3; DOM_Y=18.130e-3; DOM_Z=12.787e-3; Y_C=DOM_Y/2
N_PROBES=18; N_OFFSET=3; N_SPACING=2
PASSIVE_TOL=1.05; TARGET_GHZ=9.21

def x_coords():
    sim=Simulation(freq_max=15e9,domain=(DOM_X,DOM_Y,DOM_Z),dx=DX,cpml_layers=8,boundary="cpml")
    mp=MSLPort(feed_x=PORT_MARGIN,y_lo=Y_C-W_MSL/2,y_hi=Y_C+W_MSL/2,z_lo=4e-3+DX,
               z_hi=4e-3+DX+H_SUB,direction="+x",impedance=50.0,excitation=None)
    return np.asarray(msl_probe_x_coords_n(sim._build_grid(),mp,n_probes=N_PROBES,
        n_offset_cells=N_OFFSET,n_spacing_cells=N_SPACING),dtype=float)

def refit(v,x,beta0,idx):
    out=extract_msl_nprobe(v[idx,:].T,x[idx],np.ones(v.shape[1],dtype=complex),beta0)
    return np.abs(np.asarray(out["s11"]))

def main():
    for NP in (200,400):
        d=np.load(f"scripts/diagnostics/_artifacts/patch_rawdump_np{NP}.npz",allow_pickle=True)
        v=np.asarray(d["raw_v"])[0,0,:N_PROBES,:]
        freqs=np.asarray(d["freqs_hz"],dtype=float)
        beta0=np.real(np.asarray(d["production_beta"])).astype(float)
        if beta0.ndim==0: beta0=np.full(freqs.shape[0],float(beta0))
        x=x_coords()
        xmm=x*1e3
        print(f"\n===== num_periods={NP} =====")
        print("probe x(mm):", " ".join(f"{q:.1f}" for q in xmm))
        # selections: full, and progressively dropping near-source probes
        sels={
            "full 0..17": list(range(18)),
            "drop<7.5mm (5..17)": [i for i in range(18) if xmm[i]>=7.5],
            "drop<8.0mm (6..17)": [i for i in range(18) if xmm[i]>=8.0],
            "drop<8.5mm (7..17)": [i for i in range(18) if xmm[i]>=8.5],
            "clean 3-probe 8,12,16": [8,12,16],
        }
        ir=int(np.argmin(np.abs(freqs-TARGET_GHZ*1e9)))
        for name,idx in sels.items():
            s11=refit(v,x,beta0,idx)
            imax=int(np.argmax(s11)); idip=int(np.argmin(s11))
            tag="PASS" if s11.max()<=PASSIVE_TOL else "FAIL"
            print(f" {name:22s} n={len(idx):2d}  max|S11|={s11.max():.4f}@{freqs[imax]/1e9:5.2f}G "
                  f"[{tag}]  dip@{freqs[idip]/1e9:5.2f}G(|S11|={s11.min():.3f})  |S11|@9.21={s11[ir]:.3f}")
        # full trace for the cleanest selection
        idx=[i for i in range(18) if xmm[i]>=8.0]
        s11=refit(v,x,beta0,idx)
        print(f" -- clean(>=8mm) trace near resonance (8-11 GHz) --")
        for k in range(freqs.shape[0]):
            g=freqs[k]/1e9
            if 8.0<=g<=11.0:
                print(f"    {g:6.3f} GHz  |S11|={s11[k]:.4f}")

if __name__=="__main__":
    main()
