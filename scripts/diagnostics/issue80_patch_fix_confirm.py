"""Issue #80 — confirm BOTH fixes on the real patch (GPU).

Tests two corrections, using DEFAULT probe settings (so it exercises the new
n_probe_offset h_sub floor, not an override):

  (1) offset floor (rfx/api/__init__.py): patch default offset 4 -> ~20 cells,
      past the source higher-order transient -> max|S11| should drop <= 1.05.
  (2) geometry: trace ON the substrate (no air gap) -> resonance dip should move
      from ~10.78 GHz toward the analytic Balanis 9.21 GHz.

Runs 4 configs at num_periods=400 (settling-clean from the cleanfit analysis):
  gap=True/False  x  (default probes).  The gap=True case isolates the geometry
effect on the dip; gap=False is the corrected patch that the test should adopt.

Pass target (gap=False): max|S11| <= 1.05 AND dip in 9.21 +- 0.20 GHz.
"""
from __future__ import annotations
import numpy as np
from rfx import Box, Simulation
from rfx.sources import GaussianPulse

EPS_R=3.38; H_SUB=0.787e-3; W=10.129e-3; L=8.595e-3; W_MSL=1.8e-3; L_MSL=8.0e-3
PORT_MARGIN=5.0e-3; DX=0.197e-3; DOM_X=29.747e-3; DOM_Y=18.130e-3; DOM_Z=12.787e-3
Y_C=DOM_Y/2; TARGET_GHZ=9.21; TOL_GHZ=0.20; PASSIVE_TOL=1.05


def build(gap: bool):
    sim=Simulation(freq_max=15e9,domain=(DOM_X,DOM_Y,DOM_Z),dx=DX,cpml_layers=8,boundary="cpml")
    sim.add_material("ro4003c",eps_r=EPS_R,sigma=0.0)
    sim.add(Box((0,0,4e-3),(DOM_X,DOM_Y,4e-3+DX)),material="pec")
    sim.add(Box((0,0,4e-3+DX),(DOM_X,DOM_Y,4e-3+DX+H_SUB)),material="ro4003c")
    # trace bottom: substrate_top (+DX gap if gap=True, else flush on substrate)
    z0 = 4e-3+DX+H_SUB + (DX if gap else 0.0)
    sim.add(Box((0,Y_C-W_MSL/2,z0),(PORT_MARGIN+L_MSL,Y_C+W_MSL/2,z0+DX)),material="pec")
    sim.add(Box((PORT_MARGIN+L_MSL,Y_C-W/2,z0),(PORT_MARGIN+L_MSL+L,Y_C+W/2,z0+DX)),material="pec")
    sim.add_msl_port(position=(PORT_MARGIN,Y_C,4e-3+DX),width=W_MSL,height=H_SUB,
                     direction="+x",impedance=50.0,
                     waveform=GaussianPulse(f0=8.5e9,bandwidth=1.6))
    return sim


def main():
    for gap in (True, False):
        sim=build(gap)
        # report the default offset actually chosen (proves the floor fired)
        off=sim._msl_ports[0].n_probe_offset; nsp=sim._msl_ports[0].n_probe_spacing
        print(f"\n===== gap={gap}  default n_probe_offset={off} cells "
              f"(x_probe0={ (PORT_MARGIN+off*DX)*1e3:.2f}mm), spacing={nsp} =====")
        sim.preflight()
        res=sim.compute_msl_s_matrix(n_freqs=81,num_periods=400.0)
        f=np.asarray(res.freqs,float); s11=np.abs(np.asarray(res.S)[0,0,:])
        imax=int(np.argmax(s11)); idip=int(np.argmin(s11))
        ir=int(np.argmin(np.abs(f-TARGET_GHZ*1e9)))
        passv = s11[imax] <= PASSIVE_TOL
        dip_ok = (TARGET_GHZ-TOL_GHZ) <= f[idip]/1e9 <= (TARGET_GHZ+TOL_GHZ)
        print(f" max|S11|={s11[imax]:.4f}@{f[imax]/1e9:.3f}G  "
              f"[passivity {'PASS' if passv else 'FAIL'}]")
        print(f" dip |S11|={s11[idip]:.4f}@{f[idip]/1e9:.3f}G  "
              f"[dip-loc {'PASS' if dip_ok else 'FAIL'} vs {TARGET_GHZ}+-{TOL_GHZ}]")
        print(f" |S11|@9.21={s11[ir]:.4f}")
        # trace near resonance
        for k in range(f.shape[0]):
            if 8.0 <= f[k]/1e9 <= 11.5:
                print(f"   {f[k]/1e9:6.3f} GHz |S11|={s11[k]:.4f}")
        if not gap:
            verdict = "XPASS-READY" if (passv and dip_ok) else "still failing"
            print(f" >>> corrected patch (gap=False): {verdict}")


if __name__=="__main__":
    main()
