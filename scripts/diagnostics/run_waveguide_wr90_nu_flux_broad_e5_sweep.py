"""WR-90 nonuniform (graded dy) FLUX broad-E5 sweep vs analytic Airy.

Validates compute_waveguide_s_matrix(normalize="flux") on a GRADED transverse
mesh (dy_profile) — the issue #88 Step B path. Unlike the normalize=True NU
sweep (eps_r=2 only, capped by the ~0.077 |S11| floor on strong reflectors),
the flux extractor is tested across BOTH eps_r=2 and eps_r=4 so the headline
question — does flux extend the NU envelope to strong reflectors? — is
answered directly.

WR-90 X-band single-mode TE10, centered slab. dx=0.25mm base. Truth: analytic
Airy (independent). Settled num_periods=60 (flux is stable when settled; the
transition-num_periods overshoot documented in 20260529_flux_nu_wiring_design.md
does not apply at this settling).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0=299_792_458.0
A=22.86e-3; B=10.16e-3; FC=C0/(2*A)
BAND=(8.2e9,12.4e9); N_FREQS=11; BW=0.4
DX=0.25e-3; CPML=24; NUM_PERIODS=60
DOMAIN_X=200e-3; PL=40e-3; PR=160e-3; RL=50e-3; RR=150e-3
GRADING_RATIOS=(1.0,1.5,2.0,3.0)
# Flux extractor: test BOTH eps_r=2 (where normalize=True already passes)
# and eps_r=4 (the strong reflector that normalize=True floors at ~0.077).
# Two slab lengths each → 4 geometries × 4 ratios = 16 cases.
GEOMETRIES=((2.0,4e-3),(2.0,2e-3),(4.0,4e-3),(4.0,2e-3))
OUT=REPO/".omx/physics-gate/2026-05-29-waveguide-wr90-nu-flux-broad-e5/rfx-sweep"

def graded_dy(total, base_dx, ratio):
    n=int(round(total/base_dx)); x=np.linspace(-1,1,n)
    w=1.0+(ratio-1.0)*np.abs(x); return w/w.sum()*total

def run(ratio, eps_r, slab_L, freqs):
    dy=graded_dy(A, DX, ratio)
    sim=Simulation(freq_max=float(freqs[-1])*1.1, domain=(DOMAIN_X,A,B),
        boundary=BoundarySpec(x=Boundary(lo="cpml",hi="cpml"),
            y=Boundary(lo="pec",hi="pec"), z=Boundary(lo="pec",hi="pec")),
        cpml_layers=CPML, dx=DX, dy_profile=dy)
    c=0.5*(PL+PR)
    sim.add_material("slab", eps_r=eps_r, sigma=0.0)
    sim.add(Box((c-0.5*slab_L,0,0),(c+0.5*slab_L,A,B)), material="slab")
    pf=jnp.asarray(freqs); f0=float(np.mean(freqs))
    sim.add_waveguide_port(PL,direction="+x",mode=(1,0),mode_type="TE",freqs=pf,f0=f0,
        bandwidth=BW,waveform="modulated_gaussian",reference_plane=RL,name="left")
    sim.add_waveguide_port(PR,direction="-x",mode=(1,0),mode_type="TE",freqs=pf,f0=f0,
        bandwidth=BW,waveform="modulated_gaussian",reference_plane=RR,name="right")
    r=sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize="flux")
    s=np.asarray(r.s_params); pi={n:i for i,n in enumerate(r.port_names)}
    return np.asarray(r.freqs), s[pi["left"],pi["left"],:], s[pi["right"],pi["left"],:], float(dy.max()/dy.min()), len(dy)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freqs=np.linspace(BAND[0],BAND[1],N_FREQS)
    manifest={"schema":"rfx.waveguide_wr90_nu_flux_broad_e5_sweep_manifest","schema_version":1,
        "waveguide":"WR-90","a_m":A,"b_m":B,"band_hz":list(BAND),"fc_te10_hz":FC,
        "cpml_layers":CPML,"normalize":"flux","num_periods":NUM_PERIODS,"bandwidth_frac":BW,
        "base_dx_m":DX,"mesh_axis":"graded_dy_profile_ratio",
        "domain_m":[DOMAIN_X,A,B],"ports_x_m":[PL,PR],"reference_planes_x_m":[RL,RR],
        "freqs_hz":[float(f) for f in freqs],"cases":[]}
    print(f"WR-90 NU FLUX graded-dy: fc={FC/1e9:.2f}GHz band {BAND[0]/1e9}-{BAND[1]/1e9}, "
          f"base dx={DX*1e6:.0f}um, geometries={GEOMETRIES}")
    for ratio in GRADING_RATIOS:
        for eps_r, slab_L in GEOMETRIES:
            tag=f"ratio{ratio:g}_er{int(eps_r)}_L{int(slab_L*1e3)}mm"
            t0=time.time()
            print(f"  [{tag}] starting", flush=True)
            fr,s11,s21,adj,ncells=run(ratio,eps_r,slab_L,freqs)
            dt=time.time()-t0
            out=OUT/f"{tag}.npz"
            np.savez(out, freqs_hz=fr, s11=s11, s21=s21, grading_ratio=ratio,
                     adjacent_ratio=adj, n_cells_y=ncells, eps_r=eps_r,
                     slab_length_m=slab_L, base_dx_m=DX)
            manifest["cases"].append({"tag":tag,"grading_ratio":ratio,"adjacent_ratio":adj,
                "n_cells_y":ncells,"eps_r":eps_r,"slab_length_m":slab_L,"geometry":f"slab_er{int(eps_r)}_L{int(slab_L*1e3)}mm",
                "rfx_npz":str(out.relative_to(REPO)),"wallclock_s":dt})
            print(f"  [{tag}] done {dt:.1f}s adj-ratio={adj:.2f} "
                  f"|S11|max={np.abs(s11).max():.4f} |S21|mean={np.abs(s21).mean():.4f} -> {out.name}", flush=True)
    manifest["jax_default_backend"]=jax.default_backend(); manifest["jax_version"]=jax.__version__
    manifest["numpy_version"]=np.__version__
    mp=OUT/"rfx_wr90_nu_flux_sweep_manifest.json"; mp.write_text(json.dumps(manifest,indent=2))
    print(f"  manifest -> {mp}")

if __name__=="__main__": main()
