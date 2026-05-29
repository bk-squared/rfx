"""Build the WR-90 nonuniform (graded-dy) broad-E5 envelope vs analytic Airy.

Mesh axis: graded dy_profile ratio (1.0-3.0). Geometry axis: slab eps_r.
Per-case max |S11|/|S21| diff vs single-mode analytic Airy.
"""
from __future__ import annotations
import json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
C0=299_792_458.0; ETA0=376.730313668; MAX_TOL=0.05; RATIO_FLOOR=0.005
MANIFEST=REPO/".omx/physics-gate/2026-05-27-waveguide-wr90-nonuniform-broad-e5/rfx-sweep/rfx_wr90_nonuniform_sweep_manifest.json"
OUT=REPO/".omx/physics-gate/2026-05-27-waveguide-wr90-nonuniform-broad-e5/waveguide_wr90_nonuniform_broad_e5_envelope.json"

def airy(f,er,L,fc):
    fcd=fc/np.sqrt(er); zv=ETA0/np.sqrt(1-(fc/f)**2); zd=(ETA0/np.sqrt(er))/np.sqrt(1-(fcd/f)**2)
    rho=(zd-zv)/(zd+zv); tau=2*zd/(zd+zv); taub=2*zv/(zd+zv)
    bd=(2*np.pi*f*np.sqrt(er)/C0)*np.sqrt(1-(fcd/f)**2); d=bd*L; e2=np.exp(-2j*d)
    return rho*(1-e2)/(1-rho*rho*e2), tau*taub*np.exp(-1j*d)/(1-rho*rho*e2)

def _commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(REPO)).decode().strip()[:7]
    except Exception: return "unknown"

def _validate(text):
    req=("broad","mesh","frequency","geometry","wr-90","nonuniform","airy")
    blk=("narrow","enabling","partial","experimental","shadow","only")
    lo=text.lower(); miss=[t for t in req if t not in lo]
    if miss: raise SystemExit(f"claim_scope missing: {miss}")
    bad=[t for t in blk if t in lo]
    if bad: raise SystemExit(f"claim_scope blocking: {bad}")

def main():
    m=json.loads(MANIFEST.read_text())
    fc=float(m["fc_te10_hz"]); slab_L=float(m["slab_length_m"])
    PL,PR=m["ports_x_m"]; RL=m["reference_planes_x_m"][0]; c=0.5*(PL+PR)
    cases=[]; diffs=[]
    for case in m["cases"]:
        d=np.load(REPO/case["rfx_npz"],allow_pickle=False)
        fr=d["freqs_hz"]; s11=d["s11"]; s21=d["s21"]; er=float(d["eps_r"]); slab_L=float(d["slab_length_m"])
        s11e,s21e=airy(fr,er,slab_L,fc); bv=(2*np.pi*fr/C0)*np.sqrt(1-(fc/fr)**2)
        s11r=s11e*np.exp(-2j*bv*(c-0.5*slab_L-RL)); s21r=s21e*np.exp(+1j*bv*slab_L)
        d11=np.abs(np.abs(s11)-np.abs(s11r)); d21=np.abs(np.abs(s21)-np.abs(s21r))
        cmax=float(max(d11.max(),d21.max())); diffs.append(cmax)
        cases.append({"tag":case["tag"],"grading_ratio":float(case["grading_ratio"]),
            "adjacent_ratio":float(case["adjacent_ratio"]),"n_cells_y":int(case["n_cells_y"]),
            "eps_r":er,"geometry":case["geometry"],
            "s11_max_mag_abs_diff":float(d11.max()),"s21_max_mag_abs_diff":float(d21.max()),
            "max_mag_abs_diff":cmax,"rfx_npz":case["rfx_npz"],
            "status":"passed" if cmax<=MAX_TOL else "failed"})
    diffs=np.array(diffs); mx=float(diffs.max()); mn=float(diffs.mean())
    rs=float((diffs.max()-diffs.min())/max(diffs.max(),1e-12))
    failed=[x for x in cases if x["status"]!="passed"]
    status="passed" if not failed else "failed"
    ratios=sorted({x["grading_ratio"] for x in cases}); eps_rs=sorted({x["eps_r"] for x in cases})
    scope=(f"broad rfx WR-90 rectangular_waveguide_port nonuniform-mesh "
        f"compute_waveguide_s_matrix(normalize=True) versus analytic Airy reference "
        f"envelope spanning the graded-dy mesh refinement axis (grading_ratio "
        f"{min(ratios):g}-{max(ratios):g}, adjacent-cell ratio up to "
        f"{max(x['adjacent_ratio'] for x in cases):.2f}), the frequency axis "
        f"({m['band_hz'][0]/1e9:.1f}-{m['band_hz'][1]/1e9:.1f} GHz X-band single-mode "
        f"TE10), and the geometry axis (eps_r in {eps_rs} centered slabs). The "
        f"graded-mesh discrete TE10 mode profile uses the Galerkin symmetric "
        f"generalized eigensolve (commit 13c9651). Truth source is independent "
        f"analytic Airy, not a same-class FDTD reference.")
    _validate(scope)
    env={"schema":"rfx.waveguide_wr90_nonuniform_broad_e5_envelope","schema_version":1,
        "status":status,"evidence_level":"E5-broad-mesh-frequency-geometry-nonuniform",
        "claim":(f"rfx WR-90 nonuniform graded-dy compute_waveguide_s_matrix(normalize=True) "
            f"vs analytic Airy across {len(ratios)} grading ratios and {len(eps_rs)} "
            f"geometries over X-band {'passes' if status=='passed' else 'fails'} broad-E5 0.05."),
        "claim_scope":scope,"commit_hash":_commit(),
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "max_mag_abs_tol":MAX_TOL,"ratio_spread_floor":RATIO_FLOOR,"noise_floor_baseline":0.0021,
        "primary_reference":{"label":"analytic_airy","truth_key":"airy_slab_closed_form",
            "path":"internal_closed_form","meta":{"eps_r_values":eps_rs,"slab_length_m":slab_L}},
        "cross_check_references":[],
        "envelope_summary":{"case_count":len(cases),
            "passed_case_count":sum(1 for x in cases if x["status"]=="passed"),
            "failed_case_count":len(failed),"freq_range_hz":list(m["band_hz"]),
            "cutoff_te10_hz":fc,"grading_ratios":ratios,"eps_r_values":eps_rs,
            "max_adjacent_ratio":float(max(x["adjacent_ratio"] for x in cases)),
            "max_mag_abs_diff_across_cases":mx,"mean_max_mag_abs_diff_across_cases":mn,
            "ratio_spread":rs,"primary_reference_label":"analytic_airy",
            "mesh_axis_kind":"nonuniform_dy_profile_ratio",
            "setup_recipe":{"cpml_layers":int(m["cpml_layers"]),"normalize":m["normalize"],
                "num_periods":int(m["num_periods"]),"base_dx_m":m["base_dx_m"],
                "domain_m":list(m["domain_m"])},
            "runtime_env":{"jax_default_backend":m.get("jax_default_backend"),
                "jax_version":m.get("jax_version"),"numpy_version":m.get("numpy_version")}},
        "diagnostic_note":(f"max_mag_abs_diff_across_cases {mx:.4f} (tol {MAX_TOL}); "
            f"graded-mesh TE10 via Galerkin eigensolve (13c9651); normalize=True NU path."),
        "rfx_manifest_path":str(MANIFEST),"cases":cases}
    OUT.write_text(json.dumps(env,indent=2))
    print(f"wrote {OUT}\nstatus: {status}, case_count: {len(cases)}")
    print(f"max_mag_abs_diff_across_cases: {mx:.4f}")
    for x in cases:
        print(f"  {x['tag']:22s} adj={x['adjacent_ratio']:.2f} |S11|={x['s11_max_mag_abs_diff']:.4f} "
              f"|S21|={x['s21_max_mag_abs_diff']:.4f} -> {x['status']}")

if __name__=="__main__": main()
