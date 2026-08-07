"""Build the WR-90 nonuniform (graded-dy) FLUX broad-E5 envelope vs analytic Airy.

Mesh axis: graded dy_profile ratio (1.0-3.0). Geometry axis: slab eps_r in
{2, 4}. Extraction: compute_waveguide_s_matrix(normalize="flux") (issue #88
Step B). The eps_r=4 cases are the headline test — normalize=True floors at
~0.077 |S11| there; flux clears the broad-E5 tolerance, which since the
#574 regeneration is DERIVED from the measured envelope (MAX_TOL below),
not the flat 0.05 that predated the #576 absorber fix.
"""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[2]
C0=299_792_458.0; ETA0=376.730313668; RATIO_FLOOR=0.005

sys.path.insert(0, str(REPO))
# The FROZEN measured envelope of the #574 regeneration (16 cases, GPU,
# cpml_layers 183 = 0.75 lambda_g, num_periods 60), and the gate DERIVED from it
# through the repo-wide multiplier — the same shape the E4 producer uses, and
# #574 step 4. Not circular: this literal is a frozen record of a past
# measurement, and the current build's own envelope is cross-checked against it
# from outside by test_waveguide_nu_broad_e5_envelope_gates.py::
# test_envelope_is_recomputed_from_the_artifact_and_capped_from_outside, which
# also caps it — without that cap a regeneration that degraded 10x would
# re-derive a 10x looser gate and stay green (#576's dependency-closure trap).
#
# The 0.05 this replaces predates the #576 absorber fix: at cpml_layers=24
# (0.099 lambda_g) the envelope measured 0.015609, and after the fix it is
# 14.4x better, which left 0.05 sitting 46x above the thing it bounded — a gate
# that no longer measured anything. quantum=1000 because the residual is
# milli-scale (every unquantized-abs lane elsewhere uses 100).
from tests._gate_policy import gate_from_envelope  # noqa: E402
_MEASURED_ENVELOPE=0.001081
MAX_TOL=gate_from_envelope(_MEASURED_ENVELOPE, quantum=1000)
MANIFEST=REPO/".omx/physics-gate/2026-05-29-waveguide-wr90-nu-flux-broad-e5/rfx-sweep/rfx_wr90_nu_flux_sweep_manifest.json"
OUT=REPO/".omx/physics-gate/2026-05-29-waveguide-wr90-nu-flux-broad-e5/waveguide_wr90_nu_flux_broad_e5_envelope.json"

def airy(f,er,L,fc):
    fcd=fc/np.sqrt(er); zv=ETA0/np.sqrt(1-(fc/f)**2); zd=(ETA0/np.sqrt(er))/np.sqrt(1-(fcd/f)**2)
    rho=(zd-zv)/(zd+zv); tau=2*zd/(zd+zv); taub=2*zv/(zd+zv)
    bd=(2*np.pi*f*np.sqrt(er)/C0)*np.sqrt(1-(fcd/f)**2); d=bd*L; e2=np.exp(-2j*d)
    return rho*(1-e2)/(1-rho*rho*e2), tau*taub*np.exp(-1j*d)/(1-rho*rho*e2)

def _commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(REPO)).decode().strip()[:7]
    except Exception: return "unknown"

WITNESS_DIR=REPO/".omx/i574-step0-absorber-window/gpu"

def _settling_witness(m):
    """The MANDATORY settling witness for this lane, rebuilt from the step-0
    artifacts rather than hand-attached — a hand-attached block is dropped by
    the next regeneration, which is how a required witness goes missing quietly.

    CLAUDE.md requires one for any claims-bearing number taken from an open
    CPML domain at fixed num_periods. #576 recorded that it had not been run and
    made it a named step of #574. Form is observable-invariance + passivity, not
    energy-dB: rfx exposes no total-energy monitor, and on a LOSSLESS structure
    a truncated record shows up first as non-passive column power and then as a
    moved observable (see i574_e5_absorber_window_falsifier.py).

    The artifacts are keyed by the configuration they were measured at, so a
    regeneration that moves the absorber or the window cannot silently reuse
    them: the lookup misses and the build fails.
    """
    cpml=int(m["cpml_layers"]); npd=int(m["num_periods"])
    cells={}
    for g in ("pec_short","slab"):
        a=WITNESS_DIR/f"{g}_{cpml}x{npd}.json"; b=WITNESS_DIR/f"{g}_{cpml}x{2*npd}.json"
        if not (a.exists() and b.exists()):
            raise SystemExit(
                f"settling witness missing for cpml={cpml}, num_periods={npd}: "
                f"expected {a.name} and {b.name} under {WITNESS_DIR}. Re-measure "
                f"with i574_e5_absorber_window_falsifier.py before promoting an "
                f"envelope from this configuration (#574 named step, #576).")
        A=json.loads(a.read_text()); B=json.loads(b.read_text())
        cells[g]=dict(max_s11_np60=A["max_s11"],max_s11_np120=B["max_s11"],
            max_s11_shift=abs(A["max_s11"]-B["max_s11"]),
            max_col_power_np60=A["max_col_power"],max_col_power_np120=B["max_col_power"],
            cpml_fraction_lambda_g_low=A["cpml_fraction_lambda_g_low"])
    worst=max(c["max_s11_shift"] for c in cells.values())
    return {"form":"observable-invariance + passivity (case-19 idiom), not energy-dB",
        "why_not_energy_db":("rfx exposes no total-energy monitor; on a LOSSLESS "
            "structure a truncated record announces itself first as non-passive "
            "column power and then as a moved observable, so both are reported "
            "here (see i574_e5_absorber_window_falsifier.py)"),
        "configuration":f"cpml_layers={cpml}, num_periods={npd}, dx={m['base_dx_m']} m",
        "independent_axis":f"record window doubled {npd} -> {2*npd} periods, same absorber",
        "artifact":str(WITNESS_DIR.relative_to(REPO))+f"/{{pec_short,slab}}_{cpml}x{{{npd},{2*npd}}}.json",
        "cells":cells,"worst_max_s11_shift":worst,
        "verdict":("the record window does NOT bind at this absorber: doubling it "
            f"moves max|S11| by at most {worst:.2e}, far below the gate {MAX_TOL}. "
            "At the superseded 24-cell absorber the same doubling moved pec_short "
            "by 1.46e-02, 520x more — absorber depth and record length are "
            "co-conditions (#576), so this witness is only meaningful read at the "
            "absorber actually promoted.")}

def _validate(text):
    req=("broad","mesh","frequency","geometry","wr-90","nonuniform","airy","flux")
    blk=("narrow","enabling","partial","experimental","shadow","only")
    lo=text.lower(); miss=[t for t in req if t not in lo]
    if miss: raise SystemExit(f"claim_scope missing: {miss}")
    bad=[t for t in blk if t in lo]
    if bad: raise SystemExit(f"claim_scope blocking: {bad}")

def main():
    m=json.loads(MANIFEST.read_text())
    fc=float(m["fc_te10_hz"])
    PL,PR=m["ports_x_m"]; RL=m["reference_planes_x_m"][0]; c=0.5*(PL+PR)
    cases=[]; diffs=[]
    for case in m["cases"]:
        d=np.load(REPO/case["rfx_npz"],allow_pickle=False)
        fr=d["freqs_hz"]; s11=d["s11"]; s21=d["s21"]; er=float(d["eps_r"]); slab_L=float(d["slab_length_m"])
        s11e,s21e=airy(fr,er,slab_L,fc); bv=(2*np.pi*fr/C0)*np.sqrt(1-(fc/fr)**2)
        s11r=s11e*np.exp(-2j*bv*(c-0.5*slab_L-RL)); s21r=s21e*np.exp(+1j*bv*slab_L)
        d11=np.abs(np.abs(s11)-np.abs(s11r)); d21=np.abs(np.abs(s21)-np.abs(s21r))
        cmax=float(max(d11.max(),d21.max())); diffs.append(cmax)
        # PASSIVITY WITNESS (#496 ask 3). Every case here is a LOSSLESS slab, so
        # column power |S11|^2+|S21|^2 must be 1 and any departure is unphysical.
        # This lane shipped with NO passivity witness at all -- its builder wrote
        # no unitarity keys, which is why the #496 auditor could only print
        # "NOT MEASURED" for it while every band lane got a real reading. Same
        # key names as the band producer so one convention is read everywhere.
        colpow=np.abs(s11)**2+np.abs(s21)**2
        cases.append({"tag":case["tag"],"grading_ratio":float(case["grading_ratio"]),
            "adjacent_ratio":float(case["adjacent_ratio"]),"n_cells_y":int(case["n_cells_y"]),
            "eps_r":er,"geometry":case["geometry"],
            "s11_max_mag_abs_diff":float(d11.max()),"s21_max_mag_abs_diff":float(d21.max()),
            "max_mag_abs_diff":cmax,
            "unitarity_min":float(colpow.min()),"unitarity_max":float(colpow.max()),
            "dx_m":float(d["base_dx_m"]),
            "rfx_npz":case["rfx_npz"],
            "status":"passed" if cmax<=MAX_TOL else "failed"})
    diffs=np.array(diffs); mx=float(diffs.max()); mn=float(diffs.mean())
    rs=float((diffs.max()-diffs.min())/max(diffs.max(),1e-12))
    failed=[x for x in cases if x["status"]!="passed"]
    status="passed" if not failed else "failed"
    ratios=sorted({x["grading_ratio"] for x in cases}); eps_rs=sorted({x["eps_r"] for x in cases})
    scope=(f"broad rfx WR-90 rectangular_waveguide_port nonuniform-mesh "
        f"compute_waveguide_s_matrix(normalize='flux') Poynting power-flux extraction "
        f"versus analytic Airy reference envelope spanning the graded-dy mesh refinement "
        f"axis (grading_ratio {min(ratios):g}-{max(ratios):g}, adjacent-cell ratio up to "
        f"{max(x['adjacent_ratio'] for x in cases):.2f}), the frequency axis "
        f"({m['band_hz'][0]/1e9:.1f}-{m['band_hz'][1]/1e9:.1f} GHz X-band single-mode "
        f"TE10), and the geometry axis (eps_r in {eps_rs} centered slabs including the "
        f"strong eps_r=4 reflector that the normalize=True path floors at ~0.077). The "
        f"graded-mesh discrete TE10 mode profile uses the Galerkin symmetric generalized "
        f"eigensolve (commit 13c9651) and the flux S-matrix branch (issue #88 Step B). "
        f"Truth source is independent analytic Airy, not a same-class FDTD reference.")
    _validate(scope)
    env={"schema":"rfx.waveguide_wr90_nu_flux_broad_e5_envelope","schema_version":1,
        "status":status,"evidence_level":"E5-broad-mesh-frequency-geometry-nonuniform-flux",
        "claim":(f"rfx WR-90 nonuniform graded-dy compute_waveguide_s_matrix(normalize='flux') "
            f"vs analytic Airy across {len(ratios)} grading ratios and {len(eps_rs)} eps_r "
            f"geometries (incl. strong eps_r=4) over X-band "
            f"{'passes' if status=='passed' else 'fails'} broad-E5 {MAX_TOL}."),
        "claim_scope":scope,"commit_hash":_commit(),
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "max_mag_abs_tol":MAX_TOL,"ratio_spread_floor":RATIO_FLOOR,"noise_floor_baseline":0.0021,
        "primary_reference":{"label":"analytic_airy","truth_key":"airy_slab_closed_form",
            "path":"internal_closed_form","meta":{"eps_r_values":eps_rs}},
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
            f"graded-mesh TE10 via Galerkin eigensolve (13c9651); normalize='flux' NU path "
            f"(issue #88 Step B). eps_r=4 cases test strong-reflector extension past the "
            f"normalize=True 0.077 floor."),
        "settling_witness":_settling_witness(m),
        "rfx_manifest_path":str(MANIFEST),"cases":cases}
    OUT.write_text(json.dumps(env,indent=2))
    print(f"wrote {OUT}\nstatus: {status}, case_count: {len(cases)}")
    print(f"max_mag_abs_diff_across_cases: {mx:.4f}")
    for x in cases:
        print(f"  {x['tag']:22s} adj={x['adjacent_ratio']:.2f} |S11|={x['s11_max_mag_abs_diff']:.4f} "
              f"|S21|={x['s21_max_mag_abs_diff']:.4f} -> {x['status']}")

if __name__=="__main__": main()
