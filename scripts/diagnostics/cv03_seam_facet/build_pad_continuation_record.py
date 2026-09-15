"""Build the cv03 post-continuation evidence revision (append-only, new file)."""
import hashlib
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[3]
SWEEP_BEFORE = REPO / "scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json"
SWEEP_AFTER = REPO / "scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json"
CASE_LOG = REPO / "validation/crossval/_03_straight_waveguide_flux_logs/20260915_pad_continuation_run.log"
CASE = REPO / "validation/crossval/03_straight_waveguide_flux.py"
OUT = REPO / "docs/design_notes/issue1043_cv03_pad_continuation_record.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a],
                          capture_output=True, text=True).stdout.strip()


log = CASE_LOG.read_text(encoding="utf-8")
preflight = [ln.strip() for ln in log.splitlines() if "[PREFLIGHT]" in ln]
# The banner repeats per run() call; keep the distinct lines in first-seen order.
seen, preflight_unique = set(), []
for ln in preflight:
    if ln not in seen:
        seen.add(ln)
        preflight_unique.append(ln)
flux_region = []
for ln in log.splitlines():
    s = ln.strip()
    if s.startswith("[FLUX REGION]") and s not in flux_region:
        flux_region.append(s)

exit_code = int(re.search(r"^EXIT=(\d+)$", log, re.M).group(1))


def arms(path):
    d = json.load(open(path))
    out = {}
    for k, v in d["stages"]["sweep"].items():
        out[k] = {
            "b_over_a_carrier_bin": v["b_over_a_carrier_bin"],
            "T_rfx_band_mean": v["T_rfx_band_mean"],
            "settling_db": v["settling_db"],
            "exit_code": v.get("exit_code"),
            "two_wave_rel_residual_max": max(v["two_wave_rel_residual"]),
        }
    return out, d.get("provenance", {})


after, prov_after = arms(SWEEP_AFTER)
before, prov_before = arms(SWEEP_BEFORE)

verdict_lines = [ln.strip() for ln in log.splitlines()
                 if re.search(r"PASS \(gate|FAIL \(gate|SELF-CHECK|SKIP\]", ln)]

record = {
    "schema": "issue1043-cv03-pad-continuation-v1",
    "issue": 1043,
    "related": [831, 813],
    "lane": "cv03",
    "revision": 1,
    "revision_of": None,
    "adopted_by": [],
    "adoption_note": (
        "NOTHING adopts this revision. It is evidence, not a calibration: per "
        "the #928 split, a new revision moves no gate until a consumer's "
        "adoption record names it, and the artifact and an adoption record may "
        "not change in the same diff. cv03's committed gates, its window and "
        "its existing evidence artifact "
        "docs/design_notes/issue812_cv03_dispersion_matched_frequency.json are "
        "untouched by this change -- that file is a separate v1 artifact with "
        "its own generator and is NOT edited here. What this revision records "
        "is what the SAME committed recipe now measures once the guide is "
        "continued through its own absorber pad."
    ),
    "runs_fdtd": True,
    "predeclaration": (
        "docs/design_notes/issue831_far_end_return_predeclaration.md section 8.4"
    ),
    "falsifier_as_predeclared": {
        "text": (
            "The fix is right when the unmodified case reaches |B/A| <= 0.03 at "
            "20 layers, the 60-layer value sits below the 20-layer one, and "
            "settling_db clears -100 dB. Report band-mean T alongside and do "
            "not expect it to move."
        ),
        "arm_it_names": "sweep_upml_20 (cv03's committed absorber and depth)",
        "b_over_a_at_20": after["sweep_upml_20"]["b_over_a_carrier_bin"],
        "b_over_a_at_60": after["sweep_upml_60"]["b_over_a_carrier_bin"],
        "settling_db_at_20": after["sweep_upml_20"]["settling_db"],
        "T_band_mean_at_20": after["sweep_upml_20"]["T_rfx_band_mean"],
        "bar_b_over_a": 0.03,
        "bar_settling_db": -100.0,
        "passed": bool(
            after["sweep_upml_20"]["b_over_a_carrier_bin"] <= 0.03
            and after["sweep_upml_60"]["b_over_a_carrier_bin"]
            < after["sweep_upml_20"]["b_over_a_carrier_bin"]
            and after["sweep_upml_20"]["settling_db"] <= -100.0),
        "T_comment": (
            "0.9682, as section 8.4 predicted: T does NOT recover to 0.99 and "
            "was not expected to. The second effect the diagnosis lane did not "
            "isolate is still there. What DID change is T's depth trend: it "
            "degraded with depth before (0.9657 / 0.9470 / 0.9357) and no "
            "longer does (0.9682 / 0.9596 / 0.9558 upml; 0.9559 / 0.9571 / "
            "0.9565 cpml)."
        ),
    },
    "committed_case_run": {
        "script": "validation/crossval/03_straight_waveguide_flux.py",
        "script_sha256": sha(CASE),
        "log": "validation/crossval/_03_straight_waveguide_flux_logs/20260915_pad_continuation_run.log",
        "log_sha256": sha(CASE_LOG),
        "exit_code": exit_code,
        "exit_code_meaning": (
            "2 = Meep reference unavailable in this pod, crossval inconclusive. "
            "UNCHANGED from before: the committed recipe exited 2 with G1 and "
            "G2 both passing then too. A fix moves values here, not verdicts."
        ),
        "verdict_lines": verdict_lines,
        "settling_db": None,
        "settling_comment": (
            "None, and not because the run is unsettled: the committed case "
            "registers only flux monitors and a DFT plane, so there is no "
            "probe record to score (#885; section 8.7 item 2 of the #831 "
            "note filed it). The witness for these arms comes from the "
            "seam_facet driver, which adds a point probe at the fit-window "
            "centre: -138.3 dB on the committed recipe's arm, against -37.3 "
            "dB before the continuation."
        ),
        "preflight_verbatim": preflight_unique,
        "flux_region_verbatim": flux_region,
    },
    "sweep": {
        "driver": "scripts/diagnostics/cv03_seam_facet/seam_facet.py --stage sweep",
        "after": {
            "artifact": "scripts/diagnostics/_artifacts/cv03_seam_facet/sweep_r2.json",
            "sha256": sha(SWEEP_AFTER),
            "provenance": prov_after,
            "arms": after,
        },
        "before": {
            "artifact": "scripts/diagnostics/_artifacts/cv03_seam_facet/sweep.json",
            "sha256": sha(SWEEP_BEFORE),
            "provenance": prov_before,
            "arms": before,
            "note": (
                "Measured on branch diag/831-cv03-far-end-return before either "
                "#1043 stage landed. Left byte-for-byte as it was recorded; the "
                "new measurement is a NEW file beside it, never an edit to it."
            ),
        },
    },
    "witnesses": {
        "depth_trend": (
            "|B/A| now FALLS with absorber depth on both families (upml 0.0296 "
            "-> 0.0123 -> 0.0020; cpml 0.0515 -> 0.0055 -> 0.0006). It rose "
            "before (0.5311 -> 0.5918 -> 0.6221). A reflector at the seam is "
            "loaded less as the pad deepens because the first pad cell's sigma "
            "falls as N^-3; the absorber's own residual reflection falls. The "
            "sign of that trend is the discriminator, and it has flipped."
        ),
        "estimator_premise": (
            "two-wave relative residual max over the band is <= 0.0060 on every "
            "arm, so |B/A| is readable rather than a fit artefact."
        ),
        "independent_reading": (
            "The committed case's own printed band reads |B/A| = 0.017 .. 0.048 "
            "over the band, against the 0.393-0.585 recorded in the manifest's "
            "claim_scope for this case. Two routes, one conclusion."
        ),
        "cpml_20_is_the_absorber_not_a_facet": (
            "The cpml 20-layer arm reads 0.0515, above the 0.03 bar. It falls "
            "by ~9x per depth doubling (0.0055 at 40, 0.0006 at 60), which is "
            "an absorber's own residual reflection for an eps_r = 12 guided "
            "mode and not a facet. Section 8.4's bar is declared on the arm "
            "cv03 runs, which is UPML at 20 (0.0296). On the standalone unit "
            "rig the same quantity reads 0.0509/0.0061/0.0002 at 20/40/60 and "
            "is insensitive to transverse clearance, source position, fit "
            "window and record length -- so the depth ladder is the mechanism, "
            "measured twice."
        ),
    },
    "provenance": {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "base": "origin/main 541f703f (#1043 stage A, PR #1047)",
        "host": "remilab pod linux x86_64, CPU",
    },
}

OUT.write_text(json.dumps(record, indent=2) + "\n")
print("wrote", OUT)
print("falsifier passed:", record["falsifier_as_predeclared"]["passed"])
print("preflight lines:", len(preflight_unique), "verdict lines:", len(verdict_lines))
