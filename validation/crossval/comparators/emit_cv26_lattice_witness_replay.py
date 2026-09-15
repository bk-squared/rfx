#!/usr/bin/env python3
"""Emit cv26's derived lattice-witness numbers as a committed artifact.

Why this exists. The case script writes ``lattice.W_witness_*`` / ``GL1_*`` /
``GL2_*`` per arm, but that plumbing landed AFTER the lanes that produced the
committed records, so none of the 24 lane-20260913b artifacts nor the two
settle-60 rungs carries those keys. Prose that quotes a ``W_witness`` number
therefore had no ``path::key`` to resolve against. This recomputes the derived
witness from the committed per-arm records -- no FDTD, no re-simulation, the
same arithmetic ``evaluate_e2`` does -- and writes it where a citation can
read it.

It is a REPLAY, and says so in the document: every input is named (which
record, that record's own commit, the arm's own source tau, the standard and
section the budget comes from), so a reader can tell a recomputation from a
measurement.

    python validation/crossval/comparators/emit_cv26_lattice_witness_replay.py

Writes ``validation/crossval/_26_oblique_results/lattice_witness_replay.json``.
Append-only in the sense that matters: it never touches ``rfx.json`` or any
``rfx__*.json``, and re-running it on unchanged records reproduces it exactly.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("cv26_replay_oblique_fresnel", "validation/crossval/comparators/oblique_fresnel.py")

SCHEMA = "cv26-lattice-witness-replay/v1"
RESULTS = os.path.join(_REPO, "validation/crossval", O.RESULTS_DIRNAME)

# (key in this document, artifact file, arm inside it)
SOURCES = [(a, "rfx.json", a) for a in O.ARM_ORDER + O.GRAZE_ARMS]
SOURCES += [("te_00__settle60", "rfx__te_00_settle60.json", "te_00"),
            ("tm_00__settle60", "rfx__tm_00_settle60.json", "tm_00")]


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def replay_one(doc: dict, arm: str) -> dict:
    ad = doc["arms"][arm]
    run = ad["run"]
    cells = O.rig_cells(run["nx_interior"], run["n_cpml"], dx_div=run["dx_div"])
    e2 = O.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], O.arm_spec(arm), run["dt_s"],
                       tail=ad["tail"], cells=cells, n_cpml=run["n_cpml"],
                       inc_amp_rel=ad["inc_amp_rel"], record=run["record"])
    lat = e2["lattice"]
    out = {
        "arm": arm,
        "source_record_commit": doc.get("commit"),
        "dx_div": run["dx_div"], "n_cpml": run["n_cpml"], "n_steps": run["n_steps"],
        "n_bins_gated": e2["n_bins_gated"],
        # the budget's inputs, named so the number can be recomputed by hand
        "budget_inputs": {
            "src_tau_s": run["record"]["src_tau_s"],
            "dt_s": run["dt_s"],
            "tail_scat_refl_rel": ad["tail"]["scat_refl_rel"],
            "tail_total_trans_rel": ad["tail"]["total_trans_rel"],
            "tail_purity_inc_rel": ad["tail"]["purity_inc_rel"],
            "rate_ring_1_s": run["record"]["rate_ring_1_s"],
        },
    }
    if not lat.get("W_witness_defined", False):
        out.update({"W_witness_defined": False,
                    "W_witness_undefined_reason": lat.get("W_witness_undefined_reason")})
        return out
    out.update({
        "W_witness_defined": True,
        "tau_src_s_used": lat["tau_src_s"], "lambda_source": lat["lambda_source"],
        "witness_rate_1_s": lat["witness_rate_1_s"], "witness_rate_source": lat["witness_rate_source"],
        "rate_incident_1_s": lat["rate_incident_1_s"],
        "mean_W_witness_R_gated": lat["mean_W_witness_R_gated"],
        "mean_W_witness_T_gated": lat["mean_W_witness_T_gated"],
        "mean_dR_lattice_gated": lat["mean_dR_lattice_gated"],
        "mean_dT_lattice_gated": lat["mean_dT_lattice_gated"],
        "GL2_R": lat["GL2_R"], "GL2_T": lat["GL2_T"],
        "GL1_R_bins_beyond": lat["GL1_R_bins_beyond"], "GL1_T_bins_beyond": lat["GL1_T_bins_beyond"],
        "GL1_gated": lat["GL1_gated"], "GL1_not_gated_reason": lat["GL1_not_gated_reason"],
        # the term the budget does NOT model, carried beside it so the gap is visible
        "absorber_term_R_gated_max": lat["absorber_term_R_gated_max"],
        "absorber_term_T_gated_max": lat["absorber_term_T_gated_max"],
        "aux_echo_term_R_gated_max": lat["aux_echo_term_R_gated_max"],
        "absorber_term_over_window_R": (lat["absorber_term_R_gated_max"]
                                        / lat["mean_W_witness_R_gated"]),
        # #1015 section 14: WHICH lattice this entry's witness is judged against,
        # the geometry that chose it, and what the other reference would have
        # given. Carried so a reader can check the choice instead of trusting it.
        "witness_reference": lat["witness_reference"],
        "witness_reference_arrival_safe": lat["witness_reference_arrival_safe"],
        "witness_reference_inputs": lat["witness_reference_inputs"],
        "witness_reference_reason": lat["witness_reference_reason"],
        "mean_dR_lattice_gated_alt": lat["mean_dR_lattice_gated_alt"],
        "mean_dT_lattice_gated_alt": lat["mean_dT_lattice_gated_alt"],
        # #1015: where the window is a valid bound at all, and how the breaches
        # split across that line. Per bin, from the standard's own primitive.
        "domain_R": lat["domain_R"], "domain_T": lat["domain_T"],
    })
    return out


def build() -> dict:
    entries = {}
    for key, fname, arm in SOURCES:
        path = os.path.join(RESULTS, fname)
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        e = replay_one(doc, arm)
        e["source_record"] = f"validation/crossval/{O.RESULTS_DIRNAME}/{fname}"
        entries[key] = e
    judged = {k: v for k, v in entries.items() if v.get("W_witness_defined")}
    ref_free = sorted(k for k, v in entries.items() if v.get("witness_reference_arrival_safe"))
    ref_real = sorted(k for k in entries if k not in ref_free)
    gl2_fail = sorted(k for k, v in judged.items() if not (v["GL2_R"] and v["GL2_T"]))
    brch_in = sum(v[k]["n_bins_beyond_in_domain"] for v in judged.values()
                  for k in ("domain_R", "domain_T"))
    brch_out = sum(v[k]["n_bins_beyond_outside_domain"] for v in judged.values()
                   for k in ("domain_R", "domain_T"))
    return {
        "schema": SCHEMA, "case_id": O.CASE_ID,
        "kind": "replay",
        "what_this_is": (
            "A RECOMPUTATION, not a measurement. The derived lattice-witness window of "
            "docs/design_notes/20260903_lattice_witness_standard.md section 3, evaluated on "
            "cv26's committed per-arm records with that standard's own primitives "
            "(comparators/lattice_witness.py budget_terms / windows_from_terms / ringdown_rate) "
            "and cv26's 2-D-at-fixed-k_y lattice as the reference. No FDTD is run here. It exists "
            "because the case script gained these keys after the lanes that produced the records, "
            "so the records themselves carry none of them and prose quoting a W_witness number had "
            "nothing to resolve against."),
        "standard": "docs/design_notes/20260903_lattice_witness_standard.md",
        "standard_section": "3 (the window), 4 (GL1 per bin, GL2 band mean)",
        "source_tau_note": (
            "Each arm's budget uses THAT ARM's src_tau_s, not the slab family's TAU_SRC_S. cv26 "
            "drives a bandwidth per arm, so its tau is 2.0x to 135x the family constant, and "
            "LAMBDA = sqrt(pi) tau / dt divides every budget term."),
        "replay_commit": _commit(),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "arms": entries,
        "gl1_reference_note": (
            "#1015 section 14. WHICH lattice each entry's witness is judged against is decided by "
            "ARRIVAL, the same test standard section 3 (Z1) uses: an entry whose 3-D CPML round "
            "trip and whose #892 auxiliary-echo arrival are BOTH outside its record cannot have "
            "measured either echo, so it is judged against the ABSORBER-FREE lattice -- the slab "
            "family's own construction -- and its unmodelled term is zero BY ARRIVAL. cv26 was "
            "judging all ten entries against the lattice WITH the realized absorbers; on the five "
            "arrival-safe entries (te_00, tm_00, te_30, te_00__settle60, tm_00__settle60) that "
            "manufactured 302 all-bin and 62 in-domain GL1 breaches against content the record "
            "cannot contain, and correcting it takes them to zero with no window moved by more "
            "than 0.008 %. The five amplitude-capped entries are UNCHANGED: the same swap makes "
            "each of them markedly worse (mean|dR| 1.09x to 1.72x, and 321x / 472x on the compact "
            "grazing boxes), which is the evidence that this is a reference defect and not a "
            "looser reference. 'witness_reference' per entry, with both residuals beside it."),
        "gl1_domain_note": (
            "#1015. GL1's validity domain (standard section 13) is the bins where the term the "
            "section-3 budget does not model -- the reference's own dependence on the absorbers "
            "this rig carries INSIDE its record -- sits inside the window that omits it. The slab "
            "family declares that term zero by ARRIVAL and is unaffected; cv26 admits its echo by "
            "AMPLITUDE and so must report coverage. The predicate is necessary, not sufficient: "
            "it says where the bound is invalid, not that GL1 holds where it is valid. GL1 stays "
            "REPORTED and not gated here; GL2 is the gate. No window was widened for #1015."),
        "verdict": {
            "n_arms_with_defined_window": len(judged),
            "gl2_failing_arms": gl2_fail,
            "gl2_all_pass": not gl2_fail,
            "gl1_gated": False,
            "gl1_breaches_in_domain": brch_in,
            "gl1_breaches_outside_domain": brch_out,
            "witness_reference_absorber_free_by_arrival": ref_free,
            "witness_reference_realized_absorbers": ref_real,
        },
    }


def main(argv=None) -> int:
    doc = build()
    out = os.path.join(RESULTS, "lattice_witness_replay.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"{out}: {len(doc['arms'])} entries, "
          f"{doc['verdict']['n_arms_with_defined_window']} with a defined window")
    for k, v in doc["arms"].items():
        if not v.get("W_witness_defined"):
            print(f"  {k:18s} window UNDEFINED ({v['W_witness_undefined_reason'][:48]}...)")
            continue
        print(f"  {k:18s} [{'abs-free' if v['witness_reference_arrival_safe'] else 'realized'}] "
              f"W_R {v['mean_W_witness_R_gated']:.4e} vs |dR| {v['mean_dR_lattice_gated']:.4e} "
              f"GL2_R {str(v['GL2_R']):5s} GL1_R {v['GL1_R_bins_beyond']:4d}  "
              f"absorber/W {v['absorber_term_over_window_R']:6.2f}x  "
              f"domain {100 * v['domain_R']['domain_fraction']:5.1f}% "
              f"(breaches in/out {v['domain_R']['n_bins_beyond_in_domain']:3d}/"
              f"{v['domain_R']['n_bins_beyond_outside_domain']:3d})")
    print(f"\n  GL2 failing arms: {doc['verdict']['gl2_failing_arms'] or 'none'}")
    print(f"  GL1 breaches (R+T) inside the validity domain "
          f"{doc['verdict']['gl1_breaches_in_domain']}, outside "
          f"{doc['verdict']['gl1_breaches_outside_domain']}")
    print(f"  witness reference: absorber-free by arrival "
          f"{doc['verdict']['witness_reference_absorber_free_by_arrival']}; realized absorbers "
          f"{doc['verdict']['witness_reference_realized_absorbers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
