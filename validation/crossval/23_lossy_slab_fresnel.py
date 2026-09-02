"""Cross-validation 23: Lossy slab Fresnel R(f), T(f), A(f) -- rfx vs analytic TMM vs Meep

A conductive slab (eps' = 4, 10 mm, sigma from tan delta 0.1 / 1 / 3 at 7 GHz)
on the cv22 / cv04 TFSF rig (``comparators/slab_rig.py``: byte-identical rig,
derived record length, -40 dB witness), with the ORDINARY ``update_e`` and
``materials.sigma`` in the slab -- the ``Simulation.add_material(..., sigma=)``
loss path, never gated before. Gated observables: R, T AND the absorption
A = 1 - R - T, per bin and band-mean, against the transfer matrix with
eps' - j sigma/(omega eps0) (E2) and against Meep's D_conductivity (E4).

Pre-declaration (arms, windows, falsifiers, records -- read it first):
  docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md
Numbers live in ONE place:
  validation/crossval/comparators/cv23_lossy_gates.py (on cv22_dispersive_gates)

Material paths (note section 2): arm tand0p1 builds materials.sigma directly
(init_materials + .at[slab].set, cv22's construction); arms tand1 and tand3
go through the documented Simulation.add_material + Simulation.add(Box) +
_assemble_materials path and assert the assembled arrays equal the direct
ones bit-for-bit before the run.

Exit codes (rfx crossval convention):
  0 = E2 pass on all arms AND the Meep leg present and E4 pass on all arms
  1 = any gate failed (a falsifier arm MUST exit 1)
  2 = E2 pass but a Meep JSON is missing -- inconclusive, NOT a pass

Run:
  python validation/crossval/23_lossy_slab_fresnel.py              # all arms
  python validation/crossval/23_lossy_slab_fresnel.py --falsifier tand1_sigma_x1p5
  python validation/crossval/23_lossy_slab_fresnel.py --smoke      # <= 20 s, no gates
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "comparators"))
import cv22_dispersive_gates as G  # noqa: E402
import cv23_lossy_gates as L  # noqa: E402
import slab_rig as RIG  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, L.RESULTS_DIRNAME)
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def _git_commit() -> str:
    return RIG.staged_commit(REPO_ROOT, cwd=SCRIPT_DIR)


# =============================================================================
# The slab material: direct arrays, or the documented Simulation API path
# =============================================================================

def build_materials(rig: dict, params: dict, path: str):
    """Return (MaterialArrays, info). ``path`` is "direct" or "api" (note
    section 2). The API path builds the SAME grid through ``Simulation`` and
    asserts its eps_r / sigma / mu_r equal the direct arrays bit-for-bit."""
    import jax.numpy as jnp
    from rfx.core.yee import init_materials

    grid = rig["grid"]
    lo, hi = rig["slab_lo"], rig["slab_hi"]
    eps_r, sigma = float(params["eps_inf"]), float(params["sigma"])
    direct = init_materials(grid.shape)
    direct = direct._replace(eps_r=direct.eps_r.at[lo:hi, :, :].set(eps_r),
                             sigma=direct.sigma.at[lo:hi, :, :].set(sigma))
    info = {"path": path, "slab_cells": [int(lo), int(hi)], "eps_r": eps_r, "sigma_s_m": sigma}
    if path == "direct":
        return direct, info
    if path != "api":
        raise ValueError(path)
    from rfx import Box, Simulation
    from rfx.geometry.csg import _grid_coords
    sim = Simulation(freq_max=20e9, domain=rig["domain"], dx=rig["dx"],
                     cpml_layers=rig["n_cpml"], mode="2d_tmz")
    sim.add_material(L.API_MATERIAL_NAME, eps_r=eps_r, sigma=sigma)
    xs, _ys, _zs = _grid_coords(grid)
    # Lattice arithmetic (csg.py:86-110): the exact node coordinates of the
    # first slab cell (inclusive) and the first cell after it (exclusive).
    sim.add(Box((float(xs[lo]), -1.0, -1.0), (float(xs[hi]), 1.0, 1.0)), material=L.API_MATERIAL_NAME)
    api_grid = sim._build_grid()
    if tuple(api_grid.shape) != tuple(grid.shape) or float(api_grid.dt) != float(grid.dt):
        raise RuntimeError(f"API grid {api_grid.shape}/{api_grid.dt} != rig grid {grid.shape}/{grid.dt}")
    mats, _debye, _lorentz, pec_mask, *_rest = sim._assemble_materials(api_grid)
    same = (bool(jnp.array_equal(mats.eps_r, direct.eps_r)) and bool(jnp.array_equal(mats.sigma, direct.sigma))
            and bool(jnp.array_equal(mats.mu_r, direct.mu_r)))
    no_pec = pec_mask is None or not bool(jnp.any(pec_mask))
    if not (same and no_pec):
        raise RuntimeError("Simulation.add_material path did not reproduce the direct slab arrays")
    info.update({"api_equals_direct": True, "api_no_pec": True, "api_material": L.API_MATERIAL_NAME,
                 "box_x_m": [float(xs[lo]), float(xs[hi])], "api_grid_shape": [int(s) for s in api_grid.shape]})
    return mats, info


def run_rfx_arm(params: dict, path: str, *, nx_interior: int, n_steps_cap: int, smoke: bool,
                dx_div: int = 1, recipe: str = G.RECIPE_R3) -> dict:
    from rfx.core.yee import update_e
    holder = {}

    def setup(rig):
        materials, info = build_materials(rig, params, path)
        holder["info"] = info

        def e_update(state, dstate, dt, dx, periodic):
            return update_e(state, materials, dt, dx, periodic), dstate

        return materials, None, e_update

    run = RIG.run_slab_arm(L.MODEL, params, setup=setup, nx_interior=nx_interior, n_steps_cap=n_steps_cap,
                           smoke=smoke, dx_div=dx_div, recipe=recipe)
    if not run.get("grow"):
        run["materials"] = holder["info"]
    return run


# =============================================================================
# Plot (diagnostic; PNGs are gitignored)
# =============================================================================

def _plot_arm(arm: str, e2: dict, e4: dict | None, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    f = np.asarray(e2["freqs_hz"]) / 1e9
    g = np.asarray(e2["gated"])
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    for k, (key, lab) in enumerate((("R", "Reflectance"), ("T", "Transmittance"), ("A", "Absorption"))):
        a = ax[0, k]
        a.plot(f, e2[f"{key}_tmm"], "k-", lw=2.2, label="TMM (eps' - j sigma/w eps0)")
        a.plot(f, e2[f"{key}_rfx"], "r--", lw=1.4, label="rfx")
        if e4 and e4.get("present"):
            a.plot(f, e4[f"{key}_meep"], "b:", lw=1.8, label="Meep")
        a.axvspan(f[g].min(), f[g].max(), color="0.9", zorder=0)
        a.set_title(lab); a.grid(True, alpha=0.3); a.legend(fontsize=8)
        b = ax[1, k]
        b.plot(f, e2[f"d{key}"], "r-", lw=1, label="|rfx - TMM|")
        b.plot(f, e2[f"window_{key}"], "r:", lw=1, label="G1 window")
        if e4 and e4.get("present"):
            b.plot(f, e4[f"d{key}_meep_tmm"], "b-", lw=1, label="|Meep - TMM|")
            b.plot(f, e4[f"d{key}_rfx_meep"], "g-", lw=1, label="|rfx - Meep|")
            b.plot(f, e4[f"window5_{key}"], "g:", lw=1, label="G5 window")
        b.set_yscale("log"); b.grid(True, alpha=0.3); b.legend(fontsize=7)
        b.set_xlabel("f (GHz)")
    fig.suptitle(f"cv23 {arm}: sigma = {e2['params']['sigma']:.4f} S/m, eps' = {e2['params']['eps_inf']:g}, "
                 f"d = 10 mm -- rfx vs TMM" + (" vs Meep" if e4 and e4.get("present") else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"23_{arm}.png"), dpi=130)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--falsifier", choices=sorted(list(L.FALSIFIERS) + list(L.MEEP_FALSIFIER_CASE_NAMES)),
                    default=None, help="apply a pre-declared defect; the run MUST then exit 1")
    ap.add_argument("--arms", default=None, help="comma-separated subset of tand0p1,tand1,tand3")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, few steps, no gates, exit 0 if finite")
    ap.add_argument("--out-dir", default=None, help="artifact directory (default: _23_lossy_results)")
    ap.add_argument("--meep-dir", default=None, help="where the Meep JSONs live (default: out-dir)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dx-div", type=int, default=None, choices=(1, 2, 4),
                    help="refine the rig in cells by K (dx/K, all cell counts x K). Default: the arm's "
                         "declared primary recipe (note section 13: tand3 at dx/2, the others at dx); an "
                         "explicit value is a diagnostic and requires --tag")
    ap.add_argument("--nx-interior", type=int, default=None,
                    help="interior cells at dx (default cv22's 1000)")
    ap.add_argument("--tag", default=None,
                    help="write rfx__<tag>.json instead of rfx.json (diagnostic arms; never the baseline)")
    ap.add_argument("--recipe", choices=(G.RECIPE_R3, G.RECIPE_CV04), default=G.RECIPE_R3,
                    help="r3 (default): derived record length, nx 1000, -40 dB witness; cv04: the 719-step rule")
    ap.add_argument("--refit-tail-fits", action="store_true",
                    help="no FDTD: recompute tail.fitted_rate_* / fit_reliable of every rfx*.json in out-dir from "
                         "the STORED envelopes (post-processing of a committed artifact; note section 14)")
    ap.add_argument("--meep-ladder-summary", action="store_true",
                    help="no FDTD: read rfx.json + meep_<arm>__res{10,20,40}.json in out-dir and write "
                         "meep_ladder_summary.json (measured Meep convergence order)")
    a = ap.parse_args(argv)
    if a.refit_tail_fits:
        import glob
        od = a.out_dir or RESULTS_DIR
        for path in sorted(glob.glob(os.path.join(od, "rfx*.json"))):
            with open(path) as fh:
                doc = json.load(fh)
            for arm, ad in doc.get("arms", {}).items():
                rec = (ad.get("run") or {}).get("record")
                if rec is None or "envelope_scat_refl_rel" not in ad.get("tail", {}):
                    continue
                before = (ad["tail"].get("fitted_rate_scat_refl_1_s"), ad["tail"].get("fitted_rate_blocks"))
                ad["tail"] = G.refit_tail(ad["tail"], ad["dt_s"], ad["run"]["n_steps"], rec["n_pulse_end"])
                nb = ad["tail"]["fitted_rate_blocks"]
                ad["tail"]["fit_note"] = (
                    f"fitted_rate_* recomputed from the stored envelope after the run ({nb} post-pulse blocks of "
                    f"{G.TAIL_WINDOW} steps; fit_reliable = {'true' if nb >= 3 else 'false: a 2-block fit is a two-point estimate'}); "
                    "no FDTD rerun (note section 15)")
                print(f"{os.path.basename(path)} {arm}: rate {before[0]} ({before[1]} blocks) -> "
                      f"{ad['tail']['fitted_rate_scat_refl_1_s']:.3e} ({ad['tail']['fitted_rate_blocks']} blocks, "
                      f"reliable={ad['tail']['fit_reliable']})")
            with open(path, "w") as fh:
                json.dump(doc, fh, indent=1)
        return 0
    if a.meep_ladder_summary:
        od = a.out_dir or RESULTS_DIR
        with open(os.path.join(od, "rfx.json")) as fh:
            summ = L.meep_ladder_summary(od, json.load(fh))
        with open(os.path.join(od, "meep_ladder_summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        for arm, v in summ["arms"].items():
            print(arm, {r: (round(x.get("mean_dR_meep_tmm_gated", float("nan")), 4),
                            round(x.get("mean_dT_meep_tmm_gated", float("nan")), 4)) for r, x in v["rungs"].items()},
                  {k: round(o, 2) for k, o in v["orders"].items()})
        print("wrote", os.path.join(od, "meep_ladder_summary.json"))
        return 0
    if (a.dx_div is not None or a.nx_interior not in (None, G.NX_INTERIOR_R3)) and not a.tag and not a.smoke:
        ap.error("--dx-div / --nx-interior arms are diagnostics and require --tag")

    out_dir = a.out_dir or (tempfile.mkdtemp(prefix="cv23_smoke_") if a.smoke else RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    meep_dir = a.meep_dir or (a.out_dir or RESULTS_DIR)

    arms = list(L.ARM_ORDER)
    if a.arms:
        arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    rfx_fals, meep_fals_key = None, None
    if a.falsifier in L.FALSIFIERS:
        rfx_fals = a.falsifier
        arms = [L.FALSIFIERS[rfx_fals][0]] if a.arms is None else arms
    elif a.falsifier in L.MEEP_FALSIFIER_CASE_NAMES:
        meep_fals_key = L.MEEP_FALSIFIER_CASE_NAMES[a.falsifier]
        arms = [L.MEEP_FALSIFIER_ARM] if a.arms is None else arms

    nx_default = G.NX_INTERIOR_R3 if a.recipe == G.RECIPE_R3 else G.NX_INTERIOR
    nx_interior = 200 if a.smoke else (a.nx_interior or nx_default)
    n_steps_cap = 300 if a.smoke else 8000

    print("=" * 70)
    print(f"Crossval 23: lossy slab -- arms {arms}; falsifier={a.falsifier}; smoke={a.smoke}")
    print("=" * 70)
    print(f"  windows: W_bin={L.W_BIN}, W_mean_R={L.W_MEAN_R}, W_mean_T={L.W_MEAN_T}, "
          f"W_bin_A={L.W_BIN_A:.3f}, W_mean_A={L.W_MEAN_A:.3f} (triangle; tighter derivable "
          f"{L.W_BIN_A_TIGHT}/{L.W_MEAN_A_TIGHT} reported as A_tight_ok); gated band "
          f"{G.BAND_GATED_HZ[0]/1e9:.0f}-{G.BAND_GATED_HZ[1]/1e9:.0f} GHz")

    doc = {
        "schema": L.SCHEMA, "case_id": L.CASE_ID, "commit": _git_commit(),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "falsifier": a.falsifier, "smoke": bool(a.smoke), "tag": a.tag, "dx_div": a.dx_div,
        "arm_dx_div": dict(L.ARM_DX_DIV),
        "recipe": a.recipe,
        "rig": {"dx_m": G.DX_M, "d_slab_m": G.D_SLAB_M, "nx_interior": nx_interior, "n_cpml": G.N_CPML,
                "tfsf_f0_hz": G.TFSF_F0_HZ, "tfsf_bw": G.TFSF_BW, "band_gated_hz": list(G.BAND_GATED_HZ),
                "eps_r_slab": L.EPS_R_SLAB, "f_centre_hz": L.F_CENTRE_HZ, "arm_tan_delta": L.ARM_TAN_DELTA,
                "W_bin": L.W_BIN, "W_mean_R": L.W_MEAN_R, "W_mean_T": L.W_MEAN_T,
                "W_bin_A": L.W_BIN_A, "W_mean_A": L.W_MEAN_A,
                "W_bin_A_tight": L.W_BIN_A_TIGHT, "W_mean_A_tight": L.W_MEAN_A_TIGHT,
                "cv04_envelope": dict(G.CV04_ENVELOPE, mean_closure=L.CV04_MEAN_CLOSURE)},
        "arms": {},
    }

    any_e2_fail = any_e4_fail = any_meep_missing = False
    for arm in arms:
        base = L.ARMS[arm]
        params = dict(base["params"])       # the DECLARED material = the oracle
        params_run = dict(params)           # what the FDTD is built with
        path = base["materials_path"]
        if rfx_fals is not None and L.FALSIFIERS[rfx_fals][0] == arm:
            _, params_run = L.apply_falsifier(rfx_fals)
            print(f"\n[{arm}] FALSIFIER {rfx_fals}: {L.FALSIFIERS[rfx_fals][1]} "
                  f"(FDTD built with the defect; judged against the DECLARED material)")
        print(f"\n[{arm}] tan delta @ {L.F_CENTRE_HZ/1e9:g} GHz = {L.ARM_TAN_DELTA[arm]:g}; "
              f"params_run={ {k: float(v) for k, v in params_run.items()} }; materials path: {path}; "
              f"recipe dx/{a.dx_div if a.dx_div is not None else L.ARM_DX_DIV[arm]}")
        dx_div = a.dx_div if a.dx_div is not None else L.ARM_DX_DIV[arm]
        nx_arm = nx_interior
        grows = []
        while True:
            run = run_rfx_arm(params_run, path, nx_interior=nx_arm, n_steps_cap=n_steps_cap, smoke=a.smoke,
                              dx_div=dx_div, recipe=a.recipe)
            if not run.get("grow"):
                break
            grows.append({"nx_interior": nx_arm, "t_safe": run["t_safe"],
                          "n_steps_reached": run.get("n_steps_reached"), "tail_at_gate": run.get("tail_at_gate")})
            print(f"  record: CPML gate {run['t_safe']} reached before the -40 dB witness "
                  f"(n_steps {run.get('n_steps_reached')}, tail {run.get('tail_at_gate')}); "
                  f"growing nx_interior {nx_arm} -> {nx_arm + G.NX_GROW_CELLS}")
            nx_arm += G.NX_GROW_CELLS
            if nx_arm > 4 * G.NX_INTERIOR_R3:
                raise RuntimeError("record never settled to -40 dB within 4x the declared box")
        run["record"] = None if run["record"] is None else dict(run["record"], nx_grows=grows)
        # The oracle is ALWAYS the declared material (cv22 note section 10.1).
        e2 = L.evaluate_e2(run["freqs_hz"], run["R_rfx"], run["T_rfx"], params, run["dt_s"], tail=run["tail"],
                           dx=run["dx_m"])
        e2["params_run"] = {k: float(v) for k, v in params_run.items()}
        e2["materials_path"] = path
        e2["materials"] = run["materials"]
        e2["band_inc_ok"] = run["band_inc_ok"]
        e2["inc_amp_rel"] = np.asarray(run["inc_amp_rel"]).tolist()
        e2["run"] = {k: run[k] for k in ("dt_s", "n_steps", "nfft", "nx_interior", "grid_shape", "elapsed_s",
                                         "dx_m", "dx_div", "n_cpml", "recipe", "record", "slab_cells")}
        if run["record"] is not None:
            r_ = run["record"]
            print(f"  record: n_pulse_end {r_['n_pulse_end']} + n_ring {r_['n_ring']} "
                  f"(etalon at {r_['f_ring_hz']/1e9:.2f} GHz, w {r_['w_ring']:.2f}, rho {r_['rho_etalon']:.3f}, "
                  f"rate {r_['rate_ring_1_s']:.3e}/s; no material pole) + window {r_['tail_window']} = n_steps_min "
                  f"{r_['n_steps_min']}; reached {r_['n_steps']} after {r_['extensions']} extension(s) "
                  f"(CPML gate {r_['t_safe_cpml_steps']}, box grows {len(r_.get('nx_grows', []))}); "
                  f"tail scat/trans {run['tail']['scat_refl_rel']:.2e}/{run['tail']['total_trans_rel']:.2e} "
                  f"vs {G.SETTLING_LIMIT:g} -> {'ok' if run['tail']['ok'] else 'FAIL'}; fitted tail rate "
                  f"scat/trans {run['tail']['fitted_rate_scat_refl_1_s']:.3e}/{run['tail']['fitted_rate_total_trans_1_s']:.3e} /s")
        if not run["band_inc_ok"]:
            e2["gates"]["rig_incident_floor"] = False
            e2["e2_ok"] = False
        print(f"  E2: max|dR|={e2['max_dR_gated']:.4f} max|dT|={e2['max_dT_gated']:.4f} "
              f"max|dA|={e2['max_dA_gated']:.4f} (worst A bin {e2['worst_bin_A_hz']/1e9:.2f} GHz) | "
              f"mean|dR|={e2['mean_dR_gated']:.4f}/{e2['mean_window_R']:.4f} "
              f"mean|dT|={e2['mean_dT_gated']:.4f}/{e2['mean_window_T']:.4f} "
              f"mean|dA|={e2['mean_dA_gated']:.4f}/{e2['mean_window_A']:.4f}; A_tight_ok={e2['A_tight_ok']}; "
              f"max R+T (masked) = {1 + e2['max_RT_closure_masked']:.4f}")
        print(f"  E2 gates: {e2['gates']} -> {'PASS' if e2['e2_ok'] else 'FAIL'}")
        lat = e2["lattice"]
        print(f"  lattice witness (reported, not gated): W_lat mean R/T/A {lat['mean_W_lat_R_gated']:.5f}/"
              f"{lat['mean_W_lat_T_gated']:.5f}/{lat['mean_W_lat_A_gated']:.5f}; |rfx - lattice| mean R/T/A "
              f"{lat['mean_dR_lattice_gated']:.2e}/{lat['mean_dT_lattice_gated']:.2e}/{lat['mean_dA_lattice_gated']:.2e} "
              f"(max R {lat['max_dR_lattice_gated']:.2e})")
        for fi in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
            i = int(np.argmin(np.abs(np.asarray(e2["freqs_hz"]) - fi * 1e9)))
            print(f"    {e2['freqs_hz'][i]/1e9:6.2f} GHz  R {e2['R_tmm'][i]:.4f}/{e2['R_rfx'][i]:.4f} | "
                  f"T {e2['T_tmm'][i]:.4f}/{e2['T_rfx'][i]:.4f} | A {e2['A_tmm'][i]:.4f}/{e2['A_rfx'][i]:.4f} "
                  f"(TMM/rfx) | W_sig R/T/A {e2['w_ade_R'][i]:.1e}/{e2['w_ade_T'][i]:.1e}/{e2['w_ade_A'][i]:.1e}")

        # --- E4: committed Meep JSON ---
        meep_name = L.meep_json_name(arm, meep_fals_key if (meep_fals_key and arm == L.MEEP_FALSIFIER_ARM) else None)
        meep_path = os.path.join(meep_dir, meep_name)
        if os.path.isfile(meep_path):
            with open(meep_path) as fh:
                mdoc = json.load(fh)
            mdoc["_source"] = os.path.relpath(meep_path, SCRIPT_DIR)
            e4 = L.evaluate_e4(e2, mdoc)
            print(f"  E4 ({meep_name}): Meep-vs-TMM mean R/T/A {e4['mean_dR_meep_tmm_gated']:.4f}/"
                  f"{e4['mean_dT_meep_tmm_gated']:.4f}/{e4['mean_dA_meep_tmm_gated']:.4f} "
                  f"(max {e4['max_dR_meep_tmm_gated']:.4f}/{e4['max_dT_meep_tmm_gated']:.4f}/{e4['max_dA_meep_tmm_gated']:.4f}); "
                  f"rfx-vs-Meep mean {e4['mean_dR_rfx_meep_gated']:.4f}/{e4['mean_dT_rfx_meep_gated']:.4f}/"
                  f"{e4['mean_dA_rfx_meep_gated']:.4f}")
            print(f"  E4 gates: {e4['gates']} -> {'PASS' if e4['e4_ok'] else 'FAIL'}")
        else:
            e4 = {"present": False, "expected_path": os.path.relpath(meep_path, SCRIPT_DIR)}
            print(f"  E4: [SKIP] Meep reference missing: {meep_path}")
        e2["meep"] = e4
        doc["arms"][arm] = e2

        if not a.smoke:
            any_e2_fail |= not e2["e2_ok"]
            if e4.get("present"):
                any_e4_fail |= not e4["e4_ok"]
            else:
                any_meep_missing = True
            if not a.no_plots:
                _plot_arm(arm, e2, e4, out_dir)

    if a.smoke:
        rc = 0
        summary = "SMOKE OK (rig executed, artifact written, gates NOT evaluated for verdict)"
    elif any_e2_fail:
        rc = 1
        summary = "rfx accuracy: FAIL -- E2 gate failed on at least one arm (exit 1)"
    elif any_meep_missing:
        rc = 2
        summary = "[SKIP] Meep reference missing for at least one arm -- inconclusive (exit 2)"
    elif any_e4_fail:
        rc = 1
        summary = "E4 FAIL -- Meep-vs-TMM or rfx-vs-Meep gate failed (exit 1)"
    else:
        rc = 0
        summary = "ALL CHECKS PASSED -- E2 (TMM) and E4 (Meep) on all arms, R, T and A (exit 0)"
    if a.falsifier is not None and not a.smoke:
        summary += f"  [falsifier {a.falsifier}: {'as pre-declared' if rc == 1 else 'NOT DETECTED -- gate does not resolve the defect'}]"
    elif a.falsifier is not None:
        summary += f"  [falsifier {a.falsifier}: smoke run, verdict not evaluated -- see the E2 gates line]"
    doc["verdict"] = {"rfx_self_ok": not any_e2_fail, "meep_present": not any_meep_missing,
                      "e4_ok": not any_e4_fail, "exit_code": rc, "summary": summary}
    out_path = os.path.join(out_dir, f"rfx__{a.tag}.json" if a.tag else L.rfx_json_name(a.falsifier))
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n  artifact: {out_path}")
    print(f"\n{summary}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
