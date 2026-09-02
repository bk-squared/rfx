"""Cross-validation 22: Dispersive slab Fresnel R(f), T(f) -- rfx vs analytic TMM vs Meep

Debye / Lorentz / Drude single-pole slabs on cv04's TFSF rig
(``04_multilayer_fresnel.py``, PART 1, byte-identical except that the slab
E-update is ``update_e_debye`` / ``update_e_lorentz`` with a slab-only mask).

Pre-declaration (windows, falsifiers, band, arms -- read it first):
  docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md
Numbers live in ONE place:
  validation/crossval/comparators/cv22_dispersive_gates.py

Gates (per arm):
  E2  G1 per-bin |R_rfx - R_TMM| <= W_bin + W_ADE(f) (T likewise), 4-10 GHz
      G2 band-mean <= W_mean + mean W_ADE
      G3 cv04 tail witnesses + passivity R+T <= 1 + CONS_MAX_LIMIT
  E4  G4 Meep vs TMM (reference soundness), G5 rfx vs Meep, same window
      algebra, using the committed Meep JSON produced by
      scripts/crossval/meep_cv22_dispersive_slab.py.

Exit codes (rfx crossval convention):
  0 = E2 pass on all arms AND the Meep leg present and E4 pass on all arms
  1 = any gate failed (a falsifier arm MUST exit 1)
  2 = E2 pass but a Meep JSON is missing -- inconclusive, NOT a pass

Run:
  python validation/crossval/22_dispersive_slab_fresnel.py            # all arms
  python validation/crossval/22_dispersive_slab_fresnel.py --falsifier debye_tau_x2
  python validation/crossval/22_dispersive_slab_fresnel.py --smoke     # <= 20 s, no gates
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "comparators"))
import cv22_dispersive_gates as G  # noqa: E402
import dispersive_eps as DE  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "_22_dispersive_results")
SCHEMA = "cv22-dispersive-slab/v1"
CASE_ID = "22_dispersive_slab_fresnel"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# =============================================================================
# rfx arm (cv04 PART 1 with the dispersive E-update)
# =============================================================================

def run_rfx_arm(model: str, params: dict, *, nx_interior: int, n_steps_cap: int,
                smoke: bool, verbose: bool = True, dx_div: int = 1,
                recipe: str = G.RECIPE_R3) -> dict:
    """One rfx arm. ``dx_div = K`` refines the SAME rig in cells: dx/K with
    nx_interior, CPML layers, TFSF margin, probe offsets and tail window all
    x K (geometry identical; pre-declaration section 11.2(a))."""
    import jax
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
    from rfx.sources.tfsf import (
        init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e, apply_tfsf_e, apply_tfsf_h,
    )
    from rfx.materials.debye import DebyePole, init_debye, update_e_debye
    from rfx.materials.lorentz import (
        drude_pole, lorentz_pole, init_lorentz, update_e_lorentz,
    )

    K = int(dx_div)
    dx = G.DX_M / K
    n_cpml = G.N_CPML * K
    nx_interior = int(nx_interior) * K
    grid = Grid(freq_max=20e9, domain=(nx_interior * dx, 0.004, dx),
                dx=dx, cpml_layers=n_cpml, mode="2d_tmz")
    dt = float(grid.dt)
    periodic = (False, True, True)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, dx, dt, cpml_layers=n_cpml, tfsf_margin=5 * K,
        f0=G.TFSF_F0_HZ, bandwidth=G.TFSF_BW, amplitude=1.0,
        polarization="ez", direction="+x", ny=grid.ny, nz=grid.nz,
    )
    x_lo, x_hi, i0 = tfsf_cfg.x_lo, tfsf_cfg.x_hi, tfsf_cfg.i0

    slab_lo_g = grid.nx // 2 - int(G.D_SLAB_M / (2 * dx))
    slab_hi_g = grid.nx // 2 + int(G.D_SLAB_M / (2 * dx))
    assert x_lo + 10 < slab_lo_g < slab_hi_g < x_hi - 10, \
        f"Slab [{slab_lo_g},{slab_hi_g}) must be inside TFSF [{x_lo},{x_hi}]"
    probe_refl_x = slab_lo_g - 30 * K
    probe_trans_x = slab_hi_g + 30 * K
    assert x_lo < probe_refl_x and probe_trans_x < x_hi, "probes must sit in the TF region"
    probe_refl = (probe_refl_x, grid.ny // 2, 0)
    probe_trans = (probe_trans_x, grid.ny // 2, 0)
    ref_1d_refl = i0 + (probe_refl_x - x_lo)
    ref_1d_trans = i0 + (probe_trans_x - x_lo)

    v_cells = DE.C0 * dt / dx
    dist_to_cpml_hi = grid.nx - n_cpml - probe_trans_x
    dist_to_cpml_lo = probe_refl_x - n_cpml
    t_safe_hi = int(2 * dist_to_cpml_hi / v_cells * 0.95)
    t_safe_lo = int(2 * dist_to_cpml_lo / v_cells * 0.95)
    # Smoke ignores the CPML time gate (it only proves the rig executes).
    rec = None
    t_safe = min(t_safe_hi, t_safe_lo)
    if smoke:
        # Smoke exercises the adaptive path too: the derived minimum on this
        # tiny box, a fake gate two extensions away, the witness necessarily
        # failing (CPML contamination) -> two extensions -> "grow", which smoke
        # reports instead of acting on.
        rec = G.derive_record_length(model, params, dt, nx_interior=nx_interior // K, dx_div=K)
        n_steps = max(n_steps_cap, rec["n_steps"])
        t_safe = n_steps + 2 * G.RECORD_EXTEND_STEPS
    elif recipe == G.RECIPE_R3:
        # §12/§13: record length from the slab's own ring-down (-40 dB), not
        # cv04's CPML rule; the CPML gate must exceed the derived minimum, and
        # the record is extended adaptively below while the witness is above
        # the bar (never past the gate: then the caller grows the box).
        rec = G.derive_record_length(model, params, dt, nx_interior=nx_interior // K, dx_div=K)
        assert rec["probe_trans"] == probe_trans_x and rec["x_lo"] == x_lo, "rig bookkeeping drift"
        n_steps = rec["n_steps"]
        if n_steps > t_safe:
            return {"grow": True, "record": rec, "t_safe": t_safe}
    else:
        n_steps = min(t_safe, n_steps_cap * K)
    tail_limit = G.SETTLING_LIMIT if recipe == G.RECIPE_R3 else G.TAIL_LIMIT
    n_alloc = t_safe if (rec is not None) else n_steps

    if verbose:
        print(f"  grid {grid.shape}, dt={dt:.4e} s, slab cells [{slab_lo_g},{slab_hi_g}), "
              f"probes {probe_refl_x}/{probe_trans_x}, n_steps={n_steps} (recipe {recipe})")

    # --- materials: eps_inf in the slab; the pole lives only in the slab ---
    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=materials.eps_r.at[slab_lo_g:slab_hi_g, :, :].set(float(params["eps_inf"])))
    slab_mask = np.zeros(grid.shape, dtype=bool)
    slab_mask[slab_lo_g:slab_hi_g, :, :] = True
    args = DE.rfx_pole_args(model, params)
    if model == "debye":
        coeffs, dstate = init_debye([DebyePole(**args)], materials, dt, mask=slab_mask)
        update_e_disp = update_e_debye
    else:
        pole = lorentz_pole(**args) if model == "lorentz" else drude_pole(**args)
        coeffs, dstate = init_lorentz([pole], materials, dt, mask=slab_mask)
        update_e_disp = update_e_lorentz

    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)

    def step(state, cs, tfsf_st, dstate, t):
        state = update_h(state, materials, dt, dx, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

        state, dstate = update_e_disp(state, coeffs, dstate, dt, dx, periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)
        samples = (state.ez[probe_refl], state.ez[probe_trans],
                   tfsf_st.e1d[ref_1d_refl], tfsf_st.e1d[ref_1d_trans])
        return state, cs, tfsf_st, dstate, samples

    step_fn = jax.jit(step)

    ts = np.zeros((4, n_alloc))
    tw = G.TAIL_WINDOW * K
    t0_wall = time.time()
    n_done = 0
    extensions = 0

    def _advance(upto):
        nonlocal state, cs, tfsf_st, dstate, n_done
        for n in range(n_done, upto):
            state, cs, tfsf_st, dstate, samples = step_fn(state, cs, tfsf_st, dstate, n * dt)
            ts[:, n] = [float(s) for s in samples]
        n_done = upto

    def _witness(n):
        ts_refl, ts_trans, ts_inc_refl, ts_inc_trans = ts[:, :n]
        inc_peak = max(np.max(np.abs(ts_inc_refl)), np.max(np.abs(ts_inc_trans)))
        scat = ts_refl - ts_inc_refl
        purity = max(np.max(np.abs(ts_inc_refl[-tw:])), np.max(np.abs(ts_inc_trans[-tw:]))) / inc_peak
        refl = np.max(np.abs(scat[-tw:])) / inc_peak
        trans = np.max(np.abs(ts_trans[-tw:])) / inc_peak
        return float(purity), float(refl), float(trans), inc_peak, scat

    _advance(n_steps)
    purity, refl_rel, trans_rel, inc_peak, ts_scat_refl = _witness(n_steps)
    # §13 adaptive extension: while the -40 dB witness is not met and the CPML
    # gate allows, run RECORD_EXTEND_STEPS more (the arrays were allocated to
    # the gate). If the gate is reached with the witness still above the bar,
    # hand back to the caller to grow the box (cv04's rig rule; never clip).
    smoke_grow = False
    if rec is not None:
        while (max(refl_rel, trans_rel) >= tail_limit or purity >= G.TAIL_PURITY_LIMIT):
            if n_steps + G.RECORD_EXTEND_STEPS > t_safe:
                if smoke:
                    smoke_grow = True
                    break
                return {"grow": True, "record": rec, "t_safe": t_safe, "n_steps_reached": n_steps,
                        "tail_at_gate": [refl_rel, trans_rel]}
            n_steps += G.RECORD_EXTEND_STEPS
            extensions += 1
            _advance(n_steps)
            purity, refl_rel, trans_rel, inc_peak, ts_scat_refl = _witness(n_steps)
    elapsed = time.time() - t0_wall
    ts = ts[:, :n_steps]
    ts_refl, ts_trans, ts_inc_refl, ts_inc_trans = ts
    if not np.all(np.isfinite(ts)):
        raise RuntimeError("non-finite probe samples (ADE blow-up?)")
    if rec is not None:
        rec = dict(rec, n_steps_min=rec["n_steps"], n_steps=int(n_steps), extensions=int(extensions),
                   extend_steps=G.RECORD_EXTEND_STEPS, smoke_grow=smoke_grow)

    # --- cv04 tail witnesses (issue #341) at the reached record, plus the stored
    # envelope and its fitted decay rate (§13) ---
    tail_inc_rel, tail_refl_rel, tail_trans_rel = purity, refl_rel, trans_rel
    tail_clean = bool(tail_inc_rel < G.TAIL_PURITY_LIMIT)
    n_env = min(G.TAIL_ENVELOPE_STEPS * K, n_steps)
    env_refl = (np.abs(ts_scat_refl[-n_env:]) / inc_peak)
    env_trans = (np.abs(ts_trans[-n_env:]) / inc_peak)
    rate_refl, nb_r = G.fit_tail_rate(env_refl, dt)
    rate_trans, nb_t = G.fit_tail_rate(env_trans, dt)
    tail = {
        "window_steps": tw, "purity_inc_rel": float(tail_inc_rel),
        "scat_refl_rel": float(tail_refl_rel), "total_trans_rel": float(tail_trans_rel),
        "purity_limit": G.TAIL_PURITY_LIMIT, "limit": tail_limit,
        "ok": bool(tail_clean and tail_refl_rel < tail_limit and tail_trans_rel < tail_limit),
        "envelope_steps": int(n_env),
        "envelope_scat_refl_rel": env_refl.tolist(), "envelope_total_trans_rel": env_trans.tolist(),
        "fitted_rate_scat_refl_1_s": rate_refl, "fitted_rate_total_trans_1_s": rate_trans,
        "fitted_rate_blocks": int(min(nb_r, nb_t)),
    }

    # --- cv04 spectral analysis, unchanged ---
    nfft = int(2 ** np.ceil(np.log2(n_steps)) * G.NFFT_OVERSAMPLE)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    S_inc_t = np.fft.rfft(ts_inc_trans, n=nfft)
    S_inc_r = np.fft.rfft(ts_inc_refl, n=nfft)
    S_tot_t = np.fft.rfft(ts_trans, n=nfft)
    S_scat_r = np.fft.rfft(ts_scat_refl, n=nfft)
    inc_amp = np.abs(S_inc_t)
    mask = (freqs > G.MASK_F_LO_HZ) & (freqs < G.MASK_F_HI_HZ) & (inc_amp > inc_amp.max() * G.MASK_AMP_FRAC)
    T_rfx = np.abs(S_tot_t[mask]) ** 2 / np.abs(S_inc_t[mask]) ** 2
    R_rfx = np.abs(S_scat_r[mask]) ** 2 / np.abs(S_inc_r[mask]) ** 2
    f_masked = freqs[mask]
    if f_masked.size == 0:
        raise RuntimeError("empty spectral mask: the pulse did not reach the probes (n_steps too small)")
    gated = G.gated_mask(f_masked)
    inc_amp_rel = inc_amp[mask] / inc_amp.max()
    band_inc_ok = bool(gated.any() and np.all(inc_amp_rel[gated] >= G.GATED_BAND_MIN_INC_AMP_FRAC))

    if verbose:
        print(f"  FDTD {elapsed:.1f}s; masked band {f_masked[0]/1e9:.2f}-{f_masked[-1]/1e9:.2f} GHz "
              f"({mask.sum()} bins), gated {int(gated.sum())} bins; tail ok={tail['ok']}; "
              f"incident >= {G.GATED_BAND_MIN_INC_AMP_FRAC:.0%} over gated band: {band_inc_ok}")

    return {
        "dt_s": dt, "n_steps": int(n_steps), "nfft": int(nfft), "nx_interior": int(nx_interior),
        "dx_m": dx, "dx_div": K, "n_cpml": n_cpml, "recipe": recipe, "record": rec,
        "grid_shape": [int(s) for s in grid.shape], "elapsed_s": elapsed,
        "freqs_hz": f_masked, "R_rfx": R_rfx, "T_rfx": T_rfx, "gated": gated,
        "inc_amp_rel": inc_amp_rel, "band_inc_ok": band_inc_ok, "tail": tail,
        "smoke": smoke,
        "time_domain": None if smoke else {
            "t_s": (np.arange(n_steps) * dt), "inc_trans": ts_inc_trans, "trans": ts_trans,
            "inc_refl": ts_inc_refl, "scat_refl": ts_scat_refl},
    }


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
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, (key, lab) in enumerate((("R", "Reflectance"), ("T", "Transmittance"))):
        a = ax[0, k]
        a.plot(f, e2[f"{key}_tmm"], "k-", lw=2.2, label="TMM (continuous eps)")
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
    fig.suptitle(f"cv22 {arm}: {e2['model']} slab d=10 mm -- rfx vs TMM"
                 + (" vs Meep" if e4 and e4.get("present") else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"22_{arm}.png"), dpi=130)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--falsifier", choices=sorted(list(G.FALSIFIERS) + list(G.MEEP_FALSIFIER_CASE_NAMES)),
                    default=None, help="apply a pre-declared defect; the run MUST then exit 1")
    ap.add_argument("--arms", default=None, help="comma-separated subset of debye,lorentz,drude")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, few steps, no gates, exit 0 if finite")
    ap.add_argument("--out-dir", default=None, help="artifact directory (default: _22_dispersive_results)")
    ap.add_argument("--meep-dir", default=None, help="where the Meep JSONs live (default: out-dir)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dx-div", type=int, default=1, choices=(1, 2, 4),
                    help="refine the rig in cells by K (dx/K, all cell counts x K); diagnostic, see note 11.2(a)")
    ap.add_argument("--nx-interior", type=int, default=None,
                    help="interior cells at dx (default cv04's 600; 1500 opens the time gate, note 11.2(a'))")
    ap.add_argument("--tag", default=None,
                    help="write rfx__<tag>.json instead of rfx.json (diagnostic arms; never the baseline)")
    ap.add_argument("--recipe", choices=(G.RECIPE_R3, G.RECIPE_CV04), default=G.RECIPE_R3,
                    help="r3 (default): record length derived from the slab ring-down, nx 1000, "
                         "-40 dB settling witness; cv04: the 719-step CPML rule (rounds 1-2, truncated)")
    ap.add_argument("--meep-ladder-summary", action="store_true",
                    help="no FDTD: read rfx.json + meep_<arm>__res{10,20,40}.json in out-dir and write "
                         "meep_ladder_summary.json (measured Meep convergence order)")
    a = ap.parse_args(argv)
    if a.meep_ladder_summary:
        od = a.out_dir or RESULTS_DIR
        with open(os.path.join(od, "rfx.json")) as fh:
            summ = G.meep_ladder_summary(od, json.load(fh))
        with open(os.path.join(od, "meep_ladder_summary.json"), "w") as fh:
            json.dump(summ, fh, indent=1)
        for arm, v in summ["arms"].items():
            print(arm, {r: (round(x.get("mean_dR_meep_tmm_gated", float("nan")), 4),
                            round(x.get("mean_dT_meep_tmm_gated", float("nan")), 4)) for r, x in v["rungs"].items()},
                  {k: round(o, 2) for k, o in v["orders"].items()})
        print("wrote", os.path.join(od, "meep_ladder_summary.json"))
        return 0
    if (a.dx_div != 1 or a.nx_interior not in (None, G.NX_INTERIOR)) and not a.tag and not a.smoke:
        ap.error("--dx-div / --nx-interior arms are diagnostics and require --tag")

    out_dir = a.out_dir or (tempfile.mkdtemp(prefix="cv22_smoke_") if a.smoke else RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    meep_dir = a.meep_dir or (a.out_dir or RESULTS_DIR)

    # Which arms, which params.
    arms = list(G.ARM_ORDER)
    if a.arms:
        arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    rfx_fals, meep_fals_key = None, None
    if a.falsifier in G.FALSIFIERS:
        rfx_fals = a.falsifier
        arms = [G.FALSIFIERS[rfx_fals][0]] if a.arms is None else arms
    elif a.falsifier in G.MEEP_FALSIFIER_CASE_NAMES:
        meep_fals_key = G.MEEP_FALSIFIER_CASE_NAMES[a.falsifier]
        arms = [G.MEEP_FALSIFIER_ARM] if a.arms is None else arms

    nx_default = G.NX_INTERIOR_R3 if a.recipe == G.RECIPE_R3 else G.NX_INTERIOR
    nx_interior = 200 if a.smoke else (a.nx_interior or nx_default)
    n_steps_cap = 300 if a.smoke else 8000

    print("=" * 70)
    print(f"Crossval 22: dispersive slab -- arms {arms}; falsifier={a.falsifier}; smoke={a.smoke}")
    print("=" * 70)
    print(f"  windows: W_bin={G.W_BIN}, W_mean_R={G.W_MEAN_R}, W_mean_T={G.W_MEAN_T} "
          f"(cv04 envelope x {G.gate_from_envelope(1.0, quantum=1000):g}); gated band "
          f"{G.BAND_GATED_HZ[0]/1e9:.0f}-{G.BAND_GATED_HZ[1]/1e9:.0f} GHz")

    doc = {
        "schema": SCHEMA, "case_id": CASE_ID, "commit": _git_commit(),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "falsifier": a.falsifier, "smoke": bool(a.smoke), "tag": a.tag, "dx_div": a.dx_div,
        "recipe": a.recipe,
        "rig": {"dx_m": G.DX_M, "d_slab_m": G.D_SLAB_M, "nx_interior": nx_interior, "n_cpml": G.N_CPML,
                "tfsf_f0_hz": G.TFSF_F0_HZ, "tfsf_bw": G.TFSF_BW, "band_gated_hz": list(G.BAND_GATED_HZ),
                "W_bin": G.W_BIN, "W_mean_R": G.W_MEAN_R, "W_mean_T": G.W_MEAN_T,
                "cv04_envelope": G.CV04_ENVELOPE},
        "arms": {},
    }

    any_e2_fail = any_e4_fail = any_meep_missing = False
    for arm in arms:
        base = G.ARMS[arm]
        model, params = base["model"], dict(base["params"])   # the DECLARED material = the oracle
        params_run = dict(params)                              # what the FDTD is built with
        if rfx_fals is not None and G.FALSIFIERS[rfx_fals][0] == arm:
            _, model, params_run = G.apply_falsifier(rfx_fals)
            print(f"\n[{arm}] FALSIFIER {rfx_fals}: {G.FALSIFIERS[rfx_fals][1]} "
                  f"(FDTD built with the defect; judged against the DECLARED material)")
        print(f"\n[{arm}] model={model} params_run={ {k: float(v) for k, v in params_run.items()} }")
        nx_arm = nx_interior
        grows = []
        while True:
            run = run_rfx_arm(model, params_run, nx_interior=nx_arm, n_steps_cap=n_steps_cap, smoke=a.smoke,
                              dx_div=a.dx_div, recipe=a.recipe)
            if not run.get("grow"):
                break
            # §13: the CPML gate was reached before the witness: grow the box
            # (cv04's rig rule) and rerun; never clip the record.
            grows.append({"nx_interior": nx_arm, "t_safe": run["t_safe"],
                          "n_steps_reached": run.get("n_steps_reached"), "tail_at_gate": run.get("tail_at_gate")})
            print(f"  record: CPML gate {run['t_safe']} reached before the -40 dB witness "
                  f"(n_steps {run.get('n_steps_reached')}, tail {run.get('tail_at_gate')}); "
                  f"growing nx_interior {nx_arm} -> {nx_arm + G.NX_GROW_CELLS}")
            nx_arm += G.NX_GROW_CELLS
            if nx_arm > 4 * G.NX_INTERIOR_R3:
                raise RuntimeError("record never settled to -40 dB within 4x the declared box")
        run["record"] = None if run["record"] is None else dict(run["record"], nx_grows=grows)
        # The oracle is ALWAYS the declared material; a falsifier that were
        # judged against its own defective eps(f) would be self-consistent
        # and pass (caught in review before the first run).
        e2 = G.evaluate_e2(run["freqs_hz"], run["R_rfx"], run["T_rfx"], model, params, run["dt_s"],
                           tail=run["tail"])
        e2["params_run"] = {k: float(v) for k, v in params_run.items()}
        e2["band_inc_ok"] = run["band_inc_ok"]
        e2["inc_amp_rel"] = np.asarray(run["inc_amp_rel"]).tolist()
        e2["run"] = {k: run[k] for k in ("dt_s", "n_steps", "nfft", "nx_interior", "grid_shape", "elapsed_s",
                                         "dx_m", "dx_div", "n_cpml", "recipe", "record")}
        if run["record"] is not None:
            r_ = run["record"]
            print(f"  record (r3/§13): n_pulse_end {r_['n_pulse_end']} + n_ring {r_['n_ring']} "
                  f"(ring component at {r_['f_ring_hz']/1e9:.2f} GHz, w {r_['w_ring']:.2f}, rate {r_['rate_ring_1_s']:.3e}/s; "
                  f"material {r_['rate_material_1_s']:.3e}, slowest etalon {r_['rate_etalon_slowest_1_s']:.3e} "
                  f"at {r_['f_etalon_slowest_hz']/1e9:.2f} GHz) + window {r_['tail_window']} = n_steps_min "
                  f"{r_['n_steps_min']}; reached {r_['n_steps']} after {r_['extensions']} extension(s) "
                  f"(CPML gate {r_['t_safe_cpml_steps']}, box grows {len(r_.get('nx_grows', []))}); "
                  f"tail scat/trans {run['tail']['scat_refl_rel']:.2e}/{run['tail']['total_trans_rel']:.2e} "
                  f"vs {G.SETTLING_LIMIT:g} -> {'ok' if run['tail']['ok'] else 'FAIL'}; fitted tail rate "
                  f"scat/trans {run['tail']['fitted_rate_scat_refl_1_s']:.3e}/{run['tail']['fitted_rate_total_trans_1_s']:.3e} /s")
        if not run["band_inc_ok"]:
            e2["gates"]["rig_incident_floor"] = False
            e2["e2_ok"] = False
        print(f"  E2: max|dR|={e2['max_dR_gated']:.4f} (worst {e2['worst_bin_R_hz']/1e9:.2f} GHz) "
              f"max|dT|={e2['max_dT_gated']:.4f} (worst {e2['worst_bin_T_hz']/1e9:.2f} GHz) | "
              f"mean|dR|={e2['mean_dR_gated']:.4f}/{e2['mean_window_R']:.4f} "
              f"mean|dT|={e2['mean_dT_gated']:.4f}/{e2['mean_window_T']:.4f}")
        print(f"  E2 gates: {e2['gates']} -> {'PASS' if e2['e2_ok'] else 'FAIL'}")
        for fi in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
            i = int(np.argmin(np.abs(np.asarray(e2["freqs_hz"]) - fi * 1e9)))
            print(f"    {e2['freqs_hz'][i]/1e9:6.2f} GHz  R_tmm={e2['R_tmm'][i]:.4f} R_rfx={e2['R_rfx'][i]:.4f} | "
                  f"T_tmm={e2['T_tmm'][i]:.4f} T_rfx={e2['T_rfx'][i]:.4f} | W_ADE R/T "
                  f"{e2['w_ade_R'][i]:.1e}/{e2['w_ade_T'][i]:.1e}")

        # --- E4: committed Meep JSON ---
        meep_name = G.meep_json_name(arm, meep_fals_key if (meep_fals_key and arm == G.MEEP_FALSIFIER_ARM) else None)
        meep_path = os.path.join(meep_dir, meep_name)
        e4 = None
        if os.path.isfile(meep_path):
            with open(meep_path) as fh:
                mdoc = json.load(fh)
            mdoc["_source"] = os.path.relpath(meep_path, SCRIPT_DIR)
            e4 = G.evaluate_e4(e2, mdoc)
            print(f"  E4 ({meep_name}): Meep-vs-TMM max|dR|={e4['max_dR_meep_tmm_gated']:.4f} "
                  f"max|dT|={e4['max_dT_meep_tmm_gated']:.4f} mean {e4['mean_dR_meep_tmm_gated']:.4f}/"
                  f"{e4['mean_dT_meep_tmm_gated']:.4f}; rfx-vs-Meep max|dR|={e4['max_dR_rfx_meep_gated']:.4f} "
                  f"max|dT|={e4['max_dT_rfx_meep_gated']:.4f} mean {e4['mean_dR_rfx_meep_gated']:.4f}/"
                  f"{e4['mean_dT_rfx_meep_gated']:.4f}")
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
        summary = "ALL CHECKS PASSED -- E2 (TMM) and E4 (Meep) on all arms (exit 0)"
    if a.falsifier is not None and not a.smoke:
        summary += f"  [falsifier {a.falsifier}: {'as pre-declared' if rc == 1 else 'NOT DETECTED -- gate does not resolve the defect'}]"
    elif a.falsifier is not None:
        summary += f"  [falsifier {a.falsifier}: smoke run, verdict not evaluated -- see the E2 gates line]"
    doc["verdict"] = {"rfx_self_ok": not any_e2_fail, "meep_present": not any_meep_missing,
                      "e4_ok": not any_e4_fail, "exit_code": rc, "summary": summary}
    out_path = os.path.join(out_dir, f"rfx__{a.tag}.json" if a.tag else G.rfx_json_name(a.falsifier))
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n  artifact: {out_path}")
    print(f"\n{summary}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
