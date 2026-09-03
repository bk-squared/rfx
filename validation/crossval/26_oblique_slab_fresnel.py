"""Cross-validation 26: Oblique-incidence slab Fresnel R, T -- rfx (Bloch TFSF) vs analytic vs Meep

cv04's slab (eps = 4, 10 mm, dx = 1 mm, 2-D TMz) under the oblique Bloch
TFSF path (``rfx/sources/tfsf_2d.py``, ``method="bloch"``: the transverse
axis is Bloch-periodic at a FIXED k_y = k0(f0) sin theta0, the 3-D grid
carries the complex envelope, ``bloch_phase_tuple`` on the y roll). Every
bin is judged at its REALIZED angle theta(f) = asin(k_y c / 2 pi f) against
the Fresnel / transfer-matrix R, T at that angle (E2), and against Meep with
the same k_point (E4). TE is the injected polarization (E perpendicular to
the plane of incidence -- the only one either rfx oblique TFSF path
injects); TM is gated through the exact eps <-> mu duality (a mu_r = 4 slab
under the same ez injection), Brewster included. The grazing arms run a
compact box in which the absorber echo is INSIDE the record: a PEC (R = 1
exactly) measures the CPML's reflection at 80-85 deg against the exact
2-D Yee-lattice prediction of the discrete absorber.

Pre-declaration (arms, windows, records, falsifiers -- read it first):
  docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md
Numbers live in ONE place:
  validation/crossval/comparators/oblique_fresnel.py

Exit codes (rfx crossval convention):
  0 = every gate passes on every arm run AND the Meep leg is present and E4 passes on every Meep arm run
  1 = any gate failed (a falsifier arm MUST exit 1)
  2 = E2 passes but a Meep JSON is missing for a Meep arm -- inconclusive, NOT a pass

Run:
  python validation/crossval/26_oblique_slab_fresnel.py                      # all arms
  python validation/crossval/26_oblique_slab_fresnel.py --arm te_45
  python validation/crossval/26_oblique_slab_fresnel.py --falsifier te_45_angle_p5
  python validation/crossval/26_oblique_slab_fresnel.py --arm graze_pec --n-cpml 8 --tag graze_pec_d8
  python validation/crossval/26_oblique_slab_fresnel.py --smoke              # <= 20 s, no gates
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "comparators"))
import oblique_fresnel as O  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, O.RESULTS_DIRNAME)
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RECORD_CAP_FACTOR = 2.0     # compact box (no CPML gate): extend at most to 2 x n_steps_min
SMOKE_THETA0_DEG = 60.0     # --smoke: the compact grazing rig re-aimed at 60 deg (~2500 steps, a few seconds)


def staged_commit() -> str:
    """``.staged_commit`` first (a staged copy on a pod has no .git; the
    orchestrator writes it at staging time), then git, else unknown."""
    p = os.path.join(REPO_ROOT, ".staged_commit")
    if os.path.isfile(p):
        with open(p) as fh:
            v = fh.read().strip()
        if v:
            return v
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# =============================================================================
# The rig: cv04's PART 1 on the Bloch TFSF path
# =============================================================================

def run_rfx_arm(spec: dict, *, n_cpml: int = O.N_CPML, cpml_kwargs: dict | None = None,
                theta0_run_deg: float | None = None, eps_scale: float = 1.0,
                nx_interior: int | None = None, dx_div: int = 1, smoke: bool = False, verbose: bool = True) -> dict:
    """One arm. ``theta0_run_deg`` / ``eps_scale`` / ``n_cpml`` / ``cpml_kwargs``
    are the RUN-side knobs (falsifiers and the depth ladder); the oracle is
    always the declared one, applied by the caller. ``dx_div = K`` refines
    the SAME rig in cells (dx/K; interior, absorber depth, TFSF margin, probe
    offsets, tail window and extension x K; the aux grid's constants are
    tfsf_2d's and are not scaled)."""
    import jax
    import jax.numpy as jnp
    import rfx.boundaries.cpml as cpml_mod
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
    from rfx.sources.tfsf_2d import (init_tfsf_2d, update_tfsf_2d_h, update_tfsf_2d_e,
                                     apply_tfsf_2d_e, apply_tfsf_2d_h, bloch_phase_tuple)

    K = int(dx_div)
    dx = O.DX_M / K
    nx_int = int(spec["nx_interior"] if nx_interior is None else nx_interior)
    n_cpml_run = int(n_cpml) * K
    th_run = spec["theta0_deg"] if theta0_run_deg is None else float(theta0_run_deg)
    grid = Grid(freq_max=20e9, domain=(nx_int * K * dx, O.NY_CELLS * dx, dx), dx=dx, cpml_layers=n_cpml_run, mode="2d_tmz")
    dt = float(grid.dt)
    assert abs(dt - O.DT_S / K) < 1e-18, (dt, O.DT_S / K)
    periodic = (False, True, True)
    cells = O.rig_cells(nx_int, n_cpml, dx_div=K)
    assert cells["nx"] == grid.nx, (cells["nx"], grid.nx)

    cfg, aux = init_tfsf_2d(grid.nx, grid.ny, dx, dt, cpml_layers=n_cpml_run, tfsf_margin=O.TFSF_MARGIN * K,
                            f0=spec["f0_hz"], bandwidth=spec["bw"], amplitude=1.0, polarization="ez",
                            direction="+x", theta_deg=th_run)
    assert cfg.x_lo == cells["x_lo"] and cfg.x_hi == cells["x_hi"], "rig bookkeeping drift"
    assert cfg.i0_x - cfg.src_x == cells["aux_src_to_x_lo"]
    ky_run = -cfg.k_transverse * cfg.direction_sign      # tfsf_2d: k_transverse = -direction_sign k0 sin theta
    lo, hi = cells["slab_lo"], cells["slab_hi"]
    assert cells["x_lo"] + 10 < lo < hi < cells["x_hi"] - 10
    p_r, p_t = cells["probe_refl"], cells["probe_trans"]
    assert cells["x_lo"] < p_r and p_t < cells["x_hi"]
    yc = grid.ny // 2
    ref_r = cfg.i0_x + (p_r - cells["x_lo"])
    ref_t = cfg.i0_x + (p_t - cells["x_lo"])

    materials = init_materials(grid.shape)
    eps_s, mu_s = spec["eps_slab_rfx"] * (eps_scale if spec["pol"] == "te" else 1.0), \
        spec["mu_slab_rfx"] * (eps_scale if spec["pol"] == "tm" else 1.0)
    if spec["slab"]:
        materials = materials._replace(eps_r=materials.eps_r.at[lo:hi, :, :].set(float(eps_s)),
                                       mu_r=materials.mu_r.at[lo:hi, :, :].set(float(mu_s)))
    pec = bool(spec.get("pec", False))

    state = init_state(grid.shape, field_dtype=jnp.complex64)
    if cpml_kwargs:
        # falsifier F5b: the SAME profile builder with a different R_asymptotic
        orig = cpml_mod._cpml_profile
        cpml_mod._cpml_profile = functools.partial(orig, **cpml_kwargs)
        try:
            cp, cs = init_cpml(grid, field_dtype=jnp.complex64)
        finally:
            cpml_mod._cpml_profile = orig
    else:
        cp, cs = init_cpml(grid, field_dtype=jnp.complex64)
    bloch = bloch_phase_tuple(cfg, dx)

    def step(state, cs, aux, t):
        state = update_h(state, materials, dt, dx, periodic, bloch=bloch)
        state = apply_tfsf_2d_h(state, cfg, aux, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        aux = update_tfsf_2d_h(cfg, aux, dx, dt)
        state = update_e(state, materials, dt, dx, periodic, bloch=bloch)
        state = apply_tfsf_2d_e(state, cfg, aux, dx, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        if pec:
            state = state._replace(ez=state.ez.at[lo:hi, :, :].set(0.0))
        aux = update_tfsf_2d_e(cfg, aux, dx, dt, t)
        samples = jnp.stack([state.ez[p_r, yc, 0], state.ez[p_t, yc, 0], aux.ez_2d[ref_r, 0], aux.ez_2d[ref_t, 0]])
        return state, cs, aux, samples

    step_fn = jax.jit(step)

    rec = O.derive_record(spec, dt, n_cpml=n_cpml, nx_interior=nx_int, dx_div=K)
    if theta0_run_deg is not None:
        rec["theta0_run_deg"] = th_run
    n_steps = rec["n_steps"]
    # Round 2 (note section 13): the record is the DERIVED settling step of the
    # exact lattice and the cap is RECORD_CAP_FACTOR x it for every arm.  Round 1
    # capped the wide rig at the ARRIVAL of the first absorber echo
    # (``t_safe_cpml_steps``, still reported) and grew the box when the witness
    # had not settled by then; that echo is the echo of the FASTEST gated
    # component, whose CPML reflection is ~1e-10, and growing the box makes the
    # settle grow faster than the cap.  The echo is now gated by its AMPLITUDE
    # over the record (``rec['e_absorber']``), not by its arrival time.
    t_cap = int(RECORD_CAP_FACTOR * n_steps)
    extend = O.RECORD_EXTEND_STEPS * K
    if smoke and not spec["compact"]:
        n_steps = min(n_steps, 1200)
        t_cap = n_steps + 2 * extend
    n_alloc = max(t_cap, n_steps)
    if verbose:
        print(f"  grid {grid.shape} dt={dt:.4e} s; theta0 run {th_run:g} deg (k_y {ky_run:.3f} rad/m); "
              f"bw {spec['bw']}; slab cells [{lo},{hi}) eps/mu {eps_s:g}/{mu_s:g} pec={pec}; probes {p_r}/{p_t}; "
              f"n_cpml {n_cpml}; record n_pulse_end {rec['n_pulse_end']} + n_echo {rec['n_echo']} + n_ring {rec['n_ring']} "
              f"+ {rec['tail_window']} = {rec['n_steps']} (cap {t_cap}; theta gated {rec['theta_gate_lo_deg']:.1f}-"
              f"{rec['theta_gate_hi_deg']:.1f} deg)")

    ts = np.zeros((4, n_alloc), complex)
    tw = O.TAIL_WINDOW * K
    n_done = 0
    t0_wall = time.time()

    def _advance(upto):
        nonlocal state, cs, aux, n_done
        for n in range(n_done, upto):
            state, cs, aux, s = step_fn(state, cs, aux, n * dt)
            ts[:, n] = np.asarray(s)
        n_done = upto

    def _witness(n):
        tot_r, tot_t, inc_r, inc_t = ts[:, :n]
        inc_peak = max(np.max(np.abs(inc_r)), np.max(np.abs(inc_t)))
        scat = tot_r - inc_r
        purity = max(np.max(np.abs(inc_r[-tw:])), np.max(np.abs(inc_t[-tw:]))) / inc_peak
        refl = np.max(np.abs(scat[-tw:])) / inc_peak
        trans = np.max(np.abs(tot_t[-tw:])) / inc_peak
        return float(purity), float(refl), float(trans), float(inc_peak), scat

    _advance(n_steps)
    purity, refl_rel, trans_rel, inc_peak, scat = _witness(n_steps)
    extensions = 0
    cap_hit = False
    if not smoke:
        while max(refl_rel, trans_rel) >= O.SETTLING_LIMIT or purity >= O.TAIL_PURITY_LIMIT:
            if n_steps + extend > t_cap:
                cap_hit = True
                break
            n_steps += extend
            extensions += 1
            _advance(n_steps)
            purity, refl_rel, trans_rel, inc_peak, scat = _witness(n_steps)
    elapsed = time.time() - t0_wall
    ts = ts[:, :n_steps]
    if not np.all(np.isfinite(ts)):
        raise RuntimeError("non-finite probe samples")
    tot_r, tot_t, inc_r, inc_t = ts
    rec = dict(rec, n_steps_min=rec["n_steps"], n_steps=int(n_steps), extensions=int(extensions),
               extend_steps=extend, cap_steps=int(t_cap), cap_reached=bool(cap_hit))
    tail = {"window_steps": tw, "purity_inc_rel": purity, "scat_refl_rel": refl_rel, "total_trans_rel": trans_rel,
            "purity_limit": O.TAIL_PURITY_LIMIT, "limit": O.SETTLING_LIMIT,
            "ok": bool(purity < O.TAIL_PURITY_LIMIT and refl_rel < O.SETTLING_LIMIT and trans_rel < O.SETTLING_LIMIT
                       and not cap_hit)}

    # --- spectra: the envelope carries exp(-j 2 pi f0 t); the +j kernel puts
    # the carrier at +f0, so DFT the conjugate and read the positive bins ---
    nfft = int(2 ** math.ceil(math.log2(n_steps)) * O.NFFT_OVERSAMPLE)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    npos = freqs.size

    def _spec(x):
        return np.fft.fft(np.conj(x), n=nfft)[:npos]
    S_inc_t, S_inc_r, S_tot_t, S_scat = _spec(inc_t), _spec(inc_r), _spec(tot_t), _spec(scat)
    inc_amp = np.abs(S_inc_t)
    mask = (freqs > O.MASK_F_LO_HZ) & (freqs < O.MASK_F_HI_HZ) & (inc_amp > inc_amp.max() * O.MASK_AMP_FRAC)
    if not mask.any():
        raise RuntimeError("empty spectral mask")
    f = freqs[mask]
    R = np.abs(S_scat[mask]) ** 2 / np.abs(S_inc_r[mask]) ** 2
    T = np.abs(S_tot_t[mask]) ** 2 / np.abs(S_inc_t[mask]) ** 2
    inc_rel = inc_amp[mask] / inc_amp.max()
    g = O.gated_mask(f, spec)
    if verbose:
        print(f"  FDTD {elapsed:.1f}s, {n_steps} steps ({extensions} ext); masked {f[0]/1e9:.2f}-{f[-1]/1e9:.2f} GHz "
              f"({mask.sum()} bins), gated {int(g.sum())}; tail purity {purity:.1e} scat {refl_rel:.2e} trans {trans_rel:.2e} "
              f"-> {'ok' if tail['ok'] else 'FAIL'}")
    n_env = min(300 * K, n_steps)
    return {"dt_s": dt, "n_steps": int(n_steps), "nfft": int(nfft), "nx_interior": nx_int, "n_cpml": int(n_cpml),
            "dx_m": dx, "dx_div": K, "n_cpml_run": n_cpml_run,
            "cpml_kwargs": cpml_kwargs, "grid_shape": [int(s) for s in grid.shape], "elapsed_s": elapsed,
            "record": rec, "cells": cells, "theta0_run_deg": th_run, "ky_run_rad_m": float(ky_run),
            "eps_slab_run": float(eps_s), "mu_slab_run": float(mu_s), "pec": pec,
            "freqs_hz": f, "R_rfx": R, "T_rfx": T, "inc_amp_rel": inc_rel, "gated": g, "tail": dict(
                tail, envelope_steps=n_env, envelope_scat_refl_rel=(np.abs(scat[-n_env:]) / inc_peak).tolist(),
                envelope_total_trans_rel=(np.abs(tot_t[-n_env:]) / inc_peak).tolist()),
            "smoke": smoke}


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
    th = np.asarray(e2["theta_deg"])
    g = np.asarray(e2["gated"], bool)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, key in enumerate(("R", "T")):
        a = ax[0, k]
        a.plot(th, e2[f"{key}_an"], "k-", lw=2, label="Fresnel at theta(f)")
        a.plot(th, e2[f"{key}_rfx"], "r--", lw=1.3, label="rfx")
        if "lattice" in e2:
            a.plot(th, e2["lattice"][f"{key}_lattice"], "c:", lw=1, label="exact lattice")
        if e4 and e4.get("present"):
            a.plot(th, e4[f"{key}_meep"], "b:", lw=1.6, label="Meep")
        if g.any():
            a.axvspan(th[g].min(), th[g].max(), color="0.9", zorder=0)
        a.set_title(f"{key} ({arm})"); a.grid(True, alpha=0.3); a.legend(fontsize=8); a.set_xlabel("realized theta (deg)")
        b = ax[1, k]
        b.plot(th, e2[f"d{key}"], "r-", lw=1, label="|rfx - Fresnel|")
        b.plot(th, e2[f"window_{key}"], "r:", lw=1, label="G1 window")
        if e4 and e4.get("present"):
            b.plot(th, e4[f"d{key}_meep_tmm"], "b-", lw=1, label="|Meep - Fresnel|")
            b.plot(th, e4[f"d{key}_rfx_meep"], "g-", lw=1, label="|rfx - Meep|")
        b.set_yscale("log"); b.grid(True, alpha=0.3); b.legend(fontsize=7); b.set_xlabel("realized theta (deg)")
    fig.suptitle(f"cv26 {arm}: theta0 {e2['theta0_deg']:g} deg, {e2['pol'].upper()}, bw {e2['bw']}, f0 10 GHz")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"26_{arm}.png"), dpi=120)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def _serial(d):
    if isinstance(d, dict):
        return {k: _serial(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_serial(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.floating, np.integer, np.bool_)):
        return d.item()
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, help="comma-separated subset of " + ",".join(O.ARM_ORDER + O.GRAZE_ARMS))
    ap.add_argument("--falsifier", default=None,
                    choices=sorted(list(O.FALSIFIERS) + list(O.MEEP_FALSIFIER_CASE_NAMES)),
                    help="apply a pre-declared defect; the run MUST then exit 1 (except the declared not-to-fail ones)")
    ap.add_argument("--smoke", action="store_true", help="tiny box, few steps, no gates, exit 0 if finite")
    ap.add_argument("--out-dir", default=None, help=f"artifact directory (default: {O.RESULTS_DIRNAME})")
    ap.add_argument("--meep-dir", default=None, help="where the Meep JSONs live (default: out-dir)")
    ap.add_argument("--n-cpml", type=int, default=None, help="absorber depth (ladder rung); requires --tag")
    ap.add_argument("--dx-div", type=int, default=None, choices=(1, 2),
                    help="refine the rig in cells by K (default: the arm's declared primary recipe, ARM_DX_DIV; "
                         "an explicit value is a diagnostic rung and requires --tag)")
    ap.add_argument("--tag", default=None, help="write rfx__<tag>.json instead of rfx.json (diagnostic arms)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    if (a.n_cpml is not None or a.dx_div is not None) and not a.tag and not a.smoke:
        ap.error("--n-cpml / --dx-div are diagnostics and require --tag")

    out_dir = a.out_dir or (tempfile.mkdtemp(prefix="cv26_smoke_") if a.smoke else RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    meep_dir = a.meep_dir or (a.out_dir or RESULTS_DIR)

    arms = list(O.ARM_ORDER + O.GRAZE_ARMS)
    if a.arm:
        arms = [s.strip() for s in a.arm.split(",") if s.strip()]
    rfx_fals = meep_fals_key = None
    if a.falsifier in O.FALSIFIERS:
        rfx_fals = a.falsifier
        arms = [O.FALSIFIERS[rfx_fals][0]] if a.arm is None else arms
    elif a.falsifier in O.MEEP_FALSIFIER_CASE_NAMES:
        meep_fals_key = O.MEEP_FALSIFIER_CASE_NAMES[a.falsifier]
        arms = [O.MEEP_FALSIFIER_ARM] if a.arm is None else arms
    if a.smoke and a.arm is None:
        arms = ["graze_pec"]      # the compact box at SMOKE_THETA0_DEG: whole gate pipeline in a few seconds

    print("=" * 70)
    print(f"Crossval 26: oblique slab Fresnel -- arms {arms}; falsifier={a.falsifier}; smoke={a.smoke}; tag={a.tag}")
    print("=" * 70)
    print(f"  windows: W_bin={O.W_BIN}, W_mean_R={O.W_MEAN_R}, W_mean_T={O.W_MEAN_T} (+ W_disp(f), W_inj(f)); "
          f"leak bar {O.LEAK_BAR:g}; grazing gate PML_REL {O.PML_REL} + floor {O.PML_FLOOR_R:.2e}")

    doc = {"schema": O.SCHEMA, "case_id": O.CASE_ID, "commit": staged_commit(),
           "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "falsifier": a.falsifier, "smoke": bool(a.smoke), "tag": a.tag, "n_cpml_override": a.n_cpml,
           "dx_div_override": a.dx_div, "arm_dx_div": dict(O.ARM_DX_DIV),
           "rig": {"dx_m": O.DX_M, "d_slab_m": O.D_SLAB_M, "eps_r_slab": O.EPS_R_SLAB, "n_cpml": O.N_CPML,
                   "nx_interior": O.NX_INTERIOR, "nx_interior_graze": O.NX_INTERIOR_GRAZE, "f0_hz": O.TFSF_F0_HZ,
                   "arm_bw": O.ARM_BW, "graze_bw": O.GRAZE_BW, "graze_theta0_deg": O.GRAZE_THETA0_DEG,
                   "theta_gate_max_deg": O.THETA_GATE_MAX_DEG, "W_bin": O.W_BIN, "W_mean_R": O.W_MEAN_R,
                   "W_mean_T": O.W_MEAN_T, "leak_bar": O.LEAK_BAR, "pml_rel": O.PML_REL, "pml_floor_R": O.PML_FLOOR_R,
                   "cv04_envelope": O.CV04_ENVELOPE, "tfsf_path": "rfx.sources.tfsf_2d (bloch), polarization ez"},
           "arms": {}}

    any_fail = any_meep_missing = False
    for arm in arms:
        spec = O.arm_spec(arm)
        if a.smoke and spec["compact"]:
            th = SMOKE_THETA0_DEG
            spec = dict(spec, theta0_deg=th, bw=O.bandwidth_for(th), ky=O.ky_from(O.TFSF_F0_HZ, th),
                        f_cutoff_hz=O.cutoff_hz(O.ky_from(O.TFSF_F0_HZ, th)), theta_gate_deg=(0.0, O.THETA_GATE_MAX_DEG))
        run_kw = {"n_cpml": O.N_CPML if a.n_cpml is None else a.n_cpml}
        or_kw = {}
        if rfx_fals is not None and O.FALSIFIERS[rfx_fals][0] == arm:
            _, desc, run_def, or_def = O.FALSIFIERS[rfx_fals]
            print(f"\n[{arm}] FALSIFIER {rfx_fals}: {desc} (FDTD built with the defect; judged against the DECLARED oracle)")
            run_kw.update({k: v for k, v in run_def.items()})
            or_kw.update(or_def)
        run_kw = {("theta0_run_deg" if k == "theta0_deg" else k): v for k, v in run_kw.items()}
        dx_div = a.dx_div if a.dx_div is not None else O.ARM_DX_DIV.get(arm, 1)
        if a.smoke:
            dx_div = 1
        print(f"\n[{arm}] theta0 {spec['theta0_deg']:g} deg, pol {spec['pol']}, bw {spec['bw']}, "
              f"f_cutoff {spec['f_cutoff_hz']/1e9:.3f} GHz, k_y {spec['ky']:.3f} rad/m, box {spec['nx_interior']}, "
              f"recipe dx/{dx_div}")
        run = run_rfx_arm(spec, smoke=a.smoke, nx_interior=spec["nx_interior"], dx_div=dx_div, **run_kw)
        cells = run["cells"]
        e2 = O.evaluate_e2(run["freqs_hz"], run["R_rfx"], run["T_rfx"], spec, run["dt_s"], tail=run["tail"],
                           cells=cells, n_cpml=run["n_cpml"], **or_kw)
        e2["run"] = {k: run[k] for k in ("dt_s", "n_steps", "nfft", "nx_interior", "n_cpml", "n_cpml_run", "dx_m", "dx_div",
                                         "cpml_kwargs", "grid_shape", "elapsed_s", "record", "theta0_run_deg",
                                         "ky_run_rad_m", "eps_slab_run", "mu_slab_run", "pec")}
        e2["inc_amp_rel"] = np.asarray(run["inc_amp_rel"]).tolist()
        e2["falsifier"] = rfx_fals if (rfx_fals and O.FALSIFIERS[rfx_fals][0] == arm) else None
        r_ = run["record"]
        # the absorber term is GATED on the arm's declared primary recipe only; on a
        # diagnostic rung (a different dx or absorber depth) it is a measurement, and
        # the measurement is exactly why the oblique arms run at dx/2 (note section 13.4)
        # ... and never on the COMPACT box, where the echo is INSIDE the record by
        # design (note section 4.5) and the arm is read against the lattice WITH the
        # absorber, not against Fresnel.
        primary = ((dx_div == O.ARM_DX_DIV.get(arm, 1)) and (a.n_cpml is None)
                   and not a.smoke and not spec["compact"])
        print(f"  record: derived {r_['n_steps_min']} -> {r_['n_steps']} ({r_['extensions']} ext, cap {r_['cap_steps']}, "
              f"source {r_['record_source']}, closed form {r_['n_closed_form']}, theta_eff {r_['theta_eff_deg']:.1f} deg); "
              f"absorber echo over the record {r_['e_absorber']:.2e} -> W_abs R/T "
              f"{r_['W_absorber_R_max']:.3f}/{r_['W_absorber_T_max']:.3f} vs W_bin {O.W_BIN:g} -> "
              f"{'ok' if r_['absorber_ok'] else ('n/a' if not np.isfinite(r_['e_absorber']) else 'OVER')}"
              f"{'' if primary else ' (reported, not gated: diagnostic rung)'}; "
              f"tail scat/trans {run['tail']['scat_refl_rel']:.2e}/{run['tail']['total_trans_rel']:.2e} "
              f"vs {O.SETTLING_LIMIT:g} -> {'ok' if run['tail']['ok'] else 'FAIL'}")
        print(f"  E2 ({e2['n_bins_gated']} bins, theta {e2['theta_gated_deg'][0]:.1f}-{e2['theta_gated_deg'][1]:.1f} deg): "
              f"max|dR|={e2['max_dR_gated']:.4f} max|dT|={e2['max_dT_gated']:.4f} | mean|dR|={e2['mean_dR_gated']:.4f}/"
              f"{e2['mean_window_R']:.4f} mean|dT|={e2['mean_dT_gated']:.4f}/{e2['mean_window_T']:.4f}; "
              f"max closure {e2['max_closure_gated']:.4f}")
        lat = e2["lattice"]
        print(f"  lattice witness (reported): |rfx - lattice| mean R/T {lat['mean_dR_lattice_gated']:.2e}/"
              f"{lat['mean_dT_lattice_gated']:.2e} (max R {lat['max_dR_lattice_gated']:.2e}); W_lat mean R/T "
              f"{lat['mean_W_lat_R_gated']:.4f}/{lat['mean_W_lat_T_gated']:.4f}; absorber term max R "
              f"{lat['absorber_term_R_gated_max']:.2e}")
        arm_ok = e2["e2_ok"] and (bool(r_["absorber_ok"]) or not primary)
        gates_line = dict(e2["gates"], G3_absorber=(bool(r_["absorber_ok"]) if primary else "reported"))
        if spec["slab"] and not spec["compact"]:
            th_ = np.asarray(e2["theta_deg"]); g_ = np.asarray(e2["gated"], bool)
            for tq in (0.0, 30.0, 45.0, 55.0, 60.0, 63.43, 65.0, 70.0):
                if g_.any() and th_[g_].min() - 0.5 <= tq <= th_[g_].max() + 0.5:
                    i = int(np.argmin(np.abs(np.where(g_, th_, 1e9) - tq)))
                    print(f"    theta {th_[i]:6.2f} ({e2['freqs_hz'][i]/1e9:5.2f} GHz)  R {e2['R_an'][i]:.4f}/{e2['R_rfx'][i]:.4f}"
                          f"  T {e2['T_an'][i]:.4f}/{e2['T_rfx'][i]:.4f} (Fresnel/rfx)  W_disp R/T "
                          f"{e2['w_disp_R'][i]:.1e}/{e2['w_disp_T'][i]:.1e}  lattice R {lat['R_lattice'][i]:.4f}")
        if arm == O.BREWSTER_ARM:
            bw_ = O.evaluate_brewster(e2)
            e2["brewster"] = bw_
            gates_line["G_brewster"] = bw_["ok"]
            arm_ok = arm_ok and bw_["ok"]
            print(f"  Brewster: bin {bw_['bin_hz']/1e9:.3f} GHz (theta {bw_['theta_bin_deg']:.2f} vs {bw_['theta_brewster_deg']:.2f}): "
                  f"R_rfx {bw_['R_rfx_at_brewster']:.4f} vs floor {bw_['floor']:.4f} -> {'ok' if bw_['ok'] else 'FAIL'}; "
                  f"measured minimum at {bw_['theta_of_measured_min_deg']:.2f} deg (R {bw_['R_measured_min']:.4f})")
        if arm == "te_00":
            # F2 control: at normal incidence the TE and TM oracles coincide, so the swapped reference must PASS
            e2_sw = O.evaluate_e2(run["freqs_hz"], run["R_rfx"], run["T_rfx"], spec, run["dt_s"], tail=run["tail"], oracle_pol="tm")
            e2["swap_ref_at_normal"] = {"e2_ok": e2_sw["e2_ok"], "mean_dR_gated": e2_sw["mean_dR_gated"], "mean_dT_gated": e2_sw["mean_dT_gated"]}
            print(f"  F2 control (TM oracle on the theta0 = 0 arm): {'PASS' if e2_sw['e2_ok'] else 'FAIL'} "
                  f"(mean|dR| {e2_sw['mean_dR_gated']:.4f})")
        if arm == "graze_vac":
            lk = O.evaluate_leakage(run["freqs_hz"], run["R_rfx"], spec)
            e2["leak"] = lk
            gates_line["G_leak"] = lk["G_leak"]
            arm_ok = lk["G_leak"] and e2["gates"]["G3_tail"]      # the vacuum arm has no R/T oracle to pass
            e2["e2_ok"] = arm_ok
            print(f"  leakage witness: max |scat/inc| gated {lk['max_leak_gated']:.2e} vs {O.LEAK_BAR:g} -> {'ok' if lk['G_leak'] else 'FAIL'}")
        if arm == "graze_pec":
            decl_kw = None
            pg = O.evaluate_grazing_pec(run["freqs_hz"], run["R_rfx"], spec, run["dt_s"],
                                        O.rig_cells(spec["nx_interior"], O.N_CPML, dx_div=run["dx_div"]),
                                        n_cpml=run["n_cpml"], declared_cpml_kwargs=decl_kw)
            e2["grazing_pec"] = pg
            gates_line["G6_absorber"] = pg["G6_absorber"]
            arm_ok = pg["G6_absorber"] and e2["gates"]["G3_tail"] and e2["gates"]["G3_passivity"]
            e2["e2_ok"] = arm_ok
            print(f"  G6 absorber ({pg['n_bins_gated']} bins, theta {pg['theta_gated_deg']}): max|R - R_lat| {pg['max_abs_dev_gated']:.2e}; "
                  f"measured excess |R-1| max {pg['max_excess_meas_band']:.3e} vs a-priori absorber term max "
                  f"{pg['max_absorber_term_band']:.3e} -> {'ok' if pg['G6_absorber'] else 'FAIL'}")
        if arm == "graze_te":
            gs = O.evaluate_grazing_slab(run["freqs_hz"], run["R_rfx"], run["T_rfx"], spec, run["dt_s"], cells, n_cpml=run["n_cpml"])
            e2["grazing_slab"] = gs
            gates_line.update({"G7_R": gs["G7_R"], "G7_T": gs["G7_T"]})
            arm_ok = gs["G7_R"] and gs["G7_T"] and e2["gates"]["G3_tail"] and e2["gates"]["G3_passivity"]
            e2["e2_ok"] = arm_ok
            print(f"  G7 slab vs lattice-with-absorber: max|dR| {gs['max_dR_lattice_gated']:.3e} max|dT| {gs['max_dT_lattice_gated']:.3e}; "
                  f"excess over Fresnel max {gs['max_excess_over_fresnel_R_gated']:.3e} vs a-priori absorber term "
                  f"{gs['max_absorber_term_R_gated']:.3e} -> {'ok' if gs['G7_R'] and gs['G7_T'] else 'FAIL'}")
        e2["gates_all"] = gates_line
        print(f"  gates: {gates_line} -> {'PASS' if arm_ok else 'FAIL'}")

        # --- E4 ---
        e4 = {"present": False}
        if arm in O.MEEP_ARMS:
            fk = meep_fals_key if (meep_fals_key and arm == O.MEEP_FALSIFIER_ARM) else None
            meep_name = O.meep_json_name(arm, fk)
            meep_path = os.path.join(meep_dir, meep_name)
            mdoc = None
            if os.path.isfile(meep_path):
                with open(meep_path) as fh:
                    mdoc = json.load(fh)
                mdoc["_source"] = os.path.relpath(meep_path, SCRIPT_DIR)
            # Round 2 (note section 14): an ABSENT reference and a REJECTED one are the
            # same verdict -- "reference unavailable" -- and neither may be read as a
            # number.  Round 1's leg wrote R = -inf / T = +inf and the E4 gate FAILED
            # on infinities, which reads as "rfx disagrees with Meep"; it does not.
            unavailable = O.meep_unavailable_reason(mdoc, meep_path, SCRIPT_DIR)
            if unavailable is None:
                e4 = O.evaluate_e4(e2, mdoc)
                print(f"  E4 ({meep_name}, {e4['resolution']} px/cm, k_point {e4['k_point']}): Meep-vs-Fresnel mean R/T "
                      f"{e4['mean_dR_meep_tmm_gated']:.4f}/{e4['mean_dT_meep_tmm_gated']:.4f} (max {e4['max_dR_meep_tmm_gated']:.4f}/"
                      f"{e4['max_dT_meep_tmm_gated']:.4f}); rfx-vs-Meep mean {e4['mean_dR_rfx_meep_gated']:.4f}/"
                      f"{e4['mean_dT_rfx_meep_gated']:.4f}; gates {e4['gates']} -> {'PASS' if e4['e4_ok'] else 'FAIL'}")
            else:
                e4 = {"present": False, "expected_path": os.path.relpath(meep_path, SCRIPT_DIR),
                      "unavailable_reason": unavailable,
                      "rejection_reasons": list((mdoc or {}).get("rejection_reasons", []))}
                print(f"  E4: [SKIP] reference unavailable -- {unavailable}")
        e2["meep"] = e4
        doc["arms"][arm] = _serial(e2)

        if not a.smoke:
            any_fail |= not arm_ok
            if arm in O.MEEP_ARMS:
                if e4.get("present"):
                    any_fail |= not e4["e4_ok"]
                else:
                    any_meep_missing = True
            if not a.no_plots:
                _plot_arm(arm, e2, e4, out_dir)

    if a.smoke:
        rc, summary = 0, "SMOKE OK (rig executed, artifact written, gates NOT evaluated for verdict)"
    elif any_fail:
        rc, summary = 1, "rfx accuracy: FAIL -- a gate failed on at least one arm (exit 1)"
    elif any_meep_missing:
        rc, summary = 2, ("[SKIP] the Meep reference is UNAVAILABLE (absent, or written and rejected by the leg's "
                          "own acceptance) for at least one Meep arm -- inconclusive, NOT a disagreement (exit 2)")
    else:
        rc, summary = 0, "ALL CHECKS PASSED -- E2 (Fresnel at the realized angle), lattice-gated grazing arms and E4 (Meep) (exit 0)"
    if a.falsifier is not None and not a.smoke:
        summary += f"  [falsifier {a.falsifier}: {'exit 1 as pre-declared' if rc == 1 else 'exit ' + str(rc) + ' -- read the note'}]"
    doc["verdict"] = {"rfx_self_ok": not any_fail, "meep_present": not any_meep_missing, "exit_code": rc, "summary": summary}
    out_path = os.path.join(out_dir, f"rfx__{a.tag}.json" if a.tag else O.rfx_json_name(a.falsifier))
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n  artifact: {out_path}")
    print(f"\n{summary}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
