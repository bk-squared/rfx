"""The cv04 TFSF slab rig as ONE shared FDTD runner (cv22, cv23).

``run_slab_arm`` is cv22's ``run_rfx_arm`` (``22_dispersive_slab_fresnel.py``,
rounds 1-4) with the slab MATERIAL factored out into a ``setup`` hook, so the
lossy-slab case (cv23) runs byte-identically the same rig -- grid, dt, TFSF,
probes, CPML, the derived record length, the adaptive -40 dB witness with box
growth, the stored tail envelope and its fit, the spectral analysis and mask
-- with a different slab. The hook receives the grid bookkeeping and returns
``(materials, dstate, e_update)``:

    materials : rfx MaterialArrays for the whole grid (eps_r / sigma set in the
                slab cells; used by update_h and by e_update)
    dstate    : the E-update's auxiliary state (ADE polarization; None if none)
    e_update  : (state, dstate, dt, dx, periodic) -> (state, dstate)

Importing this module does not import rfx or jax; ``run_slab_arm`` imports
them lazily, as cv22's runner did, so the comparator package stays
importable in a Meep-only environment.
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np

import cv22_dispersive_gates as G
import dispersive_eps as DE


def staged_commit(repo_root: str, cwd: str | None = None) -> str:
    """The source commit: ``.staged_commit`` first (a staged copy on a pod has
    no .git; the orchestrator writes it at staging time -- cv22 review
    finding 6), then ``git rev-parse HEAD``, else "unknown"."""
    staged = os.path.join(repo_root, ".staged_commit")
    if os.path.isfile(staged):
        with open(staged) as fh:
            val = fh.read().strip()
        if val:
            return val
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd or repo_root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_slab_arm(model: str, params: dict, *, setup, nx_interior: int, n_steps_cap: int,
                 smoke: bool, verbose: bool = True, dx_div: int = 1,
                 recipe: str = G.RECIPE_R3) -> dict:
    """One arm on the cv04 rig. ``dx_div = K`` refines the SAME rig in cells:
    dx/K with nx_interior, CPML layers, TFSF margin, probe offsets and tail
    window all x K (geometry identical; cv22 note section 11.2(a))."""
    import jax
    from rfx.grid import Grid
    from rfx.core.yee import init_state, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
    from rfx.sources.tfsf import (
        init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e, apply_tfsf_e, apply_tfsf_h,
    )

    K = int(dx_div)
    dx = G.DX_M / K
    n_cpml = G.N_CPML * K
    nx_interior = int(nx_interior) * K
    domain = (nx_interior * dx, 0.004, dx)
    grid = Grid(freq_max=20e9, domain=domain, dx=dx, cpml_layers=n_cpml, mode="2d_tmz")
    dt = float(grid.dt)
    periodic = (False, True, True)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx, dx, dt, cpml_layers=n_cpml, tfsf_margin=G.TFSF_MARGIN * K,
        f0=G.TFSF_F0_HZ, bandwidth=G.TFSF_BW, amplitude=1.0,
        polarization="ez", direction="+x", ny=grid.ny, nz=grid.nz,
    )
    x_lo, x_hi, i0 = tfsf_cfg.x_lo, tfsf_cfg.x_hi, tfsf_cfg.i0

    slab_lo_g = grid.nx // 2 - int(G.D_SLAB_M / (2 * dx))
    slab_hi_g = grid.nx // 2 + int(G.D_SLAB_M / (2 * dx))
    assert x_lo + 10 < slab_lo_g < slab_hi_g < x_hi - 10, \
        f"Slab [{slab_lo_g},{slab_hi_g}) must be inside TFSF [{x_lo},{x_hi}]"
    probe_refl_x = slab_lo_g - G.PROBE_OFFSET_CELLS * K
    probe_trans_x = slab_hi_g + G.PROBE_OFFSET_CELLS * K
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
        # cv22 sections 12/13: record length from the slab's own ring-down
        # (-40 dB), not cv04's CPML rule; the CPML gate must exceed the derived
        # minimum, and the record is extended adaptively below while the
        # witness is above the bar (never past the gate: then the caller grows
        # the box).
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

    # --- the slab material: the case's hook ---
    rig = {"grid": grid, "dt": dt, "dx": dx, "n_cpml": n_cpml, "domain": domain,
           "nx_interior_cells": nx_interior, "slab_lo": slab_lo_g, "slab_hi": slab_hi_g, "dx_div": K}
    materials, dstate, e_update = setup(rig)

    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)

    def step(state, cs, tfsf_st, dstate, t):
        state = update_h(state, materials, dt, dx, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

        state, dstate = e_update(state, dstate, dt, dx, periodic)
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
    # cv22 section 13 adaptive extension: while the -40 dB witness is not met
    # and the CPML gate allows, run RECORD_EXTEND_STEPS more (the arrays were
    # allocated to the gate). If the gate is reached with the witness still
    # above the bar, hand back to the caller to grow the box (cv04's rig rule;
    # never clip).
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
        raise RuntimeError("non-finite probe samples (update blow-up?)")
    if rec is not None:
        rec = dict(rec, n_steps_min=rec["n_steps"], n_steps=int(n_steps), extensions=int(extensions),
                   extend_steps=G.RECORD_EXTEND_STEPS, smoke_grow=smoke_grow)

    # --- cv04 tail witnesses (issue #341) at the reached record, plus the stored
    # envelope and its fitted decay rate (cv22 section 13) ---
    tail_inc_rel, tail_refl_rel, tail_trans_rel = purity, refl_rel, trans_rel
    tail_clean = bool(tail_inc_rel < G.TAIL_PURITY_LIMIT)
    n_env = min(G.TAIL_ENVELOPE_STEPS * K, n_steps)
    env_refl = (np.abs(ts_scat_refl[-n_env:]) / inc_peak)
    env_trans = (np.abs(ts_trans[-n_env:]) / inc_peak)
    tail = {
        "window_steps": tw, "purity_inc_rel": float(tail_inc_rel),
        "scat_refl_rel": float(tail_refl_rel), "total_trans_rel": float(tail_trans_rel),
        "purity_limit": G.TAIL_PURITY_LIMIT, "limit": tail_limit,
        "ok": bool(tail_clean and tail_refl_rel < tail_limit and tail_trans_rel < tail_limit),
        "envelope_steps": int(n_env),
        "envelope_scat_refl_rel": env_refl.tolist(), "envelope_total_trans_rel": env_trans.tolist(),
    }
    # The fit starts after the incident pulse (rec is None only for the cv04
    # recipe, where n_pulse_end is not derived: fit the whole envelope).
    n_pulse_end_fit = rec["n_pulse_end"] if rec is not None else (n_steps - n_env)
    tail = G.refit_tail(tail, dt, n_steps, n_pulse_end_fit)

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
        "smoke": smoke, "slab_cells": [int(slab_lo_g), int(slab_hi_g)],
        "time_domain": None if smoke else {
            "t_s": (np.arange(n_steps) * dt), "inc_trans": ts_inc_trans, "trans": ts_trans,
            "inc_refl": ts_inc_refl, "scat_refl": ts_scat_refl},
    }
