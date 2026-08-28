"""Issue #764 falsifier battery: whole-port driven-diagonal normalization.

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/issue764_wireport_norm_predeclaration.md

NU lane only (POST-injection ordering, measured correct per #683). Every
gate below is evaluated verbatim from the pre-declaration; nothing here may
widen an envelope. A miss is reported as a miss with the residual mechanism
named.

Run (from THIS worktree — PYTHONPATH required so python resolves the
worktree rfx, not the venv's installed main-repo copy):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/issue764_wireport_norm_falsifiers.py

  # F10 state dump (run under BOTH the branch and main, then compare):
  ... issue764_wireport_norm_falsifiers.py --f10-dump /path/state.npz

  # FIX-C thru reported prediction (NOT a kill gate):
  ... issue764_wireport_norm_falsifiers.py --fixc
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

import rfx
from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

EPS_0 = 8.8541878128e-12

# ------------------------------------------------------------- bindings ----
FREQS = np.array([0.2e9, 0.5e9, 1.0e9, 2.0e9])   # in-band set, all four
LOW = 0                                          # lowest in-band bin
TOP = 3                                          # band top
QS = (0, 1)                                      # quasi-static bins (F5, F8)
LE1GHZ = (0, 1, 2)                               # bins <= 1 GHz (F1)
Z0 = 50.0
PULSE = GaussianPulse(f0=2e9, bandwidth=0.9)

# FIX-A geometry (metres)
DX_A = 0.5e-3
DOMAIN_A = (10e-3, 10e-3, 8e-3)
NZ_A = 16
N_STEPS_A = 3000
DRV_XY = (5.0e-3, 5.0e-3)
GAP_Z0, GAP_Z1 = 3.5e-3, 4.5e-3       # live gap edges
EXTENT_A = 0.5e-3                      # -> 2 live cells (k=7,8)
PLATE_XY0, PLATE_XY1 = 3.5e-3, 6.5e-3
LOAD_XY = [(4.5e-3, 5.0e-3), (5.5e-3, 5.0e-3),
           (5.0e-3, 4.5e-3), (5.0e-3, 5.5e-3)]
RL_SWEEP = [12.5, 25.0, 50.0, 100.0, 200.0]

# FIX-A' refinement arm (F6)
DX_AP = 0.25e-3
NZ_AP = 32
N_STEPS_AP = 6000
EXTENT_AP = 0.75e-3                    # same physical column -> 4 live cells


def build_fix_a(load: str | float, *, dx=DX_A, nz=NZ_A, extent=EXTENT_A):
    """FIX-A / FIX-A': clean gap-attached load between two 3x3 mm plates.

    load: 'open', 'short', or Z_L in ohms (four symmetric 4*Z_L columns).
    """
    sim = Simulation(freq_max=10e9, domain=DOMAIN_A, dx=dx,
                     dz_profile=np.full(nz, dx), boundary="pec")
    # Electrode plates: one FIX-A cell thick, abutting the gap ends.
    sim.add(Box((PLATE_XY0, PLATE_XY0, 3.0e-3),
                (PLATE_XY1, PLATE_XY1, GAP_Z0)), material="pec")
    sim.add(Box((PLATE_XY0, PLATE_XY0, GAP_Z1),
                (PLATE_XY1, PLATE_XY1, 5.0e-3)), material="pec")
    sim.add_port(position=(DRV_XY[0], DRV_XY[1], GAP_Z0), component="ez",
                 impedance=Z0, extent=extent, excite=True, waveform=PULSE)
    if load == "short":
        for (x, y) in LOAD_XY:
            sim.add(Box((x, y, GAP_Z0), (x + DX_A, y + DX_A, GAP_Z1)),
                    material="pec")
    elif load == "open":
        pass
    else:
        z_col = 4.0 * float(load)
        for (x, y) in LOAD_XY:
            sim.add_port(position=(x, y, GAP_Z0), component="ez",
                         impedance=z_col, extent=extent, excite=False)
    return sim


def run_nu(sim, n_steps, extra_sample_cells=None):
    """Run on the NU lane; capture grid/materials/sources/wire specs/result.

    extra_sample_cells: optional list of (i, j, k) — appended as
    sampling-only wire-port spec entries (no sigma, no source; DFT
    accumulators only) pinned to those cells, for the F9 per-cell current
    profile.  The field trajectory is untouched.
    """
    import rfx.runners.nonuniform as nur
    cap = {}
    orig = nur.run_nonuniform

    def spy(grid, materials, n_steps_, **kwargs):
        wps = kwargs.get("wire_ports")
        if extra_sample_cells and wps:
            drv = next(w for w in wps if w["excite"])
            for cell in extra_sample_cells:
                w = dict(drv)
                w["mid_i"], w["mid_j"], w["mid_k"] = cell
                w["excite"] = False
                w["live_cells"] = (tuple(cell),)
                w["n_live"] = 1
                wps = list(wps) + [w]
            kwargs["wire_ports"] = wps
        r = orig(grid, materials, n_steps_, **kwargs)
        cap.update(grid=grid, materials=materials,
                   sources=kwargs.get("sources"),
                   wire_ports=kwargs.get("wire_ports"), r=r)
        return r

    nur.run_nonuniform = spy
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sim.run(n_steps=n_steps, compute_s_params=True,
                    s_param_freqs=jnp.asarray(FREQS))
    finally:
        nur.run_nonuniform = orig
    cap["warnings"] = [f"{w.category.__name__}: {w.message}" for w in wlist]
    return cap


def driven_acc(cap):
    """(v, i, vinc, vp) complex128 arrays for the (single) excited port."""
    idx = [n for n, w in enumerate(cap["wire_ports"]) if w["excite"]]
    assert len(idx) == 1, idx
    raw = cap["r"]["wire_sparams_raw"][idx[0]]
    return idx[0], tuple(np.asarray(a, dtype=np.complex128) for a in raw)


def s11_driven(vp, i):
    return (vp - Z0 * i) / (vp + Z0 * i)


def a_wave(vp, i):
    return (vp + Z0 * i) / (2.0 * np.sqrt(Z0))


def vsrc_hat(cap, n_steps):
    """Whole-port Thevenin EMF V_src(w) = Z0 * What_cell(w).

    What_cell = rect-DFT (x dt) of the per-cell injected CURRENT at the
    driven mid cell.  The captured source table is in E-add units
    (cb/dV) * I_cell(t) (make_current_source), so de-normalize by dV/cb
    computed exactly as make_current_source does (concrete materials =
    pre-port-fold: vacuum at the port cells in these fixtures).
    """
    from rfx.nonuniform import e_node_dual_spacing_at
    grid = cap["grid"]
    drv = next(w for w in cap["wire_ports"] if w["excite"])
    mid = (drv["mid_i"], drv["mid_j"], drv["mid_k"])
    table = None
    for s in cap["sources"] or []:
        if (int(s[0]), int(s[1]), int(s[2])) == mid and s[3] == "ez":
            table = np.asarray(s[4], dtype=np.float64)
    assert table is not None, "no captured source at driven mid cell"
    dt = float(grid.dt)
    eps = 1.0 * EPS_0                     # vacuum port cell, sigma_conc = 0
    cb = dt / eps
    dxn = np.asarray(grid.dx_arr, dtype=np.float64)
    dyn = np.asarray(grid.dy_arr, dtype=np.float64)
    dzn = np.asarray(grid.dz, dtype=np.float64)
    dV = (float(e_node_dual_spacing_at(dxn, mid[0]))
          * float(e_node_dual_spacing_at(dyn, mid[1]))
          * float(dzn[mid[2]]))
    i_cell = table * (dV / cb)            # amperes, already / n_live
    n = min(n_steps, len(i_cell))
    t = np.arange(n) * dt
    what = (i_cell[None, :n]
            * np.exp(-2j * np.pi * FREQS[:, None] * t)).sum(axis=1) * dt
    return Z0 * what


def gate(ok, label, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}: {detail}")
    return bool(ok)


def main():
    print(f"rfx: {rfx.__file__}")
    print(f"bins (GHz): {FREQS / 1e9}")
    verdicts = {}

    # ---------------- FIX-A family runs -----------------------------------
    runs = {}
    for name, load in [("matched", 50.0), ("short", "short"),
                       ("open", "open")] + [
                       (f"load{int(r)}", r) for r in RL_SWEEP if r != 50.0]:
        sim = build_fix_a(load)
        cap = run_nu(sim, N_STEPS_A)
        runs[name] = cap
        k, (v, i, vinc, vp) = driven_acc(cap)
        s = s11_driven(vp, i)
        print(f"[FIX-A {name:9s}] |S11|={np.abs(s).round(4)} "
              f"arg={np.degrees(np.angle(s)).round(1)}")

    # G0 fixture sanity: realized n_live on every wire port; preflight ran.
    ok = True
    for name, cap in runs.items():
        for w in cap["wire_ports"]:
            if w["n_live"] != 2:
                ok = False
                print(f"  G0 VIOLATION: {name} port at "
                      f"({w['mid_i']},{w['mid_j']},{w['mid_k']}) "
                      f"n_live={w['n_live']}")
    verdicts["G0"] = gate(ok, "G0 fixture sanity",
                          "n_live == 2 on every wire port, preflight ON")

    # Production-extraction wiring check: S[k,k] from result matches the
    # harness formula on the raw accumulators.
    cap = runs["matched"]
    k, (v, i, vinc, vp) = driven_acc(cap)
    s_prod = np.asarray(cap["r"]["s_params"])[k, k, :]
    s_harn = s11_driven(vp, i)
    dwire = float(np.max(np.abs(s_prod.astype(np.complex128) - s_harn)))
    verdicts["WIRING"] = gate(dwire < 1e-5, "extraction wiring",
                              f"max|S_prod - S_harness| = {dwire:.2e}")

    # ---------------- F0: source-sense discriminator ----------------------
    a_by = {}
    for name in ("matched", "short", "open"):
        _, (v, i, _, vp) = driven_acc(runs[name])
        a_by[name] = a_wave(vp, i)
    A = np.array([a_by[n] for n in ("matched", "short", "open")])
    spread = (np.max(np.abs(A[:, None, :] - A[None, :, :]), axis=(0, 1))
              / np.max(np.abs(A), axis=0))
    verdicts["F0"] = gate(np.all(spread <= 0.05), "F0 a-invariance",
                          f"per-bin rel spread = {spread.round(4)} (gate 0.05)")

    # ---------------- F1: matched ----------------------------------------
    _, (v, i, _, vp) = driven_acc(runs["matched"])
    s_m = s11_driven(vp, i)
    ok = np.all(np.abs(s_m) < 0.05) and all(
        np.abs(s_m[b]) < 0.02 for b in LE1GHZ)
    verdicts["F1"] = gate(ok, "F1 matched",
                          f"|S11| = {np.abs(s_m).round(4)} "
                          f"(gate <0.05 all, <0.02 at <=1 GHz)")

    # ---------------- F2: PEC short --------------------------------------
    _, (v_s, i_s, _, vp_s) = driven_acc(runs["short"])
    s_s = s11_driven(vp_s, i_s)
    mag_ok = np.all((np.abs(s_s) >= 0.95) & (np.abs(s_s) <= 1.02))
    ph = np.degrees(np.angle(s_s))
    dev180 = np.abs(((ph - 180.0) + 180.0) % 360.0 - 180.0)
    ph_ok = dev180[LOW] <= 10.0 and dev180[TOP] <= 25.0
    re_ok = np.all(np.real(s_s) < 0)
    verdicts["F2"] = gate(mag_ok and ph_ok and re_ok, "F2 short",
                          f"|S11|={np.abs(s_s).round(4)} "
                          f"dev180={dev180.round(1)} deg "
                          f"Re={np.real(s_s).round(3)}")

    # ---------------- F3: open -------------------------------------------
    _, (v_o, i_o, _, vp_o) = driven_acc(runs["open"])
    s_o = s11_driven(vp_o, i_o)
    ph_o = np.degrees(np.angle(s_o))
    ok = (0.95 <= np.abs(s_o[LOW]) <= 1.02 and np.all(np.abs(s_o) >= 0.93)
          and -10.0 <= ph_o[LOW] <= 2.0)
    verdicts["F3"] = gate(ok, "F3 open",
                          f"|S11|={np.abs(s_o).round(4)} "
                          f"arg(low)={ph_o[LOW]:.2f} deg")

    # ---------------- F4: passivity order-of-operations -------------------
    ok = True
    detail = []
    for name in list(runs):
        _, (v, i, _, vp) = driven_acc(runs[name])
        z = vp / np.where(np.abs(i) > 0, i, 1e-30)
        s = s11_driven(vp, i)
        re_ok = np.all(np.real(z) >= -1e-3 * np.abs(z))
        alg_ok = not np.any((np.abs(s) > 1.02) & (np.real(z) >= 0))
        detail.append(f"{name}: minRe(Z)={np.min(np.real(z)):.3f}")
        ok = ok and re_ok and alg_ok
    verdicts["F4"] = gate(ok, "F4 passivity", "; ".join(detail))

    # ---------------- F5: #683 circuit law + load law ---------------------
    ok = True
    s_at, gamma_an = [], []
    for r_l in RL_SWEEP:
        name = "matched" if r_l == 50.0 else f"load{int(r_l)}"
        cap_l = runs[name]
        _, (v, i, _, vp) = driven_acc(cap_l)
        vsrc = vsrc_hat(cap_l, N_STEPS_A)
        for b in QS:
            i_pred = vsrc[b] / (Z0 + r_l)
            rel = abs(abs(i[b]) - abs(i_pred)) / abs(i_pred)
            if rel >= 0.05:
                ok = False
            print(f"  F5a Z_L={r_l:6.1f} f={FREQS[b]/1e9:.1f}GHz "
                  f"|I|={abs(i[b]):.4e} pred={abs(i_pred):.4e} rel={rel:.3f}")
        s = s11_driven(vp, i)
        s_at.append(np.real(s[QS[0]]))
        gamma_an.append((r_l - Z0) / (r_l + Z0))
    A = np.vstack([gamma_an, np.ones(len(gamma_an))]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, np.array(s_at), rcond=None)
    ok = ok and (0.9 <= slope <= 1.1) and abs(intercept) < 0.05
    verdicts["F5"] = gate(ok, "F5 circuit+load law",
                          f"slope={slope:.4f} intercept={intercept:.4f} "
                          f"S(qs)={np.array(s_at).round(4)} vs "
                          f"Gamma={np.array(gamma_an).round(4)}")

    # ---------------- F6: n_live invariance (FIX-A') ----------------------
    sim = build_fix_a(50.0, dx=DX_AP, nz=NZ_AP, extent=EXTENT_AP)
    cap_p = run_nu(sim, N_STEPS_AP)
    n_live_p = next(w for w in cap_p["wire_ports"] if w["excite"])["n_live"]
    _, (v_p, i_p, _, vp_p) = driven_acc(cap_p)
    s_p = s11_driven(vp_p, i_p)
    move = np.abs(np.abs(s_p) - np.abs(s_m))
    verdicts["F6"] = gate(n_live_p == 4 and np.all(move < 0.05),
                          "F6 n_live invariance",
                          f"n_live={n_live_p}, per-bin move="
                          f"{move.round(4)} (defect frame predicts ~0.267)")

    # ---------------- F7: power bookkeeping -------------------------------
    _, (v, i, _, vp) = driven_acc(runs["matched"])
    a = a_wave(vp, i)
    b_w = (vp - Z0 * i) / (2.0 * np.sqrt(Z0))
    wiring = np.abs((np.abs(a) ** 2 - np.abs(b_w) ** 2)
                    - np.real(vp * np.conj(i)))
    ok7a = np.all(wiring <= 1e-5 * np.max(np.abs(a) ** 2))
    _, (vs2, is2, _, vps2) = driven_acc(runs["short"])
    s_sh = s11_driven(vps2, is2)
    ok7c = np.all(1.0 - np.abs(s_sh) ** 2 <= 0.10)
    verdicts["F7"] = gate(
        ok7a and ok7c, "F7 power bookkeeping",
        f"wiring max={np.max(wiring):.2e} "
        f"(gate {1e-5 * np.max(np.abs(a) ** 2):.2e}); "
        f"short 1-|S11|^2={np.round(1 - np.abs(s_sh) ** 2, 4)}")
    print("  F7b flux-box referee: NOT-RUN — the NU lane implements only "
          "full-plane flux monitors (runners/nonuniform.py raises "
          "NotImplementedError for finite-region add_flux_monitor(size=...)),"
          " so the closed box around the driven column is not expressible; "
          "the gate is not re-aimed at another referee.")

    # ---------------- F8: KVL witness (short) -----------------------------
    _, (v_mid, i_s8, _, vp_s8) = driven_acc(runs["short"])
    ratio = np.abs(vp_s8) / np.abs(v_mid)
    ok = all(ratio[b] < 0.1 for b in QS)
    verdicts["F8"] = gate(ok, "F8 KVL witness",
                          f"|V_port|/|V_mid| at qs bins = "
                          f"{ratio[list(QS)].round(4)} (gate <0.1)")

    # ---------------- F9: current-uniformity premise ----------------------
    sim = build_fix_a(50.0)
    drv_i, drv_j = (int(round(DRV_XY[0] / DX_A)),
                    int(round(DRV_XY[1] / DX_A)))
    k0 = int(round(GAP_Z0 / DX_A))
    live_ks = [k0, k0 + 1]
    cap9 = run_nu(sim, N_STEPS_A,
                  extra_sample_cells=[(drv_i, drv_j, kk) for kk in live_ks])
    n_wp = len(cap9["wire_ports"])
    i_cells = []
    for n in range(n_wp - len(live_ks), n_wp):
        raw = cap9["r"]["wire_sparams_raw"][n]
        i_cells.append(np.asarray(raw[1], dtype=np.complex128))
    I = np.abs(np.array(i_cells))
    spread9 = (I.max(axis=0) - I.min(axis=0)) / I.max(axis=0)
    verdicts["F9"] = gate(np.all(spread9 <= 0.05), "F9 current uniformity",
                          f"per-bin live-cell |I| spread = "
                          f"{spread9.round(4)} (gate 0.05)")

    # ---------------- summary --------------------------------------------
    print("\n=== VERDICTS ===")
    n_fail = 0
    for k_, v_ in verdicts.items():
        print(f"  {k_:7s}: {'PASS' if v_ else 'FAIL'}")
        n_fail += (not v_)
    print(f"F7b: NOT-RUN (NU finite-region flux monitors unimplemented)")
    print(f"F10/F11: run separately (see module docstring / test suite)")
    if n_fail:
        print(f"\n{n_fail} falsifier(s) FIRED — stop and report; "
              "do not tune.")
        return 1
    print("\nAll evaluated falsifiers PASS.")
    return 0


def f10_dump(out_path):
    """400-step FIX-A matched run; dump final field state (bit-identity)."""
    sim = build_fix_a(50.0)
    cap = run_nu(sim, 400)
    st = cap["r"]["state"]
    np.savez(out_path, **{f: np.asarray(getattr(st, f))
                          for f in ("ex", "ey", "ez", "hx", "hy", "hz")})
    print(f"saved final state to {out_path}")


def fixc():
    """FIX-C thru-with-posts: reported prediction ONLY (never a gate)."""
    sim = Simulation(freq_max=10e9, domain=(14e-3, 10e-3, 8e-3), dx=DX_A,
                     dz_profile=np.full(NZ_A, DX_A), boundary="pec")
    sim.add(Box((4.0e-3, 4.0e-3, 3.0e-3), (9.5e-3, 6.0e-3, GAP_Z0)),
            material="pec")
    sim.add(Box((4.0e-3, 4.0e-3, GAP_Z1), (9.5e-3, 6.0e-3, 5.0e-3)),
            material="pec")
    sim.add_port(position=(4.5e-3, 5.0e-3, GAP_Z0), component="ez",
                 impedance=Z0, extent=EXTENT_A, excite=True, waveform=PULSE)
    sim.add_port(position=(9.0e-3, 5.0e-3, GAP_Z0), component="ez",
                 impedance=Z0, extent=EXTENT_A, excite=False)
    cap = run_nu(sim, N_STEPS_A)
    _, (v, i, _, vp) = driven_acc(cap)
    s = s11_driven(vp, i)
    print(f"[FIX-C thru] |S11| = {np.abs(s).round(4)} at {FREQS/1e9} GHz")
    print("  reported against the #318-measured 0.033-0.086 V-shape class "
          "(envelope <= 0.15 in-band; 0.15-0.19 with post-reactance shape "
          "= fixture physics).")


if __name__ == "__main__":
    if "--f10-dump" in sys.argv:
        f10_dump(sys.argv[sys.argv.index("--f10-dump") + 1])
    elif "--fixc" in sys.argv:
        fixc()
    else:
        sys.exit(main())
