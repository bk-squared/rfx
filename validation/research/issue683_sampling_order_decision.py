"""Issue #683 decision experiment: wire-port V/I sampling order.

Protocol (binding, committed BEFORE this file existed):
    docs/design_notes/issue683_sampling_order_decision_protocol.md

Two arms on bit-identical geometry:
    PRE  = uniform lane, forward(port_s11_freqs=...)   (samples BEFORE injection)
    POST = NU lane, uniform-valued dz_profile, run()   (samples AFTER injection)

Decision quantity: rho(f) = +v_dft/i_dft at the driven wire port's mid cell,
fitted against a known external load R_L. Gates G0-G2 and falsifiers F1/F2
are pre-declared in the protocol; this script only evaluates them.

Run:  JAX_PLATFORMS=cpu .venv/bin/python validation/research/issue683_sampling_order_decision.py
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

# ---------------------------------------------------------------- fixture ---
DX = 1e-3
DOMAIN = (16e-3, 12e-3, 12e-3)   # protocol section 3
NZ = int(round(DOMAIN[2] / DX))
FREQS = np.array([0.05e9, 0.1e9, 0.2e9, 0.5e9])
F1, F2 = 0, 1                    # decision bins (indices into FREQS)
N_STEPS = 4096
Z0_DRV = 50.0
RL_SWEEP = [12.5, 25.0, 50.0, 100.0, 200.0, 400.0]
N_LIVE = 2

PORT_Y = 6e-3
PORT_Z = 5e-3                    # column live cells k=5,6 (edges z in [5,7] mm)
EXTENT = 1e-3
X_DRV = 5e-3
X_LOAD = 11e-3
# PEC bars (2 cells thick) closing the loop; attach at column end nodes.
BAR_LO = Box((4e-3, 5e-3, 3e-3), (12e-3, 7e-3, 5e-3))     # cells k=3,4
BAR_HI = Box((4e-3, 5e-3, 7e-3), (12e-3, 7e-3, 9e-3))     # cells k=7,8
SHORT_COLUMN = Box((11e-3, 5e-3, 5e-3), (12e-3, 7e-3, 7e-3))

MID_DRV = (5, 6, 6)              # mid cell = upper live cell (len//2 of [5,6])
MID_LOAD = (11, 6, 6)

PULSE = GaussianPulse(f0=2e9, bandwidth=0.9)

# Gate/falsifier constants (protocol section 5-6)
G1A_RATIO_MIN = 2.0
G1B_RATIO_MIN = 1.5
G2_TOL_FRAC = 0.10
SLOPE_LO, SLOPE_HI = 0.90, 1.10
INTERCEPT_MAX_OHM = 10.0


def build(nu: bool, r_load: float | None, boundary: str = "pec"):
    kw = {"dz_profile": np.full(NZ, DX)} if nu else {}
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary=boundary,
                     **kw)
    sim.add(BAR_LO, material="pec")
    sim.add(BAR_HI, material="pec")
    sim.add_port(position=(X_DRV, PORT_Y, PORT_Z), component="ez",
                 impedance=Z0_DRV, extent=EXTENT, excite=True,
                 waveform=PULSE, direction="+x")
    if r_load is None:
        sim.add(SHORT_COLUMN, material="pec")
    else:
        sim.add_port(position=(X_LOAD, PORT_Y, PORT_Z), component="ez",
                     impedance=r_load, extent=EXTENT, excite=False,
                     direction="+x")
    return sim


# ------------------------------------------------------- raw-acc capture ----
def run_pre(sim):
    """Uniform lane (PRE-injection sampling). Returns (accs_list, dt, W_mid)."""
    import rfx.simulation as sim_mod
    cap = {}
    orig = sim_mod.run

    def spy(grid, materials, n_steps, **kwargs):
        cap["grid"] = grid
        cap["sources"] = kwargs.get("sources")
        return orig(grid, materials, n_steps, **kwargs)

    sim_mod.run = spy
    try:
        fr = sim.forward(n_steps=N_STEPS, port_s11_freqs=jnp.asarray(FREQS))
    finally:
        sim_mod.run = orig
    dt = float(cap["grid"].dt)
    w_mid = None
    for s in cap["sources"] or []:
        if (int(s.i), int(s.j), int(s.k)) == MID_DRV and s.component == "ez":
            w_mid = np.asarray(s.waveform, dtype=np.float64)
    assert w_mid is not None, "PRE arm: no source at driven mid cell"
    accs = []
    for spec, acc in fr.wire_port_sparams:
        accs.append(((int(spec.mid_i), int(spec.mid_j), int(spec.mid_k)),
                     tuple(np.asarray(a) for a in acc)))
    return accs, dt, w_mid


def run_post(sim):
    """NU lane (POST-injection sampling). Returns (accs_list, dt, W_mid)."""
    import rfx.runners.nonuniform as nur
    cap = {}
    orig = nur.run_nonuniform

    def spy(grid, materials, n_steps, **kwargs):
        r = orig(grid, materials, n_steps, **kwargs)
        cap["grid"] = grid
        cap["sources"] = kwargs.get("sources")
        cap["wire_ports"] = kwargs.get("wire_ports")
        cap["r"] = r
        return r

    nur.run_nonuniform = spy
    try:
        sim.run(n_steps=N_STEPS, compute_s_params=True,
                s_param_freqs=jnp.asarray(FREQS))
    finally:
        nur.run_nonuniform = orig
    dt = float(cap["grid"].dt)
    w_mid = None
    for s in cap["sources"] or []:
        if (int(s[0]), int(s[1]), int(s[2])) == MID_DRV and s[3] == "ez":
            w_mid = np.asarray(s[4], dtype=np.float64)
    assert w_mid is not None, "POST arm: no source at driven mid cell"
    raw = cap["r"].get("wire_sparams_raw")
    assert raw is not None, "POST arm: wire_sparams_raw missing from result"
    accs = []
    for wp, acc in zip(cap["wire_ports"], raw):
        accs.append(((int(wp["mid_i"]), int(wp["mid_j"]), int(wp["mid_k"])),
                     tuple(np.asarray(a) for a in acc)))
    return accs, dt, w_mid


def w_dft(w_table, dt):
    n = np.arange(len(w_table))
    t = n[None, :] * dt
    return (w_table[None, :] * np.exp(-2j * np.pi * FREQS[:, None] * t)).sum(
        axis=1) * dt


def pick(accs, mid):
    for m, acc in accs:
        if m == mid:
            return acc
    raise AssertionError(f"no port acc at {mid}; have {[m for m, _ in accs]}")


def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def main():
    print(f"rfx: {rfx.__file__}")
    results = {"PRE": {}, "POST": {}}
    dts = {}
    warns = {"PRE": [], "POST": []}

    boundary = "pec"
    for r_l in RL_SWEEP:
        for arm, runner, nu in (("PRE", run_pre, False), ("POST", run_post, True)):
            sim = build(nu, r_l, boundary)
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                accs, dt, w_mid = runner(sim)
            warns[arm] += [f"R_L={r_l}: {w.category.__name__}: {w.message}"
                           for w in wlist]
            dts[arm] = dt
            v_drv, i_drv, _ = pick(accs, MID_DRV)
            v_ld, i_ld, _ = pick(accs, MID_LOAD)
            results[arm][r_l] = dict(v=v_drv.astype(np.complex128),
                                     i=i_drv.astype(np.complex128),
                                     v_load=v_ld.astype(np.complex128),
                                     w_hat=w_dft(w_mid, dt))
            print(f"[{arm}] R_L={r_l:6.1f}  "
                  f"rho(f1)={v_drv[F1]/i_drv[F1]:.4f}  "
                  f"|I|={abs(i_drv[F1]):.3e}  |V_load|={abs(v_ld[F1]):.3e}")

    # ---- G0: preflight ran (no skip flag anywhere); surface warnings ----
    for arm in ("PRE", "POST"):
        uniq = sorted(set(warns[arm]))
        print(f"[G0] {arm}: {len(uniq)} unique warnings")
        for u in uniq[:10]:
            print(f"      {u[:200]}")

    print(f"[dt] PRE={dts['PRE']:.6e}  POST={dts['POST']:.6e}")
    dt_ok = abs(dts["PRE"] - dts["POST"]) <= 1e-18 + 1e-12 * dts["PRE"]

    # ---- G1: coupling gate (ordering-independent observables) ----
    g1 = True
    for arm in ("PRE", "POST"):
        i_mag = np.array([abs(results[arm][r]["i"][F1]) for r in RL_SWEEP])
        vl_mag = np.array([abs(results[arm][r]["v_load"][F1]) for r in RL_SWEEP])
        mono = bool(np.all(np.diff(i_mag) < 0))
        ratio_i = i_mag[0] / i_mag[-1]
        ratio_v = vl_mag.max() / vl_mag.min()
        ok = mono and ratio_i >= G1A_RATIO_MIN and ratio_v >= G1B_RATIO_MIN
        g1 &= ok
        print(f"[G1] {arm}: |I| monotone-dec={mono} ratio_I={ratio_i:.2f} "
              f"(>= {G1A_RATIO_MIN})  ratio_Vload={ratio_v:.2f} "
              f"(>= {G1B_RATIO_MIN})  -> {'PASS' if ok else 'FAIL'}")
    if not g1:
        print("VERDICT: FIXTURE INVALID (G1 coupling gate failed). Stopping "
              "per protocol - no tuning, no second fixture.")
        return 2

    # ---- G2: lane difference == same-step injection increment ----
    r50_pre, r50_post = results["PRE"][50.0], results["POST"][50.0]
    g_pre = r50_pre["v"] / r50_pre["w_hat"]
    g_post = r50_post["v"] / r50_post["w_hat"]
    dg = g_pre - g_post
    g2 = True
    for fb in (F1, F2):
        err = abs(dg[fb] - DX) / DX
        ok = err <= G2_TOL_FRAC and dt_ok
        g2 &= ok
        print(f"[G2] f={FREQS[fb]/1e9:.2f}GHz  G_PRE-G_POST={dg[fb]:.6e} "
              f"(target {DX:.1e}, rel err {err:.3f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    if not g2:
        print("VERDICT: INCONCLUSIVE (G2 separability gate failed - the two "
              "lanes do not differ by exactly the injection increment, so "
              "lane-vs-lane is not a clean pre/post toggle).")
        return 3

    # ---- Decision fits (F1/F2) ----
    verdicts = {}
    x = np.array(RL_SWEEP)
    for arm in ("PRE", "POST"):
        passes = {}
        for fb in (F1, F2):
            rho = np.array([(results[arm][r]["v"][fb] / results[arm][r]["i"][fb])
                            for r in RL_SWEEP])
            a, b = fit(x, rho.real)
            ok = (SLOPE_LO <= N_LIVE * a <= SLOPE_HI
                  and N_LIVE * abs(b) <= INTERCEPT_MAX_OHM)
            passes[fb] = ok
            print(f"[FIT] {arm} f={FREQS[fb]/1e9:.2f}GHz  "
                  f"n*a={N_LIVE*a:+.4f} (in [{SLOPE_LO},{SLOPE_HI}])  "
                  f"n*b={N_LIVE*b:+.3f} Ohm (|.|<={INTERCEPT_MAX_OHM})  "
                  f"-> {'PASS' if ok else 'FAIL'}")
            print(f"       Re rho*n vs R_L: " + "  ".join(
                f"{r:g}:{N_LIVE*(results[arm][r]['v'][fb]/results[arm][r]['i'][fb]).real:+.2f}"
                for r in RL_SWEEP))
        verdicts[arm] = passes

    f1_pre, f1_post = verdicts["PRE"][F1], verdicts["POST"][F1]
    f2_pre, f2_post = verdicts["PRE"][F2], verdicts["POST"][F2]
    consistent = (f1_pre == f2_pre) and (f1_post == f2_post)

    # ---- Corroborating: discrete Ampere/update identity (section 7) ----
    from rfx.grid import C0  # noqa: F401  (import parity with repo style)
    EPS0 = 8.8541878128e-12
    d = DX
    a_dual = DX * DX
    sigma = N_LIVE * d / (Z0_DRV * a_dual)
    dt = dts["POST"]
    loss = sigma * dt / (2.0 * EPS0)
    ca = (1.0 - loss) / (1.0 + loss)
    cb = (dt / EPS0) / (1.0 + loss)
    for arm in ("PRE", "POST"):
        rr = results[arm][50.0]
        e_hat = -rr["v"] / d
        shift = np.exp(-2j * np.pi * FREQS * dt)
        lhs = (1.0 - ca * shift) * e_hat - cb * rr["i"] / a_dual - rr["w_hat"]
        for fb in (F1, F2):
            rel = abs(lhs[fb]) / abs(cb * rr["i"][fb] / a_dual)
            print(f"[AMPERE] {arm} f={FREQS[fb]/1e9:.2f}GHz  rel residual = "
                  f"{rel:.3e}")

    # ---- PEC-short anchor (diagnostic only) ----
    for arm, runner, nu in (("PRE", run_pre, False), ("POST", run_post, True)):
        sim = build(nu, None, boundary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accs, dt, _ = runner(sim)
        v, i, _ = pick(accs, MID_DRV)
        rho = v[F1] / i[F1]
        print(f"[SHORT] {arm}: rho(f1) = {rho:.4f} (expected ~0 for the "
              f"terminal-consistent arm)")

    # ---- Verdict ----
    if not consistent:
        print("VERDICT: INCONCLUSIVE (f1/f2 robustness clause fired).")
        return 3
    if f1_pre and not f1_post:
        print("VERDICT: PRE-injection sampling (uniform lane's #72 contract) "
              "satisfies the circuit law; POST fails. The NU lane's ordering "
              "is refuted.")
        return 0
    if f1_post and not f1_pre:
        print("VERDICT: POST-injection sampling (NU lane status quo) "
              "satisfies the circuit law; PRE fails. The uniform lane's #72 "
              "wire-port ordering is refuted at excited ports.")
        return 0
    print("VERDICT: INCONCLUSIVE (both arms "
          + ("passed" if f1_pre else "failed") + " the circuit law).")
    return 3


if __name__ == "__main__":
    sys.exit(main())
