# ruff: noqa: E741, E702  (EM notation + compact diagnostic statements)
"""Issue #80 Stage-6: VOLTAGE-ONLY standing-wave DE-EMBEDDING (β FIXED, NO current).

THE CANDIDATE (coordinator directive, distinct from production extract_msl_nprobe):
  * Sample the MODAL voltage V(x) = ∫ Ez·dz over the substrate column at the trace
    centreline, at MANY (N=8) well-separated Ez DFT planes spanning > λ/4 along the
    feed line. (Production used only 3 planes over λ/8 — too short to resolve a/b.)
  * Fit V(x) = a·e^{-jβx} + b·e^{+jβx} by linear LEAST-SQUARES with β FIXED from the
    femwell n_eff (β = 2π f n_eff / c0). DO NOT fit/scan β — fitting β is exactly the
    fragile step that made production extract_msl_nprobe blow to 4.16 (stage-4).
  * De-embed to a chosen reference plane: S11 = (b/a)·e^{-2jβ·x_ref}.
  * |S11| = |b/a| ≤ 1 is STRUCTURAL for a passive load → passivity is FREE.

THE REAL TEST IS ACCURACY (fidelity-first). A bounded-but-wrong S11 is FALSIFIED.
  TEST 1 — LOSSLESS shorted stub (grounded, stage0e geom): true |S11|≡1 at all f;
           de-embedded |S11| must be ≈1 AND ∠S11 must track analytic -2βL_short.
  TEST 2 — RADIATING patch (stage1b geom): de-embedded S11 must MATCH the lumped
           reference S11 (dip freq + depth) within a stated tolerance.
  TEST 3 — mesh convergence: accuracy holds/improves dx → dx/1.6.

R5 GUARDS: report fit RESIDUAL; report |a|,|b| (confirm |b/a|≤1 not trivial via a→0);
headline = accuracy-vs-reference, never bare |S11|≤1.

β source: femwell GMSH-free modesolver (stage-2), n_eff 0.3% accurate vs analytic.
  STUB feed (εr=3.66, h=254µm, W=600µm):   n_eff = 1.6889 (HJ 1.6939)
  PATCH feed (εr=3.38, h=787µm, W=1.8mm):  n_eff = 1.6574 (HJ 1.6328)
CPU only. Prototype — NO production rfx/ edits.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

C0 = 299792458.0

# --- femwell-derived n_eff (fixed-β source) -------------------------------
N_EFF_STUB = 1.6889    # femwell, εr=3.66 h=254µm W=600µm  (stage-2)
N_EFF_PATCH = 1.6574   # femwell, εr=3.38 h=787µm W=1.8mm  @9GHz


# ==========================================================================
# Voltage-only standing-wave de-embed core (β FIXED, NO current).
# ==========================================================================
def fit_ab_fixed_beta(V, xs, beta):
    """Fit V_n = a e^{-jβ x_n} + b e^{+jβ x_n} per frequency, β FIXED.

    V    : (n_freqs, N) complex modal-voltage phasors at probe planes
    xs   : (N,) probe x positions (m)
    beta : (n_freqs,) real propagation constant (rad/m), FIXED (not fit)
    Returns a,b (n_freqs,) complex at x=0, residual (n_freqs,) L2 norm.
    """
    nf, N = V.shape
    a = np.zeros(nf, complex); b = np.zeros(nf, complex); res = np.zeros(nf)
    for k in range(nf):
        col_f = np.exp(-1j * beta[k] * xs)
        col_b = np.exp(+1j * beta[k] * xs)
        A = np.stack([col_f, col_b], axis=-1)          # (N,2)
        sol, *_ = np.linalg.lstsq(A, V[k], rcond=None)
        a[k], b[k] = sol[0], sol[1]
        res[k] = np.linalg.norm(V[k] - A @ sol)
    return a, b, res


def deembed_s11(a, b, beta, x_ref):
    """S11 at reference plane x_ref:  (b/a)·e^{-2jβ x_ref}.

    a,b referenced to x=0; the incident wave at x_ref has phase e^{-jβ x_ref}
    and reflected e^{+jβ x_ref}, so Γ(x_ref) = (b/a) e^{+2jβ x_ref}... but our
    coordinate runs +x AWAY from the load.  With x increasing toward the load,
    Γ(x_ref) = (b e^{+jβ x_ref}) / (a e^{-jβ x_ref}) = (b/a) e^{+2jβ x_ref}.
    The caller passes the signed (x_ref - x0) consistent with that convention.
    """
    return (b / (a + 1e-30)) * np.exp(2j * beta * x_ref)


# ==========================================================================
# FDTD harness: MSL-port-driven feed line + N well-separated Ez planes.
# ==========================================================================
def _v_integrator(grid, y_c, z_sub_lo, z_sub_hi):
    """Return (j_centre, k_lo, k_hi, dz_slice) for V=∫Ez·dz at trace centre."""
    j_c = grid.position_to_index((0.0, y_c, 0.0))[1]
    k_lo = grid.position_to_index((0.0, 0.0, z_sub_lo))[2]
    k_hi = grid.position_to_index((0.0, 0.0, z_sub_hi))[2]
    k_lo, k_hi = sorted((k_lo, k_hi))
    dz = getattr(grid, "dz_profile", None)
    dz_slice = (np.asarray(dz)[k_lo:k_hi + 1] if dz is not None
                else np.full(k_hi - k_lo + 1, float(grid.dx)))
    return j_c, k_lo, k_hi, dz_slice


def _V_from_plane(res, name, j_c, k_lo, k_hi, dz_slice):
    """V_f = Σ_k Ez_plane[:, j_centre, k]·dz_k on the x-plane named `name`."""
    # res.dft_planes is a dict keyed by the registered name.
    acc = np.asarray(res.dft_planes[name].accumulator)   # (n_freqs, ny, nz)
    col = acc[:, j_c, k_lo:k_hi + 1]                     # (n_freqs, nk)
    return np.sum(col * dz_slice[None, :], axis=-1)


def run_feed_and_probe(build_fn, dx, freqs, n_planes, span_frac, n_eff,
                       num_periods, label):
    """Build geom via build_fn(dx)->(sim, y_c, z_sub_lo, z_sub_hi, x_feed,
    x_load, dirn), register N Ez planes from just past the feed toward the
    load, run, return dict(xs, V (nf,N), beta, x_feed, x_load, dirn, residual,
    a, b).
    span_frac: probe array spans span_frac·λ_eff(at f_mid) — set >0.25 for λ/4+.
    """
    sim, y_c, z_sub_lo, z_sub_hi, x_feed, x_load, dirn = build_fn(dx)
    sign = 1.0 if dirn == "+x" else -1.0

    # Distribute N planes EVENLY across the usable line span
    # [x_feed + 6 cells (out of source near-field) , x_load - 3 cells]. This
    # maximises the baseline without running planes off the trace (the earlier
    # span_frac·λ overshot the 6mm stub and collapsed 3 planes onto the clamp).
    x_lo = x_feed + sign * 6 * dx
    x_hi = x_load - sign * 3 * dx
    if sign < 0:
        x_lo, x_hi = x_hi, x_lo
    xs = np.linspace(min(x_lo, x_hi), max(x_lo, x_hi), n_planes)
    _ = span_frac  # retained for signature compat; span now line-bounded

    # x-axis Ez planes: z (substrate mid) is selected by the V integrator below.
    for q, xq in enumerate(xs):
        sim.add_dft_plane_probe(axis="x", coordinate=float(xq), component="ez",
                                freqs=freqs, name=f"{label}_ez{q}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = sim.preflight()
        g = sim._build_grid()
        n = int(g.num_timesteps(num_periods=num_periods))
        # We only need the Ez DFT planes; the MSL port still excites the source.
        # Avoid compute_s_params (which rejects MSL ports in Result.s_params).
        res = sim.run(n_steps=n)
        # CRITICAL: sim.run(n_steps=...) returns LAZY (async-dispatched) DFT
        # accumulators.  Touching them one-by-one re-blocks on (and can re-run)
        # the whole forward scan per plane → minutes of stall.  Stack all N
        # accumulators into ONE jax array and block_until_ready ONCE.
        import jax
        import jax.numpy as jnp
        stk = jax.block_until_ready(jnp.stack(
            [res.dft_planes[f"{label}_ez{q}"].accumulator
             for q in range(n_planes)]))                  # (N, nf, ny, nz)
        accs = np.asarray(stk)
    j_c, k_lo, k_hi, dz_slice = _v_integrator(g, y_c, z_sub_lo, z_sub_hi)
    V = np.stack([
        np.sum(accs[q][:, j_c, k_lo:k_hi + 1] * dz_slice[None, :], axis=-1)
        for q in range(n_planes)], axis=-1)              # (nf, N)
    # quote any non-trivial preflight messages
    pf = [m for m in (msgs or []) if "PASS" not in str(m).upper()]
    beta = 2 * np.pi * freqs * n_eff / C0
    a, b, residual = fit_ab_fixed_beta(V, xs, beta)
    return dict(xs=xs, V=V, beta=beta, x_feed=x_feed, x_load=x_load, dirn=dirn,
                a=a, b=b, residual=residual, pf=pf, g=g, sign=sign)


# ==========================================================================
# Geometry builders.
# ==========================================================================
def build_stub(dx):
    """Grounded LOSSLESS shorted stub (stage0e geom), MSL-port driven."""
    EPS_R, H_SUB, W = 3.66, 254e-6, 600e-6
    F_MAX, PORT_MARGIN, STUB_LEN = 12e9, 2e-3, 6e-3
    z_gnd_hi = dx
    z_sub_lo, z_sub_hi = dx, dx + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + dx
    lx = PORT_MARGIN + STUB_LEN + PORT_MARGIN
    ly = W + 2 * (2 * H_SUB + 8 * dx)
    lz = z_tr_hi + 0.6e-3
    sim = Simulation(freq_max=F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro4350b", eps_r=EPS_R)
    y_c = ly / 2.0
    tl, th = y_c - W / 2, y_c + W / 2
    x_end = PORT_MARGIN + STUB_LEN
    sim.add(Box((0, 0, 0), (lx, ly, z_gnd_hi)), material="pec")
    sim.add(Box((0, 0, z_sub_lo), (lx, ly, z_sub_hi)), material="ro4350b")
    sim.add(Box((0, tl, z_tr_lo), (x_end, th, z_tr_hi)), material="pec")
    sim.add(Box((x_end, tl, 0), (x_end + dx, th, z_tr_hi)), material="pec")  # via short
    sim.add_msl_port(position=(PORT_MARGIN, y_c, z_sub_lo),
                     width=W, height=H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1.6),
                     eps_r_sub=EPS_R, mode="laplace")
    # x_load = the shorting via plane
    return sim, y_c, z_sub_lo, z_sub_hi, PORT_MARGIN, x_end, "+x"


def build_patch(dx):
    """Radiating patch (stage1b geom), MSL-port driven."""
    EPS_R, H_SUB = 3.38, 0.787e-3
    W, L, W_MSL, L_MSL = 10.129e-3, 8.595e-3, 1.8e-3, 8.0e-3
    PORT_MARGIN = 5.0e-3
    DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
    Y_C, Z_GND = DOM_Y / 2.0, 4e-3
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=dx, cpml_layers=8, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    z_gnd_hi = Z_GND + dx
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + dx
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec")
    sim.add(Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi)), material="ro4003c")
    sim.add(Box((0, Y_C - W_MSL / 2, z_tr_lo),
                (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2, z_tr_hi)), material="pec")
    sim.add(Box((PORT_MARGIN + L_MSL, Y_C - W / 2, z_tr_lo),
                (PORT_MARGIN + L_MSL + L, Y_C + W / 2, z_tr_hi)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, Y_C, z_sub_lo),
                     width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
                     eps_r_sub=EPS_R, mode="laplace")
    return sim, Y_C, z_sub_lo, z_sub_hi, PORT_MARGIN, PORT_MARGIN + L_MSL, "+x"


# ==========================================================================
# Tests.
# ==========================================================================
def test_stub(meshes=(80e-6, 50e-6), num_periods=40.0):
    print("\n" + "=" * 74)
    print("TEST 1 — LOSSLESS shorted stub: true |S11|≡1, ∠S11 = -2βL_short")
    print("=" * 74)
    freqs = np.linspace(2e9, 11e9, 40)
    for dx in meshes:
        out = run_feed_and_probe(build_stub, dx, freqs, n_planes=8,
                                 span_frac=0.30, n_eff=N_EFF_STUB,
                                 num_periods=num_periods, label=f"stub{int(dx*1e9)}")
        if out["pf"]:
            print(f"  [preflight dx={dx*1e6:.0f}µm] " + " | ".join(map(str, out["pf"])))
        a, b, beta, xs = out["a"], out["b"], out["beta"], out["xs"]
        x_load = out["x_load"]; sign = out["sign"]
        # de-embed to the short plane. coordinate run is +x toward load; the
        # model is referenced at x=0 (absolute), so reference offset = x_load.
        s11 = deembed_s11(a, b, beta, x_load) if sign > 0 else \
              deembed_s11(a, b, beta, -(x_load))
        # analytic: distance from short to... we de-embedded TO the short, so
        # ideal ∠S11 = π (short = Γ=-1) independent of f.  |S11|=1.
        absS = np.abs(s11); ang = np.degrees(np.angle(s11))
        # R5 SNR GATE: the Gaussian source has energy only near its centre band.
        # The absolute fit residual is a flat numerical floor (~2.5e-14); where
        # |V| collapses (band edges) the NORMALISED residual blows up and |S11|
        # is meaningless noise.  Restrict the accuracy headline to bins with
        # real source energy (|V|/|V|max > 0.15) — anything else is SNR garbage.
        Vmag = np.abs(out["V"]).mean(axis=1)
        gate = Vmag / Vmag.max() > 0.15
        rv = out["residual"] / (Vmag + 1e-30)
        print(f"\n  --- dx={dx*1e6:.0f}µm  sub_cells={254e-6/dx:.2f}  "
              f"N=8 planes span={(xs[-1]-xs[0])*1e3:.2f}mm  gated_bins={gate.sum()} ---")
        print(f"  [SNR-GATED] de-embed→SHORT: |S11| mean={absS[gate].mean():.3f} "
              f"(ideal 1.0)  ∠S11 mean={ang[gate].mean():.1f}° (ideal ±180°)")
        print(f"  |a| med={np.median(np.abs(a)[gate]):.3e}  "
              f"|b| med={np.median(np.abs(b)[gate]):.3e}  "
              f"|b/a| med={np.median(np.abs(b/a)[gate]):.3f}  "
              f"resid/|V| med(gated)={np.median(rv[gate]):.2e}")
        print(f"  {'f[GHz]':>7}{'|V|/max':>9}{'|S11|':>8}{'∠S11°':>8}"
              f"{'|a|':>11}{'|b|':>11}{'resid/V':>10}")
        for i in range(0, len(freqs), 3):
            g = "*" if gate[i] else " "
            print(f" {g}{freqs[i]/1e9:6.2f}{Vmag[i]/Vmag.max():9.3f}{absS[i]:8.3f}"
                  f"{ang[i]:8.1f}{abs(a[i]):11.3e}{abs(b[i]):11.3e}{rv[i]:10.2e}")


def test_patch(meshes=(300e-6, 197e-6)):
    print("\n" + "=" * 74)
    print("TEST 2 — RADIATING patch: de-embed S11 vs LUMPED reference (ground truth)")
    print("=" * 74)
    # lumped reference from stage1b saved npz (dx=300,197µm)
    ref = {}
    for dxn in (300, 197):
        try:
            d = np.load(f"/tmp/issue80_stage1b_{dxn}.npz")
            ref[dxn] = (d["freqs"], d["S11"])
        except FileNotFoundError:
            pass
    if not ref:
        print("  [warn] no /tmp/issue80_stage1b_*.npz — run stage1b first for ref.")
    freqs = np.linspace(4e9, 14e9, 41)
    mesh_map = {300e-6: 300, 197e-6: 197, 80e-6: 80, 50e-6: 50}
    for dx in meshes:
        dxn = mesh_map.get(dx, int(round(dx * 1e6)))
        out = run_feed_and_probe(build_patch, dx, freqs, n_planes=8,
                                 span_frac=0.30, n_eff=N_EFF_PATCH,
                                 num_periods=120.0, label=f"patch{int(dx*1e9)}")
        if out["pf"]:
            print(f"  [preflight dx={dx*1e6:.0f}µm] " + " | ".join(map(str, out["pf"])))
        a, b, beta, V = out["a"], out["b"], out["beta"], out["V"]
        x_load = out["x_load"]
        s11 = deembed_s11(a, b, beta, x_load)
        absS = np.abs(s11)
        Vmag = np.abs(V).mean(axis=1)
        gate = Vmag / Vmag.max() > 0.30          # patch: stricter gate (leaky line)
        rv = out["residual"] / (Vmag + 1e-30)
        print(f"\n  --- dx={dx*1e6:.0f}µm  shape={out['g'].shape}  "
              f"span={(out['xs'][-1]-out['xs'][0])*1e3:.2f}mm  gated_bins={gate.sum()} ---")
        print(f"  [SNR-GATED] de-embed: max|S11|={absS[gate].max():.3f} "
              f"min|S11|={absS[gate].min():.3f}  >1.05 bins={int((absS[gate]>1.05).sum())}")
        print(f"  |a| med={np.median(np.abs(a)[gate]):.3e} "
              f"|b| med={np.median(np.abs(b)[gate]):.3e} "
              f"|b/a| med(gated)={np.median(np.abs(b/a)[gate]):.3f} "
              f"resid/|V| med={np.median(rv[gate]):.2e}")
        if dxn in ref:
            rf, rs = ref[dxn]
            print(f"  LUMPED ref (GROUND TRUTH): band-mean|S11|={np.abs(rs).mean():.3f} "
                  f"dip@{rf[int(np.argmin(np.abs(rs)))]/1e9:.2f}GHz min={np.abs(rs).min():.3f}")
        # R5 witness: fit |b/a| vs the RAW standing-wave envelope VSWR.  If they
        # disagree, the 2-wave FIT (not the data) is the failure mode.
        print(f"  {'f[GHz]':>7}{'|V|/mx':>8}{'fit|b/a|':>9}{'env|Γ|':>8}"
              f"{'ref|S11|':>9}{'resid/V':>9}{'mono?':>6}")
        for i in range(0, len(freqs), 3):
            Vx = np.abs(V[i]); vswr = Vx.max() / (Vx.min() + 1e-30)
            genv = (vswr - 1) / (vswr + 1)
            mono = int(np.all(np.diff(Vx) < 0) or np.all(np.diff(Vx) > 0))
            rval = (np.interp(freqs[i], ref[dxn][0], np.abs(ref[dxn][1]))
                    if dxn in ref else float('nan'))
            g = "*" if gate[i] else " "
            print(f" {g}{freqs[i]/1e9:6.2f}{Vmag[i]/Vmag.max():8.3f}"
                  f"{abs(b[i]/a[i]):9.3f}{genv:8.3f}{rval:9.3f}{rv[i]:9.2e}{mono:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["stub", "patch", "both"], default="both")
    ap.add_argument("--dx", type=float, default=None,
                    help="single dx in µm (else both meshes)")
    args = ap.parse_args()
    if args.test in ("stub", "both"):
        test_stub(meshes=(args.dx * 1e-6,) if args.dx else (80e-6, 50e-6))
    if args.test in ("patch", "both"):
        test_patch(meshes=(args.dx * 1e-6,) if args.dx else (300e-6, 197e-6))
    print("\n=== READING ===")
    print("ACCURATE+passive ⇒ propose. Bounded-but-wrong (wrong dip/phase, large")
    print("residual, or a→0 trivializing |b/a|≤1) ⇒ FALSIFIED. Fidelity-first.")


if __name__ == "__main__":
    main()
