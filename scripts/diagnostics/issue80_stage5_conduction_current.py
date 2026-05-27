# ruff: noqa: E741, E702  (V, I EM notation + compact diagnostic statements)
"""Issue #80 Stage-5: PASSIVE MSL current that AVOIDS the cross-section transverse-H.

Established failure family (memory: project_issue80_halfstep_rootcause): EVERY
mechanism that reconstructs the port current from the recorded cross-section
transverse-H fails on a standing-wave reflector at MSL-feasible mesh —
  - ∮H·dl loop:           diverges 1.39→2.56 under refinement
  - single-cell discrete curl: cell-choice-sensitive, never passive (stage-0d)
  - modal overlap I=⟨H_sim,h_mode⟩ (ad-hoc + femwell 0.3%-accurate): I≈0, |Zin|~1e5Ω
  - voltage-only γ/α:      |S11| 4.16→1.30 (>1)
The BINDING CONSTRAINT: under-resolved transverse-H (3-5 substrate cells,
staircase-biased) cannot reconstruct a balanced current. The Ez VOLTAGE is
well-resolved and survives.

THIS SCRIPT tries conduction-current mechanisms that SIDESTEP transverse-H:

  C-T (TELEGRAPHER, prioritized): the lossless TL conjugate relation
        dV/dx = -jω L' I   ⇒   I(x) = -(1/(jω L')) · dV/dx
      uses ONLY the longitudinal gradient of the well-resolved modal voltage
      V(x)=∫Ez dz. L' = Z0_modal·n_eff/c0 from the femwell mode (stage-2, 0.3%).
      NO transverse-H anywhere. This is genuinely distinct from all 5 falsified
      mechanisms. V and I are a TL conjugate pair by the telegrapher eqn ⇒ the
      passivity Re(Zin)≥0 comes from the analytic L', not from a fragile H sum.

  C-S (SURFACE current Js=n̂×H over the FULL trace perimeter): mission candidate
      #1. CAUTION required by the brief: by discrete Ampère a closed perimeter
      sum of tangential-H == the ∮H·dl loop. We make it GENUINELY different by
      sampling tangential-H ON the conductor surface (the Yee H-edges that touch
      the PEC trace) rather than 1-cell-out, and we DISCARD it fast if it just
      reproduces the loop number. (Sanity / falsification rail, not the bet.)

VALIDATION (R5, mandatory): per-freq V, I, Zin(re/im), |S11| on (a) grounded
shorted stub and (b) radiating patch, at TWO meshes. PHYSICAL-CURRENT GUARD
(stage-3 trap): a "passive" headline is worthless if Z0·|I|/|V| ≈ 0 (I≈0 ⇒
trivial |S11|≈1). Report Z0·|I|/|V| ~ O(1) ALONGSIDE passivity, always.
Compare to the lumped reference (stage-1b) and the loop (stage-0c).
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.sources.msl_port import (
    MSLPort, _msl_yz_cells, _axis_cell_size, msl_loop_current as _loop,
)
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

_C0 = 299792458.0

# Shorted-stub geometry (identical to stage-0c / stage-3).
_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_F_MAX, _PM, _SL = 12e9, 2e-3, 6e-3
_Z0 = 50.0


def build_stub(dx):
    """Grounded shorted MSL stub (matches stage-0c). Lossless => infinite-Q."""
    lx = _PM + _SL + _PM
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = _H_SUB + 0.6e-3
    sim = Simulation(freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, _H_SUB)), material="ro")
    y_c = ly / 2
    tl, th = y_c - _W_TRACE / 2, y_c + _W_TRACE / 2
    xe = _PM + _SL
    sim.add(Box((0, tl, _H_SUB), (xe, th, _H_SUB + dx)), material="pec")
    sim.add(Box((xe, tl, 0), (xe + dx, th, _H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(_PM, y_c, 0.0), width=_W_TRACE, height=_H_SUB,
                     direction="+x", impedance=_Z0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1.6), eps_r_sub=_EPS_R)
    return sim, y_c, _H_SUB, _W_TRACE, _EPS_R


def _port_indices(g, y_c, h_sub, w_trace):
    port = MSLPort(feed_x=_PM, y_lo=y_c - w_trace / 2, y_hi=y_c + w_trace / 2,
                   z_lo=0.0, z_hi=h_sub, direction="+x", impedance=_Z0, excitation=None)
    cells = _msl_yz_cells(g, port)
    j_set = sorted({c[1] for c in cells}); k_set = sorted({c[2] for c in cells})
    j_lo, j_hi = j_set[0], j_set[-1]; j_c = (j_lo + j_hi) // 2
    k_sub_lo, k_sub_hi = k_set[0], k_set[-1]; k_tr = k_sub_hi + 1
    i_feed, _, _ = g.position_to_index((_PM, y_c, 0.0))
    return port, j_lo, j_hi, j_c, k_sub_lo, k_sub_hi, k_tr, i_feed


def _V_of_x(ez_plane, j_c, k_sub_lo, k_sub_hi, dz_a):
    """Modal voltage V = ∫ Ez dz at trace-centre column (well-resolved)."""
    V = np.zeros(ez_plane.shape[0], complex)
    for k in range(k_sub_lo, k_sub_hi + 1):
        V += ez_plane[:, j_c, k] * float(dz_a[k])
    return V


def run_stub(dx, freqs, n_eff, z0_modal, n_offset=5, n_spacing=3):
    """One stub solve. Records Ez plane at 3 downstream x (for dV/dx), plus
    transverse H at the central plane (for the loop + surface comparison)."""
    sim, y_c, h_sub, w_trace, eps_r = build_stub(dx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.preflight()
    g = sim._build_grid()
    (port, j_lo, j_hi, j_c, k_sub_lo, k_sub_hi, k_tr, i_feed) = _port_indices(
        g, y_c, h_sub, w_trace)
    dy_a = np.array([_axis_cell_size(g, "y", j) for j in range(g.shape[1])])
    dz_a = np.array([_axis_cell_size(g, "z", k) for k in range(g.shape[2])])
    dx_phys = float(g.dx)
    pad = getattr(g, "pad_x_lo", 0)

    # three x-planes for a centred dV/dx: i-1, i, i+1 (sign by direction).
    i_mid = i_feed + n_offset
    i_arr = [i_mid - n_spacing, i_mid, i_mid + n_spacing]
    x_arr = [float((max(0, min(ii, g.nx - 1)) - pad) * dx_phys) for ii in i_arr]
    dx_probe = (x_arr[2] - x_arr[0])  # span for centred difference

    fa = np.asarray(freqs, float)
    for n, xx in enumerate(x_arr):
        sim.add_dft_plane_probe(axis="x", coordinate=xx, component="ez",
                                freqs=fa, name=f"_ez{n}")
    # transverse-H at the MIDDLE plane (for loop + surface comparison only)
    for comp in ("hy", "hz"):
        sim.add_dft_plane_probe(axis="x", coordinate=x_arr[1], component=comp,
                                freqs=fa, name=f"_{comp}")
    n_steps = int(g.num_timesteps(num_periods=60.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=n_steps, compute_s_params=False)
    pl = res.dft_planes
    ez = [np.asarray(pl[f"_ez{n}"].accumulator) for n in range(3)]
    hy_p = np.asarray(pl["_hy"].accumulator)
    hz_p = np.asarray(pl["_hz"].accumulator)

    V = [_V_of_x(ezn, j_c, k_sub_lo, k_sub_hi, dz_a) for ezn in ez]
    Vmid = V[1]
    w = 2.0 * np.pi * fa

    # --- C-T telegrapher current: I = -(1/(jωL')) dV/dx, L' = Z0·n_eff/c0 ---
    Lp = z0_modal * n_eff / _C0
    dVdx = (V[2] - V[0]) / dx_probe        # centred difference
    I_tel = -dVdx / (1j * w * Lp)
    # direction sign: pick the sign giving predominantly Re(Zin)>0
    Z_tel = Vmid / (I_tel + 1e-30)
    if np.sum(Z_tel.real > 0) < np.sum((-Z_tel).real > 0):
        I_tel = -I_tel; Z_tel = Vmid / (I_tel + 1e-30)
    S11_tel = (Vmid - _Z0 * I_tel) / (Vmid + _Z0 * I_tel + 1e-30)

    # --- C-S surface current: tangential-H ON the trace surface edges ---
    # Trace block occupies y in [j_lo,j_hi], z = k_tr-1..k_tr (one cell thick
    # above substrate top). ON-surface tangential-H: the Yee H-edges that lie
    # on the conductor faces. Bottom face z=k_tr-1 (=k_sub_hi): Hy[.,j,k_sub_hi]
    # sits at z=(k_sub_hi+½)dz = trace bottom. Top face: Hy[.,j,k_tr]. Sides:
    # Hz[.,j_lo-1,.] and Hz[.,j_hi,.]. This is on-surface (k_sub_hi/k_tr) vs the
    # loop's 1-cell-out (k_tr-1=k_sub_hi works out same bottom; difference is we
    # take the ON-edge value, and compare top/bottom face contributions).
    def surf_current(k_bot, k_top):
        I = np.zeros(len(fa), complex)
        for j in range(j_lo, j_hi + 1):
            I += hy_p[:, j, k_bot] * float(dy_a[j])    # bottom face (+x current)
            I -= hy_p[:, j, k_top] * float(dy_a[j])    # top face
        for k in range(k_sub_lo, k_tr + 1):
            I += hz_p[:, j_hi, k] * float(dz_a[k])     # right face
            I -= hz_p[:, j_lo - 1, k] * float(dz_a[k]) # left face
        return I
    I_surf = surf_current(k_sub_hi, k_tr)   # on-surface bottom/top edges
    Z_surf = Vmid / (I_surf + 1e-30)
    if np.sum(Z_surf.real > 0) < np.sum((-Z_surf).real > 0):
        I_surf = -I_surf; Z_surf = Vmid / (I_surf + 1e-30)
    S11_surf = (Vmid - _Z0 * I_surf) / (Vmid + _Z0 * I_surf + 1e-30)

    # --- LOOP baseline (1-cell-out, production convention) ---
    I_loop = _loop(hy_p, hz_p, j_lo=j_lo, j_hi=j_hi, k_trace_lo=k_tr,
                   k_trace_hi=k_tr, dy_arr=dy_a, dz_arr=dz_a, direction="+x")
    Z_loop = Vmid / (I_loop + 1e-30)
    S11_loop = (Vmid - _Z0 * I_loop) / (Vmid + _Z0 * I_loop + 1e-30)

    return dict(f=fa, V=Vmid, Lp=Lp,
                I_tel=I_tel, Z_tel=Z_tel, S11_tel=S11_tel,
                I_surf=I_surf, Z_surf=Z_surf, S11_surf=S11_surf,
                I_loop=I_loop, Z_loop=Z_loop, S11_loop=S11_loop)


def _report(tag, r, sl):
    for key, In, Zn, Sn in (("C-T tel", "I_tel", "Z_tel", "S11_tel"),
                            ("C-S surf", "I_surf", "Z_surf", "S11_surf"),
                            ("LOOP   ", "I_loop", "Z_loop", "S11_loop")):
        S = np.abs(r[Sn])[sl]; Z = r[Zn][sl]; I = r[In][sl]; V = r["V"][sl]
        phys = _Z0 * np.median(np.abs(I)) / (np.median(np.abs(V)) + 1e-30)
        print(f"  {key}: max|S11|={S.max():.3f} min={S.min():.3f}  "
              f"ReZin<0:{int(np.sum(Z.real<0))}/{len(S)}  "
              f"|Zin|med={np.median(np.abs(Z)):8.1f}Ω  Z0|I|/|V|={phys:.3e}")


def main():
    z0_hj, eps_eff = hammerstad_jensen_z0_eps_eff(_W_TRACE, _H_SUB, _EPS_R)
    n_eff = float(np.sqrt(eps_eff))
    freqs = np.linspace(1.2e9, 12e9, 81)
    print("\n=== Stage-5: conduction current sidestepping transverse-H ===")
    print(f"analytic n_eff={n_eff:.4f}  Z0_HJ={z0_hj:.1f}Ω  eps_eff={eps_eff:.3f}")
    print("PHYSICAL-CURRENT GUARD: Z0|I|/|V| must be O(1); ~0 ⇒ I≈0 trivial-pass trap.")

    R = {}
    for dx in (80e-6, 50.8e-6):
        r = run_stub(dx, freqs, n_eff, z0_hj)
        R[dx] = r
        f = r["f"]; sl = slice(len(f)//8, -len(f)//8)
        print(f"\n--- STUB dx={dx*1e6:.1f}µm  L'={r['Lp']*1e6:.3f} µH/m ---")
        _report("stub", r, sl)
        # R5 per-freq trace (C-T, the bet)
        print("    per-freq C-T: f[GHz] |S11| Re(Zin) Im(Zin) Z0|I|/|V|")
        for i in range(len(f)//8, len(f)-len(f)//8, max(1, (len(f)*3//4)//6)):
            phys = _Z0 * abs(r["I_tel"][i]) / (abs(r["V"][i]) + 1e-30)
            print(f"      {f[i]/1e9:5.2f}  {abs(r['S11_tel'][i]):.3f}  "
                  f"{r['Z_tel'][i].real:8.1f}  {r['Z_tel'][i].imag:8.1f}  {phys:.3e}")
        np.savez(f"/tmp/issue80_stage5_stub_{int(dx*1e9)}.npz",
                 **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})

    print("\n=== STUB CONVERGENCE (dx=80→50.8µm) ===")
    for nm, Sk, Zk in (("C-T tel", "S11_tel", "Z_tel"),
                       ("C-S surf", "S11_surf", "Z_surf"),
                       ("LOOP", "S11_loop", "Z_loop")):
        o = []
        for dx in (80e-6, 50.8e-6):
            r = R[dx]; f = r["f"]; sl = slice(len(f)//8, -len(f)//8)
            o.append((float(np.max(np.abs(r[Sk])[sl])), int(np.sum(r[Zk].real[sl] < 0))))
        print(f"  {nm:9}: max|S11| {o[0][0]:.3f}→{o[1][0]:.3f}  ReZin<0 {o[0][1]}→{o[1][1]}")

    print("\n=== READING ===")
    print("C-T PASSES if: Re(Zin)≥0 band-wide, NON-worsening 80→50.8 (vs LOOP diverging),")
    print("AND Z0|I|/|V|~O(1) (physical current, NOT the I≈0 trap). Lossless stub is")
    print("infinite-Q ⇒ |S11|≈1 expected; the gate is PASSIVITY + physical current.")


if __name__ == "__main__":
    main()
