"""WR-90 PEC-short S11 phase cross-check across rfx, OpenEMS, Meep.

Companion to ``no_fp_null_phase_check.py``. That diagnostic verified
rfx PEC-short S11 phase against the analytic round-trip exp(-j·2·β·d)
to ~10° max (Yee-dispersion-limited at dx=1 mm). This script extends
the same comparison to OpenEMS and Meep dump fields so we have a
3-tool + analytic phase consensus on a no-FP-null geometry.

Per-tool dump conventions (verified 2026-04-29 in
``cross_tool_half_step_audit.py``):
  rfx     : NEEDS the exp(+jω·dt/2) Yee half-step correction client-
            side; H_y of the dump probe is at (n+1/2)·dt.
  OpenEMS : already corrected internally.
  Meep    : already corrected internally.

Both magnitude AND complex S11 are computed via the s11_from_dumps
cell-centred TE10 V/I recipe so all three tools share an identical
post-processing path. The reference plane is ``mon_left`` (rfx
+50 mm = OE/Meep -50 mm). The ideal PEC-short reflection at that
plane is ``-1 · exp(-j·2·β_v·d)`` where ``d = pec_short - mon_left``
= 95 mm of empty guide.

Output: per-frequency table (∠S11 from each tool + analytic +
residual), plot, and a 1-line verdict per tool.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_cross_tool_phase"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
MU_0 = 1.2566370614e-6


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_s11_module():
    """Reuse read_dump from the canonical comparator."""
    spec = importlib.util.spec_from_file_location(
        "s11_from_dumps", REPO / "scripts" / "diagnostics" / "wr90_port"
        / "s11_from_dumps.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wrap(deg):
    return ((deg + 180.0) % 360.0) - 180.0


def _complex_s11_recipe(E_z, H_y, y, z, freqs, a):
    """Same as s11_from_field but returns COMPLEX S11(f), not |S11|."""
    Ny, Nz = E_z.shape[1], E_z.shape[2]
    y_ref = y - y.min()
    sin_y = np.sin(np.pi * y_ref / a)
    dy = float(np.abs(y[1] - y[0])) if Ny > 1 else a
    dz = float(np.abs(z[1] - z[0])) if Nz > 1 else 1.0
    weight = sin_y[:, None] * np.ones((1, Nz)) * dy * dz
    V = np.sum(E_z * weight[None, :, :], axis=(1, 2))
    I = np.sum(H_y * weight[None, :, :], axis=(1, 2))
    omega = 2.0 * np.pi * freqs
    k0 = omega / C0
    kc = np.pi / a
    beta = np.sqrt(np.maximum(k0 ** 2 - kc ** 2, 1e-30))
    Z = omega * MU_0 / beta
    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    return a_ref / np.where(np.abs(a_fwd) > 1e-30, a_fwd, 1e-30)


def run_rfx_dump_with_correction():
    """Run rfx PEC-short and return DFT planes with the Yee half-step
    correction applied to H_y (matches s11_from_dumps fix)."""
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cv = _load_cv11()
    dx_m = 1e-3
    plane_x = cv.MON_LEFT_X
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2,
                             float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS, dx=dx_m,
    )
    sim.add(
        Box((cv.PEC_SHORT_X, 0.0, 0.0),
            (cv.PEC_SHORT_X + 2 * dx_m, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050, name="left",
        mode_profile="discrete",
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(axis="x", coordinate=plane_x,
                                component=comp, freqs=pf, name=f"{comp}_p")
    res = sim.run(num_periods=200, compute_s_params=False)
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)

    if hasattr(sim, "_grid") and hasattr(sim._grid, "dt"):
        dt_sim = float(sim._grid.dt)
    else:
        dt_sim = 0.5 * dx_m / C0
    omega = 2.0 * np.pi * np.asarray(freqs)
    Hy = Hy * np.exp(+1j * omega * (0.5 * dt_sim))[:, None, None]

    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)
    return {"freqs": np.asarray(freqs), "y": y, "z": z,
            "E_z": Ez, "H_y": Hy, "a": cv.A_WG}


def main():
    cv = _load_cv11()
    s11mod = _load_s11_module()
    a_m = cv.A_WG

    # Dump dirs (defaults match the audit script).
    oe_dir = (REPO / "scripts" / "spikes" / "2026-04-28"
              / "out_openems_soft_current" / "dumps_r1")
    mp_dir = Path("/root/workspace/byungkwan-workspace/research/microwave-energy/"
                  "results/wr90_meep_field_dumps_20260428T0311Z/dumps_r1")
    if not oe_dir.exists():
        sys.exit(f"OpenEMS dumps missing at {oe_dir}")
    if not mp_dir.exists():
        sys.exit(f"Meep dumps missing at {mp_dir}")

    print("=== Cross-tool PEC-short S11 phase (R=1, mon_left plane) ===\n")

    # rfx
    print("Running rfx PEC-short ...")
    rfx = run_rfx_dump_with_correction()
    s11_rfx = _complex_s11_recipe(
        rfx["E_z"], rfx["H_y"], rfx["y"], rfx["z"],
        rfx["freqs"], a_m
    )
    freqs = rfx["freqs"]

    # OpenEMS
    print("Loading OpenEMS dumps ...")
    oe_e = s11mod.read_dump(str(oe_dir / "pec_short_r1_Efield_mon_left_plane.h5"))
    oe_h = s11mod.read_dump(str(oe_dir / "pec_short_r1_Hfield_mon_left_plane.h5"))
    Ez_oe = oe_e["field"][:, 0, :, :, 2]
    Hy_oe = oe_h["field"][:, 0, :, :, 1]
    s11_oe = _complex_s11_recipe(
        Ez_oe, Hy_oe, oe_e["y"], oe_e["z"], oe_e["freqs"], a_m
    )

    # Meep
    print("Loading Meep dumps ...")
    mp_e = s11mod.read_dump(str(mp_dir / "pec_short_r1_Efield_mon_left_plane.h5"))
    mp_h = s11mod.read_dump(str(mp_dir / "pec_short_r1_Hfield_mon_left_plane.h5"))
    Ez_mp = mp_e["field"][:, 0, :, :, 2]
    Hy_mp = mp_h["field"][:, 0, :, :, 1]
    s11_mp = _complex_s11_recipe(
        Ez_mp, Hy_mp, mp_e["y"], mp_e["z"], mp_e["freqs"], a_m
    )

    # Analytic round-trip reference at mon_left plane.
    omega = 2.0 * np.pi * freqs
    kc = 2.0 * np.pi * cv.F_CUTOFF_TE10 / C0
    beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))
    d_pec = cv.PEC_SHORT_X - cv.MON_LEFT_X   # 95 mm
    s11_ana = -np.exp(-1j * beta_v * 2.0 * d_pec)

    print(f"\nd_pec (mon_left -> PEC face) = {d_pec*1000:.3f} mm  | "
          f"round-trip 2d = {2*d_pec*1000:.3f} mm\n")

    # Per-tool residuals (deg)
    res_rfx = _wrap(np.degrees(np.angle(s11_rfx) - np.angle(s11_ana)))
    res_oe  = _wrap(np.degrees(np.angle(s11_oe)  - np.angle(s11_ana)))
    res_mp  = _wrap(np.degrees(np.angle(s11_mp)  - np.angle(s11_ana)))

    print(f"{'f[GHz]':>7s} | {'∠ana':>7s} | {'∠rfx':>7s} {'res':>6s} | "
          f"{'∠OE':>7s} {'res':>6s} | {'∠Meep':>7s} {'res':>6s}")
    print("-" * 85)
    for i, f in enumerate(freqs):
        print(f"{f/1e9:7.2f} | {np.degrees(np.angle(s11_ana[i])):+7.1f} | "
              f"{np.degrees(np.angle(s11_rfx[i])):+7.1f} {res_rfx[i]:+6.1f} | "
              f"{np.degrees(np.angle(s11_oe[i])):+7.1f} {res_oe[i]:+6.1f} | "
              f"{np.degrees(np.angle(s11_mp[i])):+7.1f} {res_mp[i]:+6.1f}")

    print(f"\n{'tool':<8s} | {'mean':>8s} | {'std':>6s} | {'max|·|':>8s}")
    print("-" * 40)
    for label, res in [("rfx", res_rfx), ("openems", res_oe), ("meep", res_mp)]:
        print(f"{label:<8s} | {np.mean(res):+8.2f} | "
              f"{np.std(res):6.2f} | {np.max(np.abs(res)):8.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.axhspan(-15, +15, color="green", alpha=0.10, label="±15° band")
    f_ghz = freqs / 1e9
    ax.plot(f_ghz, res_rfx, "o-", color="tab:orange",
            label=f"rfx     (max|·| {np.max(np.abs(res_rfx)):.2f}°)")
    ax.plot(f_ghz, res_oe, "s-", color="tab:blue",
            label=f"OpenEMS (max|·| {np.max(np.abs(res_oe)):.2f}°)")
    ax.plot(f_ghz, res_mp, "^-", color="tab:green",
            label=f"Meep    (max|·| {np.max(np.abs(res_mp)):.2f}°)")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("∠S11_tool − ∠S11_analytic [deg]")
    ax.set_title("WR-90 PEC-short S11 phase cross-tool check (R=1, mon_left)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / "cross_tool_phase_check_R1.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")

    out_json = OUT / "cross_tool_phase_check_R1.json"
    out_json.write_text(json.dumps({
        "freqs_hz": freqs.tolist(),
        "d_pec_m": float(d_pec),
        "s11_rfx_deg": np.degrees(np.angle(s11_rfx)).tolist(),
        "s11_oe_deg": np.degrees(np.angle(s11_oe)).tolist(),
        "s11_mp_deg": np.degrees(np.angle(s11_mp)).tolist(),
        "s11_ana_deg": np.degrees(np.angle(s11_ana)).tolist(),
        "residual_rfx": res_rfx.tolist(),
        "residual_openems": res_oe.tolist(),
        "residual_meep": res_mp.tolist(),
        "summary": {
            "rfx":     {"mean": float(np.mean(res_rfx)), "std": float(np.std(res_rfx)),
                        "max_abs": float(np.max(np.abs(res_rfx)))},
            "openems": {"mean": float(np.mean(res_oe)),  "std": float(np.std(res_oe)),
                        "max_abs": float(np.max(np.abs(res_oe)))},
            "meep":    {"mean": float(np.mean(res_mp)),  "std": float(np.std(res_mp)),
                        "max_abs": float(np.max(np.abs(res_mp)))},
        },
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
