"""Cross-tool audit: which dump tools need the Yee half-step correction?

After the 2026-04-28 finding that rfx's dump-derived |S11| recipe
needed exp(+jω·dt/2) on H_y to match the production extractor's
``_co_located_current_spectrum``, the open question is whether the
same correction is needed (or already applied internally) by
OpenEMS and Meep dump writers.

Test: run a (3 tools) × (correction on/off) matrix on the same
geometry (PEC-short, R=1, mon_left plane) and inspect the
|S11(f)| spreads.

For each tool, the "correct" answer is the one closer to the ideal
|S11|=1 with smaller per-frequency spread. Whichever toggle state
gives that answer is the convention that tool's dump tool uses.

Outputs the full toggle matrix and writes a per-tool conclusion to
JSON.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_cross_tool_audit"
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


def _s11_recipe(E_z, H_y, y, z, freqs, a):
    Ny, Nz = E_z.shape[1], E_z.shape[2]
    # Robust to (0, A) and (-A/2, A/2) frames — same convention as
    # scripts/diagnostics/wr90_port/s11_from_dumps.py::s11_from_field.
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
    s11 = np.abs(a_ref / np.where(np.abs(a_fwd) > 1e-30, a_fwd, 1e-30))
    return s11


def _read_h5_dump(path):
    """Read OpenEMS / Meep frequency-domain HDF5 field dump.

    Reuses the schema unifier from
    scripts/diagnostics/wr90_port/s11_from_dumps.py::read_dump.
    Returns dict with freqs, mesh axes, and (Nf, Nx, Ny, Nz, 3) field array.
    """
    spec = importlib.util.spec_from_file_location(
        "s11_from_dumps", REPO / "scripts" / "diagnostics" / "wr90_port"
        / "s11_from_dumps.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_dump(path)


def run_rfx_dump():
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
    if hasattr(sim, "_grid") and hasattr(sim._grid, "dt"):
        dt_sim = float(sim._grid.dt)
    else:
        dt_sim = 0.5 * dx_m / C0
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)
    z = np.linspace(0.0, cv.B_WG, Nz)
    return {"freqs": np.asarray(freqs), "y": y, "z": z,
            "E_z": Ez, "H_y_raw": Hy, "dt": dt_sim, "a": cv.A_WG}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openems-dump-dir", type=str,
                   default=str(REPO / "scripts" / "spikes" / "2026-04-28"
                               / "out_openems_soft_current" / "dumps_r1"))
    p.add_argument("--meep-dump-dir", type=str,
                   default="/root/workspace/byungkwan-workspace/research/"
                           "microwave-energy/results/"
                           "wr90_meep_field_dumps_20260428T0311Z/dumps_r1")
    p.add_argument("--R", type=int, default=1)
    args = p.parse_args()

    cv = _load_cv11()
    a_m = cv.A_WG

    print(f"=== Cross-tool half-step correction audit (R={args.R}) ===\n")

    # ---- rfx (run live; save dt and raw H_y) -----------------------------
    print("Running rfx PEC-short ...")
    rfx = run_rfx_dump()
    print(f"  rfx dt = {rfx['dt']*1e12:.3f} ps")

    # ---- OpenEMS dump ----------------------------------------------------
    print("Loading OpenEMS dumps ...")
    oe_e = _read_h5_dump(os.path.join(
        args.openems_dump_dir,
        f"pec_short_r{args.R}_Efield_mon_left_plane.h5"))
    oe_h = _read_h5_dump(os.path.join(
        args.openems_dump_dir,
        f"pec_short_r{args.R}_Hfield_mon_left_plane.h5"))
    Ez_oe = oe_e["field"][:, 0, :, :, 2]
    Hy_oe = oe_h["field"][:, 0, :, :, 1]
    print(f"  OpenEMS E shape {Ez_oe.shape}, freqs[0]={oe_e['freqs'][0]/1e9:.2f} GHz")

    # ---- Meep dump (optional) -------------------------------------------
    have_meep = os.path.exists(args.meep_dump_dir)
    if have_meep:
        print("Loading Meep dumps ...")
        try:
            mp_e = _read_h5_dump(os.path.join(
                args.meep_dump_dir,
                f"pec_short_r{args.R}_Efield_mon_left_plane.h5"))
            mp_h = _read_h5_dump(os.path.join(
                args.meep_dump_dir,
                f"pec_short_r{args.R}_Hfield_mon_left_plane.h5"))
            Ez_mp = mp_e["field"][:, 0, :, :, 2]
            Hy_mp = mp_h["field"][:, 0, :, :, 1]
            print(f"  Meep E shape {Ez_mp.shape}")
        except (FileNotFoundError, OSError):
            have_meep = False
            print("  Meep dumps not available — skipping")

    # ---- Build correction factors per tool ------------------------------
    # rfx: dt_sim from sim object.
    # OE / Meep: dt is implementation-defined; we don't know without
    # inspecting their source. As a first pass, use the *same* phase
    # factor magnitude as rfx (= ω·dt_rfx/2). If their dt actually
    # differs, the cancellation won't be clean; but it's enough to see
    # whether the correction direction matters for them at all.
    omega = 2.0 * np.pi * rfx["freqs"]
    rfx_corr = np.exp(+1j * omega * (0.5 * rfx["dt"]))
    # OE/Meep: try the same magnitude as rfx, sign +
    ref_corr = np.exp(+1j * omega * (0.5 * rfx["dt"]))

    # ---- Toggle matrix --------------------------------------------------
    rows = []
    # rfx mesh y, z
    Ny, Nz = rfx["E_z"].shape[1], rfx["E_z"].shape[2]
    y_rfx = np.linspace(0.0, cv.A_WG, Ny)
    z_rfx = np.linspace(0.0, cv.B_WG, Nz)

    # rfx
    for label, Hy_use in [
        ("rfx no-correction", rfx["H_y_raw"]),
        ("rfx + half-step", rfx["H_y_raw"] * rfx_corr[:, None, None]),
    ]:
        s11 = _s11_recipe(rfx["E_z"], Hy_use, y_rfx, z_rfx, rfx["freqs"], a_m)
        rows.append(("rfx", label, s11))

    # OpenEMS
    for label, Hy_use in [
        ("OE no-correction", Hy_oe),
        ("OE + half-step", Hy_oe * ref_corr[:, None, None]),
    ]:
        s11 = _s11_recipe(Ez_oe, Hy_use, oe_e["y"], oe_e["z"], oe_e["freqs"], a_m)
        rows.append(("openems", label, s11))

    # Meep
    if have_meep:
        for label, Hy_use in [
            ("Meep no-correction", Hy_mp),
            ("Meep + half-step", Hy_mp * ref_corr[:, None, None]),
        ]:
            s11 = _s11_recipe(Ez_mp, Hy_use, mp_e["y"], mp_e["z"],
                              mp_e["freqs"], a_m)
            rows.append(("meep", label, s11))

    # ---- Print matrix ---------------------------------------------------
    print("\n%-7s | %-22s | %s | %s | %s | %s"
          % ("tool", "variant", "min   ", "mean  ", "max   ", "spread"))
    print("-" * 80)
    for tool, label, s11 in rows:
        print(f"{tool:<7s} | {label:<22s} | {s11.min():.4f} | "
              f"{s11.mean():.4f} | {s11.max():.4f} | "
              f"{s11.max()-s11.min():.4f}")

    # ---- Per-tool conclusion --------------------------------------------
    print("\nper-tool verdict:")
    summary = {}
    for tool in ("rfx", "openems", "meep"):
        tool_rows = [r for r in rows if r[0] == tool]
        if len(tool_rows) < 2:
            continue
        no_corr = next(r for r in tool_rows if "no-correction" in r[1])
        corr = next(r for r in tool_rows if "half-step" in r[1])
        spread_no = float(no_corr[2].max() - no_corr[2].min())
        spread_yes = float(corr[2].max() - corr[2].min())
        if spread_yes < 0.5 * spread_no:
            verdict = "NEEDS correction (correction reduces spread >2x)"
        elif spread_no < 0.5 * spread_yes:
            verdict = "ALREADY applies correction internally (extra correction makes it worse)"
        else:
            verdict = "AMBIGUOUS (spread similar on/off — correction is small effect at this resolution)"
        summary[tool] = {
            "spread_no_correction": spread_no,
            "spread_with_correction": spread_yes,
            "verdict": verdict,
        }
        print(f"  {tool:<8s}: no-corr {spread_no:.4f}, +corr {spread_yes:.4f} "
              f"-> {verdict}")

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short")
    f_ghz = rfx["freqs"] / 1e9
    cmap_map = {"rfx": "tab:orange", "openems": "tab:blue", "meep": "tab:green"}
    for tool, label, s11 in rows:
        ls = "--" if "no-correction" in label else "-"
        col = cmap_map[tool]
        ax.plot(f_ghz, s11, ls, color=col, lw=1.4, ms=3,
                label=f"{label}  (spread {s11.max()-s11.min():.4f})")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11(f)|")
    ax.set_title("Cross-tool Yee half-step correction audit "
                 f"(WR-90 PEC-short, R={args.R})")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / f"cross_tool_half_step_audit_R{args.R}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")

    out_json = OUT / f"cross_tool_half_step_audit_R{args.R}.json"
    out_json.write_text(json.dumps({
        "freqs_hz": rfx["freqs"].tolist(),
        "rfx_dt": rfx["dt"],
        "rows": [
            {"tool": t, "label": l, "s11": s11.tolist(),
             "spread": float(s11.max() - s11.min())}
            for t, l, s11 in rows
        ],
        "summary": summary,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
