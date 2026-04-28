"""Apples-to-apples |S11(f)| from independent simulator dumps.

For each simulator (rfx, OpenEMS, optionally Meep) compute |S11(f)| by:
  1. Loading raw E_z, H_y at the mon_left plane (-50 mm OE, 50 mm rfx).
  2. Computing V = ∫ E_z · sin(πy/a) dA  and  I = ∫ H_y · sin(πy/a) dA
     (TE10 modal projection; same template on every side).
  3. Wave decomposition with analytic TE10 impedance Z_TE(f) =
     ωμ₀/β(f), β = sqrt(k² - kc²).
        a_fwd = 0.5·(V + I·Z),  a_ref = V − a_fwd
        |S11| = |a_ref / a_fwd|

Source-spectrum factor cancels within each simulator (V and I share it).
PEC-short → |S11(f)| should be ~1.0 across all f.

This isolates the per-frequency oscillation bug:
  - All three flat → no bug.
  - rfx oscillates but references flat → bug is rfx-side (mode template,
    aperture weighting, E/H staggering, β/Z choice).
  - References oscillate too → it's a real geometry effect or source-purity
    issue in the test setup itself, not an rfx bug.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Note: codex 2026-04-28 #1 (continuous-coordinate soft-current source) and
# #2 (continuous-coordinate box-integrated probe) were tried as fixes for the
# rfx PEC-short |S11| oscillation and refuted (spread 0.13258 → 0.13145, both
# below threshold).  The implementations are archived under
# ``scripts/spikes/2026-04-28/refuted_codex_archive/`` and documented in
# ``docs/research_notes/2026-04-28_codex_arch_attempts.md``.  This comparator
# now keeps only the canonical cell/TFSF baseline that establishes the
# rfx-vs-OpenEMS-vs-Meep apples-to-apples spread numbers.

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_compare"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
MU_0 = 1.2566370614e-6


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def read_dump(path: str):
    """Read openEMS-schema HDF5 dump; works for both OpenEMS and Meep
    (both write the same schema after 2026-04-28 unification)."""
    with h5py.File(path, "r") as f:
        x = np.asarray(f["Mesh/x"])
        y = np.asarray(f["Mesh/y"])
        z = np.asarray(f["Mesh/z"])
        fd = f["FieldData/FD"]
        idx = sorted(int(k[1:-len("_real")])
                     for k in fd.keys() if k.endswith("_real"))
        sample = np.asarray(fd[f"f{idx[0]}_real"])
        Nc, Nz_arr, Ny_arr, Nx_arr = sample.shape
        if Nc != 3:
            raise RuntimeError(f"{path}: expected 3 components, got {Nc}")
        freqs = np.zeros(len(idx))
        field = np.zeros((len(idx), Nx_arr, Ny_arr, Nz_arr, 3), dtype=complex)
        for k, i in enumerate(idx):
            re = np.asarray(fd[f"f{i}_real"])
            im = np.asarray(fd[f"f{i}_imag"])
            field[k] = np.transpose(re + 1j * im, (3, 2, 1, 0))
            freqs[k] = float(np.asarray(
                fd[f"f{i}_real"].attrs["frequency"]).ravel()[0])
    return {"x": x, "y": y, "z": z, "freqs": freqs, "field": field}


def s11_from_field(E_z: np.ndarray, H_y: np.ndarray, y: np.ndarray,
                   z: np.ndarray, freqs: np.ndarray, a: float) -> np.ndarray:
    """E_z, H_y: (Nf, Ny, Nz) complex; y, z: (Ny,), (Nz,) cell coords (m).
    a: broad-wall length (m). Returns |S11(f)| length Nf."""
    Nf, Ny, Nz = E_z.shape
    if y.size != Ny or z.size != Nz:
        raise ValueError(f"y/z size mismatch: y={y.size}, z={z.size}, "
                         f"Ny={Ny}, Nz={Nz}")
    # TE10 mode template: sin(π·(y - y_min) / a). Use the array's actual y
    # to be robust to centered (-A/2, A/2) vs offset (0, A) frames.
    y_ref = y - y.min()
    e_func = np.sin(np.pi * y_ref / a)
    dy = float(np.abs(y[1] - y[0]) if Ny > 1 else 1.0)
    dz = float(np.abs(z[1] - z[0]) if Nz > 1 else 1.0)
    dA = dy * dz
    weight = e_func[None, :, None]  # broadcast over (Nf, Ny, Nz)
    V = np.sum(E_z * weight, axis=(1, 2)) * dA
    I = np.sum(H_y * weight, axis=(1, 2)) * dA

    omega = 2 * np.pi * freqs
    f_c = C0 / (2 * a)
    k0 = omega / C0
    kc = 2 * np.pi * f_c / C0
    beta = np.sqrt(np.maximum(k0**2 - kc**2, 0.0) + 0j)
    Z = omega * MU_0 / beta

    a_fwd = 0.5 * (V + I * Z)
    a_ref = V - a_fwd
    s11 = np.abs(a_ref / a_fwd)
    return s11, V, I, Z


def s11_from_field_with_method(
    E_z: np.ndarray,
    H_y: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    freqs: np.ndarray,
    a: float,
    *,
    b: float | None = None,
    h_y: np.ndarray | None = None,
    h_z: np.ndarray | None = None,
    method: str = "cell",
    box_quad: int = 4,
) -> np.ndarray:
    """Dispatch dump-derived |S11| extraction by probe recipe."""
    if method == "cell":
        return s11_from_field(E_z, H_y, y, z, freqs, a)
    raise ValueError(
        f"method must be 'cell' (only canonical baseline retained); got {method!r}. "
        "Box-probe (codex #2) is archived under refuted_codex_archive/."
    )


def run_rfx_dump(R: int, plane_name: str = "mon_left",
                 mode_profile: str = "discrete"):
    """Run rfx PEC-short locally, return field arrays at the requested plane.

    mode_profile: "discrete" (rfx canonical Yee-grid TE10) or "analytic"
                  (continuous sin(πy/a) sampled on rfx's template grid).
    """
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cv = _load_cv11()
    dx_m = 1e-3 / R
    plane_x = cv.MON_LEFT_X if plane_name == "mon_left" else cv.PORT_LEFT_X
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
        mode_profile=mode_profile,
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(axis="x", coordinate=plane_x,
                                component=comp, freqs=pf, name=f"{comp}_p")
    res = sim.run(num_periods=200, compute_s_params=False)

    # rfx returns (Nf, Ny, Nz). Build OpenEMS-style mesh y,z for compatibility.
    Ez = np.asarray(res.dft_planes["ez_p"].accumulator)  # (Nf, Ny, Nz)
    Hy = np.asarray(res.dft_planes["hy_p"].accumulator)
    Ny, Nz = Ez.shape[1], Ez.shape[2]
    y = np.linspace(0.0, cv.A_WG, Ny)  # rfx uses (0, A) frame
    z = np.linspace(0.0, cv.B_WG, Nz)
    # Raw Yee H_y lives on the transverse dual mesh.  The legacy cell method
    # intentionally ignores this (matching the historical baseline); the box
    # probe can opt into these coordinates via h_y/h_z.
    dy = float(y[1] - y[0]) if Ny > 1 else cv.A_WG
    dz = float(z[1] - z[0]) if Nz > 1 else cv.B_WG
    h_y = y + 0.5 * dy
    h_z = z + 0.5 * dz
    return {"freqs": np.asarray(freqs), "y": y, "z": z, "E_z": Ez, "H_y": Hy,
            "h_y": h_y, "h_z": h_z, "a": cv.A_WG}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openems-dump-dir", type=str, required=True)
    p.add_argument("--meep-dump-dir", type=str, default=None)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--plane", choices=["mon_left", "source"], default="mon_left")
    p.add_argument("--rfx-mode-profile", choices=["discrete", "analytic"],
                   default="discrete")
    args = p.parse_args()
    # Canonical baseline only.  Codex #1 (soft_current source) and #2 (box
    # probe) were refuted 2026-04-28; archived under refuted_codex_archive/.
    args.probe_method = "cell"
    args.box_quad = 4

    plane_suffix = f"{args.plane}_plane"
    cv = _load_cv11()
    a_m = cv.A_WG

    # OpenEMS
    print(f"\n=== OpenEMS R={args.R}, plane={args.plane} ===", flush=True)
    oe_e = read_dump(os.path.join(
        args.openems_dump_dir,
        f"pec_short_r{args.R}_Efield_{plane_suffix}.h5"))
    oe_h = read_dump(os.path.join(
        args.openems_dump_dir,
        f"pec_short_r{args.R}_Hfield_{plane_suffix}.h5"))
    # Squeeze x to (Nf, Ny, Nz). component idx: Ez=2, Hy=1.
    Ez_oe = oe_e["field"][:, 0, :, :, 2]
    Hy_oe = oe_h["field"][:, 0, :, :, 1]
    s11_oe, V_oe, I_oe, _ = s11_from_field_with_method(
        Ez_oe, Hy_oe, oe_e["y"], oe_e["z"], oe_e["freqs"], a_m,
        b=cv.B_WG, h_y=oe_h["y"], h_z=oe_h["z"],
        method=args.probe_method, box_quad=args.box_quad)
    print(f"openems |S11| min/mean/max = "
          f"{s11_oe.min():.4f}/{s11_oe.mean():.4f}/{s11_oe.max():.4f} "
          f"(spread={s11_oe.max()-s11_oe.min():.4f})")

    # rfx
    print(f"\n=== rfx R={args.R}, plane={args.plane} ===", flush=True)
    rfx = run_rfx_dump(R=args.R, plane_name=args.plane,
                       mode_profile=args.rfx_mode_profile)
    s11_rfx, V_rfx, I_rfx, _ = s11_from_field_with_method(
        rfx["E_z"], rfx["H_y"], rfx["y"], rfx["z"], rfx["freqs"], a_m,
        b=cv.B_WG, h_y=rfx.get("h_y"), h_z=rfx.get("h_z"),
        method=args.probe_method, box_quad=args.box_quad)
    print(f"rfx     |S11| min/mean/max = "
          f"{s11_rfx.min():.4f}/{s11_rfx.mean():.4f}/{s11_rfx.max():.4f} "
          f"(spread={s11_rfx.max()-s11_rfx.min():.4f})")

    # Meep (optional)
    s11_mp = None
    if args.meep_dump_dir:
        print(f"\n=== Meep R={args.R}, plane={args.plane} ===", flush=True)
        try:
            mp_e = read_dump(os.path.join(
                args.meep_dump_dir,
                f"pec_short_r{args.R}_Efield_{plane_suffix}.h5"))
            mp_h = read_dump(os.path.join(
                args.meep_dump_dir,
                f"pec_short_r{args.R}_Hfield_{plane_suffix}.h5"))
            Ez_mp = mp_e["field"][:, 0, :, :, 2]
            Hy_mp = mp_h["field"][:, 0, :, :, 1]
            s11_mp, _, _, _ = s11_from_field_with_method(
                Ez_mp, Hy_mp, mp_e["y"], mp_e["z"], mp_e["freqs"], a_m,
                b=cv.B_WG, h_y=mp_h["y"], h_z=mp_h["z"],
                method=args.probe_method, box_quad=args.box_quad)
            print(f"meep    |S11| min/mean/max = "
                  f"{s11_mp.min():.4f}/{s11_mp.mean():.4f}/{s11_mp.max():.4f} "
                  f"(spread={s11_mp.max()-s11_mp.min():.4f})")
        except FileNotFoundError as exc:
            print(f"  meep dump not available yet: {exc}")

    # Plot
    f_ghz = oe_e["freqs"] / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ideal PEC-short = 1.0")
    ax.plot(f_ghz, s11_oe, "s-", color="#27a", lw=1.6, ms=4,
            label=f"OpenEMS (spread {s11_oe.max()-s11_oe.min():.4f})")
    ax.plot(f_ghz, s11_rfx, "o-", color="#c52", lw=1.6, ms=4,
            label=f"rfx tfsf (spread {s11_rfx.max()-s11_rfx.min():.4f})")
    if s11_mp is not None:
        ax.plot(f_ghz, s11_mp, "^-", color="#0a6", lw=1.6, ms=4,
                label=f"Meep (spread {s11_mp.max()-s11_mp.min():.4f})")
    ax.set_ylabel("|S11(f)| from dumps")
    ax.set_title(f"Apples-to-apples |S11| — same V/I recipe, dump-derived  "
                 f"(R={args.R}, plane={args.plane}, method={args.probe_method})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.plot(f_ghz, s11_oe - 1.0, "s-", color="#27a", label="OpenEMS")
    ax.plot(f_ghz, s11_rfx - 1.0, "o-", color="#c52", label="rfx")
    if s11_mp is not None:
        ax.plot(f_ghz, s11_mp - 1.0, "^-", color="#0a6", label="Meep")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("|S11| − 1.0")
    ax.set_title("Departure from ideal — this is the per-freq oscillation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / f"s11_from_dumps_r{args.R}_{args.plane}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")


if __name__ == "__main__":
    main()
