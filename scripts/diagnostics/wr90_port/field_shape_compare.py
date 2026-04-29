"""Visualize |E_z(y, z)| and |E_z(y)| profiles at f0=10.3 GHz for rfx vs OpenEMS
at the mon_left plane. If field shapes match sin(πy/a), the bug is in phase.
If they differ, the rfx source is imprinting a non-analytic mode shape.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _s11_from_dumps import read_dump, run_rfx_dump, _load_cv11

OUT = Path(__file__).parent / "out_compare"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--openems-dump-dir", required=True)
    p.add_argument("--R", type=int, default=1)
    args = p.parse_args()

    plane_suffix = "mon_left_plane"
    cv = _load_cv11()

    oe = read_dump(os.path.join(
        args.openems_dump_dir,
        f"pec_short_r{args.R}_Efield_{plane_suffix}.h5"))
    Ez_oe = oe["field"][:, 0, :, :, 2]  # (Nf, Ny, Nz) E_z
    f_target_idx = int(np.argmin(np.abs(oe["freqs"] - 10.3e9)))
    f_target = oe["freqs"][f_target_idx] / 1e9
    print(f"target freq: {f_target:.3f} GHz")

    rfx = run_rfx_dump(R=args.R, plane_name="mon_left")
    Ez_rfx = rfx["E_z"]  # (Nf, Ny, Nz)
    f_rfx_idx = int(np.argmin(np.abs(rfx["freqs"] - 10.3e9)))

    # Slice at center z (z = b/2)
    Nz_oe = Ez_oe.shape[2]
    z_mid_oe = Nz_oe // 2
    Ez_oe_slice = Ez_oe[f_target_idx, :, z_mid_oe]
    Nz_rfx = Ez_rfx.shape[2]
    z_mid_rfx = Nz_rfx // 2
    Ez_rfx_slice = Ez_rfx[f_rfx_idx, :, z_mid_rfx]

    # Normalize each by its own peak |.| so shapes are comparable
    Ez_oe_norm = Ez_oe_slice / np.max(np.abs(Ez_oe_slice))
    Ez_rfx_norm = Ez_rfx_slice / np.max(np.abs(Ez_rfx_slice))

    # Analytic sin reference
    a = cv.A_WG
    y_oe = oe["y"]; y_oe_ref = y_oe - y_oe.min()
    y_rfx = rfx["y"]; y_rfx_ref = y_rfx - y_rfx.min()
    sin_oe = np.sin(np.pi * y_oe_ref / a)
    sin_rfx = np.sin(np.pi * y_rfx_ref / a)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(y_oe_ref * 1e3, np.abs(Ez_oe_norm), "s-", label="OpenEMS |E_z|", color="#27a")
    ax.plot(y_oe_ref * 1e3, sin_oe, "k--", label="analytic sin(πy/a)", alpha=0.7)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("|E_z| (normalized)")
    ax.set_title(f"OpenEMS |E_z(y)| at z=b/2, f={f_target:.2f} GHz")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(y_rfx_ref * 1e3, np.abs(Ez_rfx_norm), "o-", label="rfx |E_z|", color="#c52")
    ax.plot(y_rfx_ref * 1e3, sin_rfx, "k--", label="analytic sin(πy/a)", alpha=0.7)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("|E_z| (normalized)")
    ax.set_title(f"rfx |E_z(y)| at z=b/2, f={f_target:.2f} GHz")
    ax.legend(); ax.grid(alpha=0.3)

    # Phase plot — arg(E_z(y)) relative to peak. PEC short standing wave →
    # arg should be ~constant across y (mode is real-valued), with ±π flips
    # only at sign changes of sin.
    ax = axes[1, 0]
    ph_oe = np.degrees(np.angle(Ez_oe_norm * np.exp(-1j * np.angle(Ez_oe_norm[Nz_oe // 2]))))  # ref to mid
    ax.plot(y_oe_ref * 1e3, ph_oe, "s-", color="#27a", label="OpenEMS")
    ax.set_xlabel("y [mm]"); ax.set_ylabel("arg(E_z) − arg(center) [deg]")
    ax.set_title("OpenEMS arg(E_z(y))")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ph_rfx = np.degrees(np.angle(Ez_rfx_norm * np.exp(-1j * np.angle(Ez_rfx_norm[Nz_rfx // 2]))))
    ax.plot(y_rfx_ref * 1e3, ph_rfx, "o-", color="#c52", label="rfx")
    ax.set_xlabel("y [mm]"); ax.set_ylabel("arg(E_z) − arg(center) [deg]")
    ax.set_title("rfx arg(E_z(y))")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = OUT / f"field_shape_compare_r{args.R}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    # Also numerical: report rms deviation between rfx |E_z(y)| and analytic
    rms_oe = np.sqrt(np.mean((np.abs(Ez_oe_norm) - sin_oe)**2))
    rms_rfx = np.sqrt(np.mean((np.abs(Ez_rfx_norm) - sin_rfx)**2))
    print(f"\n=== shape RMS deviation from analytic sin(πy/a) ===")
    print(f"OpenEMS: {rms_oe:.4f}")
    print(f"rfx    : {rms_rfx:.4f}")


if __name__ == "__main__":
    main()
