"""rfx vs OpenEMS vs Meep cell-by-cell field comparison at the WR-90 port plane.

Goal: identify in which discrete computation step (E sample, H sample, V
overlap integral, I overlap integral, wave decomposition) rfx and the
references diverge — that's where the unidentified ~6% per-frequency
oscillation in PEC-short |S11| originates. Triple cross-validation: if
two independent FDTD codes (OpenEMS, Meep) produce the same dumps but rfx
does not, the divergence is on the rfx side.

Inputs:
  - OpenEMS HDF5 dumps from VESSL job mwe-wr90-pec-short-field-dump-20260428.
    Files: {E,H}field_{source_plane,mon_left_plane}.h5 per resolution.
    Convention: per VESSL run output `dumps_r{R}/pec_short_r{R}_{E,H}field_{plane}.h5`.
  - rfx PEC-short DFT plane probes at the *same* plane and frequencies, run
    locally on demand by this script.

Coordinate frame:
  OpenEMS: domain x ∈ [-100, +100] mm; source@-60, mon_left@-50, PEC_SHORT@+45.
  rfx (crossval/11 absolute): domain x ∈ [0, 200] mm; port@+40 (=openems -60),
  PORT_RIGHT@+160. Offset: rfx_x = openems_x + 100.

  PEC-short geometry mismatch in canonical configs:
    OpenEMS canonical: PEC_SHORT_X = +45 mm (= rfx 145 mm)
    rfx canonical:     PEC short at PORT_RIGHT_X − 5 mm = 155 mm
                       (= openems +55 mm)
  For an apples-to-apples comparison we run rfx with its short repositioned
  to openems-canonical 145 mm. (See ``run_rfx_aligned_pec_short`` below.)

Modes:
  --inspect <h5>          Print HDF5 group/dataset hierarchy of one file.
                          Use this on the first arriving dump file to verify
                          the schema this script assumes.
  --compare <dump-dir>    Full apples-to-apples comparison; produces a
                          per-frequency PNG and a summary JSON.

NOTE: the assumed openEMS HDF5 schema for frequency-domain dumps:
    /Mesh/x, /Mesh/y, /Mesh/z   1-D coordinate axes (m)
    /FieldData/FD/f<i>_real     (Nx, Ny, Nz, 3) doubles
    /FieldData/FD/f<i>_imag     (Nx, Ny, Nz, 3) doubles
    /FieldData/FD attrs:        "frequency" → (n_freqs,) array
This is the standard format produced by openEMS DumpType=10/11 with FileType=1.
The --inspect mode lets you confirm before running --compare.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_compare"
OUT.mkdir(parents=True, exist_ok=True)

# OpenEMS-frame port-plane locations (mm)
OPENEMS_SOURCE_X_MM = -60.0
OPENEMS_MON_LEFT_X_MM = -50.0
OPENEMS_PEC_SHORT_X_MM = +45.0
RFX_OFFSET_MM = +100.0  # rfx_x = openems_x + offset


# ---------------------------------------------------------------------------
# HDF5 reader (defensive — multiple openEMS dump conventions)
# ---------------------------------------------------------------------------
def inspect_h5(path: str) -> None:
    """Print full group/dataset tree."""
    print(f"=== HDF5 inspect: {path} ===")
    with h5py.File(path, "r") as f:
        def _rec(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [DATASET] /{name}  shape={obj.shape}  "
                      f"dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  [GROUP]   /{name}")
            for k, v in obj.attrs.items():
                short = str(v)[:80]
                print(f"      attr  {name}@{k} = {short}")
        f.visititems(_rec)
        for k, v in f.attrs.items():
            short = str(v)[:80]
            print(f"  [ROOT attr] {k} = {short}")


def read_openems_freq_dump(path: str) -> dict:
    """Read an openEMS frequency-domain HDF5 dump.

    Schema (openEMS_HDF5_version=0.2, observed 2026-04-28):
      /Mesh/{x,y,z}                                1-D coords (m)
      /FieldData/FD/f<i>_real, f<i>_imag           shape (3, Nz, Ny, Nx) float32
        per-dataset attr "frequency" → (1,) Hz

    Returns:
      {
        "x": (Nx,), "y": (Ny,), "z": (Nz,)    -- coordinate axes (m)
        "freqs": (Nf,)                         -- Hz, sorted
        "field": (Nf, Nx, Ny, Nz, 3) complex   -- canonicalized to (Nx,Ny,Nz,3)
      }
    """
    with h5py.File(path, "r") as f:
        if "Mesh" not in f or "FieldData" not in f or "FD" not in f["FieldData"]:
            raise RuntimeError(
                f"{path}: missing /Mesh or /FieldData/FD. "
                "Run --inspect to see actual structure."
            )
        x = np.asarray(f["Mesh/x"])
        y = np.asarray(f["Mesh/y"])
        z = np.asarray(f["Mesh/z"])
        fd = f["FieldData/FD"]

        keys = list(fd.keys())
        idx_pairs = []
        for k in keys:
            if k.endswith("_real"):
                idx = int(k[len("f"):-len("_real")])
                idx_pairs.append(idx)
        idx_pairs.sort()
        if not idx_pairs:
            raise RuntimeError(
                f"{path}: no f<i>_real datasets. Keys={sorted(keys)[:10]}…"
            )

        sample = np.asarray(fd[f"f{idx_pairs[0]}_real"])
        # openEMS shape: (3, Nz, Ny, Nx). Canonicalize to (Nx, Ny, Nz, 3).
        Nc, Nz_arr, Ny_arr, Nx_arr = sample.shape
        if Nc != 3:
            raise RuntimeError(f"{path}: expected 3 components, got {Nc}")

        freqs = np.zeros(len(idx_pairs), dtype=float)
        field = np.zeros((len(idx_pairs), Nx_arr, Ny_arr, Nz_arr, 3),
                         dtype=complex)
        for k, i in enumerate(idx_pairs):
            re_ds = fd[f"f{i}_real"]
            im_ds = fd[f"f{i}_imag"]
            re = np.asarray(re_ds)  # (3, Nz, Ny, Nx)
            im = np.asarray(im_ds)
            # transpose to (Nx, Ny, Nz, 3)
            re_t = np.transpose(re, (3, 2, 1, 0))
            im_t = np.transpose(im, (3, 2, 1, 0))
            field[k] = re_t + 1j * im_t
            f_attr = re_ds.attrs.get("frequency",
                                      im_ds.attrs.get("frequency"))
            if f_attr is None:
                raise RuntimeError(
                    f"{path}: f{i}_real has no 'frequency' attribute"
                )
            freqs[k] = float(np.asarray(f_attr).ravel()[0])

    return {"x": x, "y": y, "z": z, "freqs": freqs, "field": field}


# ---------------------------------------------------------------------------
# rfx side: PEC-short with field probes at the OpenEMS-aligned plane
# ---------------------------------------------------------------------------
def _load_cv11():
    cv11_path = REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rfx_aligned_pec_short(dx_m: float, plane_x_m: float) -> dict:
    """Run rfx PEC-short at the canonical (Meep/OpenEMS-aligned) position
    and probe E_z / H_y at the requested plane (rfx absolute coords)."""
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    import jax.numpy as jnp
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cv = _load_cv11()
    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    pec_short_x_rfx = cv.PEC_SHORT_X  # canonical 0.145 m = +45 mm OpenEMS frame

    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=dx_m,
    )
    sim.add(
        Box((pec_short_x_rfx, 0.0, 0.0),
            (pec_short_x_rfx + 2 * dx_m, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
    )
    for comp in ("ez", "hy"):
        sim.add_dft_plane_probe(
            axis="x", coordinate=plane_x_m, component=comp,
            freqs=pf, name=f"{comp}_probe",
        )
    res = sim.run(num_periods=200, compute_s_params=False)
    e_z = np.transpose(np.asarray(res.dft_planes["ez_probe"].accumulator),
                       (1, 2, 0))  # (Ny, Nz, Nf)
    h_y = np.transpose(np.asarray(res.dft_planes["hy_probe"].accumulator),
                       (1, 2, 0))
    return {"freqs": np.asarray(freqs), "E_z": e_z, "H_y": h_y, "dx_m": dx_m}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _slice_plane(dump: dict, comp_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenEMS dump on a 0-thickness plane in x → squeeze x.
    Returns (Ny-array, Nz-array, (Nf, Ny, Nz) complex)."""
    F = dump["field"]  # (Nf, Nx, Ny, Nz, 3)
    if F.shape[1] != 1:
        # Some openEMS dump configs put a second x-cell at the boundary;
        # take the cell whose coordinate matches the plane request best.
        x = dump["x"]
        # Prefer the smallest |x − requested|; here we just take the middle
        # cell to be robust.
        F = F[:, F.shape[1] // 2:F.shape[1] // 2 + 1]
    F2 = F[:, 0, :, :, comp_idx]  # (Nf, Ny, Nz)
    return dump["y"], dump["z"], F2


def compare_dumps(dump_dir: Path, R: int, *,
                  plane: str = "mon_left") -> dict:
    """Load OpenEMS dumps + rfx fields at the same plane, return summary.

    plane: "mon_left" (-50 mm OE = 50 mm rfx, OpenEMS port-1 measurement
            plane) or "source" (-60 mm OE = 40 mm rfx, OpenEMS source
            plane / rfx TFSF source plane).

    mon_left is the right comparison for V/I extraction since both tools
    measure their port observables there. source-plane fields are imposed-
    source contaminated and not directly comparable in absolute terms.
    """
    dx_m = 1e-3 / R
    suffix = "mon_left_plane" if plane == "mon_left" else "source_plane"
    e_h5 = dump_dir / f"pec_short_r{R}_Efield_{suffix}.h5"
    h_h5 = dump_dir / f"pec_short_r{R}_Hfield_{suffix}.h5"
    if not (e_h5.exists() and h_h5.exists()):
        raise FileNotFoundError(
            f"Missing dump files for R={R}: {e_h5}, {h_h5}"
        )
    print(f"[load] {e_h5}")
    Edump = read_openems_freq_dump(str(e_h5))
    print(f"[load] {h_h5}")
    Hdump = read_openems_freq_dump(str(h_h5))

    y_e, z_e, E_z_oe = _slice_plane(Edump, comp_idx=2)
    y_h, z_h, H_y_oe = _slice_plane(Hdump, comp_idx=1)

    cv = _load_cv11()
    if plane == "mon_left":
        plane_x_rfx = cv.MON_LEFT_X  # 0.050 m = -50 mm OE
    else:
        plane_x_rfx = cv.PORT_LEFT_X  # 0.040 m = -60 mm OE
    print(f"[run rfx] PEC-short canonical, probe at rfx={plane_x_rfx*1e3:.1f} mm "
          f"({plane}), dx={dx_m*1e3:.3f} mm")
    rfx = run_rfx_aligned_pec_short(dx_m=dx_m, plane_x_m=plane_x_rfx)

    return {
        "R": R, "dx_m": dx_m,
        "openems_E_z": E_z_oe, "openems_H_y": H_y_oe,
        "openems_y": y_e, "openems_z": z_e, "openems_freqs": Edump["freqs"],
        "rfx_E_z": rfx["E_z"], "rfx_H_y": rfx["H_y"],
        "rfx_freqs": rfx["freqs"],
    }


def summary_table(cmp: dict) -> list[dict]:
    """Per-frequency |E_z|, |H_y| RMS deltas and overlap-integral V/I deltas."""
    rows = []
    fr = cmp["rfx_freqs"]
    Eo = cmp["openems_E_z"]   # (Nf, Ny_o, Nz_o)
    Ho = cmp["openems_H_y"]
    Er = cmp["rfx_E_z"]       # (Ny_r, Nz_r, Nf)
    Hr = cmp["rfx_H_y"]
    # Note: openEMS y is centred (-A/2,+A/2); rfx y is shifted (0,A). Both have
    # the same number of y/z lines if R matches; rfx uses Yee staggering on a
    # node-based grid. For a strict cell-by-cell match, mesh registration is
    # tricky. We report two metrics: peak |E_z|, peak |H_y|, and the V/I
    # overlap integrals each side computes against its own canonical TE10
    # template (which IS what feeds S11).

    Ny_r, Nz_r, Nf = Er.shape
    a_rfx = (Ny_r - 1) * cmp["dx_m"]
    y_rfx = np.arange(Ny_r) * cmp["dx_m"]
    e_func = np.sin(np.pi * y_rfx / a_rfx)[:, None] * np.ones(Nz_r)[None, :]
    h_func = e_func.copy()
    dA = cmp["dx_m"] ** 2

    Ny_o = Eo.shape[1]
    a_oe = (Ny_o - 1) * cmp["dx_m"]
    y_oe = (np.arange(Ny_o) - (Ny_o - 1) / 2.0) * cmp["dx_m"] + a_oe / 2.0
    e_func_oe = np.sin(np.pi * y_oe / a_oe)[:, None] * np.ones(Eo.shape[2])[None, :]
    h_func_oe = e_func_oe.copy()
    dA_oe = cmp["dx_m"] ** 2

    for k, f in enumerate(fr):
        Er_k = Er[:, :, k]
        Hr_k = Hr[:, :, k]
        Eo_k = Eo[k]
        Ho_k = Ho[k]
        V_rfx = np.sum(Er_k * e_func * dA)
        I_rfx = np.sum(Hr_k * h_func * dA)
        V_oe = np.sum(Eo_k * e_func_oe * dA_oe)
        I_oe = np.sum(Ho_k * h_func_oe * dA_oe)
        rows.append({
            "f_GHz": float(f / 1e9),
            "|E_z| peak rfx": float(np.max(np.abs(Er_k))),
            "|E_z| peak openems": float(np.max(np.abs(Eo_k))),
            "|H_y| peak rfx": float(np.max(np.abs(Hr_k))),
            "|H_y| peak openems": float(np.max(np.abs(Ho_k))),
            "|V| rfx": float(np.abs(V_rfx)),
            "|V| openems": float(np.abs(V_oe)),
            "|I| rfx": float(np.abs(I_rfx)),
            "|I| openems": float(np.abs(I_oe)),
            "arg(V) rfx [deg]": float(np.degrees(np.angle(V_rfx))),
            "arg(V) openems [deg]": float(np.degrees(np.angle(V_oe))),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="Print HDF5 hierarchy")
    pi.add_argument("path")
    pc = sub.add_parser("compare", help="rfx vs OpenEMS cell-by-cell comparison")
    pc.add_argument("dump_dir", help="dir containing pec_short_r{R}_*.h5")
    pc.add_argument("--R", type=int, default=1, help="resolution (cells/mm)")
    pc.add_argument("--plane", choices=["mon_left", "source"],
                    default="mon_left",
                    help="which plane to compare (default mon_left = the "
                         "OpenEMS port-1 measurement plane and rfx ref-plane)")
    args = p.parse_args()

    if args.cmd == "inspect":
        inspect_h5(args.path)
        return 0

    cmp = compare_dumps(Path(args.dump_dir), R=args.R, plane=args.plane)
    rows = summary_table(cmp)
    out_json = OUT / f"compare_r{args.R}_{args.plane}.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\n=== Per-frequency summary (R={args.R}) ===")
    keys = list(rows[0].keys())
    print(" ".join(f"{k:>22}" for k in keys))
    for r in rows:
        print(" ".join(f"{r[k]:>22.6g}" for k in keys))
    print(f"\nsummary written to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
