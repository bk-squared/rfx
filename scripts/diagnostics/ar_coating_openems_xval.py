"""OpenEMS reference for the multilayer AR coating demo — DEFERRED.

This script was prototyped on 2026-05-05 to add an OpenEMS leg to the
``examples/inverse_design/multilayer_ar_coating.py`` cross-validation
(rfx-vs-TMM-vs-OpenEMS).  It is **not** wired into the demo.

Why deferred:
  - OpenEMS Python bindings (v0.0.35) do not expose periodic boundary
    conditions through ``SetBoundaryCond`` — only PEC, PMC, MUR, PML-8
    are available.  PEC and PMC each kill at least one cross-field
    component of an Ey-polarised normal-incidence plane wave (PEC kills
    tangential H_z, PMC kills normal E_y).  MUR is an absorbing BC and
    interferes with the source plane (openEMS warns: "Excitation inside
    the Mur-ABC found!!! Mur-ABC will be switched on after excitation
    is done").
  - Workarounds (large-cross-section rectangular waveguide far above
    cutoff, or a custom subprocess running openEMS with periodic BC
    via XML config) are non-trivial and exceed the scope of this demo.

What works locally (verified):
  - Geometry / mesh build via CSXCAD: ✓
  - Soft E-field excitation (``exc_type=1``) with non-zero x-extent: ✓
  - HDF5 field-dump readback via h5py: ✓
  - Local sim with PMC-only BC: source fires, energy is conserved BUT
    Ey is killed at the PMC y-boundaries → wave does not propagate in x.
  - Local sim with MUR transverse: source delayed until step 754, by
    which time the structure response has decayed.

Recommended replacement: launch the equivalent crossval on VESSL via
the ``microwave-energy`` infra (``research/microwave-energy/openems_simulation/``
already has working OpenEMS pipelines for related geometries — cv05
and cv06b use this pattern).  Use periodic BC via the openEMS XML
config or a wide rectangular waveguide above all cutoffs in X-band.

The script body is preserved so the next session can resume from a
known starting point.  See the in-line comments for the failure modes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

# Fix numpy deprecation in openEMS v0.0.35 (must be BEFORE openEMS import).
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "mat"):
    np.mat = np.matrix

try:
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS
    from openEMS.physical_constants import C0
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: openEMS Python bindings not importable ({exc}).", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Geometry — must match examples/inverse_design/multilayer_ar_coating.py
# ---------------------------------------------------------------------------
EPS_SUB = 12.0
N_LAYERS = 3
F_LO, F_HI = 8e9, 12e9
F0 = 0.5 * (F_LO + F_HI)
F_MAX = 15e9
LAMBDA0 = C0 / F0
N_REF = EPS_SUB ** (0.5 / (N_LAYERS + 1))
LAYER_THK = LAMBDA0 / (4.0 * N_REF)         # ≈ 5.494 mm
DX = 0.5e-3                                  # 0.5 mm
UNIT = 1e-3                                  # CSXCAD unit = mm

# x-axis layout (metres → mm via /UNIT)
X_PML_PAD = 6e-3
X_SRC = 10e-3
X_REFL = 22e-3
X_DESIGN_LO = 35e-3
X_DESIGN_HI = X_DESIGN_LO + N_LAYERS * LAYER_THK
X_TRANS = X_DESIGN_HI + 8e-3
LX = X_TRANS + 18e-3                         # extends through right PML

# Default εr triple = analytic TMM optimum from the rfx demo
DEFAULT_LAYER_EPS = (1.943, 4.652, 4.720)


def _build_csx(layer_eps: tuple[float, float, float] | None) -> ContinuousStructure:
    """Build CSX geometry. layer_eps=None → vacuum reference."""
    csx = ContinuousStructure()
    grid = csx.GetGrid()
    grid.SetDeltaUnit(UNIT)

    # x-mesh: regular dx through the structure
    nx = int(round(LX / DX)) + 1
    grid.AddLine("x", np.linspace(0.0, LX / UNIT, nx))
    # y, z meshes — openEMS requires ≥3 disc-lines per axis (2 cells), with
    # PMC walls outside.  Two cells in y, z is enough since the wave is
    # transverse-uniform (PMC = magnetic mirror = infinite extent).
    grid.AddLine("y", [-DX / UNIT, 0.0, DX / UNIT])
    grid.AddLine("z", [-DX / UNIT, 0.0, DX / UNIT])

    if layer_eps is not None:
        # 3 design layers (one box per layer)
        for i, eps_r in enumerate(layer_eps):
            x_lo = (X_DESIGN_LO + i * LAYER_THK) / UNIT
            x_hi = (X_DESIGN_LO + (i + 1) * LAYER_THK) / UNIT
            mat = csx.AddMaterial(f"layer{i}", epsilon=float(eps_r))
            mat.AddBox(start=[x_lo, -DX / UNIT, -DX / UNIT],
                       stop=[x_hi,  DX / UNIT,  DX / UNIT])
        # Substrate: from design end through end of domain (= into PML)
        sub = csx.AddMaterial("substrate", epsilon=EPS_SUB)
        sub.AddBox(start=[X_DESIGN_HI / UNIT, -DX / UNIT, -DX / UNIT],
                   stop=[LX / UNIT,            DX / UNIT,  DX / UNIT])

    # Soft E-field current excitation along Ey on a 1-cell-thick yz-slab at
    # x=X_SRC.  exc_type=1 (E_SOFT) adds the source to the Ey edges rather
    # than overwriting them, so it both emits a propagating wave AND lets
    # reflected waves pass through the source plane unattenuated.
    exc = csx.AddExcitation("planewave_src", 1, [0.0, 1.0, 0.0])
    exc.AddBox(start=[X_SRC / UNIT,         -DX / UNIT, -DX / UNIT],
               stop=[(X_SRC + DX) / UNIT,    DX / UNIT,  DX / UNIT])

    # Field dump on a yz-plane at the refl probe — gives us Ey directly.
    # File-type 0 = vtk, 1 = HDF5; dump-type 0 = E-field; we read the HDF5.
    dump = csx.AddDump("dump_refl", dump_type=0, file_type=1, sub_sampling=[1, 1, 1])
    dump.AddBox(start=[X_REFL / UNIT, -DX / UNIT, -DX / UNIT],
                stop=[(X_REFL + DX) / UNIT, DX / UNIT,  DX / UNIT])
    return csx


def _run_openems(layer_eps, sim_dir: str, n_periods: float = 35.0) -> dict:
    """One openEMS run; returns dict with time series at refl probe."""
    # NrTS up to 60000 (~58 ns), EndCriteria 1e-5 (-50 dB) so the response
    # rings out cleanly inside the substrate before stopping.  Source pulse
    # is short (~7e-10s); the rest is structure response we need for the
    # |R(f)| spectrum across the X-band.
    fdtd = openEMS(NrTS=60000, EndCriteria=1e-5)
    # Probes record at the Nyquist rate of the excitation by default; set
    # an oversampling factor so we get enough time-samples to FFT cleanly
    # in [8, 12] GHz.  factor=10 → ~10 samples per period at 10 GHz.
    fdtd.SetOverSampling(10)
    # SetGaussExcite: f0=center, fc=halfwidth.  Centred at F0=10GHz with fc=4GHz
    # gives a Gaussian envelope spanning roughly 6–14 GHz at -10 dB.
    fdtd.SetGaussExcite(f0=F0, fc=4e9)
    # Boundaries: PML on x (open propagation), MUR absorbing on y, z.
    # openEMS doesn't expose periodic BCs through the Python API; PEC/PMC
    # always kill at least one cross-field component of an Ey-polarised
    # propagating wave (PEC kills tangential H_z, PMC kills tangential E_y),
    # so they cannot be used.  MUR is a 1st-order absorbing BC that simply
    # passes through a transverse-uniform plane wave (no transverse
    # propagation → nothing to absorb).
    fdtd.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "MUR", "MUR"])

    csx = _build_csx(layer_eps)
    fdtd.SetCSX(csx)

    if os.path.isdir(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)

    t0 = time.time()
    fdtd.Run(sim_dir, verbose=2, cleanup=False)
    elapsed = time.time() - t0

    # Read field dump (HDF5) at the refl probe yz-plane.
    import h5py
    h5_candidates = [p for p in os.listdir(sim_dir)
                     if p.startswith("dump_refl") and p.endswith(".h5")]
    if not h5_candidates:
        raise RuntimeError(f"openEMS did not write the refl HDF5 dump in {sim_dir}; "
                           f"contents: {os.listdir(sim_dir)}")
    h5_path = os.path.join(sim_dir, h5_candidates[0])
    with h5py.File(h5_path, "r") as f:
        # Schema: /FieldData/TD/<step_index> → (3, nx, ny, nz) E-vector
        td = f["/FieldData/TD"]
        steps = sorted(td.keys(), key=int)
        # Time axis in attrs.  Look for "time" attribute on each step or root.
        times = []
        ey_series = []
        for step_key in steps:
            ds = td[step_key]
            t = ds.attrs.get("time", None)
            if t is None:
                # Fall back: use step index × dt_orig (we set OverSampling above)
                t = float(step_key)
            times.append(float(t))
            arr = ds[()]  # (3, nx, ny, nz)  — Ex, Ey, Ez
            # arr shape varies: H5DumpReader uses (nx, ny, nz, 3) sometimes
            if arr.shape[0] == 3:
                ey = arr[1]            # (nx, ny, nz)
            elif arr.shape[-1] == 3:
                ey = arr[..., 1]
            else:
                raise RuntimeError(f"unexpected dump shape {arr.shape}")
            # mean over the small refl probe box → scalar Ey at probe location
            ey_series.append(float(np.mean(ey)))
        ts = np.asarray(ey_series, dtype=float)
        if len(times) > 1:
            dt_eff = float(times[1] - times[0])
        else:
            dt_eff = None
    return {
        "ts": ts.astype(float).tolist(),
        "dt": dt_eff,
        "n": int(ts.size),
        "elapsed_s": float(elapsed),
    }


def _compute_R(ts_total: np.ndarray, ts_inc: np.ndarray, dt: float,
               band: tuple[float, float] = (F_LO, F_HI)) -> dict:
    """R(f) = |FFT(total - inc) / FFT(inc)|² ; returns spectrum + band stats."""
    n = min(len(ts_total), len(ts_inc))
    ts_total = np.asarray(ts_total[:n])
    ts_inc = np.asarray(ts_inc[:n])
    nfft = int(2 ** np.ceil(np.log2(n)))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    S_inc = np.fft.rfft(ts_inc, n=nfft)
    S_scat = np.fft.rfft(ts_total - ts_inc, n=nfft)
    R = (np.abs(S_scat) / (np.abs(S_inc) + 1e-30)) ** 2
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return {
        "freqs_hz": freqs.tolist(),
        "R": R.tolist(),
        "freqs_band_hz": freqs[mask].tolist(),
        "R_band": R[mask].tolist(),
        "mean_R_band": float(np.mean(R[mask])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
                    help="JSON path to write R(f) results to")
    ap.add_argument("--layer-eps", nargs=N_LAYERS, type=float,
                    default=list(DEFAULT_LAYER_EPS),
                    help="3 design-layer εr values (default: TMM optimum)")
    ap.add_argument("--sim-dir", default=None,
                    help="openEMS scratch dir (default: temp)")
    args = ap.parse_args()

    layer_eps = tuple(args.layer_eps)
    print(f"Layer εr = {layer_eps}")
    print(f"Geometry: LX={LX*1e3:.1f}mm, dx={DX*1e3:.2f}mm, "
          f"layer_thk={LAYER_THK*1e3:.3f}mm")

    sim_dir_root = args.sim_dir or tempfile.mkdtemp(prefix="ar_coating_oe_")
    print(f"openEMS scratch dir: {sim_dir_root}")

    # ---- run #1: vacuum reference --------------------------------------
    print("\n[1/2] vacuum reference run...")
    ref_dir = os.path.join(sim_dir_root, "ref")
    res_ref = _run_openems(None, ref_dir)
    print(f"  done in {res_ref['elapsed_s']:.1f}s, n_steps = {res_ref['n']}")

    # ---- run #2: design (TMM optimum) ----------------------------------
    print("\n[2/2] design run (3 layers + substrate)...")
    des_dir = os.path.join(sim_dir_root, "design")
    res_des = _run_openems(layer_eps, des_dir)
    print(f"  done in {res_des['elapsed_s']:.1f}s, n_steps = {res_des['n']}")

    if res_ref["dt"] is None or res_des["dt"] is None:
        print("ERROR: openEMS probe did not include a time column.", file=sys.stderr)
        return 2
    if abs(res_ref["dt"] - res_des["dt"]) / res_ref["dt"] > 1e-6:
        print(f"ERROR: dt mismatch ref={res_ref['dt']} vs des={res_des['dt']}",
              file=sys.stderr)
        return 2

    print("\nComputing R(f)...")
    ts_inc = np.asarray(res_ref["ts"])
    ts_des = np.asarray(res_des["ts"])
    spec = _compute_R(ts_des, ts_inc, res_ref["dt"])
    print(f"  X-band mean R = {spec['mean_R_band']:.4e}")

    out = {
        "meta": {
            "solver": "openEMS",
            "geometry": "1D-equivalent (PMC y,z + PML x), 3-layer AR coating + εr=12 substrate",
            "layer_eps": list(layer_eps),
            "lx_mm": LX * 1e3, "dx_mm": DX * 1e3,
            "layer_thk_mm": LAYER_THK * 1e3,
            "f_band_ghz": [F_LO / 1e9, F_HI / 1e9],
            "x_src_mm": X_SRC * 1e3,
            "x_refl_mm": X_REFL * 1e3,
            "x_design_lo_mm": X_DESIGN_LO * 1e3,
            "x_design_hi_mm": X_DESIGN_HI * 1e3,
        },
        "ref": {"n_steps": res_ref["n"], "elapsed_s": res_ref["elapsed_s"]},
        "design": {"n_steps": res_des["n"], "elapsed_s": res_des["elapsed_s"]},
        "dt": res_ref["dt"],
        **spec,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
