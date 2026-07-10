#!/usr/bin/env python3
"""OpenEMS external referee for the rfx issue #313 final gate.

Independent (different solver, different port model) bracket of the
16 mm air-microstrip thru that ``tests/test_refplane_port_waves.py`` /
``tests/test_lumped_twoport_vi_validation_battery.py`` measure in rfx.
This script does NOT judge pass/fail — it reports |S11|, |S21|,
unwrapped arg(S21), and group delay for an independent MSL-port model
of the SAME geometry class, so a human/reviewer can compare against the
rfx PR #320 numbers (measured 2026-07-10, reference-plane port waves):

  |S21| = 0.998 -> 0.983 over 3-7 GHz
  line Zc (two-plane measurement)  ~= 47.9-48.6 ohm
  beta/(w/c) (measured slow-wave)  ~= 1.05-1.06
  group delay (incl. feed-post excess) ~= 70 ps

Geometry — matches the rfx canonical thru fixture exactly (physical
lengths; port MODEL differs by construction, see below):
  Ground    : PEC boundary condition at z_lo (z=0) -- NOT a filled
              substrate; this is an air microstrip, matching rfx's
              Boundary(lo="pec", hi="cpml") with no dielectric fill.
  Trace     : W = 5 mm, at z = H = 1 mm, 1-cell-thick PEC strip
              spanning x = [8, 24] mm (L = 16 mm port-to-port).
  Ports     : two MSL ports (50 ohm), feed planes at x = 8 mm and
              x = 24 mm -- the SAME 16 mm reference-plane span rfx's
              reference-plane architecture de-embeds back to.
  Domain    : LX=32 mm, LY=20 mm, LZ=10 mm -- identical to the rfx
              fixture's domain=(0.032, 0.020, 0.010) m.
  Boundaries: PML on x (both ends), y (both ends), z_hi; PEC on z_lo
              (ground plane) -- identical boundary topology to rfx.
  Mesh      : DX = 500 um uniform by default (matches rfx's dx=0.5mm
              exactly; override via --dx-um for a convergence check).
  Frequency : 0.5-7 GHz (27 pts) -- covers rfx's 3-7 GHz gate band plus
              low-frequency points for a cleaner group-delay estimate.

Port-model caveat (structural, not a bug, state honestly in review):
rfx's wire port is a POINT feed (a single vertical wire cell-column
from the ground to the trace, "extent"). openEMS's MSLPort instead
occupies a short SPAN along the propagation axis (``--port-w-cells``,
default 6 cells = 3 mm at the default mesh) inside which it launches/
absorbs the line mode -- the canonical pattern already used in this
repo's other openEMS MSL crossvals (see
research/microwave-energy/openems_simulation/msl_thru_reference.py,
itself following the upstream openEMS MSL_NotchFilter.py tutorial:
each port's ``prop_dir`` = sign(stop[axis] - start[axis]) points INTO
the line). This referee therefore brackets the physical thru-line
S-parameter CLASS (matched 50-ohm air microstrip, 16 mm long), not a
bit-for-bit reproduction of rfx's specific port implementation.

Output JSON schema::

  {
    "meta": { geometry + run summary },
    "freqs_ghz": [...],
    "s11": [[re, im], ...],
    "s21": [[re, im], ...],
    "s11_mag": [...], "s21_mag": [...],
    "s21_phase_rad_unwrapped": [...],
    "group_delay_ps": [...],           # -d(phase)/d(omega), numerical
    "z0_port0": [[re, im], ...],
    "z0_port1": [[re, im], ...],
    "band_3_7ghz_summary": { mean/min/max of |S21|, |S11|, group delay },
    "elapsed_s": float
  }

Usage (VESSL-only -- see vessl_openems_thru.yaml; openEMS python
bindings are not expected to be importable outside that lane)::

    python thru_openems.py --output results/thru_openems_dx500um.json
    python thru_openems.py --dx-um 250 --output results/thru_openems_dx250um.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Fix numpy deprecation in openEMS v0.0.35 (must be BEFORE openEMS import;
# matches research/microwave-energy/openems_simulation/msl_thru_reference.py).
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
    from openEMS.ports import MSLPort
except ImportError as exc:
    print(f"ERROR: openEMS Python bindings not importable ({exc}).\n"
          "  This script is VESSL-only (source-built openEMS); see "
          "scripts/diagnostics/openems_thru_referee/vessl_openems_thru.yaml.",
          file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Geometry constants (mm units; CSXCAD UNIT = 1e-3 m) -- matches the rfx
# canonical thru fixture (tests/test_refplane_port_waves.py _build_thru /
# tests/test_lumped_twoport_vi_validation_battery.py _build_thru) exactly.
# ---------------------------------------------------------------------------
UNIT = 1e-3

H = 1.0            # trace height above ground [mm]  (rfx _THRU_H_M = 1.0e-3)
W_TRACE = 5.0       # trace width [mm]                (rfx _THRU_W_M = 5.0e-3)
X1 = 8.0            # port 1 feed x [mm]               (rfx _THRU_X1_M = 0.008)
X2 = 24.0           # port 2 feed x [mm]               (rfx _THRU_X2_M = 0.024)
L_LINE = X2 - X1    # 16 mm port-to-port               (rfx _THRU_L_M)
LX, LY, LZ = 32.0, 20.0, 10.0   # domain [mm] (rfx _THRU_DOMAIN_M = (0.032,0.020,0.010))
CPML_CELLS = 8      # matches rfx cpml_layers=8

# Frequency grid -- 0.5 to 7 GHz, 27 points (0.25 GHz step). Covers rfx's
# committed 3-7 GHz gate band (9 pts, 0.5 GHz step) plus low-f points for
# a cleaner numerical group-delay estimate.
F_START_GHZ = 0.5
F_STOP_GHZ = 7.0
N_FREQS = 27
FREQS_GHZ = np.linspace(F_START_GHZ, F_STOP_GHZ, N_FREQS)
FREQS_HZ = FREQS_GHZ * 1e9

# Gaussian excite: centre / corner frequency chosen the same way as the
# precedent openEMS MSL scripts (F0 = midband, FC = 0.85 * f_stop).
F0_GHZ = 0.5 * (F_START_GHZ + F_STOP_GHZ)   # 3.75 GHz
FC_GHZ = F_STOP_GHZ * 0.85                   # 5.95 GHz

# Summary band -- matches rfx's own committed gate window exactly.
GATE_F_LO_GHZ = 3.0
GATE_F_HI_GHZ = 7.0


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------
def _run(*, dx_mm: float, sim_path: str, threads: int, nrts: int,
         end_criteria: float, port_w_cells: int) -> dict:
    """Build CSXCAD/openEMS geometry, run, extract S-params.

    Returns a dict with complex arrays for S11, S21, Z0_0, Z0_1, and elapsed.
    """
    DX = dx_mm
    y_centre = LY / 2.0

    FDTD = openEMS(NrTS=nrts, EndCriteria=end_criteria)
    FDTD.SetGaussExcite(F0_GHZ * 1e9, FC_GHZ * 1e9)

    # Boundaries: z_lo PEC (ground plane, matches rfx Boundary(lo="pec")),
    # PML_8 everywhere else (matches rfx BoundarySpec(x="cpml", y="cpml",
    # z=Boundary(hi="cpml")) with cpml_layers=8). openEMS order:
    # [xmin, xmax, ymin, ymax, zmin, zmax].
    FDTD.SetBoundaryCond(["PML_8", "PML_8",
                          "PML_8", "PML_8",
                          "PEC",   "PML_8"])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(UNIT)

    x_lines = np.arange(0.0, LX + 0.5 * DX, DX)
    y_lines = np.arange(0.0, LY + 0.5 * DX, DX)
    z_lines = np.arange(0.0, LZ + 0.5 * DX, DX)
    grid.SetLines("x", x_lines)
    grid.SetLines("y", y_lines)
    grid.SetLines("z", z_lines)

    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0

    # --- Air microstrip: NO dielectric substrate box. The ground plane is
    # the z_lo PEC boundary condition itself (see Boundaries above) --
    # matches the rfx fixture, which has no material fill either (vacuum
    # everywhere, PEC trace + PEC boundary).
    #
    # --- PEC trace strip between the two port columns. MSLPort adds its
    # own metal within its own x-span (port width); this box fills the
    # clear line between the two port spans (matches
    # msl_thru_reference.py's precedent pattern exactly).
    trace_metal = CSX.AddMetal("trace")
    trace_metal.AddBox([X1, trace_y_lo, H],
                       [X2, trace_y_hi, H + DX], priority=10)

    # --- MSL ports: canonical openEMS pattern (matches upstream
    # MSL_NotchFilter.py / this repo's msl_thru_reference.py precedent).
    #   - start[z]=H (trace plane), stop[z]=0 (ground) so the Ez
    #     excitation points from trace down to ground.
    #   - prop_dir = sign(stop[x] - start[x]): port0 stop > start =>
    #     +1 (+x, INTO the line); port1 stop < start => -1 (-x, INTO
    #     the line). Both ports' propagation direction points into the
    #     shared trace, opposite in sense to rfx's "direction" kwarg
    #     (which names the OUTWARD/away-from-the-line normal) -- an
    #     inherent convention difference between the two port models,
    #     not a bug; see the module docstring's port-model caveat.
    PORT_W = port_w_cells * DX

    port0_metal = CSX.AddMetal("port0_metal")
    port0 = MSLPort(
        CSX, port_nr=1,
        metal_prop=port0_metal,
        start=[X1,           trace_y_lo, H],
        stop=[X1 + PORT_W,   trace_y_hi, 0.0],
        prop_dir=0,   # +x, into the line
        exc_dir=2,    # Ez excitation (vertical)
        excite=1.0,
        Feed_R=50.0,
    )
    port1_metal = CSX.AddMetal("port1_metal")
    port1 = MSLPort(
        CSX, port_nr=2,
        metal_prop=port1_metal,
        start=[X2,           trace_y_lo, H],
        stop=[X2 - PORT_W,   trace_y_hi, 0.0],
        prop_dir=0,   # -x, into the line (sign(stop-start) = sign(-PORT_W))
        exc_dir=2,
        excite=0.0,   # passive matched load
        Feed_R=50.0,
    )

    os.makedirs(sim_path, exist_ok=True)
    CSX.Write2XML(os.path.join(sim_path, "thru.xml"))

    orig_cwd = os.getcwd()
    t0 = time.time()
    FDTD.Run(sim_path, cleanup=True, verbose=1, numThreads=threads)
    try:
        os.chdir(orig_cwd)
    except OSError:
        os.chdir("/tmp")
    elapsed = time.time() - t0

    # --- Extract S-params. Do NOT pass ref_impedance as a scalar float --
    # that triggers a bug in the base Port.CalcPort when Z_ref is
    # array-valued (msl_thru_reference.py precedent, verified working).
    port0.CalcPort(sim_path, FREQS_HZ)
    port1.CalcPort(sim_path, FREQS_HZ)

    s11 = np.asarray(port0.uf_ref, dtype=complex) / np.asarray(port0.uf_inc, dtype=complex)
    # port1 propagates -x; its uf_ref channel carries the +x-going
    # transmitted wave (same convention as msl_thru_reference.py).
    s21 = np.asarray(port1.uf_ref, dtype=complex) / np.asarray(port0.uf_inc, dtype=complex)

    z0_port0 = np.asarray(port0.Z_ref, dtype=complex)
    z0_port1 = np.asarray(port1.Z_ref, dtype=complex)

    return {
        "s11": s11,
        "s21": s21,
        "z0_port0": z0_port0,
        "z0_port1": z0_port1,
        "elapsed_s": elapsed,
    }


def _group_delay(freqs_hz: np.ndarray, s21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unwrapped arg(S21) and group delay tau_g = -d(phase)/d(omega).

    Numerical derivative (np.gradient, central differences) over the
    supplied frequency grid -- no fitted model, just the raw measured
    curve, so a kink or artefact is visible rather than smoothed away.
    """
    phase = np.unwrap(np.angle(s21))
    omega = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    gd_s = -np.gradient(phase, omega)
    return phase, gd_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="results/thru_openems.json",
                   help="Output JSON path")
    p.add_argument("--sim-root", default="/tmp/openems_thru_referee",
                   help="Scratch directory for openEMS run files")
    p.add_argument("--dx-um", type=float, default=500.0,
                   help="Uniform cell size in um (default 500 = 0.5mm, "
                        "matches the rfx fixture's dx exactly)")
    p.add_argument("--port-w-cells", type=int, default=6,
                   help="MSL port length in cells along propagation "
                        "(default 6, the canonical openEMS MSL_NotchFilter "
                        "pattern already used in this repo)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nrts", type=int, default=500000,
                   help="Max timesteps (default 500000, matches precedent)")
    p.add_argument("--end-criteria", type=float, default=1e-4,
                   help="Energy end criteria (default 1e-4, matches precedent)")
    args = p.parse_args()

    dx_mm = args.dx_um * 1e-3

    print("=== OpenEMS thru referee (rfx issue #313 final gate) ===")
    print(f"  Air microstrip: H={H} mm  W_TRACE={W_TRACE} mm  L_LINE={L_LINE} mm")
    print(f"  Domain: LX={LX} mm  LY={LY} mm  LZ={LZ} mm  CPML={CPML_CELLS} cells")
    print(f"  DX = {args.dx_um:.0f} um  PORT_W = {args.port_w_cells * dx_mm:.2f} mm "
          f"({args.port_w_cells} cells)")
    print(f"  Freqs: {F_START_GHZ}-{F_STOP_GHZ} GHz ({N_FREQS} pts)")
    print(f"  Threads: {args.threads}  NrTS: {args.nrts}  EndCriteria: {args.end_criteria}")

    sim_path = os.path.abspath(args.sim_root)
    result = _run(
        dx_mm=dx_mm,
        sim_path=sim_path,
        threads=args.threads,
        nrts=args.nrts,
        end_criteria=args.end_criteria,
        port_w_cells=args.port_w_cells,
    )

    s11 = result["s11"]
    s21 = result["s21"]
    z0_p0 = result["z0_port0"]
    z0_p1 = result["z0_port1"]
    elapsed = result["elapsed_s"]

    phase, gd_s = _group_delay(FREQS_HZ, s21)
    gd_ps = gd_s * 1e12

    mask = (FREQS_GHZ >= GATE_F_LO_GHZ) & (FREQS_GHZ <= GATE_F_HI_GHZ)
    if not np.any(mask):
        mask = np.ones_like(FREQS_GHZ, dtype=bool)

    s11_mag = np.abs(s11)
    s21_mag = np.abs(s21)

    band_summary = {
        "f_lo_ghz": GATE_F_LO_GHZ,
        "f_hi_ghz": GATE_F_HI_GHZ,
        "n_pts": int(np.sum(mask)),
        "mean_s21_mag": float(np.mean(s21_mag[mask])),
        "min_s21_mag": float(np.min(s21_mag[mask])),
        "max_s21_mag": float(np.max(s21_mag[mask])),
        "mean_s11_mag": float(np.mean(s11_mag[mask])),
        "max_s11_mag": float(np.max(s11_mag[mask])),
        "mean_z0_re_ohm": float(np.mean(z0_p0[mask].real)),
        "mean_group_delay_ps": float(np.mean(gd_ps[mask])),
        "min_group_delay_ps": float(np.min(gd_ps[mask])),
        "max_group_delay_ps": float(np.max(gd_ps[mask])),
    }

    print(f"\n=== Band {GATE_F_LO_GHZ}-{GATE_F_HI_GHZ} GHz ({band_summary['n_pts']} pts) ===")
    print(f"  |S21|: mean={band_summary['mean_s21_mag']:.4f}  "
          f"range=[{band_summary['min_s21_mag']:.4f}, {band_summary['max_s21_mag']:.4f}]")
    print(f"  |S11|: mean={band_summary['mean_s11_mag']:.4f}  "
          f"max={band_summary['max_s11_mag']:.4f}")
    print(f"  Re(Z0) port0: mean={band_summary['mean_z0_re_ohm']:.2f} ohm")
    print(f"  Group delay: mean={band_summary['mean_group_delay_ps']:.2f} ps  "
          f"range=[{band_summary['min_group_delay_ps']:.2f}, "
          f"{band_summary['max_group_delay_ps']:.2f}] ps")
    print(f"\n  Elapsed: {elapsed:.1f} s")

    out = {
        "meta": {
            "solver": "openEMS",
            "version": "v1",
            "purpose": "rfx issue #313 external referee (final gate) -- "
                       "brackets, does not judge",
            "h_mm": H,
            "w_trace_mm": W_TRACE,
            "l_line_mm": L_LINE,
            "x1_mm": X1,
            "x2_mm": X2,
            "lx_mm": LX,
            "ly_mm": LY,
            "lz_mm": LZ,
            "cpml_cells": CPML_CELLS,
            "dx_um": args.dx_um,
            "port_w_cells": args.port_w_cells,
            "port_w_mm": args.port_w_cells * dx_mm,
            "f_start_ghz": F_START_GHZ,
            "f_stop_ghz": F_STOP_GHZ,
            "n_freqs": N_FREQS,
            "threads": args.threads,
            "nrts": args.nrts,
            "end_criteria": args.end_criteria,
        },
        "freqs_ghz": FREQS_GHZ.tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in s11],
        "s21": [[float(c.real), float(c.imag)] for c in s21],
        "s11_mag": s11_mag.tolist(),
        "s21_mag": s21_mag.tolist(),
        "s21_phase_rad_unwrapped": phase.tolist(),
        "group_delay_ps": gd_ps.tolist(),
        "z0_port0": [[float(c.real), float(c.imag)] for c in z0_p0],
        "z0_port1": [[float(c.real), float(c.imag)] for c in z0_p1],
        "band_3_7ghz_summary": band_summary,
        "elapsed_s": round(elapsed, 1),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== Written to {out_path} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
