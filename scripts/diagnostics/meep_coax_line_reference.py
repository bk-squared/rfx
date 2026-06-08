#!/usr/bin/env python3
"""External MEEP POWER-FLUX coaxial |S11| reference (MPB-free) for rfx crossval.

Mirrors the proven two-run flux pattern of
``scripts/diagnostics/meep_tjunction_reference.py`` (straight-line P_inc
normalization + device run, ``stop_when_fields_decayed``,
``force_complex_fields=True``, ``amp_func``, ``.npz`` output) but for a 3D coax
(TEM) line along z. MEEP's MPB eigenmode solver rejects PEC (mp.metal) walls
("invalid dielectric function for MPB"), so we use power flux through a transverse
port plane: the PEC shell confines all fields to r<b, so the plane flux IS the
coax modal power, and |S11|^2 = (P_inc - P_dut) / P_inc (magnitudes only — exactly
what the broad-E5 magnitude gate compares; |S11| is reference-plane independent).

Geometry (from /tmp/meep_coax_spec.md, units a = 1 meter so all lengths are SI):
  pin   r < A = 0.635 mm           PEC (mp.metal)
  PTFE  A < r < B = 2.055 mm       eps_r = 2.1
  shell B < r < SHELL_OUT          PEC (mp.metal)
  SMA Z0 = 60/sqrt(eps)*ln(b/a) = 48.6 ohm (matches rfx).

TEM source: coax TEM has E purely RADIAL with E_r ~ 1/r. Decompose into Cartesian
Ex, Ey on a transverse (xy) source plane, masked to the PTFE annulus a<r<b:
  E_x = E_r * x/r = k*x/r^2 ;  E_y = k*y/r^2 ; zero outside the annulus.

SCOPE: short / open ONLY — the two |Gamma| = 1 full-reflection cross-checks, which
are robust in MEEP (geometry-only terminations, no material tuning). 'matched' is
NOT modelled: a PML-terminated line is byte-identical to the P_inc normalization
geometry so deterministic FDTD gives |S11| = 0 BY CONSTRUCTION (no independent
evidence), and a matched R=Z0 resistive sheet hits the same staircased-sheet trap
as the resistive loads. The matched (|Gamma|=0) and resistive 25/100 ohm
(|Gamma|=0.32/0.35) terminations are covered by the exact-analytic broad-E5 envelope
(coax_line_broad_e5_envelope.json) instead — a 1-cell MEEP resistive sheet would
merely re-test rfx's own resistor stamp and is staircase-sensitive (an R5 trap).

Run (LINEAGE A — conda-forge pymeep, see scripts/physics_gate_coaxial_meep.yaml):
  export PATH=/opt/conda/bin:$PATH
  for T in short open matched; do
    python scripts/diagnostics/meep_coax_line_reference.py \
      --termination $T --output-dir .omx/physics-gate/coaxial-meep-reference
  done

``--stub`` emits the EXACT analytic |Gamma| per termination (no meep import) so the
file + downstream plumbing can be smoke-tested locally without meep installed; the
stub flag is recorded in the npz and printout so it can never be mistaken for a
real full-wave run.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

C0 = 299792458.0

# --- SMA coax constants (verified vs rfx/sources/coaxial_port.py SMA_*_RADIUS) ---
A = 0.635e-3          # pin radius (m)               SMA_PIN_RADIUS
B = 2.055e-3          # PTFE / shell-inner radius (m) SMA_OUTER_RADIUS
EPS = 2.1             # PTFE_EPS_R
Z0 = 48.6             # analytic 60/sqrt(eps)*ln(b/a) = 48.6 ohm (matches rfx)
SHELL_T = 0.6e-3      # PEC shell wall thickness (~8 cells at RES=3200)
SHELL_OUT = B + SHELL_T   # outer PEC radius = 2.655e-3 m

# Frequency band 4-12 GHz (SI Hz; MEEP units divide by C0).
FMIN_HZ = 4.0e9
FMAX_HZ = 12.0e9

# Exact analytic |Gamma| for the three calibration terminations (used by --stub
# and as an independent witness, R5). short/open ride |Gamma|=1, matched -> 0.
ANALYTIC_ABS_GAMMA = {"short": 1.0, "open": 1.0}


# ----------------------------------------------------------------------------
# MEEP run (imported LAZILY so --stub works without meep installed; meep is
# cluster-only via the conda-forge pymeep LINEAGE A recipe).
# ----------------------------------------------------------------------------
def _run_meep(termination: str, resolution: int, nfreq: int,
              fmin_hz: float, fmax_hz: float):
    import meep as mp

    fmin = fmin_hz / C0
    fmax = fmax_hz / C0
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    res = float(resolution)
    # Transverse window just enclosing the shell + a thin air guard.
    w_tr = 2.0 * SHELL_OUT + 4.0 / res
    # Axial layout (z): [ -z PML | clearance | DUT | line | src | +z PML ].
    # z_dut MUST sit in the CLEAN region with several cells of clearance from the
    # -z PML inner edge; otherwise the short cap / open end lands inside the
    # absorber (attenuated reflection + PEC-in-PML instability).
    dpml = 6.0e-3                       # PML each z-end (>~1/3 lambda_g in PTFE)
    l_line = 38.0e-3                    # clean region length (between the PMLs)
    z_cell = 2.0 * dpml + l_line
    z_pml_lo = -z_cell / 2.0 + dpml     # -z PML inner edge
    z_pml_hi = z_cell / 2.0 - dpml      # +z PML inner edge
    z_dut = z_pml_lo + 8.0e-3           # DUT plane: 8 mm (~25 cells) clear of -z PML
    z_src = z_pml_hi - 6.0e-3           # TEM source: 6 mm clear of the +z PML edge
    z_port = 0.5 * (z_dut + z_src)      # flux monitor: mid clean-line

    cell = mp.Vector3(w_tr, w_tr, z_cell)
    pml = [mp.PML(dpml, direction=mp.Z)]   # PML only on +/-z; transverse = PEC shell

    def cyl(z0, z1, radius, material):
        return mp.Cylinder(radius=radius, height=(z1 - z0),
                           center=mp.Vector3(0, 0, 0.5 * (z0 + z1)),
                           axis=mp.Vector3(0, 0, 1), material=material)

    def coax_blocks(z_lo, z_hi):
        # MEEP: LATER objects override EARLIER at overlapping points, so order =
        # [outer PEC shell, PTFE annulus carving the inside, inner PEC pin].
        return [
            cyl(z_lo, z_hi, SHELL_OUT, mp.metal),
            cyl(z_lo, z_hi, B, mp.Medium(epsilon=EPS)),
            cyl(z_lo, z_hi, A, mp.metal),
        ]

    def build_dut_geometry(term):
        # Coax line + the chosen termination at z_dut (in the CLEAN region). The
        # line runs up through the +z PML (z_hi = +z cell edge) so the source's +z
        # half is absorbed.
        z_hi = z_cell / 2.0
        if term == "short":
            # PEC cap disk (radius=SHELL_OUT) bridging pin->shell over ~2 cells at
            # z_dut (CLEAN region, clear of the -z PML). Coax above z_dut only; the
            # cap reflects everything. Gamma = -1, |S11| = 1.
            return coax_blocks(z_dut, z_hi) + [
                cyl(z_dut - 2.0 / res, z_dut, SHELL_OUT, mp.metal),
            ]
        if term == "open":
            # Pin retracts a GAP short of z_dut (open at the pin end, CLEAN region);
            # shell + PTFE continue DOWN through the -z PML so the below-cutoff
            # stub's evanescent tail is absorbed, not vented. Gamma = +1, |S11| = 1.
            gap = 3.0 / res
            return [
                cyl(-z_cell / 2.0, z_hi, SHELL_OUT, mp.metal),
                cyl(-z_cell / 2.0, z_hi, B, mp.Medium(epsilon=EPS)),
                cyl(z_dut + gap, z_hi, A, mp.metal),
            ]
        raise ValueError(f"unsupported termination: {term!r}")

    def make_tem_sources():
        # amp_func(p) gets p RELATIVE to the source center (cf tjunction line 80).
        # E_r ~ 1/r TEM profile -> Ex = E_r*(x/r) = x/r^2, Ey = y/r^2. Scale by A so
        # the peak amplitude is O(1) (raw x/r^2 ~ 1/A ~ 1575 near the pin would inject
        # a near-singular field on the cells next to the ~2-cell pin). Magnitude is
        # irrelevant for the |S11| ratio; the bound is purely for numerical safety.
        def amp_ex(p):
            r2 = p.x * p.x + p.y * p.y
            r = np.sqrt(r2)
            if A < r < B:
                return complex(A * p.x / r2)
            return 0.0 + 0.0j

        def amp_ey(p):
            r2 = p.x * p.x + p.y * p.y
            r = np.sqrt(r2)
            if A < r < B:
                return complex(A * p.y / r2)
            return 0.0 + 0.0j

        src_size = mp.Vector3(w_tr, w_tr, 0)
        common = dict(center=mp.Vector3(0, 0, z_src), size=src_size)
        # is_integrated=True: the source plane sits a few cells from the +z PML and
        # the coax metal passes through the PML; an un-integrated J near PML injects a
        # DC offset the absorber amplifies. Integrated current avoids that artifact.
        return [
            mp.Source(mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
                      component=mp.Ex, amp_func=amp_ex, **common),
            mp.Source(mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
                      component=mp.Ey, amp_func=amp_ey, **common),
        ]

    def add_port_flux(sim):
        # Full transverse plane at z_port; PEC shell confines fields to r<B so
        # plane flux == coax modal power. weight=-1 => +reading = inward (-z, toward
        # the DUT) (mirrors tjunction weight=float(-outsign)).
        fr = mp.FluxRegion(center=mp.Vector3(0, 0, z_port),
                           size=mp.Vector3(w_tr, w_tr, 0),
                           direction=mp.Z, weight=-1.0)
        return sim.add_flux(fcen, df, nfreq, fr)

    def run_flux(geometry):
        sim = mp.Simulation(cell_size=cell, resolution=resolution,
                            geometry=geometry, sources=make_tem_sources(),
                            boundary_layers=pml, dimensions=3,
                            force_complex_fields=True,
                            # PEC (mp.metal, eps=-inf) must NOT be subpixel-smoothed:
                            # averaging -inf into a sub-cell pin voxel yields a garbage
                            # effective-eps tensor that breaks CFL -> instant NaN
                            # (MEEP subpixel smoothing is valid only for lossless
                            # dielectrics). Staircase the metal instead.
                            eps_averaging=False,
                            # metal / curved-feature stabilizer.
                            Courant=0.3)
        fobj = add_port_flux(sim)
        # Stop on FLUX-DFT convergence, NOT raw-field decay. A high-Q shorted/open
        # coax traps energy (only the PML leaks it), so raw |Ex| reaches 1e-6 only
        # after ~hours; but the flux DFT (the |S11| observable, flat ~1 for a full
        # reflector) converges in seconds. stop_when_dft_decayed watches the flux
        # monitor; maximum_run_time caps it as a hard safety.
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=1.0e-3, maximum_run_time=60.0))
        freqs = np.array(mp.get_flux_freqs(fobj))
        flux = np.array(mp.get_fluxes(fobj))   # inward (weight=-1 already applied)
        return freqs, flux

    # (1) normalization: straight matched line spanning the FULL cell (coax through
    #     BOTH +-z PMLs -> reflectionless both ends) -> P_inc (incident -z TEM).
    geo_inc = coax_blocks(-z_cell / 2.0, z_cell / 2.0)
    freqs, p_inc = run_flux(geo_inc)
    # (2) device run with the chosen termination.
    _, p_dut = run_flux(build_dut_geometry(termination))

    s11_mag = np.sqrt(np.clip((p_inc - p_dut) / np.abs(p_inc), 0.0, None))
    return freqs * C0, s11_mag, p_inc, p_dut


def run(termination: str, resolution: int, nfreq: int, stub: bool,
        fmin_hz: float = FMIN_HZ, fmax_hz: float = FMAX_HZ):
    if stub:
        # Plumbing-only: emit the EXACT analytic |Gamma| with no meep import. The
        # flat band makes a stub trivially distinguishable from a real run, and the
        # stub flag is persisted so downstream cannot mistake it for evidence.
        freqs_hz = np.linspace(fmin_hz, fmax_hz, nfreq)
        s11_mag = np.full(nfreq, ANALYTIC_ABS_GAMMA[termination], dtype=float)
        return freqs_hz, s11_mag, None, None
    return _run_meep(termination, resolution, nfreq, fmin_hz, fmax_hz)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--termination", choices=["short", "open"], required=True)
    ap.add_argument("--resolution", type=int, default=3200,
                    help="cells per meter; >=2817 gives >=4 cells across the "
                         "1.42 mm annulus (default 3200 -> ~4.5 cells/annulus)")
    ap.add_argument("--fmin", type=float, default=FMIN_HZ)
    ap.add_argument("--fmax", type=float, default=FMAX_HZ)
    ap.add_argument("--nfreq", type=int, default=9)
    ap.add_argument("--output-dir",
                    default=".omx/physics-gate/coaxial-meep-reference")
    ap.add_argument("--stub", action="store_true",
                    help="emit the EXACT analytic |Gamma| (no meep import) to "
                         "smoke-test the file/plumbing without meep installed")
    args = ap.parse_args(argv)

    freqs_hz, s11_mag, p_inc, p_dut = run(
        args.termination, args.resolution, args.nfreq, args.stub,
        fmin_hz=float(args.fmin), fmax_hz=float(args.fmax))

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"meep_coax_{args.termination}.npz")
    _empty = np.asarray([], dtype=float)
    np.savez(
        out,
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        s11_mag=np.asarray(s11_mag, dtype=float),
        # R5 witnesses persisted for downstream inspection (empty for --stub):
        p_inc=np.asarray(p_inc, dtype=float) if p_inc is not None else _empty,
        p_dut=np.asarray(p_dut, dtype=float) if p_dut is not None else _empty,
        termination=args.termination,
        resolution=int(args.resolution),
        stub=bool(args.stub),
        z0_ohm=Z0,
    )

    tag = " (STUB: analytic, NOT a meep run)" if args.stub else ""
    band_mean = float(np.mean(s11_mag))
    print(f"meep_coax termination={args.termination} |S11| band-mean={band_mean:.4f}{tag}")
    print(f"[meep_coax] wrote {out}")
    # R5: dump the full per-frequency trace, not just the band-mean headline.
    with np.printoptions(precision=4, suppress=True):
        print(f"[meep_coax] freqs_GHz = {np.asarray(freqs_hz) / 1e9}")
        print(f"[meep_coax] |S11|(f)  = {np.asarray(s11_mag)}")
    if p_inc is not None:
        # R5: dump the full per-frequency P_inc/P_dut, not just band-means — a
        # band-mean |S11|~1 can hide a per-freq dip from a buried cap / PML
        # attenuation / sign-flip. P_inc must track the source spectrum (not flat
        # or ~0) and P_dut < P_inc for a reflector.
        with np.printoptions(precision=4, suppress=True):
            print(f"[meep_coax] P_inc(f) = {np.asarray(p_inc)}")
            print(f"[meep_coax] P_dut(f) = {np.asarray(p_dut)}")
            print(f"[meep_coax] P_inc band-mean={float(np.mean(p_inc)):.4e} "
                  f"P_dut band-mean={float(np.mean(p_dut)):.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
