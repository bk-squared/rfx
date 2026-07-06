"""rfx replica of the CANONICAL openEMS Simple_Patch_Antenna — deep S11 null.

WHY
---
The cv05 FR4 probe patch (examples/crossval/05_patch_antenna.py) only ever
produced a shallow (~2 dB) rfx lumped-port S11 dip: its single-cell probe
port carries a parasitic cell reactance and the geometry was never tuned
for a matched (-15 dB or deeper) null. To cross-validate that rfx can
reproduce a genuinely MATCHED patch — a deep S11 null with the right input
impedance — we replicate the openEMS "Simple_Patch_Antenna" tutorial, which
is a proven-matched reference:

    openEMS reference : min|S11| = -25.7 dB @ 2.085 GHz, Zin = 54.5 - 3.0j ohm

Recipe (verbatim from the openEMS tutorial):
  * Substrate : Rogers-like eps_r = 3.38, thickness 1.524 mm, low loss.
                kappa = 1e-3 * 2*pi*2.45e9 * eps0 * eps_r  (tan_delta ~ 1e-3).
                Substrate + ground = 60 x 60 mm.
  * Patch     : 32 mm (x) x 40 mm (y), coplanar PEC at z = 1.524 mm.
  * Ground    : 60 x 60 mm PEC at z = 0.
  * Feed      : VERTICAL lumped port at (x = -6 mm from centre, y = 0),
                spanning z = 0 -> 1.524 mm (ground to patch), 50 ohm.
                This -6 mm inset is what matches it (openEMS AddLumpedPort
                z-directed at feed_pos = -6 mm).
  * Big air margins + absorbing boundary.

This is a STRUCTURAL / port-recipe replica. It reuses the rfx PART-1/PART-3
pattern of examples/crossval/05_patch_antenna.py: Box(material=...) stack,
add_port lumped feed, CPML + air padding, full complex S11(f) via
run(compute_s_params=True, ...), plus a Harminv ring-down resonance.

COORDINATE CONVENTION
---------------------
openEMS centres the structure at the origin and feeds at x = feed_pos = -6 mm,
y = 0. rfx uses absolute domain coordinates in [0, dom]. The structure is
centred in the domain, so the openEMS origin maps to (dom_x/2, dom_y/2) and:

    feed_x = dom_x/2 + feed_pos   (feed_pos = -6 mm  =>  dom_x/2 - 6 mm)
    feed_y = dom_y/2              (y = 0 in openEMS)

The patch (32 x 40 mm) is centred, so it spans
    x in [dom_x/2 - 16 mm, dom_x/2 + 16 mm], y in [dom_y/2 - 20 mm, dom_y/2 + 20 mm]
and the -6 mm feed sits 10 mm inside the -x radiating edge, on the y centre-line.

MESH / BOUNDARY / AIR PADDING
-----------------------------
  * z stack (bottom -> top): coarse air below (radiates into bottom CPML),
    1 PEC cell = finite ground plane, substrate (n_sub cells, default 5 ->
    ~0.30 mm/cell), coarse air above the patch. smooth_grading (ratio 1.3).
  * xy: non-uniform GRADED mesh (mirrors cv05's _refined_xy_profile): fine
    (--dx-mm) inside the patch+feed region, coarse (dx_coarse) in the air/PML.
  * boundary = CPML on all faces, cpml_layers = 8 (rfx had no instability on
    the cv05 patch; the only cv05 issue was the feed-position mismatch, fixed
    here by the -6 mm canonical inset). rfx CPML replaces openEMS PML_8.
  * Generous air: 20 mm below, 30 mm above, 20 mm xy margin around the 60 mm
    ground plane (~lambda/4 at 2.085 GHz clearance to the absorber).

Run (single resolution; caller submits to VESSL GPU):
    python scripts/crossval/rfx_patch_canonical.py --dx-mm 1.0 \
        --output scripts/crossval/out_canonical --n-steps 12000
    # finer GPU sweep:
    python scripts/crossval/rfx_patch_canonical.py --dx-mm 0.5 \
        --output scripts/crossval/out_canonical --n-steps 16000

Emits per-run JSON (freqs, complex S11, Z11, harminv, grid) + a |S11| dB
plot PNG into --output.
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8
EPS0 = 8.8541878128e-12
Z0_PORT = 50.0

# =============================================================================
# Canonical openEMS Simple_Patch_Antenna geometry (Rogers eps_r = 3.38)
# =============================================================================
eps_r = 3.38                     # Rogers-like substrate
h_sub = 1.524e-3                 # substrate thickness (1.524 mm)
# Low-loss substrate: kappa = 1e-3 * 2*pi*2.45e9 * eps0 * eps_r  (tan_delta ~ 1e-3)
sub_sigma = 1e-3 * 2 * math.pi * 2.45e9 * EPS0 * eps_r

patch_Lx = 32.0e-3               # patch extent in x
patch_Ly = 40.0e-3               # patch extent in y
sub_wx = 60.0e-3                 # substrate + ground extent in x
sub_wy = 60.0e-3                 # substrate + ground extent in y
feed_pos = -6.0e-3               # openEMS feed_pos: x offset from centre, y = 0

# openEMS reference (proven-matched) — used only for plot annotation / JSON.
REF_S11_MIN_DB = -25.7
REF_F_HZ = 2.085e9
REF_ZIN = complex(54.5, -3.0)

# =============================================================================
# Mesh / boundary / air-padding defaults
# =============================================================================
dx_coarse = 2.0e-3               # air / PML cell size
n_cpml = 8
n_sub_default = 5                # cells across the 1.524 mm substrate (~0.30 mm)
air_below = 20.0e-3              # free space below the finite ground plane
air_above = 30.0e-3              # radiation air above the patch
margin_xy = 20.0e-3              # air margin around the 60 mm ground plane


def build_geometry(dx_fine, n_sub):
    """Compute all absolute-coordinate positions + the non-uniform mesh
    profiles for one interior resolution ``dx_fine`` and ``n_sub`` substrate
    cells. Returns a dict of geometry / grid parameters.
    """
    from rfx.auto_config import smooth_grading

    dz_sub = h_sub / n_sub

    dom_x = sub_wx + 2 * margin_xy
    dom_y = sub_wy + 2 * margin_xy
    dom_z = air_below + h_sub + air_above

    # Ground / substrate footprint (centred in x/y)
    gx_lo = (dom_x - sub_wx) / 2
    gx_hi = gx_lo + sub_wx
    gy_lo = (dom_y - sub_wy) / 2
    gy_hi = gy_lo + sub_wy

    # Patch (centred, 32 x 40 mm)
    patch_x_lo = dom_x / 2 - patch_Lx / 2
    patch_x_hi = dom_x / 2 + patch_Lx / 2
    patch_y_lo = dom_y / 2 - patch_Ly / 2
    patch_y_hi = dom_y / 2 + patch_Ly / 2

    # Feed: openEMS origin -> domain centre; feed_pos is the x offset, y = 0.
    feed_x = dom_x / 2 + feed_pos
    feed_y = dom_y / 2

    # z stack (bottom -> top): air_below | 1 PEC ground cell | substrate | air_above
    #
    # CRITICAL ALIGNMENT RULE: smooth_grading INSERTS transition cells, which
    # grows the profile and SHIFTS everything after the insertion point. If the
    # geometry z-coordinates are computed from the pre-grading arithmetic
    # (air_below etc.), the fine substrate cells end up several mm away from
    # the substrate Box -> the substrate collapses into one coarse cell and the
    # ground/patch land in ADJACENT cells (measured: |S11|~1 flat, Zin ~ -j3k).
    # Fix: (a) preserve the ground+substrate fine block via preserve_regions
    # (the documented thin-PEC-on-NU-mesh convention), and (b) derive ALL
    # geometry z-coordinates from the FINAL graded profile's cumsum.
    n_below = int(math.ceil(air_below / dx_coarse))
    n_above = int(math.ceil(air_above / dx_coarse))
    raw_dz = np.concatenate([
        np.full(n_below, dx_coarse),   # air below finite ground plane
        np.full(1, dz_sub),            # PEC ground cell
        np.full(n_sub, dz_sub),        # substrate
        np.full(n_above, dx_coarse),   # air above patch
    ])
    dz_profile = smooth_grading(
        raw_dz, max_ratio=1.3,
        preserve_regions=[(n_below, n_below + 1 + n_sub)],
    )
    # Locate the preserved fine block (contiguous run of dz_sub cells) in the
    # graded output and anchor the metal/substrate stack to its actual z.
    fine = np.isclose(dz_profile, dz_sub, rtol=1e-9)
    k_fine0 = int(np.argmax(fine))
    tail = fine[k_fine0:]
    n_fine_run = int(np.argmin(tail)) if not tail.all() else int(tail.size)
    if n_fine_run < 1 + n_sub:
        raise RuntimeError(
            f"graded dz profile lost the fine block: run={n_fine_run} "
            f"< {1 + n_sub} (ground+substrate)")
    zcum = np.concatenate([[0.0], np.cumsum(dz_profile)])
    z_gnd_lo = float(zcum[k_fine0])            # 1 PEC ground cell (fine block)
    z_gnd_hi = float(zcum[k_fine0 + 1])
    z_sub_lo = z_gnd_hi                        # substrate = next n_sub cells
    z_sub_hi = float(zcum[k_fine0 + 1 + n_sub])
    z_patch_lo = z_sub_hi                      # thin patch box snaps to the
    z_patch_hi = z_sub_hi + dz_sub             # cell just above the substrate

    # Fine region = patch + feed +/- 2 mm margin
    m = 2.0e-3
    fine_x_lo = min(patch_x_lo, feed_x) - m
    fine_x_hi = max(patch_x_hi, feed_x) + m
    fine_y_lo = patch_y_lo - m
    fine_y_hi = patch_y_hi + m
    dx_profile = _graded_profile(dom_x, fine_x_lo, fine_x_hi, dx_fine, dx_coarse)
    dy_profile = _graded_profile(dom_y, fine_y_lo, fine_y_hi, dx_fine, dx_coarse)

    return dict(
        dz_sub=dz_sub, dom_x=dom_x, dom_y=dom_y, dom_z=dom_z,
        gx_lo=gx_lo, gx_hi=gx_hi, gy_lo=gy_lo, gy_hi=gy_hi,
        patch_x_lo=patch_x_lo, patch_x_hi=patch_x_hi,
        patch_y_lo=patch_y_lo, patch_y_hi=patch_y_hi,
        feed_x=feed_x, feed_y=feed_y,
        z_gnd_lo=z_gnd_lo, z_gnd_hi=z_gnd_hi,
        z_sub_lo=z_sub_lo, z_sub_hi=z_sub_hi,
        z_patch_lo=z_patch_lo, z_patch_hi=z_patch_hi,
        dz_profile=dz_profile, dx_profile=dx_profile, dy_profile=dy_profile,
    )


def build_geometry_uniform(n_sub):
    """UNIFORM-grid geometry: one cell size dx_u = h_sub/n_sub everywhere,
    every coordinate snapped to an integer number of cells.

    WHY: the NU lane turned out to be a minefield for ports — (1) the
    smooth_grading insertion misaligns geometry vs mesh unless carefully
    anchored, and (2) the NU runner never folds lumped/wire port impedance
    into the materials (setup_wire_port exists only on the uniform path),
    so run(compute_s_params=True) on NU produces an UNLOADED port (pure
    series-C Zin, no resonance in V/I — measured). The uniform lane is the
    validated one (issue #80 patch S11). Integer-cell snapping makes the
    Box rasterization and port-cell mapping exact by construction.
    """
    dx_u = h_sub / n_sub                      # 0.3048 mm for n_sub=5

    def cells(length):
        return max(1, int(round(length / dx_u)))

    n_gx = cells(sub_wx)                      # ground/substrate footprint
    n_gy = cells(sub_wy)
    n_px = cells(patch_Lx)                    # patch
    n_py = cells(patch_Ly)
    n_mx = cells(margin_xy)
    n_below = cells(air_below)
    n_above = cells(air_above)

    n_dom_x = n_gx + 2 * n_mx
    n_dom_y = n_gy + 2 * n_mx
    n_dom_z = n_below + n_sub + n_above
    dom_x = n_dom_x * dx_u
    dom_y = n_dom_y * dx_u
    dom_z = n_dom_z * dx_u

    # Centre ground + patch on integer cell boundaries
    gx_lo = ((n_dom_x - n_gx) // 2) * dx_u
    gx_hi = gx_lo + n_gx * dx_u
    gy_lo = ((n_dom_y - n_gy) // 2) * dx_u
    gy_hi = gy_lo + n_gy * dx_u
    px_lo_c = (n_dom_x - n_px) // 2
    py_lo_c = (n_dom_y - n_py) // 2
    patch_x_lo = px_lo_c * dx_u
    patch_x_hi = (px_lo_c + n_px) * dx_u
    patch_y_lo = py_lo_c * dx_u
    patch_y_hi = (py_lo_c + n_py) * dx_u

    # Feed node snapped to the nearest grid node
    feed_x = round((dom_x / 2 + feed_pos) / dx_u) * dx_u
    feed_y = round((dom_y / 2) / dx_u) * dx_u

    # z stack on exact cell boundaries
    z_gnd_lo = (n_below - 1) * dx_u
    z_gnd_hi = n_below * dx_u
    z_sub_lo = z_gnd_hi
    z_sub_hi = (n_below + n_sub) * dx_u
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dx_u

    return dict(
        uniform=True, dx_u=dx_u, dz_sub=dx_u,
        dom_x=dom_x, dom_y=dom_y, dom_z=dom_z,
        gx_lo=gx_lo, gx_hi=gx_hi, gy_lo=gy_lo, gy_hi=gy_hi,
        patch_x_lo=patch_x_lo, patch_x_hi=patch_x_hi,
        patch_y_lo=patch_y_lo, patch_y_hi=patch_y_hi,
        feed_x=feed_x, feed_y=feed_y,
        z_gnd_lo=z_gnd_lo, z_gnd_hi=z_gnd_hi,
        z_sub_lo=z_sub_lo, z_sub_hi=z_sub_hi,
        z_patch_lo=z_patch_lo, z_patch_hi=z_patch_hi,
        dz_profile=None, dx_profile=None, dy_profile=None,
    )


def _graded_profile(dom_len, interior_lo, interior_hi, dx_fine, dx_coarse):
    """dx profile: coarse (dx_coarse) outside [interior_lo, interior_hi], fine
    (dx_fine) inside, smooth-graded (max cell ratio 1.3). Mirrors cv05's
    _refined_xy_profile; first/last cells stay dx_coarse for uniform CPML cells.
    """
    from rfx.auto_config import smooth_grading
    lo_len = max(interior_lo, 0.0)
    hi_len = max(dom_len - interior_hi, 0.0)
    fine_len = max(interior_hi - interior_lo, 0.0)
    n_lo = max(1, int(round(lo_len / dx_coarse)))
    n_fine = max(1, int(round(fine_len / dx_fine)))
    n_hi = max(1, int(round(hi_len / dx_coarse)))
    raw = np.concatenate([
        np.full(n_lo, dx_coarse),
        np.full(n_fine, dx_fine),
        np.full(n_hi, dx_coarse),
    ])
    return smooth_grading(raw, max_ratio=1.3)


def build_sim(g, with_port, pulse_f0, pulse_bw, feed_mode="wire"):
    """Build the canonical patch stack. ``g`` = build_geometry() dict.

    Ground plane is an explicit finite-size PEC box BELOW the substrate (the
    physically-correct finite ground plane radiating into the bottom CPML),
    matching the cv05 root-cause fix rather than pec_faces (which would make an
    infinite PEC sheet / cavity and bias the resonance).
    """
    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources.sources import GaussianPulse

    if g.get("uniform"):
        # UNIFORM validated lane: single dx everywhere, ports get their
        # impedance folded (setup_wire_port), no NU alignment traps.
        sim = Simulation(
            freq_max=4e9,
            domain=(g["dom_x"], g["dom_y"], g["dom_z"]),
            dx=g["dx_u"],
            boundary=BoundarySpec.uniform("cpml"),
            cpml_layers=n_cpml,
        )
    else:
        sim = Simulation(
            freq_max=4e9,
            domain=(g["dom_x"], g["dom_y"], 0),
            dx=dx_coarse,
            dz_profile=g["dz_profile"],
            dx_profile=g["dx_profile"],
            dy_profile=g["dy_profile"],
            boundary=BoundarySpec.uniform("cpml"),
            cpml_layers=n_cpml,
            # no pec_faces: bottom CPML absorbs radiation below the finite ground
        )
    sim.add_material("rogers", eps_r=eps_r, sigma=sub_sigma)
    # Finite ground plane: 60 x 60 mm PEC, 1 cell thick, BELOW the substrate
    sim.add(Box((g["gx_lo"], g["gy_lo"], g["z_gnd_lo"]),
                (g["gx_hi"], g["gy_hi"], g["z_gnd_hi"])), material="pec")
    # Rogers substrate
    sim.add(Box((g["gx_lo"], g["gy_lo"], g["z_sub_lo"]),
                (g["gx_hi"], g["gy_hi"], g["z_sub_hi"])), material="rogers")
    # Patch: 1 cell thick PEC on top of substrate (32 x 40 mm)
    sim.add(Box((g["patch_x_lo"], g["patch_y_lo"], g["z_patch_lo"]),
                (g["patch_x_hi"], g["patch_y_hi"], g["z_patch_hi"])), material="pec")

    if with_port:
        # VERTICAL lumped port spanning the FULL substrate: ground (z_sub_lo,
        # top of the PEC ground) -> patch (z_sub_hi). This is the canonical
        # openEMS ground->patch feed span (not cv05's half-substrate probe).
        # Span from the CENTRE of the first substrate cell to the centre of
        # the last one so index rounding covers exactly the n_sub substrate
        # Ez cells — the wire then touches the ground PEC below and the patch
        # PEC above via shared nodes (any end-cell miss leaves a dielectric
        # gap = series capacitor = the flat-|S11| failure).
        # End at z_sub_hi - 0.75*dz (strictly inside the last substrate cell)
        # so nearest-node rounding cannot spill the wire into the patch PEC
        # cell (a spilled cell is shorted and skews the port R off 50 ohm).
        if feed_mode == "via":
            # PROBE-VIA FEED (Luebbers-style): a PEC via column shorts the
            # bottom n_sub-1 substrate cells from the ground plane up; a
            # SINGLE-CELL lumped port sits in the remaining gap cell just
            # under the patch. Uses only validated machinery: the 1-cell
            # port (N=1 -> midpoint == whole port, no multi-cell wire
            # convention ambiguity) + PEC boxes. Mirrors how a real coax
            # probe feeds a patch: metal pin + localized excitation gap.
            dz = g["dz_sub"]
            via_hw = 0.4 * dz  # sub-cell half-width -> rasterizes to 1 column
            sim.add(Box((g["feed_x"] - via_hw, g["feed_y"] - via_hw,
                         g["z_sub_lo"]),
                        (g["feed_x"] + via_hw, g["feed_y"] + via_hw,
                         g["z_sub_hi"] - dz)), material="pec")
            sim.add_port(
                position=(g["feed_x"], g["feed_y"], g["z_sub_hi"] - 0.6 * dz),
                component="ez",
                impedance=Z0_PORT,
                waveform=GaussianPulse(f0=pulse_f0, bandwidth=pulse_bw),
            )
        else:
            # Multi-cell WIRE port spanning the substrate. Tie-free offsets
            # (0.4 / 1.8 cells) so round-to-node can never banker's-round
            # the wire ends off the substrate cells.
            port_z0 = g["z_sub_lo"] + 0.4 * g["dz_sub"]
            port_ext = h_sub - 1.8 * g["dz_sub"]
            if not g.get("uniform"):
                # NU span verified cell-exact by the connectivity probe.
                port_z0 = g["z_sub_lo"] + g["dz_sub"] / 2
                port_ext = h_sub - 1.25 * g["dz_sub"]
            sim.add_port(
                position=(g["feed_x"], g["feed_y"], port_z0),
                component="ez",
                impedance=Z0_PORT,
                extent=port_ext,
                waveform=GaussianPulse(f0=pulse_f0, bandwidth=pulse_bw),
            )
    else:
        # Broadband Ez source + offset probe for Harminv ring-down resonance.
        src_z = g["z_sub_lo"] + g["dz_sub"] * 2.5
        sim.add_source(
            position=(g["feed_x"], g["feed_y"], src_z),
            component="ez",
            waveform=GaussianPulse(f0=pulse_f0, bandwidth=1.2),
        )
        sim.add_probe(
            position=(g["dom_x"] / 2 + 6e-3, g["dom_y"] / 2 + 6e-3, src_z),
            component="ez",
        )
    return sim


def run_once(args):
    import jax.numpy as jnp
    from rfx.harminv import harminv

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    dx_fine = args.dx_mm * 1e-3
    if args.uniform:
        g = build_geometry_uniform(args.n_sub)
        print(f"  UNIFORM lane: dx = {g['dx_u']*1e3:.4f} mm everywhere "
              f"(h_sub/{args.n_sub}); --dx-mm ignored for meshing")
    else:
        g = build_geometry(dx_fine, args.n_sub)
    tag = (f"uniform_{g['dx_u']*1e3:.4f}mm" if args.uniform
           else f"{args.dx_mm:.2f}mm")
    if args.feed_mode == "via":
        tag += "_via"
    if args.uniform:
        nx = int(round(g["dom_x"] / g["dx_u"]))
        ny = int(round(g["dom_y"] / g["dx_u"]))
        nz = int(round(g["dom_z"] / g["dx_u"]))
    else:
        nx = len(g["dx_profile"])
        ny = len(g["dy_profile"])
        nz = len(g["dz_profile"])
    ncells = nx * ny * nz

    freqs_hz = np.linspace(args.freq_lo_ghz * 1e9, args.freq_hi_ghz * 1e9,
                           args.nfreq)
    pulse_f0 = args.pulse_f0_ghz * 1e9

    print("=" * 72)
    print("rfx CANONICAL patch (openEMS Simple_Patch_Antenna replica)")
    print("=" * 72)
    print(f"  Substrate: eps_r={eps_r}, h={h_sub*1e3:.3f} mm, "
          f"sigma={sub_sigma:.4e} S/m (tan_delta~1e-3)")
    print(f"  Patch: {patch_Lx*1e3:.0f} x {patch_Ly*1e3:.0f} mm  "
          f"Ground/sub: {sub_wx*1e3:.0f} x {sub_wy*1e3:.0f} mm")
    print(f"  Feed (openEMS x={feed_pos*1e3:+.1f} mm, y=0) -> rfx abs "
          f"({g['feed_x']*1e3:.2f}, {g['feed_y']*1e3:.2f}) mm, "
          f"z {g['z_sub_lo']*1e3:.3f}->{g['z_sub_hi']*1e3:.3f} mm")
    print(f"  Domain: {g['dom_x']*1e3:.0f} x {g['dom_y']*1e3:.0f} x "
          f"{g['dom_z']*1e3:.0f} mm  grid {nx} x {ny} x {nz} = {ncells:,} cells")
    print(f"  dx_fine={args.dx_mm:.3f} mm  dx_coarse={dx_coarse*1e3:.1f} mm  "
          f"n_sub={args.n_sub}  cpml={n_cpml}")
    print(f"  openEMS reference: min|S11|={REF_S11_MIN_DB:.1f} dB @ "
          f"{REF_F_HZ/1e9:.3f} GHz, Zin={REF_ZIN.real:.1f}{REF_ZIN.imag:+.1f}j")
    print("=" * 72)

    # ---- Harminv ring-down resonance (clean, port-independent) ----
    print("\n[1/2] Harminv ring-down run...")
    sim_h = build_sim(g, with_port=False, pulse_f0=pulse_f0, pulse_bw=args.pulse_bw)
    t0 = time.time()
    res_h = sim_h.run(num_periods=args.num_periods)
    t_harminv = time.time() - t0
    ts = np.asarray(res_h.time_series).ravel()
    dt_h = float(res_h.dt)
    skip = int(len(ts) * 0.3)
    modes = harminv(ts[skip:], dt_h, args.freq_lo_ghz * 1e9, args.freq_hi_ghz * 1e9)
    modes_good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]
    if modes_good:
        modes_good.sort(key=lambda m: abs(m.freq - REF_F_HZ))
        f_res_harminv = float(modes_good[0].freq)
        Q_harminv = float(modes_good[0].Q)
    else:
        f_res_harminv = float("nan")
        Q_harminv = float("nan")
    print(f"  Harminv: f={f_res_harminv/1e9:.4f} GHz Q={Q_harminv:.1f} "
          f"({t_harminv:.1f}s, dt={dt_h*1e12:.4f} ps)")

    # ---- Full complex S11(f) via the lumped port ----
    print("\n[2/2] Lumped-port S11 run...")
    sim_p = build_sim(g, with_port=True, pulse_f0=pulse_f0, pulse_bw=args.pulse_bw)
    print("  Preflight:")
    sim_p.preflight(strict=False)
    freqs_s = jnp.asarray(freqs_hz)
    t0 = time.time()
    result = sim_p.run(
        n_steps=args.n_steps,
        compute_s_params=True,
        s_param_freqs=freqs_s,
        s_param_n_steps=args.n_steps,
    )
    t_s11 = time.time() - t0
    S = np.asarray(result.s_params)
    S11_raw = S[0, 0, :]

    # ---- Multi-cell wire-port midpoint-convention correction -------------
    # rfx's wire-port extractor measures V/I at the MIDPOINT CELL only
    # (probes.py wire_port_voltage: one cell's -E*dz) but the DIAGONAL S11
    # decomposition compares that per-cell impedance against the TOTAL
    # Z0=50 (decompose_wire_s_matrix). For an N-cell wire the measured
    # z is ~Zin_true/N, so the raw diagonal S11 is wrong for N>1 (the
    # off-diagonal path already normalizes by Z0/n_cells — the diagonal
    # doesn't). Invert the extractor's mapping and redo the reflection
    # against the per-cell reference, identically to the off-diagonal
    # convention:  z_meas = Z0(1+S)/(1-S);  S_corr = (z_meas - Z0/N)/(z_meas + Z0/N)
    # Physical input impedance: Zin_true = N * z_meas (should read ~54 ohm
    # at resonance if the feed chain is healthy — openEMS reference).
    n_wire = args.n_sub  # NU: probe-verified; uniform: recomputed exactly below
    if args.feed_mode == "via":
        n_wire = 1  # single-cell port: all conventions coincide
    elif args.uniform:
        from rfx.sources.sources import WirePort, _wire_port_cells
        pe = sim_p._ports[0]
        end = list(pe.position); end[2] += pe.extent
        n_wire = len(_wire_port_cells(
            sim_p._build_grid(),
            WirePort(start=pe.position, end=tuple(end), component="ez",
                     impedance=Z0_PORT)))
        print(f"  wire port cells (exact, uniform grid): {n_wire}")
    z_meas = Z0_PORT * (1 + S11_raw) / (1 - S11_raw)
    zin_true = n_wire * z_meas
    S11 = (zin_true - Z0_PORT) / (zin_true + Z0_PORT)
    S11_dB = 20 * np.log10(np.maximum(np.abs(S11), 1e-6))
    Z11 = zin_true

    # Global minimum |S11| over the band (matched patch -> deep null).
    idx_min = int(np.argmin(S11_dB))
    f_min = float(freqs_hz[idx_min])
    s11_min_dB = float(S11_dB[idx_min])
    zin_min = complex(Z11[idx_min])
    passive = bool(np.all(np.abs(S11) < 1.05))
    max_abs = float(np.max(np.abs(S11)))

    # Value nearest the openEMS reference frequency (2.085 GHz) for comparison.
    idx_ref = int(np.argmin(np.abs(freqs_hz - REF_F_HZ)))
    s11_at_ref_dB = float(S11_dB[idx_ref])
    zin_at_ref = complex(Z11[idx_ref])

    print(f"\n  S11 global min: {s11_min_dB:.2f} dB @ {f_min/1e9:.4f} GHz  "
          f"Zin={zin_min.real:.1f}{zin_min.imag:+.1f}j  ({t_s11:.1f}s)")
    print(f"  S11 @ ref {REF_F_HZ/1e9:.3f} GHz: {s11_at_ref_dB:.2f} dB  "
          f"Zin={zin_at_ref.real:.1f}{zin_at_ref.imag:+.1f}j")
    print(f"  Passivity |S11|<=1: {'PASS' if passive else 'FAIL'} "
          f"(max|S11|={max_abs:.3f})")
    print(f"  openEMS ref: {REF_S11_MIN_DB:.1f} dB @ {REF_F_HZ/1e9:.3f} GHz  "
          f"Zin={REF_ZIN.real:.1f}{REF_ZIN.imag:+.1f}j")

    payload = {
        "case": "canonical_openems_simple_patch_antenna",
        "dx_mm": args.dx_mm,
        "n_sub": args.n_sub,
        "n_steps": args.n_steps,
        "grid": {"nx": nx, "ny": ny, "nz": nz, "ncells": ncells},
        "dt_s": dt_h,
        "runtime_s": {"harminv": t_harminv, "s11": t_s11},
        "geometry": {
            "eps_r": eps_r, "h_sub_mm": h_sub * 1e3, "sub_sigma_S_per_m": sub_sigma,
            "patch_Lx_mm": patch_Lx * 1e3, "patch_Ly_mm": patch_Ly * 1e3,
            "sub_wx_mm": sub_wx * 1e3, "sub_wy_mm": sub_wy * 1e3,
            "feed_pos_mm": feed_pos * 1e3,
            "feed_abs_mm": [g["feed_x"] * 1e3, g["feed_y"] * 1e3],
            "domain_mm": [g["dom_x"] * 1e3, g["dom_y"] * 1e3, g["dom_z"] * 1e3],
        },
        "reference_openems": {
            "s11_min_db": REF_S11_MIN_DB, "f_hz": REF_F_HZ,
            "zin_ohm": [REF_ZIN.real, REF_ZIN.imag],
        },
        "harminv_hz": f_res_harminv,
        "harminv_Q": Q_harminv,
        "s11_min_db": s11_min_dB,
        "s11_min_hz": f_min,
        "zin_at_min_ohm": [zin_min.real, zin_min.imag],
        "s11_at_ref_db": s11_at_ref_dB,
        "zin_at_ref_ohm": [zin_at_ref.real, zin_at_ref.imag],
        "s11_max_abs": max_abs,
        "s11_passive": passive,
        "freqs_hz": [float(v) for v in freqs_hz],
        "s11": [[float(v.real), float(v.imag)] for v in S11],
        "s11_raw_midpoint_convention": [[float(v.real), float(v.imag)] for v in S11_raw],
        "wire_n_cells": int(n_wire),
        "z11": [[float(v.real), float(v.imag)] for v in Z11],
    }
    json_path = os.path.join(out_dir, f"rfx_patch_canonical_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n  wrote {json_path}")

    png_path = _plot(out_dir, args, freqs_hz, S11, S11_dB, f_res_harminv,
                     f_min, s11_min_dB, idx_min)
    print(f"  wrote {png_path}")
    return payload


def _plot(out_dir, args, freqs_hz, S11, S11_dB, f_res_harminv,
          f_min, s11_min_dB, idx_min):
    f_g = freqs_hz / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(f_g, S11_dB, "r-", lw=1.6, label="rfx lumped-port |S11|")
    axes[0].axvline(REF_F_HZ / 1e9, color="b", ls="--", alpha=0.7,
                    label=f"openEMS ref {REF_F_HZ/1e9:.3f} GHz")
    axes[0].scatter([REF_F_HZ / 1e9], [REF_S11_MIN_DB], color="b", s=50, zorder=5,
                    label=f"openEMS min {REF_S11_MIN_DB:.1f} dB")
    if not np.isnan(f_res_harminv):
        axes[0].axvline(f_res_harminv / 1e9, color="red", ls=":", alpha=0.8,
                        label=f"rfx Harminv {f_res_harminv/1e9:.3f} GHz")
    axes[0].scatter([f_min / 1e9], [s11_min_dB], color="r", s=50, zorder=5,
                    label=f"rfx min {s11_min_dB:.1f} dB @ {f_min/1e9:.3f} GHz")
    axes[0].axhline(-10, color="gray", ls=":", alpha=0.5)
    axes[0].axhline(-15, color="gray", ls=":", alpha=0.3)
    axes[0].set_xlabel("f (GHz)"); axes[0].set_ylabel("|S11| (dB)")
    axes[0].set_title("Return Loss |S11| — canonical matched patch")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-40, 2)

    # Smith-plane view (complex S11 on the unit circle)
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.5)
    axes[1].plot([-1, 1], [0, 0], "k-", lw=0.5, alpha=0.3)
    axes[1].plot([0, 0], [-1, 1], "k-", lw=0.5, alpha=0.3)
    axes[1].plot(S11.real, S11.imag, "r-", lw=1.2)
    axes[1].scatter([S11[idx_min].real], [S11[idx_min].imag], color="r", s=50,
                    zorder=5, label=f"min @ {f_min/1e9:.3f} GHz")
    axes[1].set_xlabel("Re(S11)"); axes[1].set_ylabel("Im(S11)")
    axes[1].set_title("Complex S11 (unit circle = passivity)")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-1.2, 1.2); axes[1].set_ylim(-1.2, 1.2); axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Canonical openEMS Simple_Patch_Antenna replica (rfx)  —  "
        f"Rogers eps_r={eps_r}, {patch_Lx*1e3:.0f}x{patch_Ly*1e3:.0f} mm patch, "
        f"-6 mm inset feed  (dx_fine={args.dx_mm:.2f} mm)",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"rfx_patch_canonical_{tag}.png")
    plt.savefig(png_path, dpi=150); plt.close()
    return png_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx-mm", type=float, default=1.0,
                   help="interior (substrate/patch/feed) fine cell size in mm "
                        "for the NU graded mesh (air/PML stay ~2 mm). Default 1.0.")
    p.add_argument("--output", default=os.path.join(SCRIPT_DIR, "out_canonical"),
                   help="output directory for per-run JSON + |S11| PNG")
    p.add_argument("--n-steps", type=int, default=12000,
                   help="FDTD steps for the S-parameter (S11) run")
    p.add_argument("--num-periods", type=float, default=60.0,
                   help="ring-down periods for the Harminv run")
    p.add_argument("--n-sub", type=int, default=n_sub_default,
                   help="z cells across the 1.524 mm substrate (4-6 typical)")
    p.add_argument("--freq-lo-ghz", type=float, default=1.5)
    p.add_argument("--freq-hi-ghz", type=float, default=3.0)
    p.add_argument("--nfreq", type=int, default=151)
    p.add_argument("--pulse-f0-ghz", type=float, default=2.1,
                   help="Gaussian source centre freq (near the 2.085 GHz null)")
    p.add_argument("--feed-mode", choices=["wire", "via"], default="wire",
                   help="Feed model: 'wire' = multi-cell wire port spanning "
                        "the substrate; 'via' = PEC probe via + single-cell "
                        "lumped port in the top gap (validated N=1 path)")
    p.add_argument("--uniform", action="store_true",
                   help="Uniform validated lane: dx = h_sub/n_sub everywhere, "
                        "integer-cell-snapped geometry (bypasses the NU port "
                        "and alignment traps)")
    p.add_argument("--pulse-bw", type=float, default=1.0,
                   help="fractional bandwidth of the lumped-port Gaussian pulse")
    args = p.parse_args()
    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
