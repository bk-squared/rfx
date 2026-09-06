"""Patch antenna end to end: multi-mode ring-down, far-field mode ID, error budget.

Geometry: the openEMS "Simple Patch Antenna" tutorial — a 32 x 40 mm patch on a
1.524 mm, eps_r = 3.38, tan_delta = 1e-3 substrate over a 60 x 60 mm finite
ground plane, probe-fed 6 mm off centre.  The identical geometry in openEMS,
with that tutorial's thirds-rule edge meshing, gives the reference numbers used
below: f_res = 2.4221 GHz, broadside directivity 6.79 dBi.

What this tutorial teaches, in order:

1. Thin-substrate stacks on a graded z-mesh must be registered to the BUILT
   mesh.  ``smooth_grading`` inserts transition cells that shift the fine band,
   so a stack placed at pre-smoothing coordinates can silently rasterize onto
   coarse cells — the resonance then shifts or splits.  Every z coordinate here
   is derived from the final ``dz_profile``.  ``nonuniform_patch_demo.py``
   teaches the mesh side in detail; this script only applies the rule.

2. A patch is MULTI-MODE.  ``harminv`` returns several ring-down modes, and the
   radiating one must be identified from the FAR FIELD — a broadside beam plus
   the peak of the radiated-power spectrum — not by amplitude rank and not by
   whichever mode sits closest to a textbook estimate.  Both shortcuts have
   mis-identified modes on this exact structure.

3. Near-to-far-field (NTFF) box placement.  Every face should sit at least half
   a wavelength from the radiator where the domain allows.  The face below the
   ground plane cannot: it sits 6 mm below the ground.  Preflight flags it, the
   run quotes the warning verbatim, and a solver-to-solver cross-check backs the
   placement.  Preflight output is part of the result; do not suppress it.

4. A settling witness (end-of-run envelope vs post-source peak, -40 dB bar) is
   printed BEFORE any frequency is quoted.

Error budget at this deliberately coarse resolution (dx = 2 mm):

- The resonance reads HIGH against openEMS-with-thirds: +11.3 % at dx = 2 mm,
  2.6953 GHz here against 2.4221 GHz.  Both numbers are the 32 mm feed-axis
  design mode; the 40 mm cross mode is a separate entry in the printed mode
  list (2.2157 GHz here).  (Mode pairing repinned 2026-08-27, #693.)  The
  committed lock ``tests/crossval/test_patch_canonical_farfield_e4.py`` gates
  the [+6 %, +16 %] magnitude envelope and the sign.
- The sign is not a settled coarse-grid bias that finer cells remove.  Two
  discretization errors push opposite ways: the substrate under-resolved in z
  reads high, the staircased PEC patch edge reads low.  Which one wins depends
  on how the one-cell PEC sheets are realized — the same canonical patch reads
  +11.3 % with single-plane sheets and -1.5 % at dx = 1 mm with the two-plane
  realization (issue #720).  The substrate-permittivity half is addressable
  with the opt-in interface treatment ``sim.run(..., subpixel_smoothing=True)``
  (cavity-oracle evidence in ``tests/oracle/test_patch_cavity_eps_oracle.py``).
  Measure "finer dx recovers it" on this fixture before assuming it.
- The far field is the observable that agrees.  This configuration prints
  D = 7.39 dBi at its radiating bin against openEMS 6.79 dBi, +0.60 dB, inside
  the committed envelope lock (``D_ABS_TOL_DB = 1.0``).  The demo trims the air
  above the patch to 84 mm with ``num_periods = 125``; the research frame
  behind the reference numbers used 95 mm and ``num_periods = 250``.

Run as::

    python examples/tutorials/patch_antenna_demo.py

Measured 2026-09-05, 64-core CPU, run alone: 1345 s (22 min).
``NUM_PERIODS = 125`` is sized from measurement: 90 periods gave -36.5 dB and
204 periods gave -52.6 dB, which puts the -40 dB settling bar near 115 on the
average slope, and the multi-mode tail beats rather than decaying smoothly.
The 2026-09-05 run measured -50.9 dB, SETTLED.  The witness below re-measures
the end-of-run envelope every run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import (
    Box,
    GaussianPulse,
    Simulation,
    compute_far_field_jax,
    directivity,
    harminv,
    smooth_grading,
)

C0 = 2.998e8
EPS0 = 8.8541878e-12

# ---- Geometry: identical to the openEMS "Simple Patch Antenna" tutorial ----
PATCH_W = 32.0e-3          # x extent of the patch
PATCH_L = 40.0e-3          # y extent of the patch
SUB_EPS_R = 3.38
SUB_THICK = 1.524e-3
TAN_DELTA = 1.0e-3
GP_SIZE = 60.0e-3          # finite ground plane, 60 x 60 mm
FEED_OFFSET_X = -6.0e-3    # probe feed 6 mm off centre
F_DESIGN = 2.2e9
SIGMA_SUB = 2 * math.pi * 2.45e9 * EPS0 * SUB_EPS_R * TAN_DELTA

# openEMS-with-thirds reference values for the identical geometry.
OPENEMS_F_RES = 2.4221e9
OPENEMS_D_DBI = 6.79

# ---- Mesh and domain (coarse on purpose: dx = 2 mm, 22 min measured) ----
DX = 2.0e-3
N_CPML = 8
N_SUB = 4                  # fine cells across the substrate thickness
DZ_SUB = SUB_THICK / N_SUB
MARGIN_XY = 85.0e-3        # air beyond the ground-plane edge, x and y
AIR_BELOW = 30.0e-3
# 84 mm keeps the top NTFF face half a wavelength above the patch at F_DESIGN
# at the measured 22-minute runtime.  The research frame behind the quoted
# openEMS reference numbers used 95 mm of air and num_periods = 250.
AIR_ABOVE = 84.0e-3

# 125 periods settles this fixture past the -40 dB bar: measured -50.9 dB,
# SETTLED (2026-09-05).  Endpoints -36.5 dB at 90 periods and -52.6 dB at 204
# put the bar near 115 on the average slope, but the multi-mode tail beats
# rather than decaying smoothly, so trust the printed witness over slope
# extrapolation.  (The research run of this geometry recorded f_res =
# 2.2147 GHz and D = 6.71 dBi at the 40 mm cross mode, not at the design mode
# this demo reports.)
#
# Why not run(until_decay=...): its total-interior-energy stop cannot fire on
# this fixture — the soft current feed leaves a static charge field across
# the patch-ground gap (~1% of peak interior energy) that neither radiates
# nor is absorbed by CPML, so a decay-terminated run walks to its step cap.
# See issue #388.
NUM_PERIODS = 125

# Far-field bins: a ladder around the expected coarse-grid resonance plus one
# bin near the second ring-down mode, so every harminv candidate gets a
# far-field verdict.
NTFF_FREQS = np.array([2.0e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.5e9, 2.8e9])

OUTPUT_PATH = Path(__file__).with_name("output") / "patch_antenna_cuts.png"


def build_simulation():
    """Build the patch on a graded z-mesh, registering the stack to the mesh."""
    n_below = int(math.ceil(AIR_BELOW / DX))
    n_above = int(math.ceil(AIR_ABOVE / DX))
    dom_x = GP_SIZE + 2 * MARGIN_XY
    dom_y = GP_SIZE + 2 * MARGIN_XY
    cx, cy = dom_x / 2, dom_y / 2

    # Fine band: 1 ground cell + N_SUB substrate cells + 1 patch cell.
    raw_dz = np.concatenate([
        np.full(n_below, DX),
        np.full(1 + N_SUB + 1, DZ_SUB),
        np.full(n_above, DX),
    ])
    dz_profile = smooth_grading(raw_dz, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz_profile), 0, 0.0)
    z_total = float(edges[-1])

    # DERIVE the stack z coordinates from where smooth_grading actually put the
    # fine band.  Do NOT reuse AIR_BELOW: the inserted transition cells shift
    # the band upward, and a stack placed at the pre-smoothing coordinate can
    # land on coarse cells.  (nonuniform_patch_demo.py shows the mesh anatomy.)
    fine = np.where(np.isclose(dz_profile, DZ_SUB, rtol=1e-6))[0]
    if len(fine) < 2 + N_SUB:
        raise RuntimeError(
            f"graded mesh lost the fine band: expected >= {2 + N_SUB} fine "
            f"cells, found {len(fine)}"
        )
    f0 = int(fine[0])
    z_gnd_lo, z_gnd_hi = edges[f0], edges[f0 + 1]
    z_sub_lo, z_sub_hi = edges[f0 + 1], edges[f0 + 1 + N_SUB]
    z_patch_lo, z_patch_hi = z_sub_hi, edges[f0 + 1 + N_SUB + 1]

    # Verify the realized rasterization, then fail loudly on a mismatch: a
    # substrate on the wrong cells produces a wrong resonance, not a crash.
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    print(
        f"z-mesh: substrate rasterizes to {sub_cells} fine cells "
        f"(intended {N_SUB}); z_sub = [{z_sub_lo * 1e3:.3f}, "
        f"{z_sub_hi * 1e3:.3f}] mm"
    )
    if sub_cells != N_SUB:
        raise RuntimeError(
            f"substrate landed on {sub_cells} cells instead of {N_SUB} — "
            "stack is mis-registered to the graded mesh"
        )

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, 0),
        dx=DX,
        dz_profile=dz_profile,
        boundary="cpml",
        cpml_layers=N_CPML,
    )
    sim.add_material("sub", eps_r=SUB_EPS_R, sigma=SIGMA_SUB)

    gx_lo, gx_hi = cx - GP_SIZE / 2, cx + GP_SIZE / 2
    gy_lo, gy_hi = cy - GP_SIZE / 2, cy + GP_SIZE / 2
    # Ground plane and patch are each exactly one fine cell thick, snapped to
    # cell edges.  Preflight prints its sub-cell PEC advisory (quoted in the run
    # output) because a 381 um sheet sits at the 1-cell floor; that thickness is
    # deliberate and edge-aligned here.
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="sub")
    sim.add(
        Box(
            (cx - PATCH_W / 2, cy - PATCH_L / 2, z_patch_lo),
            (cx + PATCH_W / 2, cy + PATCH_L / 2, z_patch_hi),
        ),
        material="pec",
    )

    # Soft Ez source through the substrate at the feed point; once the patch
    # resonates, the radiated field is patch-dominated, not feed-dominated.
    feed_x, feed_y = cx + FEED_OFFSET_X, cy
    src_z = z_sub_lo + DZ_SUB * 1.5
    sim.add_source(
        position=(feed_x, feed_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2),
        amplitude_kind="current",
    )
    sim.add_probe(position=(feed_x + 4e-3, feed_y + 4e-3, src_z), component="ez")

    # NTFF box.  Side and top faces keep >= lambda/2-class clearance.  The
    # bottom face cannot: the domain holds only 30 mm of air below the ground
    # plane, so that face sits 6 mm below the ground and preflight flags it (the
    # warning is quoted in the run output).  The ground plane sits between the
    # radiator and that face, and the placement is cross-checked against openEMS
    # at the design-mode bin: +0.60 dB in directivity, inside the committed
    # 1.0 dB envelope lock.
    pad = (N_CPML + 3) * DX
    box_lo = (pad, pad, max(pad, z_gnd_lo - 3 * DX))
    box_hi = (dom_x - pad, dom_y - pad, z_total - pad)
    sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=NTFF_FREQS)

    lam_design = C0 / F_DESIGN
    lam_fmax = C0 / float(NTFF_FREQS.max())
    print(
        f"NTFF clearances: ground-plane edge -> side face "
        f"{(gx_lo - box_lo[0]) * 1e3:.1f} mm | patch top -> top face "
        f"{(box_hi[2] - z_patch_hi) * 1e3:.1f} mm | ground -> bottom face "
        f"{(z_gnd_lo - box_lo[2]) * 1e3:.1f} mm"
    )
    print(
        f"  (lambda/2 = {lam_design / 2 * 1e3:.1f} mm at {F_DESIGN / 1e9:.1f} GHz, "
        f"{lam_fmax / 2 * 1e3:.1f} mm at {NTFF_FREQS.max() / 1e9:.1f} GHz)"
    )
    return sim


def half_power_beamwidth_deg(angle_deg, power_lin):
    """Half-power beamwidth of a single-cut pattern, NaN if no -3 dB crossing."""
    p = np.asarray(power_lin, float)
    p = p / np.max(p)
    ipk = int(np.argmax(p))

    def edge(direction):
        i = ipk
        while 0 <= i + direction < len(p) and p[i] >= 0.5:
            i += direction
        if p[i] >= 0.5:
            return np.nan
        j = i - direction
        t = (p[j] - 0.5) / (p[j] - p[i])
        return angle_deg[j] + t * (angle_deg[i] - angle_deg[j])

    lo, hi = edge(-1), edge(+1)
    return float("nan") if (np.isnan(lo) or np.isnan(hi)) else abs(hi - lo)


def principal_cut(power_f, phi_index_pos, phi_index_neg):
    """Compose a -90..+90 degree cut from two opposite azimuth columns.

    ``power_f`` is the (n_theta, n_phi) power at one frequency on the
    theta = 0..180 deg, phi = 0..360 deg sphere grid.
    """
    n_half = 91  # theta = 0..90 deg at 1 deg spacing
    pos = power_f[:n_half, phi_index_pos]           # angle = +theta
    neg = power_f[:n_half, phi_index_neg][::-1]     # angle = -theta
    angle = np.concatenate([-np.arange(90, 0, -1.0), np.arange(0, 91, 1.0)])
    return angle, np.concatenate([neg[:-1], pos])


def main():
    t_start = time.time()
    eps_eff = (SUB_EPS_R + 1) / 2 + (SUB_EPS_R - 1) / 2 * (
        1 + 12 * (SUB_THICK / PATCH_W)
    ) ** -0.5
    f_cavity_guess = C0 / (2 * PATCH_L * math.sqrt(eps_eff))
    print(
        f"Patch {PATCH_W * 1e3:.0f} x {PATCH_L * 1e3:.0f} mm on eps_r = "
        f"{SUB_EPS_R} | cavity-formula starting guess "
        f"{f_cavity_guess / 1e9:.3f} GHz (a rough guide only — NOT the "
        "mode-identification criterion)"
    )
    print(
        f"openEMS-with-thirds reference for this geometry: f_res = "
        f"{OPENEMS_F_RES / 1e9:.4f} GHz, broadside D = {OPENEMS_D_DBI:.2f} dBi"
    )

    sim = build_simulation()

    # Preflight prints each advisory verbatim.  Expected on this fixture: two
    # sub-cell PEC sheet advisories (the deliberate 1-cell ground and patch),
    # the close bottom NTFF face (see build_simulation), and the small ground
    # plane (0.56 lambda across, so edge diffraction shapes the pattern —
    # physics shared with the openEMS reference, not a solver defect).  All
    # four are conditions to interpret, not to suppress.
    report = sim.preflight()
    print(f"preflight advisories: {len(list(report))}")

    print(f"\nRunning {NUM_PERIODS} periods of {F_DESIGN / 1e9:.1f} GHz ...")
    t_run = time.time()
    result = sim.run(num_periods=NUM_PERIODS)
    print(f"FDTD run: {time.time() - t_run:.0f} s")

    # ---- Settling witness FIRST: no frequency is quotable before it ----
    ts = np.asarray(result.time_series).ravel()
    dt = float(result.dt)
    envelope = np.abs(ts)
    peak = float(np.max(envelope))
    tail = float(np.max(envelope[int(len(envelope) * 0.95) :]))
    end_db = 20 * math.log10(max(tail, 1e-300) / peak)
    settled = end_db < -40.0
    print(
        f"\nSettling witness: end-of-run envelope {end_db:.1f} dB of the "
        f"post-source peak (bar: -40 dB) -> "
        f"{'SETTLED' if settled else 'UNDER-SETTLED'}"
    )
    if not settled:
        print(
            "  The ring-down was truncated; frequencies and far fields below "
            "carry transient error. Raise NUM_PERIODS and rerun."
        )

    # ---- Full mode list: the patch rings at more than one frequency ----
    modes = [
        m
        for m in harminv(ts[int(len(ts) * 0.3) :], dt, 1.0e9, 3.5e9)
        if m.Q > 2 and m.amplitude > 1e-8
    ]
    modes.sort(key=lambda m: m.freq)
    if not modes:
        raise RuntimeError("harminv found no ring-down modes — inspect the probe trace")
    print(f"\nharminv ring-down modes ({len(modes)}):")
    for m in modes:
        print(f"  f = {m.freq / 1e9:.4f} GHz | Q = {m.Q:6.1f} | amplitude = {m.amplitude:.3g}")
    print(
        "  Amplitude rank orders PROBE coupling, not radiation - the probe "
        "sits at one point in the substrate. The radiator is identified below."
    )

    # ---- Far-field verdict per candidate bin ----
    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 49)
    ff = compute_far_field_jax(result.ntff_data, result.ntff_box, result.grid, theta, phi)
    d_dbi = directivity(ff)

    power = np.abs(np.asarray(ff.E_theta)) ** 2 + np.abs(np.asarray(ff.E_phi)) ** 2
    dth = np.gradient(theta)
    dph = np.gradient(phi)
    p_rad = np.sum(
        power * np.sin(theta)[None, :, None] * dth[None, :, None] * dph[None, None, :],
        axis=(1, 2),
    )
    p_rel_db = 10 * np.log10(p_rad / p_rad.max())
    peak_theta_deg = np.degrees(theta[np.argmax(np.max(power, axis=2), axis=1)])

    print("\nFar field per monitored bin (radiated power is relative to the peak bin):")
    for k, f in enumerate(NTFF_FREQS):
        print(
            f"  {f / 1e9:.1f} GHz: P_rad {p_rel_db[k]:6.1f} dB | beam peak at "
            f"theta = {peak_theta_deg[k]:5.1f} deg | D = {d_dbi[k]:.2f} dBi"
        )

    # The radiating mode: broadside beam AND the radiated-power peak.
    broadside = peak_theta_deg <= 15.0
    if not broadside.any():
        raise RuntimeError("no monitored bin shows a broadside beam — inspect the pattern")
    k_star = int(np.argmax(np.where(broadside, p_rel_db, -np.inf)))
    radiating = min(modes, key=lambda m: abs(m.freq - NTFF_FREQS[k_star]))
    print(
        f"\nRADIATING mode: f_res = {radiating.freq / 1e9:.4f} GHz "
        f"(Q = {radiating.Q:.1f}) — the radiated-power spectrum peaks at the "
        f"{NTFF_FREQS[k_star] / 1e9:.1f} GHz bin with a broadside beam "
        f"(theta = {peak_theta_deg[k_star]:.1f} deg). Not chosen by amplitude "
        "rank, not chosen by distance to the textbook estimate."
    )
    others = [m for m in modes if m is not radiating]
    if others:
        listed = ", ".join(f"{m.freq / 1e9:.2f} GHz" for m in others)
        print(
            f"  Non-radiating ring-down content: {listed} — real signal in the "
            "probe, but the far field does not select it."
        )

    # ---- Principal-plane cuts + headline numbers ----
    angle_xz, cut_xz = principal_cut(power[k_star], 0, 24)   # phi = 0 / 180 deg
    angle_yz, cut_yz = principal_cut(power[k_star], 12, 36)  # phi = 90 / 270 deg
    hpbw_xz = half_power_beamwidth_deg(angle_xz, cut_xz)
    hpbw_yz = half_power_beamwidth_deg(angle_yz, cut_yz)
    print(
        f"\nAt {NTFF_FREQS[k_star] / 1e9:.1f} GHz: D = {d_dbi[k_star]:.2f} dBi | "
        f"beamwidth {hpbw_xz:.0f} deg (x-z cut) / {hpbw_yz:.0f} deg (y-z cut)"
    )

    dev_pct = (radiating.freq - OPENEMS_F_RES) / OPENEMS_F_RES * 100
    print("\nAccuracy recap (error budget in the module docstring):")
    print(
        f"  f_res {radiating.freq / 1e9:.4f} GHz vs openEMS "
        f"{OPENEMS_F_RES / 1e9:.4f} GHz: {dev_pct:+.1f}% — the design mode "
        "reads HIGH at dx = 2 mm. The committed lock "
        "(tests/crossval/test_patch_canonical_farfield_e4.py) gates the sign "
        "and the [+6%, +16%] envelope. Two coarse-grid mechanisms compete "
        "here (substrate under-resolved in z reads high, staircased patch "
        "edge reads low) and are not separated, so finer dx is not a "
        "predictable direction."
    )
    print(
        f"  D {d_dbi[k_star]:.2f} dBi vs openEMS {OPENEMS_D_DBI:.2f} dBi "
        f"({d_dbi[k_star] - OPENEMS_D_DBI:+.2f} dB) — the far field is the "
        "observable that agrees, inside the committed 1.0 dB envelope."
    )

    # ---- Save the far-field cuts ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    floor = 1e-4
    for angle, cut, label in (
        (angle_xz, cut_xz, "x-z cut (feed plane)"),
        (angle_yz, cut_yz, "y-z cut"),
    ):
        cut_db = 10 * np.log10(np.maximum(cut / cut.max(), floor))
        ax.plot(angle, cut_db, label=label)
    ax.set(
        xlabel="Angle from broadside (degrees)",
        ylabel="Normalized power (dB)",
        title=f"Patch far-field cuts at {NTFF_FREQS[k_star] / 1e9:.1f} GHz",
        xlim=(-90, 90),
        ylim=(-30, 1),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")
    fig.savefig(OUTPUT_PATH, dpi=130)
    plt.close(fig)
    print(f"\nSaved far-field cuts: {OUTPUT_PATH}")
    print(f"Total wall time: {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
