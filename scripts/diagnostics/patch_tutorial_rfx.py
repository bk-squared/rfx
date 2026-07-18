"""STATUS 2026-07-13: LIVE — validated far-field lane (rfx side; D 6.71 vs openEMS 6.79 dBi). Env knobs documented below; one-off rung YAMLs archived in scripts/archive/20260712_patch_farfield_arc/.

TRACKED 2026-07-18 as the canonical-patch oracle provenance (rfx side of the
committed reference pair; openEMS side: patch_tutorial_openems.py -> fixture
tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json, gated by
tests/test_patch_canonical_farfield_e4.py). Recorded research-frame result
(cv05_investigation_results/patch_tutorial_rfx.json, num_periods=250, buf=8,
wall 4615 s): f_res 2.2143 GHz (Q 59.3), D 6.71 dBi, E/H peaks 0/-3 deg,
two-window-settled per that JSON (the energy-witness field below postdates that
run). The committed test runs the lean frame of
examples/tutorials/patch_antenna_demo.py instead (~12 min class).

Canonical patch far-field — rfx NTFF, IDENTICAL geometry to the openEMS
Simple_Patch tutorial (patch_tutorial_openems.py). Apples-to-apples solver-vs-solver.

SHARED GEOMETRY: patch 32x40mm | substrate eps_r=3.38 tan_delta=1e-3 h=1.524mm |
GP 60x60mm | probe feed 6mm from centre (x=-6). f_design ~2.2 GHz.
rfx-only: CPML, dz graded (fine substrate), NTFF box >= lambda/2 from the GP edges.
The probe feed is a soft Ez source through the substrate at the feed point (the patch
radiates dominantly once resonant, so the pattern is patch-dominated, not feed-dominated).
"""
import json
import math
import os
import time
import numpy as np
from rfx import Simulation, Box
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading
from rfx.harminv import harminv
from rfx.farfield import compute_far_field, directivity

C0 = 2.998e8
EPS0 = 8.8541878e-12
# ---- SHARED GEOMETRY ----
patch_w, patch_l = 32.0e-3, 40.0e-3
sub_epsR, sub_thick = 3.38, 1.524e-3
GP = 60.0e-3
feed_off = -6.0e-3
TAN_DELTA = 1e-3
SIGMA = 2 * math.pi * 2.45e9 * EPS0 * sub_epsR * TAN_DELTA   # == openEMS kappa
f_design = 2.2e9
# in-plane resolution (env for the dx-convergence falsifier); z AIR cell size is
# decoupled (PT_Z_AIR_MM) so halving dx does not double the z cell count — dt is
# dominated by the 0.381mm substrate cells either way.
dx = float(os.environ.get("PT_DX_MM", "2.0")) * 1e-3
z_air = float(os.environ.get("PT_Z_AIR_MM", str(dx * 1e3))) * 1e-3
SKIP_NTFF = os.environ.get("PT_SKIP_NTFF", "0") == "1"   # resonance-only lean mode
TAG = os.environ.get("PT_TAG", "")
n_cpml = 8
n_sub = 4
dz_sub = sub_thick / n_sub
lam = C0 / f_design
# pad must clear the CPML in EVERY axis — z cells are z_air-sized, so use the
# larger of (dx, z_air) or the box lands inside the z-CPML when dx < z_air.
box_pad = (n_cpml + 3) * max(dx, z_air)
margin_xy = float(os.environ.get("PT_MARGIN_XY_MM", "85")) * 1e-3
air_below = float(os.environ.get("PT_AIR_BELOW_MM", "30")) * 1e-3
air_above = float(os.environ.get("PT_AIR_ABOVE_MM", "95")) * 1e-3
NUM_PERIODS = int(os.environ.get("PT_NUM_PERIODS", "250"))
eps_eff = (sub_epsR + 1) / 2 + (sub_epsR - 1) / 2 * (1 + 12 * (sub_thick / patch_w)) ** -0.5
F_GUESS = C0 / (2 * patch_l * math.sqrt(eps_eff))

RESULT_DIR = os.path.join(os.path.dirname(__file__), "cv05_investigation_results")
os.makedirs(RESULT_DIR, exist_ok=True)


def hpbw_deg(ang_deg, power_lin):
    p = np.asarray(power_lin, float); p = p / np.max(p)
    ipk = int(np.argmax(p))

    def edge(direction):
        i = ipk
        while 0 <= i + direction < len(p) and p[i] >= 0.5:
            i += direction
        if p[i] >= 0.5:
            return np.nan
        j = i - direction
        t = (p[j] - 0.5) / (p[j] - p[i])
        return ang_deg[j] + t * (ang_deg[i] - ang_deg[j])

    lo, hi = edge(-1), edge(+1)
    return float("nan") if (np.isnan(lo) or np.isnan(hi)) else abs(hi - lo)


def build_cubic():
    """Uniform CUBIC mesh (dz = dx, no dz_profile) — the only path Stage-2
    kottke_pec supports. Requires sub_thick/dx integer (dx=0.762mm -> 2 cells).
    Geometry snapped to cell boundaries; no smooth_grading shift hazard."""
    n_sub_c = round(sub_thick / dx)
    assert abs(n_sub_c * dx - sub_thick) < 1e-9, "sub_thick must be integer cells"
    dom_x = GP + 2 * margin_xy
    dom_y = GP + 2 * margin_xy
    zb = round(air_below / dx) * dx
    dom_z = zb + sub_thick + round(air_above / dx) * dx
    cx, cy = dom_x / 2, dom_y / 2
    gx_lo, gx_hi = cx - GP / 2, cx + GP / 2
    gy_lo, gy_hi = cy - GP / 2, cy + GP / 2
    patch_x_lo, patch_x_hi = cx - patch_w / 2, cx + patch_w / 2
    patch_y_lo, patch_y_hi = cy - patch_l / 2, cy + patch_l / 2
    feed_x, feed_y = cx + feed_off, cy
    z_gnd_lo, z_gnd_hi = zb - dx, zb
    z_sub_lo, z_sub_hi = zb, zb + sub_thick
    z_patch_lo, z_patch_hi = z_sub_hi, z_sub_hi + dx

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx,
                     boundary=BoundarySpec.uniform("cpml"), cpml_layers=n_cpml)
    sim.add_material("sub", eps_r=sub_epsR, sigma=SIGMA)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="sub")
    sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
                (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")
    src_z = z_sub_lo + 0.75 * dx
    sim.add_source(position=(feed_x, feed_y, src_z), component="ez",
                   waveform=GaussianPulse(f0=f_design, bandwidth=1.2))
    sim.add_probe(position=(feed_x + 4e-3, feed_y + 4e-3, src_z), component="ez")
    meta = dict(dom_x_mm=round(dom_x * 1e3, 1), z_total_mm=round(dom_z * 1e3, 1),
                cubic=True, sub_cells=n_sub_c, dx_mm=dx * 1e3,
                gp_edge_to_box_mm=None, lambda_half_mm=round(lam / 2 * 1e3, 1),
                ntff_freqs_ghz=[])
    ntff_freqs = np.array([2.0e9])  # unused in lean mode
    return sim, meta, ntff_freqs


def build():
    n_above = int(math.ceil(air_above / z_air)); n_below = int(math.ceil(air_below / z_air))
    dom_x = GP + 2 * margin_xy; dom_y = GP + 2 * margin_xy
    cx, cy = dom_x / 2, dom_y / 2
    gx_lo, gx_hi = cx - GP / 2, cx + GP / 2
    gy_lo, gy_hi = cy - GP / 2, cy + GP / 2
    patch_x_lo, patch_x_hi = cx - patch_w / 2, cx + patch_w / 2
    patch_y_lo, patch_y_hi = cy - patch_l / 2, cy + patch_l / 2
    feed_x, feed_y = cx + feed_off, cy

    # z-mesh with a fine band of (buf + 1 ground + n_sub substrate + 1 patch + buf)
    # cells. PT_FINE_BUFFER_CELLS > 0 inserts uniform-fine BUFFER cells so the
    # coarse<->fine grading transition sits away from the resonator (mechanism
    # test: cv05 #325 verdict predicts the transition ADJACENT to the resonant
    # stack injects a spurious mode split; real modes are mesh-robust).
    buf = int(os.environ.get("PT_FINE_BUFFER_CELLS", "0"))
    raw_dz = np.concatenate([np.full(n_below, z_air),
                             np.full(buf + 1 + n_sub + 1 + buf, dz_sub),
                             np.full(n_above, z_air)])
    dz_profile = smooth_grading(raw_dz, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz_profile), 0, 0.0)
    z_total = float(edges[-1])
    # DERIVE the stack z-positions FROM where smooth_grading actually put the fine
    # band (do NOT trust a fixed air_below — smooth_grading's transition cells shift
    # it, the #325 bug: the substrate then lands on 1 coarse cell instead of n_sub).
    fi = np.where(np.isclose(dz_profile, dz_sub, rtol=1e-6))[0]
    assert len(fi) >= 2 + n_sub + 2 * buf, f"expected >= {2+n_sub+2*buf} fine cells, got {len(fi)}"
    f0 = int(fi[0]) + buf
    z_gnd_lo, z_gnd_hi = edges[f0], edges[f0 + 1]
    z_sub_lo, z_sub_hi = edges[f0 + 1], edges[f0 + 1 + n_sub]
    z_patch_lo, z_patch_hi = z_sub_hi, edges[f0 + 1 + n_sub + 1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    print(f"  z-mesh FIXED: substrate rasterizes to {sub_cells} cells (intended {n_sub}); "
          f"z_sub=[{z_sub_lo*1e3:.3f},{z_sub_hi*1e3:.3f}]mm")

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, 0), dx=dx,
                     dz_profile=dz_profile, boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=n_cpml)
    sim.add_material("sub", eps_r=sub_epsR, sigma=SIGMA)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="sub")
    sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
                (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")
    src_z = z_sub_lo + dz_sub * 1.5
    sim.add_source(position=(feed_x, feed_y, src_z), component="ez",
                   waveform=GaussianPulse(f0=f_design, bandwidth=1.2))
    sim.add_probe(position=(feed_x + 4e-3, feed_y + 4e-3, src_z), component="ez")

    pad = box_pad
    box_lo = (pad, pad, max(pad, z_gnd_lo - 3 * z_air))
    box_hi = (dom_x - pad, dom_y - pad, z_total - pad)
    ntff_freqs = np.array([2.00e9, 2.10e9, 2.20e9, 2.30e9, 2.40e9, 2.50e9])
    if not SKIP_NTFF:
        sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=ntff_freqs)

    meta = dict(dom_x_mm=round(dom_x * 1e3, 1), z_total_mm=round(z_total * 1e3, 1),
                fine_buffer_cells=buf,
                dx_mm=dx * 1e3, gp_edge_to_box_mm=round((gx_lo - box_lo[0]) * 1e3, 1),
                lambda_half_mm=round(lam / 2 * 1e3, 1),
                ntff_freqs_ghz=[round(f / 1e9, 3) for f in ntff_freqs])
    return sim, meta, ntff_freqs


def main():
    t0 = time.time()
    print(f"TUTORIAL patch rfx | f_guess={F_GUESS/1e9:.4f} GHz | dx={dx*1e3}mm np={NUM_PERIODS}")
    sim, meta, ntff_freqs = (build_cubic() if os.environ.get('PT_CUBIC','0')=='1' else build())
    print(f"domain {meta['dom_x_mm']}mm z {meta['z_total_mm']}mm | GP-edge->box "
          f"{meta['gp_edge_to_box_mm']}mm (lambda/2={meta['lambda_half_mm']})")
    issues = [str(i) for i in sim.preflight()]
    print(f"preflight ({len(issues)}):")
    for i in issues:
        print(f"  ! {i}")

    # Lane-1 (#330) hook: PT_SUBPIXEL=kottke_pec tests whether the existing
    # Kottke inverse-eps subpixel path (SPSD, Stage 2) shrinks the staircase
    # edge-length extension (+1 dx) on the patch resonance.
    _sps = os.environ.get("PT_SUBPIXEL", "")
    res = sim.run(num_periods=NUM_PERIODS,
                  subpixel_smoothing=_sps if _sps else False)
    ts = np.asarray(res.time_series).ravel(); dt = float(res.dt)

    def _hv(sig):
        return sorted([m for m in harminv(sig, dt, 1.0e9, 3.5e9)
                       if m.Q > 2 and m.amplitude > 1e-8], key=lambda m: m.freq)
    good = _hv(ts[int(len(ts) * 0.3):])
    nearest = min(good, key=lambda m: abs(m.freq - F_GUESS)) if good else None
    f_res = float(nearest.freq) if nearest else float("nan")
    spectrum = [dict(f_ghz=round(m.freq / 1e9, 4), Q=round(m.Q, 1),
                     amp=float(m.amplitude)) for m in good]
    # energy/truncation witness (rfx CLAUDE.md ring-down rule): end-of-run probe
    # envelope vs post-source peak, dB. Always a measured fact.
    env = np.abs(ts)
    peak = float(np.max(env))
    tail = float(np.max(env[int(len(env) * 0.95):]))
    end_energy_db = 20 * math.log10(max(tail, 1e-300) / peak)
    print(f"ENERGY witness: end-of-run envelope {end_energy_db:.1f} dB of peak "
          f"({'OK (< -40 dB)' if end_energy_db < -40 else 'TRUNCATION SUSPECT'})")
    de = _hv(ts[int(len(ts) * 0.30):]); dl = _hv(ts[int(len(ts) * 0.55):])
    fe = max(de, key=lambda m: m.amplitude).freq if de else float("nan")
    fl = max(dl, key=lambda m: m.amplitude).freq if dl else float("nan")
    settled = (not np.isnan(fe) and not np.isnan(fl) and abs(fe - fl) / fl < 0.02)
    print(f"SETTLING: early={fe/1e9:.4f} late={fl/1e9:.4f} -> {'SETTLED' if settled else 'UNDER-SETTLED'}")
    print(f"W1 f_res={f_res/1e9:.4f} GHz; spectrum={[(m['f_ghz'],m['Q']) for m in spectrum]}")

    out = dict(solver="rfx", geometry="tutorial_patch_32x40_eps3.38",
               settled=bool(settled), end_energy_db=round(end_energy_db, 1),
               f_res_ghz=round(f_res / 1e9, 4), spectrum=spectrum,
               meta=meta, preflight=issues, num_periods=NUM_PERIODS)

    if not SKIP_NTFF:
        fi = int(np.argmin(np.abs(ntff_freqs - f_res)))
        theta = np.linspace(-np.pi / 2, np.pi / 2, 181)
        tfull = np.linspace(0, np.pi, 181)
        ffE = compute_far_field(res.ntff_data, res.ntff_box, res.grid, np.abs(theta), np.array([0.0]))
        ffH = compute_far_field(res.ntff_data, res.ntff_box, res.grid, np.abs(theta), np.array([np.pi / 2]))
        ffD = compute_far_field(res.ntff_data, res.ntff_box, res.grid, tfull, np.linspace(0, 2 * np.pi, 49))
        D_dbi = float(directivity(ffD)[fi])
        pE = np.abs(ffE.E_theta[fi, :, 0]) ** 2 + np.abs(ffE.E_phi[fi, :, 0]) ** 2
        pH = np.abs(ffH.E_theta[fi, :, 0]) ** 2 + np.abs(ffH.E_phi[fi, :, 0]) ** 2
        tdeg = np.degrees(theta); ipE, ipH = int(np.argmax(pE)), int(np.argmax(pH))
        hE, hH = hpbw_deg(tdeg, pE), hpbw_deg(tdeg, pH)
        broadside_ok = abs(tdeg[ipE]) <= 15 and abs(tdeg[ipH]) <= 15
        print(f"W2 @ {ntff_freqs[fi]/1e9:.3f}GHz: E-peak {tdeg[ipE]:.1f} HPBW {hE:.1f} | "
              f"H-peak {tdeg[ipH]:.1f} HPBW {hH:.1f} | D {D_dbi:.2f} dBi | BROADSIDE {'OK' if broadside_ok else 'FAIL'}")
        out.update(E_plane_peak_deg=round(float(tdeg[ipE]), 1), H_plane_peak_deg=round(float(tdeg[ipH]), 1),
                   hpbw_E_deg=None if np.isnan(hE) else round(float(hE), 1),
                   hpbw_H_deg=None if np.isnan(hH) else round(float(hH), 1),
                   directivity_dbi=round(D_dbi, 2), broadside_ok=bool(broadside_ok),
                   theta_deg=[round(float(t), 2) for t in tdeg],
                   E_plane_norm=[round(float(v), 5) for v in (pE / np.max(pE))],
                   H_plane_norm=[round(float(v), 5) for v in (pH / np.max(pH))])
    out["wall_s"] = round(time.time() - t0, 1)
    fname = f"patch_tutorial_rfx_{TAG}.json" if TAG else "patch_tutorial_rfx.json"
    path = os.path.join(RESULT_DIR, fname)
    json.dump(out, open(path, "w"), indent=1)
    print(f"WROTE {path} (wall {out['wall_s']}s)")


if __name__ == "__main__":
    main()
