"""openEMS reference: MATCHED 2.4 GHz FR4 patch with a QUARTER-WAVE IMPEDANCE
TRANSFORMER edge feed, plus a small transformer-width sweep to find the deepest
|S11| null.

WHY a transformer edge feed (not an inset/notch feed): a prior inset-fed attempt
DIVERGED in openEMS (exponential energy growth) because of the thin notch-slot
cells. This design is ALL-RECTANGULAR PEC (50 ohm line -> lambda/4 transformer
-> patch edge), so there are no sub-cell notch slots to destabilise the FDTD.

STABLE FRAME (reused verbatim from ``openems_patch_s11.py``): FR4 substrate
(eps_r=4.3, tan_delta=0.02, h=1.5 mm), finite ground at z=0, PML_8 on all 6
faces, 50 mm air margin to the PML, 40 mm radiation air above, 6 cells across
the substrate, SmoothMeshLines grading ratio<=1.4, coarse<=5 mm in air.
The ONLY departures are (a) the feed (edge transformer instead of a z-probe) and
(b) the ground+substrate are extended on the FEED (-x) side ONLY, because a
lambda/4 transformer (~18 mm) plus a 50 ohm reference line cannot fit in the
15.25 mm feed-side margin of the original 60x55 mm ground (a z-probe sat inside
the patch footprint and needed no extra ground). The +x and +-y ground edges,
the substrate stack, the 50 mm PML margins, the mesh grading and the PML are
kept EXACTLY.

Transformer synthesis (Hammerstad, FR4 h=1.5 mm, 2.4 GHz):
  Z_edge (Balanis G1+G12) ~ 322 ohm  ->  Z_t = sqrt(50*Z_edge) ~ 127 ohm
  w50 (50 ohm) ~ 2.92 mm ;  w_t(Z_t~120 ohm) ~ 0.40 mm ;  L_t ~ lam_g/4 ~ 18.3 mm
Because Z_edge is uncertain, w_t is SWEPT (5 values) to bracket the match; if the
best null is shallow, L_t is nudged. Sweep stays synchronous and small.

Run (system python -- openEMS/CSXCAD bindings live there):
  python scripts/crossval/openems_patch_qwt.py
"""

import json
import math
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "out_ref")
os.makedirs(OUT_DIR, exist_ok=True)
C0 = 2.998e8
Z0_PORT = 50.0

# ---- geometry (stable frame values) --------------------------------------
f_design = 2.4e9
eps_r = 4.3
tan_delta = 0.02
h_sub = 1.5e-3
W = 38.0e-3          # patch width (non-resonant, along y)
L = 29.5e-3          # patch length (resonant, along x)
gnd_y_mm = 55.0      # ground width in y (kept EXACTLY)
front_gnd_mm = 15.25  # +x ground margin from patch edge (kept EXACTLY = (60-29.5)/2)

# ---- feed synthesis (Hammerstad) -----------------------------------------
def _z_and_eeff(w, eps_r=eps_r, h=h_sub):
    """Microstrip characteristic impedance & eff. permittivity (Hammerstad)."""
    woh = w / h
    ee = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / woh) ** -0.5
    if woh <= 1:
        z = 60 / math.sqrt(ee) * math.log(8 / woh + woh / 4)
    else:
        z = 120 * math.pi / (math.sqrt(ee) * (woh + 1.393 + 0.667 * math.log(woh + 1.444)))
    return z, ee

def _w_from_z(z0, eps_r=eps_r, h=h_sub):
    """Hammerstad synthesis: width for a target impedance."""
    A = z0 / 60 * math.sqrt((eps_r + 1) / 2) + (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)
    B = 377 * math.pi / (2 * z0 * math.sqrt(eps_r))
    woh_a = 8 * math.exp(A) / (math.exp(2 * A) - 2)
    if woh_a < 2:
        return woh_a * h
    woh_b = 2 / math.pi * (B - 1 - math.log(2 * B - 1) +
                           (eps_r - 1) / (2 * eps_r) * (math.log(B - 1) + 0.39 - 0.61 / eps_r))
    return woh_b * h

def _quarterwave_len(w):
    _, ee = _z_and_eeff(w)
    return (C0 / (f_design * math.sqrt(ee))) / 4

W50 = _w_from_z(Z0_PORT)                     # ~2.92 mm
L50 = 15.0e-3                                # 50 ohm reference line length
Z_EDGE_EST = 322.0
Z_T_SYNTH = math.sqrt(Z0_PORT * Z_EDGE_EST)  # ~127 ohm
WT_SYNTH = _w_from_z(Z_T_SYNTH)              # ~0.33 mm (thin)
LT_NOM = _quarterwave_len(max(WT_SYNTH, 0.4e-3))  # ~18.3 mm

# Swept transformer widths (meshable, >=0.4 mm) bracketing the synthesized value.
WT_SWEEP_MM = [0.40, 0.55, 0.75, 1.00, 1.30]

FREQS_HZ = np.linspace(1.5e9, 3.5e9, 101)


def _numpy_shim():
    for _n in ("float", "int", "complex"):
        if not hasattr(np, _n):
            setattr(np, _n, {"float": float, "int": int, "complex": complex}[_n])


def _uniform_pml_buffer(lines, dx_b, n_buf):
    """Replace the outer ``n_buf`` cells at each end with EQUIDISTANT lines of
    spacing ``dx_b``. openEMS PML requires an equidistant mesh across the PML
    cells (+ a buffer); a graded mesh grading up to coarse cells right at the
    PML boundary is the classic cause of late-time energy regrowth. The interior
    (structure) region keeps its graded lines; SmoothMeshLines is re-applied by
    the caller so the graded interior meets the uniform buffer smoothly."""
    lines = np.unique(np.round(np.asarray(lines, float), 4))
    lo, hi = lines[0], lines[-1]
    lo_buf = lo + np.arange(0, n_buf + 1) * dx_b
    hi_buf = hi - np.arange(0, n_buf + 1) * dx_b
    inner = lines[(lines > lo_buf[-1] + 1e-6) & (lines < hi_buf[-1] - 1e-6)]
    return np.unique(np.round(np.concatenate([lo_buf, inner, hi_buf]), 4))


def _merge_close(lines, protected, dmin=0.2):
    """Drop mesh lines closer than ``dmin`` mm to a kept neighbour, but never
    drop a ``protected`` line (feed/patch/ground edges). This removes the tiny
    slivers created where sub-mm feed edges land next to the 0.6 mm patch grid
    -- the sub-0.2 mm cells that would shrink dt and destabilise the FDTD."""
    lines = np.unique(np.round(np.asarray(lines, float), 4))
    prot = np.unique(np.round(np.asarray(protected, float), 4))
    is_prot = lambda x: bool(np.any(np.abs(prot - x) < 1e-6))
    keep = []
    for x in lines:
        if keep and (x - keep[-1]) < dmin:
            if is_prot(x) and not is_prot(keep[-1]):
                keep[-1] = x            # prefer the protected edge
            elif is_prot(x) and is_prot(keep[-1]):
                keep.append(x)          # both protected (>=dmin apart by design)
            # else: x is a non-protected sliver -> drop it
        else:
            keep.append(x)
    return np.array(keep)


def build_and_run(wt_mm, lt_mm, tag):
    """Build the transformer-fed patch for one (w_t, L_t) and run openEMS.
    Returns (freqs, s11_complex, diagnostics dict)."""
    from CSXCAD.CSXCAD import ContinuousStructure
    from CSXCAD.SmoothMeshLines import SmoothMeshLines
    from openEMS.openEMS import openEMS as OEMS

    UNIT = 1e-3
    sim_path = os.path.join(SCRIPT_DIR, f"qwt_tmp_{tag}")

    # --- lengths in mm ---
    L_mm = L * 1e3
    W_mm = W * 1e3
    h_mm = h_sub * 1e3
    w50_mm = W50 * 1e3
    l50_mm = L50 * 1e3
    lt = lt_mm
    wt = wt_mm
    back_gnd_mm = 6.0            # ground behind the port (fine-meshed, see below)

    # --- x layout (feed enters +x into the -x radiating edge) ---
    # domain x starts at 0; 50 mm margin to the -x ground edge.
    margin_mm = 50.0
    air_above_mm = 40.0
    gnd_x_lo = margin_mm
    x_port = gnd_x_lo + back_gnd_mm
    line50_x0 = x_port
    line50_x1 = x_port + l50_mm
    trans_x0 = line50_x1
    trans_x1 = trans_x0 + lt
    patch_x_lo = trans_x1
    patch_x_hi = patch_x_lo + L_mm
    gnd_x_hi = patch_x_hi + front_gnd_mm
    dom_x_mm = gnd_x_hi + margin_mm

    # --- y layout (centred) ---
    gnd_y_lo = margin_mm
    gnd_y_hi = gnd_y_lo + gnd_y_mm
    y_c = (gnd_y_lo + gnd_y_hi) / 2
    patch_y_lo = y_c - W_mm / 2
    patch_y_hi = y_c + W_mm / 2
    dom_y_mm = gnd_y_hi + margin_mm

    dom_z_mm = h_mm + air_above_mm

    # --- solver (stable-frame settings) ---
    # Truncate the ring-down once the physical field has decayed to the (surface-
    # wave) energy floor (~step 9000) but BEFORE the graded-mesh/PML late-time
    # growth region; the port V(t) up to here is passive. (Full 30000 runs into
    # the instability.) 9500 keeps a 5-width sweep synchronous (~9 min total).
    nrts = int(os.environ.get("OPENEMS_NRTS", "9500"))
    FDTD = OEMS(NrTS=nrts, EndCriteria=1e-6)
    FDTD.SetGaussExcite(f_design, 1.2e9)
    FDTD.SetBoundaryCond(['PML_8'] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    # FR4 substrate + ground. The grounded FR4 slab supports a cutoff-free TM0
    # surface wave; on a FINITE slab that wave reflects off the truncated
    # dielectric edge and rings as a ~-18 dB energy FLOOR that never drains,
    # capping the usable S11 dynamic range. Extending the slab + ground THROUGH
    # the lateral PML absorbs the guided mode (standard microstrip-FDTD cure),
    # giving a clean >=40 dB ring-down. This idealises the finite 60x55 ground to
    # an effectively-infinite grounded slab -- it sacrifices finite-ground back-
    # lobe accuracy but yields a trustworthy FEED-MATCH (S11) measurement, which
    # is the goal here. Set QWT_FINITE_GND=1 to restore the finite slab.
    # NOTE: extending the slab THROUGH the PML (QWT_FINITE_GND=0) was tried to
    # drain the surface-wave floor but the grounded dielectric-in-PML supports a
    # parallel-plate mode the PML absorbs poorly -> reintroduces the blow-up.
    # Finite ground is the verified-bounded config, so it is the DEFAULT.
    finite_gnd = os.environ.get("QWT_FINITE_GND", "1") == "1"
    if finite_gnd:
        sub_x = (gnd_x_lo, gnd_x_hi); sub_y = (gnd_y_lo, gnd_y_hi)
    else:
        sub_x = (0.0, dom_x_mm); sub_y = (0.0, dom_y_mm)
    sub = CSX.AddMaterial('FR4')
    sub.SetMaterialProperty(epsilon=eps_r,
                            kappa=2 * math.pi * f_design * 8.8541878128e-12 * eps_r * tan_delta)
    sub.AddBox([sub_x[0], sub_y[0], 0], [sub_x[1], sub_y[1], h_mm], priority=1)

    # ground (2D PEC at z=0)
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([sub_x[0], sub_y[0], 0], [sub_x[1], sub_y[1], 0], priority=10)

    # feed chain + patch (2D PEC at z=h_sub). Boxes abut on shared mesh lines and
    # connect electrically (verified: a one-cell overlap changes S11/Zin by <1%).
    # QWT_FEED_OVL>0 forces an overlap; default 0 keeps the exact synthesized geometry.
    ovl = float(os.environ.get("QWT_FEED_OVL", "0.0"))
    metal = CSX.AddMetal('top')
    metal.AddBox([line50_x0, y_c - w50_mm / 2, h_mm],
                 [line50_x1 + ovl, y_c + w50_mm / 2, h_mm], priority=10)   # 50 ohm line
    metal.AddBox([trans_x0, y_c - wt / 2, h_mm],
                 [trans_x1 + ovl, y_c + wt / 2, h_mm], priority=10)         # transformer
    metal.AddBox([patch_x_lo, patch_y_lo, h_mm],
                 [patch_x_hi, patch_y_hi, h_mm], priority=10)         # patch

    # --- mesh: fine over feed+patch, coarse graded in air (stable frame) ---
    fine_mm = 0.6
    # Fine mesh spans the ENTIRE ground top (from the -x ground edge through the
    # patch), so the lumped port has symmetric fine cells on both sides. Starting
    # the fine region at the port (with coarse-graded cells behind it) left the
    # excitation straddling a fine/coarse cell jump -> late-time FDTD blow-up.
    x_fine = np.arange(gnd_x_lo, patch_x_hi + fine_mm / 2, fine_mm)
    y_fine = np.arange(patch_y_lo, patch_y_hi + fine_mm / 2, fine_mm)
    x_fixed = np.concatenate([
        [0.0, dom_x_mm, gnd_x_lo, gnd_x_hi],
        [x_port, line50_x1, trans_x0, trans_x1, patch_x_lo, patch_x_hi],
        x_fine,
    ])
    # feed y-edges: give the (thin) transformer & 50 ohm line their own lines.
    y_feed_edges = [y_c - w50_mm / 2, y_c + w50_mm / 2,
                    y_c - wt / 2, y_c + wt / 2, y_c]
    y_fixed = np.concatenate([
        [0.0, dom_y_mm, gnd_y_lo, gnd_y_hi, patch_y_lo, patch_y_hi],
        y_feed_edges, y_fine,
    ])
    sub_cells = 6
    z_fixed = np.concatenate([[0.0, dom_z_mm, h_mm], np.linspace(0, h_mm, sub_cells + 1)])

    # gentler grading (1.25) + smaller coarse cell (3 mm) than the probe frame:
    # the transformer feed enlarges the domain, and grading up to 5 mm at 1.4
    # right at the PML seeded a late-time PML instability (exponential energy
    # regrowth). 3 mm / 1.25 keeps it bounded.
    coarse_mm = float(os.environ.get("QWT_COARSE_MM", "3.0"))
    ratio = float(os.environ.get("QWT_RATIO", "1.25"))
    x_lines = SmoothMeshLines(np.unique(np.round(x_fixed, 4)), coarse_mm, ratio=ratio)
    y_lines = SmoothMeshLines(np.unique(np.round(y_fixed, 4)), coarse_mm, ratio=ratio)
    z_lines = SmoothMeshLines(np.unique(np.round(z_fixed, 4)), coarse_mm, ratio=ratio)

    # merge sub-0.2 mm slivers where the fine feed edges collide with the
    # 0.6 mm patch/coarse grid; protect the structural + feed edges.
    x_prot = [gnd_x_lo, gnd_x_hi, x_port, line50_x1, trans_x0, trans_x1,
              patch_x_lo, patch_x_hi]
    y_prot = [gnd_y_lo, gnd_y_hi, patch_y_lo, patch_y_hi] + y_feed_edges
    x_lines = _merge_close(x_lines, x_prot, dmin=0.2)
    y_lines = _merge_close(y_lines, y_prot, dmin=0.2)

    # EQUIDISTANT buffer through the PML cells (openEMS PML needs a uniform mesh
    # across the PML; grading into the PML is the classic late-time blow-up).
    # (lateral x/y only; z keeps the fine substrate stack + graded air of the
    # stable frame, which was already stable, and a z buffer would wreck the
    # 0.25 mm substrate cells.) Gated OFF by default: with finite ground the
    # gentler grading already bounds the run, and the buffer only helped when
    # combined with the (unstable) infinite slab.
    if os.environ.get("QWT_PML_BUFFER", "0") == "1":
        n_pml_buf = 12
        x_lines = _uniform_pml_buffer(x_lines, coarse_mm, n_pml_buf)
        y_lines = _uniform_pml_buffer(y_lines, coarse_mm, n_pml_buf)

    # snap the port to FINAL mesh lines (so the excitation box lands on an edge)
    x_port_s = float(x_lines[np.argmin(np.abs(x_lines - x_port))])
    y_c_s = float(y_lines[np.argmin(np.abs(y_lines - y_c))])

    port = FDTD.AddLumpedPort(
        port_nr=1, R=Z0_PORT,
        start=[x_port_s, y_c_s, 0.0],
        stop=[x_port_s, y_c_s, h_mm],
        p_dir='z', excite=1.0,
    )

    mesh.SetLines('x', x_lines)
    mesh.SetLines('y', y_lines)
    mesh.SetLines('z', z_lines)

    # mesh sanity: smallest in-plane cell must stay >= 0.2 mm (stability guard)
    min_dx = float(np.min(np.diff(x_lines)))
    min_dy = float(np.min(np.diff(y_lines)))
    ncells = len(x_lines) * len(y_lines) * len(z_lines)
    print(f"  [{tag}] mesh {len(x_lines)}x{len(y_lines)}x{len(z_lines)}={ncells:,} cells | "
          f"min dx={min_dx:.3f} dy={min_dy:.3f} mm | wt={wt:.2f} Lt={lt:.2f} mm")
    if min(min_dx, min_dy) < 0.2:
        print(f"    WARNING: sub-0.2 mm cell present (min={min(min_dx,min_dy):.3f} mm)")

    t0 = time.time()
    FDTD.Run(sim_path, verbose=0, cleanup=True, numThreads=16)
    rt = time.time() - t0

    port.CalcPort(sim_path, FREQS_HZ)
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    s11_dB = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))

    # energy-decay check from the port voltage ring-down
    ut = np.loadtxt(os.path.join(sim_path, "port_ut_1"), comments="%")
    ut = np.atleast_2d(ut)
    v = ut[:, 1]
    n = len(v)
    if n >= 20:
        head = float(np.max(np.abs(v[: n // 5])))
        tail = float(np.max(np.abs(v[-(n // 20):])))
        decayed = bool(tail < 0.05 * head) if head > 0 else False
    else:
        head = tail = float("nan")
        decayed = False

    idx = int(np.argmin(s11_dB))
    zin = Z0_PORT * (1 + s11[idx]) / (1 - s11[idx])   # port-plane Zin at the dip
    diag = {
        "tag": tag, "wt_mm": wt, "lt_mm": lt, "w50_mm": w50_mm,
        "ncells": ncells, "min_dx_mm": min_dx, "min_dy_mm": min_dy,
        "runtime_s": rt, "n_time_samples": n,
        "v_head": head, "v_tail": tail, "energy_decayed": decayed,
        "s11_dip_db": float(s11_dB[idx]), "s11_dip_hz": float(FREQS_HZ[idx]),
        "s11_max_abs": float(np.max(np.abs(s11))),
        "zin_at_dip_re": float(zin.real), "zin_at_dip_im": float(zin.imag),
    }
    print(f"    -> dip {diag['s11_dip_db']:.2f} dB @ {diag['s11_dip_hz']/1e9:.3f} GHz | "
          f"Zin={zin.real:.0f}{zin.imag:+.0f}j | {rt:.0f}s | decayed={decayed} "
          f"(head={head:.2e} tail={tail:.2e})")
    return FREQS_HZ, s11, diag


def _write_deliverables(results):
    """Write both JSON deliverables from the results so far (called after every
    run so a truncated session still leaves valid artifacts)."""
    best = min(results, key=lambda d: d["s11_dip_db"])
    reached = best["s11_dip_db"] < -15.0
    p1 = os.path.join(OUT_DIR, "openems_patch_qwt_matched.json")
    matched = {
        "solver": "openems",
        "feed": "quarter_wave_transformer_edge",
        "reached_minus15dB": reached,
        "note": ("Best |S11| null achieved by the transformer-width sweep. If shallow, "
                 "the patch radiating-edge impedance (see zin_at_dip) is high (~300-800 ohm) "
                 "and a single lambda/4 transformer to 50 ohm needs a very thin/high-Z line "
                 "(w_t << 0.4 mm) that violates the >=0.2 mm cell / FDTD-stability limit. "
                 "Ring-down is truncated at ~step 9000-12000 before a graded-mesh/PML "
                 "late-time instability; the port V(t) up to there is passive/clean."),
        "best": {
            "wt_mm": best["wt_mm"], "lt_mm": best["lt_mm"], "w50_mm": best["w50_mm"],
            "s11_dip_db": best["s11_dip_db"], "s11_dip_hz": best["s11_dip_hz"],
            "zin_at_dip_ohm": [best["zin_at_dip_re"], best["zin_at_dip_im"]],
            "energy_decayed": best["energy_decayed"], "s11_max_abs": best["s11_max_abs"],
            "min_cell_mm": min(best["min_dx_mm"], best["min_dy_mm"]),
            "runtime_s": best["runtime_s"], "ncells": best["ncells"],
        },
        "synthesis": {
            "Z_edge_est_ohm": Z_EDGE_EST, "Z_t_synth_ohm": Z_T_SYNTH,
            "wt_synth_mm": WT_SYNTH * 1e3, "Lt_nom_mm": LT_NOM * 1e3, "w50_mm": W50 * 1e3,
        },
        "sweep": [
            {k: d[k] for k in ("wt_mm", "lt_mm", "s11_dip_db", "s11_dip_hz", "zin_at_dip_re",
                               "zin_at_dip_im", "s11_max_abs", "energy_decayed",
                               "min_dx_mm", "min_dy_mm", "runtime_s")}
            for d in results
        ],
        "freqs_hz": [float(v) for v in FREQS_HZ],
        "s11": best["s11"],
        "s11_db": [20 * math.log10(max(math.hypot(re, im), 1e-9)) for re, im in best["s11"]],
    }
    with open(p1, "w") as fh:
        json.dump(matched, fh, indent=2); fh.write("\n")

    p2 = os.path.join(SCRIPT_DIR, "matched_patch_geometry.json")
    geom = {
        "description": "2.4 GHz FR4 patch, lambda/4 transformer edge feed "
                       "(all-rectangular PEC, no notch slots). openEMS width-swept.",
        "units": "mm unless stated",
        "reached_minus15dB": reached,
        "substrate": {"material": "FR4", "eps_r": eps_r, "tan_delta": tan_delta,
                      "h_mm": h_sub * 1e3},
        "patch": {"W_mm": W * 1e3, "L_mm": L * 1e3, "resonant_axis": "x (length L)"},
        "ground": {"x_mm": round(front_gnd_mm + L * 1e3 + best["lt_mm"] + L50 * 1e3 + 6.0, 2),
                   "y_mm": gnd_y_mm,
                   "note": "extended on the feed(-x) side to host the feed; +x margin "
                           "15.25 mm and y width 55 mm kept from the stable 60x55 frame"},
        "feed_topology": "lumped port -> 50 ohm line (w50,L50) -> lambda/4 transformer "
                         "(w_t,L_t) -> patch radiating-edge centre, coplanar PEC at z=h",
        "line_50ohm": {"w50_mm": round(W50 * 1e3, 3), "L50_mm": L50 * 1e3, "Z0_ohm": 50.0},
        "transformer": {"wt_mm": best["wt_mm"], "Lt_mm": round(best["lt_mm"], 2),
                        "Z_t_ohm_synth": round(Z_T_SYNTH, 1), "Z_edge_est_ohm": Z_EDGE_EST},
        "port": {"type": "lumped", "R_ohm": 50.0, "p_dir": "z",
                 "location": "outer end of 50 ohm line, ground->line"},
        "resonance": {"s11_dip_hz": best["s11_dip_hz"], "s11_dip_db": best["s11_dip_db"],
                      "zin_at_dip_ohm": [best["zin_at_dip_re"], best["zin_at_dip_im"]]},
        "solver_frame": {"boundary": "PML_8 x6", "air_margin_mm": 50.0,
                         "air_above_mm": 40.0, "sub_cells": 6, "fine_mm": 0.6,
                         "coarse_mm": 3.0, "smooth_ratio": 1.25,
                         "gauss_f0_hz": f_design, "gauss_fc_hz": 1.2e9,
                         "note": "coarse 3 mm / ratio 1.25 (finer than the probe frame's "
                                 "5 mm/1.4) to bound the feed-triggered PML late-time growth"},
    }
    with open(p2, "w") as fh:
        json.dump(geom, fh, indent=2); fh.write("\n")
    return best, reached, p1, p2


def main():
    print("=" * 72)
    print("openEMS -- 2.4 GHz FR4 patch, lambda/4 transformer edge feed + wt sweep")
    print("=" * 72)
    print(f"Synthesis: Z_edge~{Z_EDGE_EST:.0f} ohm -> Z_t~{Z_T_SYNTH:.0f} ohm | "
          f"w50={W50*1e3:.3f} mm  wt_synth={WT_SYNTH*1e3:.3f} mm  Lt_nom={LT_NOM*1e3:.3f} mm")
    print(f"wt sweep (mm): {WT_SWEEP_MM}")
    print()
    _numpy_shim()

    lt_nom_mm = LT_NOM * 1e3
    results = []
    runs = [(wt, lt_nom_mm, f"wt{i}") for i, wt in enumerate(WT_SWEEP_MM)]
    # Optional L_t nudge (QWT_LT_NUDGE=1). OFF by default: a width-invariant ~90%
    # reflection is a Z-mismatch to a high edge impedance, which a length nudge
    # does not fix -- do not burn runs faking depth.
    for wt, lt, tag in runs:
        f, s11, d = build_and_run(wt, lt, tag)
        d["s11"] = [[float(v.real), float(v.imag)] for v in s11]
        results.append(d)
        _write_deliverables(results)   # timeout-safe incremental write

    if os.environ.get("QWT_LT_NUDGE", "0") == "1":
        best = min(results, key=lambda d: d["s11_dip_db"])
        for j, dlt in enumerate((-1.5, +1.5)):
            f, s11, d = build_and_run(best["wt_mm"], lt_nom_mm + dlt, f"lt{j}")
            d["s11"] = [[float(v.real), float(v.imag)] for v in s11]
            results.append(d)
            _write_deliverables(results)

    best, reached, p1, p2 = _write_deliverables(results)
    print("\n  === wt sweep table ===")
    print("  wt(mm)  Lt(mm)   dip(dB)   f_dip(GHz)   Zin_at_dip(ohm)   min_cell(mm)  decayed")
    for d in results:
        print(f"  {d['wt_mm']:5.2f}  {d['lt_mm']:6.2f}  {d['s11_dip_db']:8.2f}  "
              f"{d['s11_dip_hz']/1e9:9.3f}    {d['zin_at_dip_re']:6.0f}{d['zin_at_dip_im']:+6.0f}j  "
              f"  {min(d['min_dx_mm'],d['min_dy_mm']):6.3f}     {d['energy_decayed']}")
    print(f"\n  best: wt={best['wt_mm']:.2f} mm Lt={best['lt_mm']:.2f} mm -> "
          f"{best['s11_dip_db']:.2f} dB @ {best['s11_dip_hz']/1e9:.3f} GHz")
    print(f"  reached |S11| < -15 dB: {reached}")
    print(f"  wrote {p1}\n  wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
