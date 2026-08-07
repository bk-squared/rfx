#!/usr/bin/env python3
"""#589 item-2 adjudication driver: face-resolved flux accounting (C1/C2/target).

PRE-DECLARATION BINDING
-----------------------
This driver implements, verbatim, the decision rule pre-declared on issue
#589 (comment of 2026-08-07, "PRE-DECLARATION — item 2 adjudication
attempt"). The rule, witnesses, controls, and floors are NOT restated as
authority here — the issue comment is the authority; this header carries
only what that comment delegated to "the driver PR header": the exact face
coordinates, their shift ranges, and instrument notes measured during
implementation. Runs: C1 (MSL through-line control), C2 (coax two-port
control), target (settled attempt-2 fixture + flux box). Controls gate the
target: if a control hard-gate fails, the target is NOT interpreted
(rc=2), per the pre-declaration.

FACE COORDINATE TABLE (attempt-2 fixture; DX = 100 um; interior domain
12.5 x 3.4 x 3.9 mm; CPML = 8 PAD cells OUTSIDE the interior domain)
---------------------------------------------------------------------
All coordinates physical metres; "cell" = physical coordinate / DX.
Outward sign: +1 if +axis flux is outward for that face, else -1.

  face  axis coord(mm) cell  tangential footprint            outward  shifts
  xp    x    2.2       22    y [0.3,3.1], z [0.9,3.4] mm     +1       +/-2
  xm    x    0.3        3    y [0.3,3.1], z [0.9,3.4] mm     -1       +2 only
  yp    y    3.1       31    x [0.3,2.2], z [0.9,3.4] mm     +1       -2 only
  ym    y    0.3        3    x [0.3,2.2], z [0.9,3.4] mm     -1       +2 only
  zt    z    3.4       34    x [0.3,2.2], y [0.3,3.1] mm     +1       +/-2
  zb    z    0.9        9    x [0.3,2.2], y [0.3,3.1] mm     -1       +/-2 (see flag)
  x2    x    6.0       60    same footprint as xp (2-plane witness)   none
  xp_aperture (partition of xp, books to the MSL return channel):
        x    2.2       22    y Y_C+/-0.6 mm, z [2.5,3.2] mm  +1       +/-2
  strip_{xp,xm,yp,ym}: below-ground sub-strips of the side faces
        (z [0.9,2.5] mm) — coax shell tightness witness, must read ~0.

  One-sided shifts (declared): xm/ym +2 only and yp -2 only — the omitted
  member would sit 1 cell from the CPML inner edge, where absorber-fringe
  contamination could trip the invariance gate for a placement reason.
  zb's -2 member (0.7 mm) sits 2 cells above the coax source plane and is
  flagged in the output (SHIFT_FLAGS).
  xp remainder := xp - xp_aperture (arithmetic; DFT flux is per-cell
        linear, both monitors sit at the same plane index).

Placement facts checked against the fixture source (they satisfy the
pre-declaration's constraints — an earlier in-session estimate that the
fixture could NOT satisfy the >=3-cell absorber clearance was WRONG; it
mistook the 8 CPML PAD cells for cells inside the interior domain):
  * every face is >= 3 cells from every interior-domain edge (min: xm and
    ym/zb at exactly 3);
  * both resistive elements are OUTSIDE the box: coax annular resistor at
    z-index pad+3 (z = 0.3 mm) and coax source plane z = 0.5 mm sit BELOW
    zb = 0.9 mm; the MSL feed (x = 11.0 mm) sits far outside xp = 2.2 mm;
  * the below-ground coax annulus (x [0.4,1.6], y [1.1,2.3] mm) is FULLY
    inside the zb face footprint, so R_coax is measured, not bounded;
  * xp sits between the junction (x = 1.0 mm) and the first MSL ladder
    probe (x = 2.6 mm), 4 cells clear of the probe plane;
  * zb shift -2 lands at 0.7 mm, still 2 cells above the coax source
    plane — the shift member nearest a source; flagged in output.

MONITOR FREQUENCIES: the lane's committed FREQS_2 bins EXACTLY, plus a
dense 5-11 GHz reporting band (25 points) on the same monitors. Decision
arithmetic indexes the exact FREQS_2 members only.

INSTRUMENT NOTES (measured during implementation, 2026-08-07)
-------------------------------------------------------------
* Non-perturbation witness: S bit-identical with/without monitors —
  committed as tests/test_coax_msl_transition.py::
  test_extra_flux_monitors_do_not_perturb_s (slow_physics).
* Flux sign convention verified live: a top-drive coax two-port run reads
  NEGATIVE net flux at a mid-line z-plane (power travels -z), so
  "positive = +axis" holds and the outward multipliers above are correct.
* kappa(f) = P_inc,flux / |a_inc[msl,msl]|^2 is a RECORDED witness with a
  soft [0.7, 1.3] range, not a hard C1 gate: the decision rule's flux legs
  are normalized by flux-measured P_inc throughout, and its |S22|^2 leg is
  a pencil-unit RATIO, so the analytic-vs-discrete Z0 factor cancels
  everywhere it could bind. (This resolves the wording tension between the
  posted C1 bullet and the posted definitions bullet in favor of the
  definitions — declared here, before any run.)

HAND-PORTED PREFLIGHT-EQUIVALENTS (C1 runs rfx.simulation.run directly,
which has no preflight): nonzero excitation energy (witness peak > 0),
the -40 dB ring-down settling witness (same end/peak tail arithmetic as
compute_coax_msl_transition), and an in-code assert of every monitor
coordinate against the table above.

Stages:  --stage c1 | c2 | target | all     (all = c1 -> c2 -> gate -> target)
Exit codes: 0 = ran, gates evaluated (verdict in JSON, incl. NON-CLOSING);
2 = control hard-gate failed (target skipped / not interpreted); 1 = crash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

DX = 100e-6
FLOOR = 0.02          # pre-declared floor (fraction of P_inc)
FLOOR_FAIL = 0.05     # pre-declared: control floor above this fails C1
W_TOL = 0.15          # pre-declared W1/W2/two-plane tolerance
DENSE_BAND = np.linspace(5.0e9, 11.0e9, 25)

# face: (axis, coordinate_m, size_(t1,t2)_m, center_(t1,t2)_m, outward, shifts)
# tangential order follows add_flux_monitor: x-normal -> (y, z);
# y-normal -> (x, z); z-normal -> (x, y).
# Shift ranges (review finding, declared): xm/ym take +2 only and yp -2
# only — their other member would land 1 cell from the CPML inner edge,
# where absorber-fringe contamination could trip the invariance gate for a
# placement reason rather than an instrument one. zb keeps -2 with an
# explicit source-proximity flag in the output (2 cells above the coax
# source plane at its -2 member).
FACES = {
    "xp": ("x", 2.2e-3, (2.8e-3, 2.5e-3), (1.7e-3, 2.15e-3), +1, (-2, +2)),
    "xm": ("x", 0.3e-3, (2.8e-3, 2.5e-3), (1.7e-3, 2.15e-3), -1, (+2,)),
    "yp": ("y", 3.1e-3, (1.9e-3, 2.5e-3), (1.25e-3, 2.15e-3), +1, (-2,)),
    "ym": ("y", 0.3e-3, (1.9e-3, 2.5e-3), (1.25e-3, 2.15e-3), -1, (+2,)),
    "zt": ("z", 3.4e-3, (1.9e-3, 2.8e-3), (1.25e-3, 1.7e-3), +1, (-2, +2)),
    "zb": ("z", 0.9e-3, (1.9e-3, 2.8e-3), (1.25e-3, 1.7e-3), -1, (-2, +2)),
}
SHIFT_FLAGS = {
    "zb_s-2": "2 cells above the coax source plane (closest shift-to-source)",
}
# Below-ground strip witnesses (coax shell tightness): side-face strips
# z in [0.9, 2.5] mm (below the ground plane) must read ~0.
STRIPS = {
    "strip_xp": ("x", 2.2e-3, (2.8e-3, 1.6e-3), (1.7e-3, 1.7e-3)),
    "strip_xm": ("x", 0.3e-3, (2.8e-3, 1.6e-3), (1.7e-3, 1.7e-3)),
    "strip_yp": ("y", 3.1e-3, (1.9e-3, 1.6e-3), (1.25e-3, 1.7e-3)),
    "strip_ym": ("y", 0.3e-3, (1.9e-3, 1.6e-3), (1.25e-3, 1.7e-3)),
}
X2_PLANE = ("x", 6.0e-3, (2.8e-3, 2.5e-3), (1.7e-3, 2.15e-3), +1)
APERTURE = ("x", 2.2e-3, (1.2e-3, 0.7e-3), (1.7e-3, 2.85e-3), +1, (-2, +2))


def _monitor_freqs(base):
    return np.unique(np.concatenate([np.asarray(base, float), DENSE_BAND]))


def _bin_index(mfreqs, targets):
    idx = []
    for f in np.asarray(targets, float):
        j = int(np.argmin(np.abs(mfreqs - f)))
        assert abs(mfreqs[j] - f) < 1.0, (mfreqs[j], f)
        idx.append(j)
    return idx


def _build_target_entries(freqs_base):
    """All monitors for the target/C1 box, incl. shifted siblings."""
    from rfx.api import Simulation
    from tests.test_coax_msl_transition import LX_2, LY, LZ_2, FREQ_MAX_2
    mf = _monitor_freqs(freqs_base)
    scratch = Simulation(freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX,
                         boundary="cpml")
    for name, (axis, coord, size, center, _outward, shifts) in FACES.items():
        scratch.add_flux_monitor(axis=axis, coordinate=coord, freqs=mf,
                                 size=size, center=center, name=name)
        for sh in shifts:
            scratch.add_flux_monitor(
                axis=axis, coordinate=coord + sh * DX, freqs=mf,
                size=size, center=center, name=f"{name}_s{sh:+d}",
            )
    axis, coord, size, center, _o = X2_PLANE
    scratch.add_flux_monitor(axis=axis, coordinate=coord, freqs=mf,
                             size=size, center=center, name="x2")
    axis, coord, size, center, _o, shifts = APERTURE
    scratch.add_flux_monitor(axis=axis, coordinate=coord, freqs=mf,
                             size=size, center=center, name="xp_aperture")
    for sh in shifts:
        scratch.add_flux_monitor(
            axis=axis, coordinate=coord + sh * DX, freqs=mf,
            size=size, center=center, name=f"xp_aperture_s{sh:+d}",
        )
    for name, (axis, coord, size, center) in STRIPS.items():
        scratch.add_flux_monitor(axis=axis, coordinate=coord, freqs=mf,
                                 size=size, center=center, name=name)
    return scratch._flux_monitors, mf


def _assert_monitor_indices(entries, grid):
    """Header-table assert (hand-ported preflight): the MATERIALIZED plane
    index of every monitor must equal round(coordinate/DX) + pad on its
    normal axis — checked against the actual grid, pads included, so a
    rasterization surprise fails loudly before any physics is read."""
    pads = {"x": int(grid.pad_x_lo), "y": int(grid.pad_y_lo),
            "z": int(grid.pad_z_lo)}
    for pe in entries:
        expected = int(round(float(pe.coordinate) / DX)) + pads[pe.axis]
        axis_idx = {"x": 0, "y": 1, "z": 2}[pe.axis]
        pos = [0.0, 0.0, 0.0]
        pos[axis_idx] = pe.coordinate
        got = int(grid.position_to_index(tuple(pos))[axis_idx])
        assert got == expected, (
            f"monitor {pe.name}: materialized plane index {got} != "
            f"expected {expected} (coord {pe.coordinate} m)")


def _face_terms(spectra, p_inc, idx):
    """Per-face OUTWARD flux terms / P_inc at the FREQS_2 bins."""
    def g(name):
        return np.asarray(spectra[name], float)[idx]
    return {
        "xp_rem": +1 * (g("xp") - g("xp_aperture")) / p_inc,
        "xm": -1 * g("xm") / p_inc,
        "yp": +1 * g("yp") / p_inc,
        "ym": -1 * g("ym") / p_inc,
        "zt": +1 * g("zt") / p_inc,
        "zb": -1 * g("zb") / p_inc,
        "xp_aperture": +1 * g("xp_aperture") / p_inc,
    }, g


def _channels(spectra, p_inc, idx, b_nl_subtract=None):
    """Pre-declared channel arithmetic at the FREQS_2 bins.

    R_nl books the xp remainder (xp - xp_aperture) plus xm/yp/ym/zt;
    R_msl_return = 1 + outward aperture flux / P_inc; R_coax from zb.
    ``b_nl_subtract`` (per-bin, from C1) applies the pre-declared
    launch-spill background subtraction to R_nl; both raw and corrected
    values are reported, and the decision arms evaluate the corrected one.

    W2 note (review finding, declared): W2 is algebraically identical to
    W1 given these channel definitions (the aperture terms cancel), so it
    is reported as a BOOKKEEPING consistency check of the channel
    arithmetic, not counted as an independent physics witness —
    witnesses_ok gates on W1 once.
    """
    terms, g = _face_terms(spectra, p_inc, idx)
    r_nl_raw = (terms["xp_rem"] + terms["xm"] + terms["yp"]
                + terms["ym"] + terms["zt"])
    r_nl = r_nl_raw if b_nl_subtract is None else (
        r_nl_raw - np.asarray(b_nl_subtract, float))
    r_coax = terms["zb"]
    r_msl_return = 1.0 + terms["xp_aperture"]
    w1 = (g("xp") - g("xm") + g("yp") - g("ym") + g("zt") - g("zb"))
    out = dict(
        R_nl=r_nl, R_nl_raw=r_nl_raw, R_coax=r_coax,
        R_msl_return=r_msl_return,
        per_face_outward=terms,
        W1_raw_closure=np.abs(w1) / p_inc,
        W2_bookkeeping_check=np.abs(r_nl_raw + r_coax + r_msl_return - 1.0),
        two_plane_delta=(+1) * (g("x2") - g("xp")) / p_inc,
    )
    def _ser(v):
        if isinstance(v, dict):
            return {k: _ser(x) for k, x in v.items()}
        return np.asarray(v, float).tolist()
    return {k: _ser(v) for k, v in out.items()}


def _shift_table(spectra, p_inc, idx):
    """Face-shift invariance: rebuild each channel with one face shifted."""
    base = _channels(spectra, p_inc, idx)
    rows = {}
    for name in list(FACES) + ["xp_aperture"]:
        for sh in (-2, +2):
            key = f"{name}_s{sh:+d}"
            if key not in spectra:
                continue
            sub = dict(spectra)
            sub[name] = spectra[key]
            ch = _channels(sub, p_inc, idx)
            rows[key] = {
                k: (np.abs(np.asarray(ch[k]) - np.asarray(base[k]))).tolist()
                for k in ("R_nl", "R_coax", "R_msl_return")
            }
    worst = 0.0
    for row in rows.values():
        for v in row.values():
            worst = max(worst, float(np.max(v)))
    return {"per_shift_abs_delta": rows, "worst_abs_delta": worst,
            "gate_lt_floor": bool(worst < FLOOR)}


def _settling_from_witness(ts):
    ts = np.asarray(ts, float)
    power = ts ** 2
    tail = max(1, power.shape[0] // 10)
    end = power[-tail:, :].mean(axis=0)
    peak = power.max(axis=0)
    tiny = np.finfo(float).tiny
    return float(np.max(10.0 * np.log10((end + tiny) / (peak + tiny))))


# ---------------------------------------------------------------------------
# C1 — MSL through-line control (junction removed, trace into the -x absorber)
# ---------------------------------------------------------------------------

def run_c1(n_steps, outdir):
    from rfx.api import Simulation
    from rfx.geometry.csg import Box
    from rfx.runners.uniform import build_flux_monitor_cfgs
    from rfx.probes.probes import flux_spectrum
    from rfx.simulation import run as _run, ProbeSpec
    from rfx.sources.msl_port import (  # same helpers the extractor imports
        MSLPort as _MSLPortLL,
        compute_msl_mode_profile, setup_msl_port, make_msl_port_sources,
    )
    from rfx import GaussianPulse
    import tests.test_coax_msl_transition as T

    sim = Simulation(freq_max=T.FREQ_MAX_2, domain=(T.LX_2, T.LY, T.LZ_2),
                     dx=DX, cpml_layers=8, boundary="cpml")
    sim.add_material("sub", eps_r=T.EPS_SUB)
    gnd_lo, gnd_hi = T._half_cell_box_z(T.N_GND, T.N_GND)
    sim.add(Box((0.0, 0.0, gnd_lo), (T.LX_2, T.LY, gnd_hi)), material="pec")
    sub_lo, sub_hi = T._half_cell_box_z(T.N_SUB_LO, T.N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (T.LX_2, T.LY, sub_hi)), material="sub")
    trc_lo, trc_hi = T._half_cell_box_z(T.N_TRACE, T.N_TRACE)
    # junction removed: the trace spans the FULL x-domain into both absorbers
    sim.add(Box((0.0, T.Y_C - T.W_TRACE / 2, trc_lo),
                (T.LX_2, T.Y_C + T.W_TRACE / 2, trc_hi)), material="pec")
    sim.add_msl_port(position=(T.FEED_X_2, T.Y_C, T.N_SUB_LO * DX),
                     width=T.W_TRACE, height=T.H_SUB, direction="-x",
                     impedance=50.0, eps_r_sub=T.EPS_SUB)

    entries, mf = _build_target_entries(T.FREQS_2)
    from rfx.api import Simulation as _S
    plane_scratch = _S(freq_max=T.FREQ_MAX_2, domain=(T.LX_2, T.LY, T.LZ_2),
                       dx=DX, boundary="cpml")
    for nm, x in (("planeA", 9.0e-3), ("planeB", 3.0e-3)):
        plane_scratch.add_flux_monitor(
            axis="x", coordinate=x, freqs=mf,
            size=(2.8e-3, 2.5e-3), center=(1.7e-3, 2.15e-3), name=nm)
    entries = list(entries) + list(plane_scratch._flux_monitors)

    grid = sim._build_grid()
    materials, _debye, _lorentz, pec_mask, _, _, _ = sim._assemble_materials(grid)
    msl_pe = sim._msl_ports[0]
    # mirror the extractor's entry -> low-level MSLPort conversion verbatim
    x_feed, y_centre, msl_z_lo = (float(c) for c in msl_pe.position)
    msl_port_base = _MSLPortLL(
        feed_x=x_feed,
        y_lo=y_centre - msl_pe.width / 2, y_hi=y_centre + msl_pe.width / 2,
        z_lo=msl_z_lo, z_hi=msl_z_lo + msl_pe.height,
        direction=msl_pe.direction, impedance=msl_pe.impedance,
        excitation=None,
    )
    eps_r = float(T.EPS_SUB)
    mode_profile = compute_msl_mode_profile(grid, msl_port_base, eps_r)
    materials = setup_msl_port(grid, msl_port_base, materials,
                               mode_profile=mode_profile)
    import dataclasses as _dc
    wf = GaussianPulse(f0=sim._freq_max / 2, bandwidth=0.8)
    msl_driven = _dc.replace(msl_port_base, excitation=wf)
    sources = make_msl_port_sources(grid, msl_driven, materials, int(n_steps),
                                    mode_profile=mode_profile)

    y_c = T.Y_C
    k_probe = int(round((T.N_SUB_LO * DX + 0.5 * msl_pe.height) / DX)) + int(grid.pad_z_lo)
    i_probe = int(grid.position_to_index((6.0e-3, y_c, T.N_SUB_LO * DX))[0])
    j_probe = int(grid.pad_y_lo) + int(round(y_c / DX))
    witness = [ProbeSpec(i=i_probe, j=j_probe, k=k_probe, component="ez")]

    _assert_monitor_indices(entries, grid)
    flux_cfgs = build_flux_monitor_cfgs(sim, grid, int(n_steps), entries=entries)
    result = _run(grid, materials, int(n_steps), boundary="cpml",
                  cpml_axes="xyz", sources=sources, mag_sources=[],
                  probes=witness, dft_planes=[], pec_mask=pec_mask,
                  return_state=False, flux_monitors=flux_cfgs)

    spectra = {e.name: np.asarray(flux_spectrum(fm), float)
               for e, fm in zip(entries, result.flux_monitors or ())}
    settling = _settling_from_witness(result.time_series)
    assert float(np.max(np.asarray(result.time_series) ** 2)) > 0.0, \
        "hand-ported preflight: witness saw zero excitation energy"

    idx = _bin_index(mf, T.FREQS_2)
    phi_a = np.asarray(spectra["planeA"], float)[idx]
    phi_b = np.asarray(spectra["planeB"], float)[idx]
    p_inc = -phi_a  # incident travels -x: net flux at planeA = -P_inc (matched)
    plane_inv = np.abs(phi_a - phi_b) / np.maximum(np.abs(phi_a), 1e-300)
    ch = _channels(spectra, p_inc, idx)
    # Launch-spill background (review fix): the through-line trace passes
    # THROUGH both x-faces, so xm carries the full transmitted power and
    # must be EXCLUDED — the background covers the faces the target's R_nl
    # shares minus xm (xm background is unmeasurable on a through line;
    # disclosed). Composition: xp remainder + yp + ym + zt.
    terms = {k: np.asarray(v, float)
             for k, v in ch["per_face_outward"].items()}
    b_nl = terms["xp_rem"] + terms["yp"] + terms["ym"] + terms["zt"]
    strips = {k: np.asarray(spectra[k], float)[idx].tolist()
              for k in STRIPS}
    strip_max = max(float(np.max(np.abs(np.asarray(v)))) for v in strips.values())

    gates = {
        "settling_db": settling,
        "settling_gate_le_-40": bool(settling <= -40.0),
        "p_inc_positive": bool(np.all(p_inc > 0.0)),
        "plane_invariance": plane_inv.tolist(),
        "plane_invariance_gate_le_floor": bool(np.max(plane_inv) <= FLOOR_FAIL),
        "measured_floor": float(np.max(plane_inv)),
        "launch_spill_B_nl": b_nl.tolist(),
        "launch_spill_composition": "xp_rem + yp + ym + zt (xm EXCLUDED: "
                                    "through-line exit face; disclosed)",
        "spill_gate_le_0.05": bool(np.max(np.abs(b_nl)) <= 0.05),
        "below_ground_strips": strips,
        "below_ground_strip_max_over_pinc": strip_max / max(
            float(np.max(np.abs(p_inc))), 1e-300),
    }
    gates["c1_pass"] = bool(
        gates["settling_gate_le_-40"] and gates["p_inc_positive"]
        and gates["plane_invariance_gate_le_floor"] and gates["spill_gate_le_0.05"]
    )
    out = {
        "stage": "c1", "n_steps": int(n_steps),
        "monitor_freqs_hz": mf.tolist(), "freqs2_idx": idx,
        "p_inc_w": p_inc.tolist(), "gates": gates, "channels": ch,
        "spectra": {k: v.tolist() for k, v in spectra.items()},
    }
    (outdir / "c1_result.json").write_text(json.dumps(out, indent=1))
    print(f"[c1] settling {settling:.2f} dB | plane-inv "
          f"{np.max(plane_inv):.4f} | spill {np.max(np.abs(b_nl)):.4f} "
          f"| PASS={gates['c1_pass']}")
    return out


# ---------------------------------------------------------------------------
# C2 — coax two-port control (validated lane, attenuation-model gate)
# ---------------------------------------------------------------------------

def run_c2(n_steps, outdir):
    from rfx.api import Simulation
    from rfx import GaussianPulse
    import tests.test_coax_two_port_fdtd as T2

    band = T2.BAND
    mf = _monitor_freqs(band)
    dom = (0.008, 0.008, 0.060)
    scratch = Simulation(domain=dom, freq_max=40.0e9, boundary="cpml")
    z1, z2 = 0.025, 0.035
    for nm, z in (("c2_z1", z1), ("c2_z2", z2)):
        scratch.add_flux_monitor(axis="z", coordinate=z, freqs=mf,
                                 size=(6.0e-3, 6.0e-3), center=(4.0e-3, 4.0e-3),
                                 name=nm)
    for nm, ax, c in (("c2_xm", "x", 0.5e-3), ("c2_xp", "x", 7.5e-3),
                      ("c2_ym", "y", 0.5e-3), ("c2_yp", "y", 7.5e-3)):
        scratch.add_flux_monitor(axis=ax, coordinate=c, freqs=mf,
                                 size=(6.0e-3, 10.0e-3), center=(4.0e-3, 30.0e-3),
                                 name=nm)

    sim = Simulation(domain=dom, freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    res = sim.compute_coaxial_two_port(
        n_steps=int(n_steps), freqs=band,
        extra_flux_monitors=scratch._flux_monitors,
    )
    idx = _bin_index(mf, band)
    gamma = np.asarray(res.gamma)
    # Attenuation model (review fix): the committed mechanism-check record
    # attributes the own/other-drive Re(gamma) split to loss at the
    # RECEIVING feed, which does not occur between the mid-line planes —
    # so the bulk ratio is modeled by the OWN-DRIVE gamma pair; the
    # all-fits mean is recorded alongside for attribution if the gate
    # trips near the band edge.
    if gamma.ndim == 3:
        re_own = np.mean(np.abs(np.real(gamma[(0, 1), (0, 1), :])), axis=0)
    else:
        re_own = np.mean(np.abs(np.real(gamma)).reshape(-1, gamma.shape[-1]),
                         axis=0)
    re_all = np.mean(np.abs(np.real(gamma)).reshape(-1, gamma.shape[-1]), axis=0)
    # snapped plane separation from the actual grid indices
    g2 = sim._build_grid()
    k1 = int(g2.position_to_index((0.0, 0.0, z1))[2])
    k2 = int(g2.position_to_index((0.0, 0.0, z2))[2])
    d12 = (k2 - k1) * float(g2.dx)
    model = np.exp(-2.0 * re_own * d12)
    model_allfits = np.exp(-2.0 * re_all * d12)

    def flux(drive, name):
        return np.asarray(res.flux_monitors[drive][name], float)[idx]

    # drive port1 = top: power travels -z through both planes (negative net)
    f1, f2 = flux("port1", "c2_z1"), flux("port1", "c2_z2")
    # power decays toward -z: |flux| at z1 (farther from top source) is the
    # attenuated one: |f1| ~= |f2| * exp(-2 Re(gamma) d12)
    ratio = np.abs(f1) / np.maximum(np.abs(f2), 1e-300)
    atten_ok = np.abs(ratio / model - 1.0) <= 0.15
    # W1 Poynting closure over the closed C2 box (posted gate): outward =
    # +f2 (z2, +z face) - f1 (z1, -z face) + outward side faces.
    side_out = (flux("port1", "c2_xp") - flux("port1", "c2_xm")
                + flux("port1", "c2_yp") - flux("port1", "c2_ym"))
    w1_c2 = np.abs(f2 - f1 + side_out) / np.maximum(np.abs(f2), 1e-300)
    # per-bin shield gate (posted: per-bin, floor-scaled)
    side_abs = np.max(np.stack([np.abs(flux("port1", n)) for n in
                                ("c2_xm", "c2_xp", "c2_ym", "c2_yp")]), axis=0)
    shield_ok = side_abs <= FLOOR * np.maximum(np.abs(f2), 1e-300)
    col = np.sum(np.abs(np.asarray(res.s_params)[:, 0, :]) ** 2, axis=0)

    gates = {
        "signs_negative_towards_minus_z": bool(np.all(f1 < 0) and np.all(f2 < 0)),
        "atten_ratio": ratio.tolist(), "atten_model_own_drive": model.tolist(),
        "atten_model_all_fits": model_allfits.tolist(),
        "d12_snapped_m": d12,
        "atten_gate_pm15pct": bool(np.all(atten_ok)),
        "w1_closure": w1_c2.tolist(),
        "w1_gate_le_0.15": bool(np.all(w1_c2 <= W_TOL)),
        "shield_side_over_f2_per_bin": (side_abs / np.maximum(np.abs(f2), 1e-300)).tolist(),
        "shield_gate_le_floor_per_bin": bool(np.all(shield_ok)),
        "column_power_port1_drive": col.tolist(),
        "column_power_committed_range": [0.546, 0.923],
        "column_power_in_widened_range_0.50_0.97": bool(
            np.all((col > 0.50) & (col < 0.97))),
        "settling_db": np.asarray(res.settling_db, float).tolist(),
        "settling_gate_le_-40": bool(np.all(np.asarray(res.settling_db) <= -40.0)),
    }
    gates["c2_pass"] = bool(
        gates["atten_gate_pm15pct"] and gates["shield_gate_le_floor_per_bin"]
        and gates["w1_gate_le_0.15"]
        and gates["settling_gate_le_-40"]
        and gates["signs_negative_towards_minus_z"]
    )
    out = {"stage": "c2", "n_steps": int(n_steps),
           "monitor_freqs_hz": mf.tolist(), "gates": gates,
           "flux_monitors": {d: {k: v.tolist() for k, v in s.items()}
                             for d, s in res.flux_monitors.items()}}
    (outdir / "c2_result.json").write_text(json.dumps(out, indent=1))
    print(f"[c2] atten dev {np.max(np.abs(ratio / model - 1.0)):.4f} | "
          f"W1 {np.max(w1_c2):.4f} | shield "
          f"{np.max(side_abs / np.maximum(np.abs(f2), 1e-300)):.2e} | "
          f"PASS={gates['c2_pass']}")
    return out


# ---------------------------------------------------------------------------
# target — settled attempt-2 fixture + flux box, pre-declared decision rule
# ---------------------------------------------------------------------------

def run_target(n_steps, outdir, c1_data, floor_eff=FLOOR):
    import tests.test_coax_msl_transition as T

    p_inc_c1 = c1_data["p_inc_w"]
    b_nl_bg = np.asarray(c1_data["gates"]["launch_spill_B_nl"], float)
    entries, mf = _build_target_entries(T.FREQS_2)
    sim = T._build_coax_msl_transition_sim_attempt2()
    _assert_monitor_indices(entries, sim._build_grid())
    res = sim.compute_coax_msl_transition(
        junction_x=T.JUNCTION_X, eps_r_sub=T.EPS_SUB, n_steps=int(n_steps),
        freqs=T.FREQS_2,
        probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        msl_probe_count=T.PROBE_COUNT_2, msl_probe_start_cells=T.PROBE_START_2,
        msl_probe_spacing_cells=T.PROBE_SPACING_2,
        skip_preflight=True, strict_passivity=False,
        extra_flux_monitors=entries,
    )
    idx = _bin_index(mf, T.FREQS_2)
    spectra = res.flux_monitors["msl"]
    p_inc = np.asarray(p_inc_c1, float)
    S = np.asarray(res.s_params)
    s22 = np.abs(S[1, 1, :]) ** 2
    s12 = np.abs(S[0, 1, :]) ** 2
    deficit = 1.0 - s22 - s12

    ch = _channels(spectra, p_inc, idx, b_nl_subtract=b_nl_bg)
    shifts = _shift_table(spectra, p_inc, idx)
    a_msl = np.abs(np.asarray(res.a_inc)[1, 1, :]) ** 2
    kappa = (p_inc / np.maximum(a_msl, 1e-300)).tolist()
    kappa_in_soft_range = [bool(0.7 <= k <= 1.3) for k in kappa]
    if not all(kappa_in_soft_range[:2]):
        print("[target] WARNING: kappa outside soft range [0.7, 1.3] at a "
              f"binding bin: {np.round(kappa, 3).tolist()} — investigate "
              "the P_inc normalization before trusting this verdict "
              "(pre-declared investigate-before-interpret trigger).")

    # spill-corrected R_nl is the DECISION value (posted C1 bullet:
    # background 'reported and subtracted explicitly'); raw kept alongside.
    r_nl = np.asarray(ch["R_nl"])
    r_ret = np.asarray(ch["R_msl_return"])
    w1 = np.asarray(ch["W1_raw_closure"])
    w2 = np.asarray(ch["W2_bookkeeping_check"])
    two_plane = np.asarray(ch["two_plane_delta"])
    settling_ok = bool(np.all(np.asarray(res.settling_db) <= -40.0))
    strips = {k: np.asarray(spectra[k], float)[idx].tolist() for k in STRIPS}
    strip_max_over_pinc = max(
        float(np.max(np.abs(np.asarray(v) / p_inc))) for v in strips.values())

    b0, b1 = 0, 1  # binding bins: 6, 8 GHz; bin 2 (10 GHz) = trend witness
    shifts["gate_lt_floor"] = bool(shifts["worst_abs_delta"] < floor_eff)
    w1_per_bin_ok = (w1 <= W_TOL).tolist()
    # W1 gates at the binding bins; W2 is a bookkeeping identity of the
    # channel arithmetic (== W1 by construction), reported, not counted.
    witnesses_ok = bool(np.all(w1[[b0, b1]] <= W_TOL)
                        and shifts["gate_lt_floor"] and settling_ok)
    ret_leg_ok = bool(np.all(np.abs(two_plane[[b0, b1]]) <= W_TOL))

    # Contamination rule, symmetric (review fix, matches the posted text
    # "only R_nl binds"): when the two-plane witness declares the return
    # leg contaminated, BOTH arms reduce to their R_nl condition.
    arm_a = bool(
        np.all(r_nl[[b0, b1]] >= 0.7 * deficit[[b0, b1]])
        and np.all(r_nl[[b0, b1]] <= 1.3 * deficit[[b0, b1]])
        and ((not ret_leg_ok) or np.all(
            (r_ret - s22)[[b0, b1]] <= 0.15))
    )
    arm_b = bool(
        np.all(r_nl[[b0, b1]] <= 0.3 * deficit[[b0, b1]])
        and ((not ret_leg_ok) or np.any(
            (r_ret - s22)[[b0, b1]] >= 0.5 * deficit[[b0, b1]]))
    )
    if not witnesses_ok:
        verdict = "NON-CLOSING (witness failure)"
    elif arm_a and not arm_b:
        verdict = "SUPPORTS (a) PHYSICAL"
    elif arm_b and not arm_a:
        verdict = "SUPPORTS (b) INSTRUMENT"
    else:
        verdict = "NON-CLOSING"

    out = {
        "stage": "target", "n_steps": int(n_steps), "verdict": verdict,
        "monitor_freqs_hz": mf.tolist(), "freqs2_idx": idx,
        "p_inc_from_c1_w": p_inc.tolist(), "kappa_witness": kappa,
        "kappa_in_soft_range_0.7_1.3": kappa_in_soft_range,
        "launch_spill_bg_from_c1": b_nl_bg.tolist(),
        "decision_uses": "R_nl spill-corrected (raw kept in channels.R_nl_raw)",
        "deficit": deficit.tolist(), "s22_sq": s22.tolist(),
        "s12_sq": s12.tolist(),
        "settling_db": np.asarray(res.settling_db, float).tolist(),
        "channels": ch, "shift_invariance": shifts,
        "shift_member_flags": SHIFT_FLAGS,
        "below_ground_strips": strips,
        "below_ground_strip_max_over_pinc": strip_max_over_pinc,
        "w1_per_bin_ok": w1_per_bin_ok,
        "ret_leg_ok_two_plane": ret_leg_ok,
        "trend_witness_10ghz_R_nl": float(r_nl[2]),
        "trend_witness_10ghz_w1_ok": bool(w1_per_bin_ok[2]),
        "trend_expected_if_a": [0.14, 0.26],
        "spectra_msl_drive": {k: np.asarray(v).tolist()
                              for k, v in spectra.items()},
        "spectra_coax_drive": {k: np.asarray(v).tolist()
                               for k, v in res.flux_monitors["coax"].items()},
    }
    if verdict == "SUPPORTS (b) INSTRUMENT":
        out["predeclared_consequence"] = (
            "Per the #589 pre-declaration: the follow-up is a Gwarek V-I "
            "MSL-side extraction check per the repo's port-class fallback "
            "discipline — flux numbers never become S numbers. NOTE "
            "(PI decision 2026-08-07): the lane hard-stops after this "
            "verdict is reported; the Gwarek follow-up is NOT auto-started."
        )
        print("[target] predeclared consequence:", out["predeclared_consequence"])
    (outdir / "target_result.json").write_text(json.dumps(out, indent=1))
    print(f"[target] verdict: {verdict}")
    print(f"  Deficit {deficit.round(4).tolist()}  R_nl {r_nl.round(4).tolist()}")
    print(f"  R_msl_return-|S22|^2 {(r_ret - s22).round(4).tolist()}  "
          f"W1 {w1.round(4).tolist()}  W2 {w2.round(4).tolist()}")
    print(f"  shift worst {shifts['worst_abs_delta']:.4f}  "
          f"two-plane {two_plane.round(4).tolist()}  kappa {np.round(kappa,3).tolist()}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("c1", "c2", "target", "all"),
                    default="all")
    ap.add_argument("--n-steps-c1", type=int, default=20000)
    ap.add_argument("--n-steps-c2", type=int, default=6000)
    ap.add_argument("--n-steps-target", type=int, default=135000)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("c1", "all"):
        c1 = run_c1(args.n_steps_c1, args.output_dir)
        if args.stage == "all" and not c1["gates"]["c1_pass"]:
            print("[all] C1 hard-gate FAILED -> target skipped (rc=2)")
            return 2
    if args.stage in ("c2", "all"):
        c2 = run_c2(args.n_steps_c2, args.output_dir)
        if args.stage == "all" and not c2["gates"]["c2_pass"]:
            print("[all] C2 hard-gate FAILED -> target skipped (rc=2)")
            return 2
    if args.stage in ("target", "all"):
        # both controls must EXIST and PASS regardless of invocation path
        # (review fix: --stage target alone previously skipped the gates)
        gate_data = {}
        for st in ("c1", "c2"):
            p = args.output_dir / f"{st}_result.json"
            if not p.exists():
                print(f"[target] {st}_result.json missing — run --stage {st} "
                      "first (controls gate the target, per the "
                      "pre-declaration)")
                return 2
            gate_data[st] = json.loads(p.read_text())
            if not gate_data[st]["gates"][f"{st}_pass"]:
                print(f"[target] {st} hard-gate FAILED in its recorded "
                      "result — target not interpreted (rc=2)")
                return 2
        c1_data = gate_data["c1"]
        # pre-declared: floor = 0.02 unless C1 measured worse (fail > 0.05)
        floor_eff = min(max(FLOOR, c1_data["gates"]["measured_floor"]),
                        FLOOR_FAIL)
        run_target(args.n_steps_target, args.output_dir, c1_data,
                   floor_eff=floor_eff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
