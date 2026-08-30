#!/usr/bin/env python3
"""F-M1b RETRY measurement: paper-faithful Fig. 8 fixture (arXiv:1606.08761 V-C).

Fixture (pre-declared in portgrid_m1b_retry_predeclaration.md; geometry decoded
from the Fig. 8 vector data): 66 x 40 mm guide, coarse dx = dy = 1 mm, PEC at
y = 0/40 mm, split-field x-PML (15 cells, m = 3, R0 = 1e-5) at both ends,
Jy line source column x = 17 mm, probe = y-mean of Ey on column x = 19 mm,
8 x 8 mm subgrid over coarse cells [39,47) x [16,24); rod arm adds four copper
rods (sigma = 5.8e7 S/m, radius 1 mm, centers (41,18)/(41,22)/(45,18)/(45,22)
mm). dt = 0.99 x fine CFL per run; the reference of each run uses the SAME dt.
|S11|(f) = |FFT(probe_run - probe_ref)| / |FFT(probe_ref)|, no gating.

Arms
  interface  r in {2,3,4,5,6}, no rods.   Windows (FROZEN):
             max|S11| [2,20] GHz <= -46.24 dB AND [2,30] GHz <= -29.29 dB.
  null       r = 1 chain null: max|S11| [2,30] GHz <= -200 dB.
  floor      PML reflection floor, 400 x 40 mm guide, src col 100, probe col
             150; gates (Correction R1): direct [0, 0.30] ns, echo
             [0.55, 1.15] ns; window |R_PML| <= -50 dB on [2,30] GHz at
             dt(r=2) and dt(r=6).
  rods       subgrid r in {2,4,6} + all-fine (r=6 uniform) + all-coarse, all
             at dt(r=6) except subgrid r in {2,4} at their own dt.  Window
             (FROZEN, r=6 only): max linear | |S11_sub| - |S11_fine| | over
             [2,30] GHz <= 0.0941.  r in {2,4} and all-coarse recorded.
  xcheck     RECORDING ONLY (cannot fire), pre-declaration Correction R3:
             (a) the interface arm re-read through a point probe as well as the
             y-mean, against the SAME frozen windows; (b) our ALL-FINE (no
             subgridding anywhere) absolute |S11| against the paper's own
             all-fine curve -- pure Yee vs pure Yee, so any gap is fixture or
             observable, never the scheme; (c) the R3(c) rescaled rod window.

NOTE on the probe (Correction R3): the y-mean of the Ey column is the TEM
(n = 0) modal projection.  It is what the pre-declaration declared and what the
frozen verdicts were taken under, but it is BLIND to reflected energy that the
interface or the scatterer converts into higher-order modes.  Read the xcheck
arm before quoting any |S11| from this script as "the" reflection.

Run:
  PYTHONPATH=<worktree> .venv/bin/python \
      validation/research/portgrid/m1b_retry.py --arm all --out results.json

Mur-1 rejection derivation (kept per the pre-declaration; not used by any arm):
the discrete Mur-1 reflection follows from the plane-wave ansatz on the update
u0' = u1 + kappa (u1' - u0), kappa = (c dt - dx)/(c dt + dx), with k(w) from
sin(w dt/2)/(c dt) = sin(k dx/2)/dx:
  R(w) = (P - T + kappa (T P - 1)) / (T - 1/P + kappa (1 - T/P)),
  T = exp(i w dt), P = exp(i k dx)
giving |R(30 GHz)| = -31.8 dB at dt = 0.99 fine-CFL(r=6), dx = 1 mm.
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)  # script entrypoint

import numpy as np

C0 = 299792458.0
NX, NY = 66, 40
DX = DY = 1e-3
ISLAND = (39, 47, 16, 24)
SRC_COL, PROBE_COL = 17, 19
NPML = 15
F0, HWHM = 16e9, 10e9
SIGMA_CU = 5.8e7
ROD_CENTERS = [(41e-3, 18e-3), (41e-3, 22e-3), (45e-3, 18e-3), (45e-3, 22e-3)]
ROD_RADIUS = 1e-3
WIN_20, WIN_30 = -46.24, -29.29
# F-M1b-rod window as FROZEN in the pre-declaration (§5): the paper-extracted
# r=6-vs-all-fine mismatch 0.0529 x 10^(5/20).  Keeps its verdict authority.
#
# CAVEAT (retry pre-declaration Correction R3(c), review finding): 0.0529 is an
# ABSOLUTE linear difference measured on the paper's curves, whose all-fine peak
# is 0.3968 (-8.03 dB); OUR all-fine curve under this fixture's declared
# y-averaged (TEM-projection) observable peaks at 0.2073 (-13.67 dB).  The
# "+5 dB rule" the window claims to implement is a RELATIVE class rule, so
# transplanting the absolute difference delivers
#   20 log10(0.3968 / 0.2073) = 5.64 dB
# MORE allowance than +5 dB intends -- a factor 1.91 looser.  The rescaling rule
# below was declared in Correction R3(c) BEFORE this number was computed; it is
# recorded and reported as a strictly TIGHTER re-judgment of the same, unchanged
# measurement.  The frozen window is not moved.
WIN_ROD_LINEAR = 0.0941
PAPER_ROD_MISMATCH_LINEAR = 0.0529     # portgrid_fig9_extraction.json
PAPER_ALLFINE_PEAK_LINEAR = 0.3968     # portgrid_fig9_allfine_absolute.json
WIN_FLOOR = -50.0
WIN_NULL = -200.0
WIN_ABS_DB = 5.0                       # F-M1b-abs, Correction R3(b)
PAPER_ABS_JSON = "docs/design_notes/portgrid_fig9_allfine_absolute.json"


def rescaled_rod_window(our_allfine_peak_linear: float) -> float:
    """Correction R3(c) rule, declared before evaluation:

        0.0529 x 10^(5/20) x (our all-fine peak / paper all-fine peak)

    i.e. the +5 dB class allowance carried as a RELATIVE fraction of the
    all-fine signal level and re-applied at this fixture's own level.  The
    normaliser is the REFERENCE arm's peak; no part of the judged difference
    enters its own window.
    """
    return (PAPER_ROD_MISMATCH_LINEAR * 10.0 ** (5.0 / 20.0)
            * our_allfine_peak_linear / PAPER_ALLFINE_PEAK_LINEAR)


def _dt_for(sim2d, r):
    spec = sim2d.TwoRegionSpec(nx=NX, ny=NY, dx=DX, dy=DY, i0=ISLAND[0],
                               i1=ISLAND[1], j0=ISLAND[2], j1=ISLAND[3],
                               r=r, dt=np.nan)
    return 0.99 * sim2d.fine_cfl_dt(spec), spec


def _s11(p_run, p_ref, dt):
    n = len(p_ref)
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.fft.rfftfreq(nfft, dt)
    inc = np.fft.rfft(np.asarray(p_ref), nfft)
    refl = np.fft.rfft(np.asarray(p_run - p_ref), nfft)
    s11 = np.abs(refl) / np.maximum(np.abs(inc), 1e-300)
    return f, s11


def _band_max_db(f, s11, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.max(20.0 * np.log10(np.maximum(s11[m], 1e-300))))


def subgrid_pair(sim2d, r, t_total, sigma_f=None):
    """Return (f, s11, dt) for a subgrid run vs its uniform reference."""
    dt, spec = _dt_for(sim2d, r)
    spec.dt = dt
    n_steps = int(np.ceil(t_total / dt))
    wf = sim2d.gaussian_modulated(n_steps, dt, F0, HWHM)

    kw = {}
    if sigma_f is not None:
        kw = dict(sigma_fx=sigma_f[0], sigma_fy=sigma_f[1])
    step, init, _ = sim2d.make_stepper_pml(
        spec, src_col=SRC_COL, probe_col=PROBE_COL, npml=NPML, **kw)
    p_run = np.asarray(jax.jit(
        lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])

    ustep, uinit, _ = sim2d.make_uniform_pml(
        NX, NY, DX, DY, dt, src_col=SRC_COL, probe_col=PROBE_COL, npml=NPML)
    p_ref = np.asarray(jax.jit(
        lambda s, w: jax.lax.scan(ustep, s, w))(uinit(), wf)[1])

    f, s11 = _s11(p_run, p_ref, dt)
    return f, s11, dt, float(np.max(np.abs(p_run - p_ref)))


def uniform_pair(sim2d, nx, ny, d, dt, t_total, npml, src_col, probe_col,
                 sigma_maps):
    """(f, s11) for a uniform grid with materials vs its vacuum reference."""
    n_steps = int(np.ceil(t_total / dt))
    wf = sim2d.gaussian_modulated(n_steps, dt, F0, HWHM)
    step, init, _ = sim2d.make_uniform_pml(
        nx, ny, d, d, dt, src_col=src_col, probe_col=probe_col, npml=npml,
        sigma_x=sigma_maps[0], sigma_y=sigma_maps[1])
    p_run = np.asarray(jax.jit(
        lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])
    rstep, rinit, _ = sim2d.make_uniform_pml(
        nx, ny, d, d, dt, src_col=src_col, probe_col=probe_col, npml=npml)
    p_ref = np.asarray(jax.jit(
        lambda s, w: jax.lax.scan(rstep, s, w))(rinit(), wf)[1])
    return _s11(p_run, p_ref, dt)


def arm_interface(sim2d, out):
    fired = False
    for r in (2, 3, 4, 5, 6):
        t0 = time.perf_counter()
        f, s11, dt, dmax = subgrid_pair(sim2d, r, 4.0e-9)
        m20 = _band_max_db(f, s11, 2e9, 20e9)
        m30 = _band_max_db(f, s11, 2e9, 30e9)
        ok = (m20 <= WIN_20) and (m30 <= WIN_30)
        fired |= not ok
        out[f"interface_r{r}"] = {
            "dt_s": dt, "max_s11_db_2_20GHz": m20, "max_s11_db_2_30GHz": m30,
            "windows_db": [WIN_20, WIN_30], "fired": not ok,
            "wall_s": round(time.perf_counter() - t0, 1)}
        print(f"[interface r={r}] max|S11| [2,20]={m20:7.2f} dB (win {WIN_20}) "
              f"[2,30]={m30:7.2f} dB (win {WIN_30}) -> {'FIRE' if not ok else 'PASS'}")
    out["F_M1b_r2_fired"] = fired
    return fired


def arm_null(sim2d, out):
    f, s11, dt, dmax = subgrid_pair(sim2d, 1, 4.0e-9)
    m30 = _band_max_db(f, s11, 2e9, 30e9)
    ok = m30 <= WIN_NULL
    out["chain_null_r1"] = {"max_s11_db_2_30GHz": m30, "window_db": WIN_NULL,
                           "max_abs_time_diff": dmax, "fired": not ok}
    print(f"[null r=1] max|S11| [2,30] = {m30:.1f} dB (win {WIN_NULL}) "
          f"time-domain max diff {dmax:.2e} -> {'FIRE' if not ok else 'PASS'}")
    return not ok


def arm_floor(sim2d, out):
    fired = False
    nx, ny = 400, 40
    src, prb = 100, 150
    for r in (2, 6):
        dt, _ = _dt_for(sim2d, r)
        n_steps = int(np.ceil(2.2e-9 / dt))
        wf = sim2d.gaussian_modulated(n_steps, dt, F0, HWHM)
        step, init, _ = sim2d.make_uniform_pml(
            nx, ny, DX, DY, dt, src_col=src, probe_col=prb, npml=NPML)
        p = np.asarray(jax.jit(
            lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])
        t = np.arange(n_steps) * dt
        direct = np.where(t <= 0.30e-9, p, 0.0)
        echo = np.where((t >= 0.55e-9) & (t <= 1.15e-9), p, 0.0)
        nfft = 1 << int(np.ceil(np.log2(4 * n_steps)))
        f = np.fft.rfftfreq(nfft, dt)
        rr = np.abs(np.fft.rfft(echo, nfft)) / np.maximum(
            np.abs(np.fft.rfft(direct, nfft)), 1e-300)
        m30 = _band_max_db(f, rr, 2e9, 30e9)
        ok = m30 <= WIN_FLOOR
        fired |= not ok
        out[f"pml_floor_dt_r{r}"] = {"dt_s": dt, "max_R_db_2_30GHz": m30,
                                     "window_db": WIN_FLOOR, "fired": not ok}
        print(f"[floor dt(r={r})] max|R_PML| [2,30] = {m30:6.1f} dB "
              f"(win {WIN_FLOOR}) -> {'FIRE' if not ok else 'PASS'}")
    out["F_M1b_abc_fired"] = fired
    return fired


def arm_rods(sim2d, out):
    t_total = 6.0e-9
    dt6, _ = _dt_for(sim2d, 6)

    # subgrid runs (rods live on the fine grid of each r)
    sub = {}
    for r in (2, 4, 6):
        dxf = DX / r
        nfx = (ISLAND[1] - ISLAND[0]) * r
        nfy = (ISLAND[3] - ISLAND[2]) * r
        sig = sim2d.disk_sigma_maps(
            nfx, nfy, dxf, dxf, (ISLAND[0] * DX, ISLAND[2] * DY),
            ROD_CENTERS, ROD_RADIUS, SIGMA_CU)
        f, s11, dt, _ = subgrid_pair(sim2d, r, t_total, sigma_f=sig)
        sub[r] = (f, s11)
        out[f"rods_subgrid_r{r}"] = {"dt_s": dt}

    # all-fine r = 6 uniform (same dt as the r = 6 subgrid run)
    rf = 6
    dfine = DX / rf
    nxf, nyf = NX * rf, NY * rf
    sigf = sim2d.disk_sigma_maps(nxf, nyf, dfine, dfine, (0.0, 0.0),
                                 ROD_CENTERS, ROD_RADIUS, SIGMA_CU)
    t0 = time.perf_counter()
    ff, s11_fine = uniform_pair(sim2d, nxf, nyf, dfine, dt6, t_total,
                                NPML * rf, SRC_COL * rf, PROBE_COL * rf, sigf)
    out["rods_allfine"] = {"dt_s": dt6, "wall_s": round(time.perf_counter() - t0, 1)}

    # all-coarse uniform (staircased rods, same dt)
    sigc = sim2d.disk_sigma_maps(NX, NY, DX, DY, (0.0, 0.0),
                                 ROD_CENTERS, ROD_RADIUS, SIGMA_CU)
    fc, s11_coarse = uniform_pair(sim2d, NX, NY, DX, dt6, t_total,
                                  NPML, SRC_COL, PROBE_COL, sigc)

    # mismatch metric on the all-fine frequency grid, [2,30] GHz
    band = (ff >= 2e9) & (ff <= 30e9)
    fine_b = s11_fine[band]

    def mismatch(f_o, s_o):
        s_i = np.interp(ff[band], f_o, s_o)
        return float(np.max(np.abs(s_i - fine_b)))

    mm = {r: mismatch(*sub[r]) for r in (2, 4, 6)}
    mm["all_coarse"] = mismatch(fc, s11_coarse)
    ok = mm[6] <= WIN_ROD_LINEAR
    out["rods_mismatch_linear_vs_allfine_2_30GHz"] = {
        "r2": mm[2], "r4": mm[4], "r6": mm[6], "all_coarse": mm["all_coarse"],
        "window_linear_r6": WIN_ROD_LINEAR, "F_M1b_rod_fired": not ok,
        "max_s11_fine_linear": float(np.max(fine_b))}
    for k, v in mm.items():
        print(f"[rods {k}] max linear mismatch vs all-fine = {v:.4f}"
              + (f" (win {WIN_ROD_LINEAR}) -> "
                 f"{'FIRE' if not ok else 'PASS'}" if k == 6 else ""))
    return not ok


# --------------------------------------------------------------- xcheck arm
# Correction R3: the declared observable (y-mean of the Ey probe column) is the
# TEM (n = 0) projection.  These helpers run the SAME fixtures and return the
# whole Ey column so several projections can be read off ONE run.

PROBE_PROJECTIONS = ("y_mean", "point_quarter", "point_centre")


def _project(cols):
    """Projections of an (n_steps, ny) Ey-probe-column trace.

    y_mean          the declared observable: TEM amplitude only.
    point_quarter   Ey at the cell nearest y = H/4.  Source and island are
                    symmetric about y = H/2, so the scattered field carries
                    only even modes cos(n pi y/H), n = 0, 2, 4...; cos(2 pi y/H)
                    has a NODE at y = H/4, so this row is the least
                    n = 2-contaminated point available.
    point_centre    Ey at y = H/2: TEM + n = 2 at full weight (opposite
                    extreme, recorded as the bound in the other direction).
    """
    ny = cols.shape[1]
    return {"y_mean": cols.mean(axis=1),
            "point_quarter": cols[:, ny // 4],
            "point_centre": cols[:, ny // 2]}


def _subgrid_cols(sim2d, r, t_total, sigma_f=None):
    dt, spec = _dt_for(sim2d, r)
    spec.dt = dt
    n_steps = int(np.ceil(t_total / dt))
    wf = sim2d.gaussian_modulated(n_steps, dt, F0, HWHM)
    kw = dict(sigma_fx=sigma_f[0], sigma_fy=sigma_f[1]) if sigma_f else {}
    step, init, _ = sim2d.make_stepper_pml(
        spec, src_col=SRC_COL, probe_col=PROBE_COL, npml=NPML,
        probe_full=True, **kw)
    run = np.asarray(jax.jit(lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])
    ustep, uinit, _ = sim2d.make_uniform_pml(
        NX, NY, DX, DY, dt, src_col=SRC_COL, probe_col=PROBE_COL, npml=NPML,
        probe_full=True)
    ref = np.asarray(jax.jit(lambda s, w: jax.lax.scan(ustep, s, w))(uinit(), wf)[1])
    return run, ref, dt


def _uniform_cols(sim2d, nx, ny, d, dt, t_total, npml, src_col, probe_col,
                  sigma_maps):
    n_steps = int(np.ceil(t_total / dt))
    wf = sim2d.gaussian_modulated(n_steps, dt, F0, HWHM)
    step, init, _ = sim2d.make_uniform_pml(
        nx, ny, d, d, dt, src_col=src_col, probe_col=probe_col, npml=npml,
        sigma_x=sigma_maps[0], sigma_y=sigma_maps[1], probe_full=True)
    run = np.asarray(jax.jit(lambda s, w: jax.lax.scan(step, s, w))(init(), wf)[1])
    rstep, rinit, _ = sim2d.make_uniform_pml(
        nx, ny, d, d, dt, src_col=src_col, probe_col=probe_col, npml=npml,
        probe_full=True)
    ref = np.asarray(jax.jit(lambda s, w: jax.lax.scan(rstep, s, w))(rinit(), wf)[1])
    return run, ref


def _committed_r6_mismatch(root):
    """The rods arm's measured r=6 mismatch, read from its COMMITTED json.

    Read rather than re-measured on purpose: Correction R3(c) re-expresses the
    WINDOW only; the measurement it judges is unchanged and must stay the one
    already on the record.
    """
    import pathlib

    f = pathlib.Path(root) / "docs/design_notes/portgrid_m1b_retry_rods.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get("rods_mismatch_linear_vs_allfine_2_30GHz", {}).get("r6")


def arm_xcheck(sim2d, out):
    """Correction R3 cross-checks.  NEITHER part re-judges a frozen falsifier.

    (a) F-M1b-r2-pt: the interface arm read through the R3(a) projections,
        against the ALREADY-FROZEN F-M1b-r2 windows.  Recording only.
    (b) F-M1b-abs: our ALL-FINE (uniform r=6, four rods, no subgrid anywhere)
        absolute |S11| against the paper's own all-fine curve.  Pure Yee vs
        pure Yee -- no scheme content, so any gap is fixture/observable.
        Declared BURNED for this lane (the review reported its outcome before
        the window was written); committed for M2 and for re-runs.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    paper = json.loads((root / PAPER_ABS_JSON).read_text())
    pa = paper["top_panel_absolute_anchors"]
    band = pa["band_GHz"]                       # [2.0, 29.8] as extracted
    fine_paper = pa["all_fine6"]

    # ---- (a) interface arm, three projections, frozen windows ----
    inc_unif = {}
    for r in (2, 4, 6):
        run, ref, dt = _subgrid_cols(sim2d, r, 4.0e-9)
        pr, pf = _project(run), _project(ref)
        row = {"dt_s": dt}
        for name in PROBE_PROJECTIONS:
            f, s11 = _s11(pr[name], pf[name], dt)
            row[name] = {
                "max_s11_db_2_20GHz": _band_max_db(f, s11, 2e9, 20e9),
                "max_s11_db_2_30GHz": _band_max_db(f, s11, 2e9, 30e9)}
        # the |S11| DENOMINATOR is the incident trace: check it is y-uniform, so
        # the projection cannot be moving the normalisation
        m = np.abs(ref).max()
        row["incident_y_nonuniformity_rel"] = float(
            np.abs(ref - ref.mean(axis=1, keepdims=True)).max() / m)
        out[f"xcheck_interface_r{r}"] = row
        inc_unif[r] = row["incident_y_nonuniformity_rel"]
        print(f"[xcheck interface r={r}] "
              + "  ".join(f"{n}: {row[n]['max_s11_db_2_20GHz']:7.2f}/"
                          f"{row[n]['max_s11_db_2_30GHz']:7.2f} dB"
                          for n in PROBE_PROJECTIONS)
              + f"  (incident y-nonuniformity {row['incident_y_nonuniformity_rel']:.1e})")

    # ---- (b) all-fine absolute vs the paper's own all-fine ----
    dt6, _ = _dt_for(sim2d, 6)
    rf = 6
    dfine = DX / rf
    nxf, nyf = NX * rf, NY * rf
    sigf = sim2d.disk_sigma_maps(nxf, nyf, dfine, dfine, (0.0, 0.0),
                                 ROD_CENTERS, ROD_RADIUS, SIGMA_CU)
    t0 = time.perf_counter()
    run, ref = _uniform_cols(sim2d, nxf, nyf, dfine, dt6, 6.0e-9,
                             NPML * rf, SRC_COL * rf, PROBE_COL * rf, sigf)
    pr, pf = _project(run), _project(ref)
    abs_rows = {}
    fired_any = False
    for name in PROBE_PROJECTIONS:
        f, s11 = _s11(pr[name], pf[name], dt6)
        fg = f / 1e9
        sel = (fg >= band[0]) & (fg <= band[1])
        db = 20.0 * np.log10(np.maximum(s11, 1e-300))
        row = {"max_db": float(np.max(db[sel])),
               "max_linear": float(np.max(s11[sel])),
               "f_at_max_GHz": float(fg[sel][np.argmax(db[sel])])}
        deltas = {"max_db": row["max_db"] - fine_paper["max_db"]}
        for fq in (10, 25, 29):
            v = float(np.interp(fq, fg, db))
            row[f"db_at_{fq}GHz"] = v
            deltas[f"db_at_{fq}GHz"] = v - fine_paper[f"db_at_{fq}GHz"]
        row["delta_vs_paper_db"] = deltas
        row["max_abs_delta_db"] = float(max(abs(v) for v in deltas.values()))
        row["within_5dB_class"] = bool(row["max_abs_delta_db"] <= WIN_ABS_DB)
        fired_any |= not row["within_5dB_class"]
        abs_rows[name] = row
        print(f"[xcheck all-fine {name:14s}] max {row['max_db']:7.2f} dB "
              f"(paper {fine_paper['max_db']:.2f})  worst |delta| "
              f"{row['max_abs_delta_db']:5.2f} dB -> "
              f"{'INSIDE' if row['within_5dB_class'] else 'OUTSIDE'} the 5 dB class")

    ymean_peak = abs_rows["y_mean"]["max_linear"]
    out["xcheck_allfine_absolute_vs_paper"] = {
        "band_GHz": band,
        "paper_all_fine6": fine_paper,
        "ours": abs_rows,
        "window_db": WIN_ABS_DB,
        "wall_s": round(time.perf_counter() - t0, 1),
        "F_M1b_abs_outside_class_under_some_projection": bool(fired_any),
        "verdict_authority": (
            "NONE in this lane -- BURNED: the adversarial review reported this "
            "arm's outcome before its window was written (pre-declaration "
            "Correction R3(b)).  Committed so M2 and any re-run inherit an "
            "unburned fixture-fidelity gate."),
    }
    out["xcheck_rod_window_rescaling"] = {
        "rule": ("0.0529 x 10^(5/20) x (our all-fine peak / paper all-fine "
                 "peak), y-mean observable, [2,30] GHz -- Correction R3(c), "
                 "declared before evaluation"),
        "frozen_window_linear": WIN_ROD_LINEAR,
        "our_allfine_peak_linear_ymean": ymean_peak,
        "paper_allfine_peak_linear": PAPER_ALLFINE_PEAK_LINEAR,
        "rescaled_window_linear": rescaled_rod_window(ymean_peak),
        "frozen_window_looseness_db": float(
            20.0 * np.log10(PAPER_ALLFINE_PEAK_LINEAR / ymean_peak)),
        "measured_r6_mismatch_linear": _committed_r6_mismatch(root),
        "note": ("measured_r6_mismatch_linear is filled from the rods arm "
                 "(portgrid_m1b_retry_rods.json) -- the measurement is "
                 "unchanged; only the window it is compared against is "
                 "re-expressed."),
    }
    print(f"[xcheck rod window] frozen {WIN_ROD_LINEAR:.4f} -> rescaled "
          f"{rescaled_rod_window(ymean_peak):.4f} "
          f"(frozen is {20.0 * np.log10(PAPER_ALLFINE_PEAK_LINEAR / ymean_peak):.2f} dB looser "
          f"than the +5 dB rule intends)")
    return False   # recording arm: never fires a verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    choices=["all", "floor", "null", "interface", "rods",
                             "xcheck"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from validation.research.portgrid import sim2d

    out = {}
    fired = False
    if args.arm in ("all", "floor"):
        fired |= arm_floor(sim2d, out)
    if args.arm in ("all", "null"):
        fired |= arm_null(sim2d, out)
    if args.arm in ("all", "interface"):
        fired |= arm_interface(sim2d, out)
    if args.arm in ("all", "rods"):
        fired |= arm_rods(sim2d, out)
    if args.arm in ("all", "xcheck"):
        arm_xcheck(sim2d, out)      # recording only, cannot fire (R3)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
