#!/usr/bin/env python3
"""Isolated-patch ring-down under a padded lateral domain and a thin CPML (#801).

ONE ARM PER PROCESS.  ``build()`` and ``raster()`` below are copied VERBATIM from
``scripts/diagnostics/_artifacts/patch_close_20260830/harnesses/wt-refnull/refute_nulltf_ladder.py``
(the harness issue #801 names; untracked in the primary checkout).  Nothing about the
geometry, the drive, the probe quad or the absorber is changed -- if it were, the arm would
no longer be the arm the issue measured.  What is added is reporting only:

  * the FULL per-probe envelope trace, decimated, so the growth can be read as a curve and
    not as one headline number (workspace rule R5);
  * a growth-rate fit over the last 30 % of the run, the window #801 quotes;
  * the SOLVED permittivity rows inside the lateral CPML pads, which is what tells a
    continued dielectric from a vacuum facet;
  * provenance: the repo root this driver was read from, its git sha and dirty flag, the
    ``rfx`` package actually imported and ITS sha, so a run cannot silently measure a
    different tree than the one it claims.

The mode readout (harminv + parity classification) is deliberately NOT carried over: this
lane asks whether the run is stable, and a mode census of a diverging series is not a
number.  Read the arm's frequencies from the original harness if they are wanted.

Usage:
    PYTHONPATH=<tree> python3 scripts/diagnostics/patch_pad_cpml_ringdown.py \
        --n 4 --pad 10 --periods 150 --tag main_n4_pad10 --out-dir <dir> \
        --rfx-tree-sha <sha> --expect-rfx-root <path>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

C0 = 299792458.0
EPS_R = 3.38
H = 0.787e-3                 # substrate thickness; the unit of every length
L_H, W_H = 11, 13            # patch = 11h x 13h  (8.657 x 10.231 mm)
DOMX_H, DOMY_H, DOMZ_H = 38, 23, 16
ZGND_H = 5
XP0_H = 16                   # patch spans x in [16h, 27h]
BAR_DB = -40.0
F0, BW = 8.5e9, 1.6


def balanis(L, W, h, er):
    ee = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
    dL = 0.412 * h * (ee + 0.3) * (W / h + 0.264) / ((ee - 0.258) * (W / h + 0.8))
    return C0 / (2 * (L + 2 * dL) * math.sqrt(ee)), ee, dL


# --- VERBATIM from refute_nulltf_ladder.py (build) ------------------------------------
def build(n, shift_x=0.0, shift_y=0.0, pad_h=0, swap=False, cpml=None,
          gnd_cell="dielectric", pad_z_h=0, patch_plane="top", shrink_domain_ulp=0,
          sheet_conductors=False):
    """One arm's Simulation.  See the source harness's docstring for the registration,
    ground-cell and cavity reasoning; this copy changes nothing.

    Two knobs are added, both OFF by default, so the default call is the rig unchanged.
    Each exists for one pre-declared falsifier:

    ``shrink_domain_ulp`` nudges the declared lateral domain lengths down by that many
    ULPs.  ``(38 + 2*pad)*h`` can carry a one-ULP excess over an exact multiple of ``dx``,
    which makes rfx allocate one more cell than any declared Box fills; the extra node is
    vacuum and ``extend_cpml_pad_materials`` then replicates IT through that face's
    absorber.  See ``_artifacts/patch_pad_cpml_ringdown/pad_facet_rounding.py``.

    ``sheet_conductors`` declares the ground and the patch as zero-thickness Boxes instead
    of one-cell volumes, which reconstructs the board this issue's numbers were taken on
    (see the comment at the declaration).  Whether it succeeded is decided by the raster,
    not by the flag: the caller compares the realized wall planes against the recorded ones.
    """
    from rfx import Box, Simulation
    from rfx.sources import GaussianPulse
    dx = H / n
    cpml = 2 * n if cpml is None else cpml
    nud = -0.1 * dx
    sx, sy = shift_x * dx + nud, shift_y * dx + nud   # geometry shift vs lattice
    px, py = pad_h * H, pad_h * H                     # domain growth each side
    domx, domy, domz = DOMX_H * H + 2 * px, DOMY_H * H + 2 * py, (DOMZ_H + pad_z_h) * H
    for _ in range(int(shrink_domain_ulp)):
        domx = math.nextafter(domx, 0.0)
        domy = math.nextafter(domy, 0.0)
    L, W = L_H * H, W_H * H
    x0, y0 = XP0_H * H + px + sx, (DOMY_H / 2 - W_H / 2) * H + py + sy
    z_gnd = ZGND_H * H + nud
    z_sub_lo = z_gnd + dx
    if patch_plane == "inner":
        z_sub_hi = z_gnd + H
    else:
        z_sub_hi = z_sub_lo + H
    z_tr_hi = z_sub_hi + dx
    z_diel_lo = z_sub_lo if gnd_cell == "vacuum" else z_gnd

    def P(x, y):                      # axis permutation, applied to every point
        return (y, x) if swap else (x, y)

    dom = P(domx, domy)
    sim = Simulation(freq_max=15e9, domain=(dom[0], dom[1], domz), dx=dx,
                     cpml_layers=cpml, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    o = P(sx, sy)                     # same footprint as the committed fixture: [0, dom)
    if sheet_conductors:
        # BOARD RECONSTRUCTION, off by default.  Both PEC Boxes are one cell thick, and
        # #931's lattice-ownership contract realizes such a Box as a filled slab with
        # walls on BOTH faces.  Before #931 the same declaration realized as a SHEET --
        # one node plane with its normal E edge live -- which is the board this issue's
        # numbers were taken on.  Declaring them zero-thickness is what the preflight
        # itself now advises ("declare a SHEET: a zero-thickness Box via add()"), and it
        # puts the cavity back at 5 cells.  The raster assert below is the falsifier: the
        # reconstruction must reproduce the recorded walls, not merely resemble them.
        sim.add(Box((o[0], o[1], z_gnd), (dom[0] + o[0], dom[1] + o[1], z_gnd)), material="pec")
    else:
        sim.add(Box((o[0], o[1], z_gnd), (dom[0] + o[0], dom[1] + o[1], z_sub_lo)), material="pec")
    sim.add(Box((o[0], o[1], z_diel_lo), (dom[0] + o[0], dom[1] + o[1], z_sub_hi)), material="ro4003c")
    a, b = P(x0, y0), P(x0 + L, y0 + W)
    if sheet_conductors:
        sim.add(Box((a[0], a[1], z_sub_hi), (b[0], b[1], z_sub_hi)), material="pec")
    else:
        sim.add(Box((a[0], a[1], z_sub_hi), (b[0], b[1], z_tr_hi)), material="pec")
    s = P(x0 + 0.31 * L, y0 + W / 2 - 0.27 * W)
    zm = 0.5 * (z_gnd + z_sub_hi) if patch_plane == "inner" else 0.5 * (z_sub_lo + z_sub_hi)
    sim.add_source(position=(s[0], s[1], zm),
                   component="ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=BW))
    xc, yc = x0 + L / 2, y0 + W / 2
    quad = [(xc - 3.5 * H, yc - 3.5 * H), (xc + 3.5 * H, yc - 3.5 * H),
            (xc - 3.5 * H, yc + 3.5 * H), (xc + 3.5 * H, yc + 3.5 * H)]
    for (qx, qy) in quad:
        q = P(qx, qy)
        sim.add_probe(position=(q[0], q[1], zm), component="ez")
    geom = dict(n=n, dx=dx, cpml_layers=cpml, shift_x=shift_x, shift_y=shift_y,
                nudge_cells=-0.1, gnd_cell=gnd_cell, pad_z_h=pad_z_h, patch_plane=patch_plane,
                shrink_domain_ulp=int(shrink_domain_ulp),
                sheet_conductors=bool(sheet_conductors),
                domx_over_dx=domx / dx, domy_over_dx=domy / dx,
                pad_h=pad_h, swap=swap, L=L, W=W, h=H, domain=(dom[0], dom[1], domz),
                z_gnd=z_gnd, z_sub_lo=z_sub_lo, z_sub_hi=z_sub_hi, z_diel_lo=z_diel_lo,
                patch_box=(a, b), quad=[P(*q) for q in quad], source=s)
    return sim, geom


# --- VERBATIM from refute_nulltf_ladder.py (raster), plus a pad-permittivity dump -----
def raster(sim, geom):
    """Realized patch raster + cavity walls, read from the masks the solve uses."""
    # #931 replaced ``tangential_edge_masks`` with ``realized_pec_edge_masks``; the two
    # trees compared in this lane straddle that rename, so the REPORTING adapts.  The
    # rename is reported in the record (``edge_mask_api``) because #931 also changed the
    # sheet realization RULE -- a confounder for any cross-tree comparison, named here so
    # it is not mistaken for a CPML effect.
    try:
        from rfx.boundaries.pec import realized_pec_edge_masks as _edge_masks
        _edge_api = "realized_pec_edge_masks (post-#931)"
    except ImportError:  # pragma: no cover - only on pre-#931 trees
        from rfx.boundaries.pec import tangential_edge_masks as _edge_masks
        _edge_api = "tangential_edge_masks (pre-#931)"
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    grid = sim._build_grid()
    # #931 §1.3: a zero-thickness PEC Box is a SHEET, owns no cell, and is NOT in
    # pec_mask; _assemble_materials refuses to hand back materials for such a model
    # unless the caller passes collectors, precisely so a reader cannot mistake a
    # sheet-carrying board for a conductor-free one.  Collect them and realize the
    # conductor from the edge masks, which is where a sheet actually lives.
    pec_sheets, pec_wires = [], []
    eps = np.asarray(sim._assemble_materials(
        grid, pec_sheets=pec_sheets, pec_wires=pec_wires)[0].eps_r, dtype=float)
    cell_mask = np.asarray(sim.conductor_mask(grid), dtype=bool) if not pec_sheets else None
    mex, mey, mez = _edge_masks(cell_mask, sheets=pec_sheets, wires=pec_wires,
                                periodic=(False, False, False))
    mex, mey = np.asarray(mex), np.asarray(mey)
    # "conductor here" for REPORTING: a cell for a volume, a tangential edge for a sheet.
    cond = cell_mask if cell_mask is not None else (mex | mey)
    c = coords_from_uniform_grid(grid)
    z = np.asarray(c.z, dtype=float)
    dx = geom["dx"]
    (ax, ay), (bx, by) = geom["patch_box"]
    off = int(geom["cpml_layers"])
    ic = int(round(0.5 * (ax + bx) / dx)) + off
    jc = int(round(0.5 * (ay + by) / dx)) + off
    ks = np.flatnonzero(cond[ic, jc, :])
    k_patch = int(ks.max()); k_gnd = int(ks.min())
    ii = np.flatnonzero(cond[:, jc, k_patch]); jj = np.flatnonzero(cond[ic, :, k_patch])
    walls = np.flatnonzero(mex[ic, jc, :] | mey[ic, jc, :])
    lo = walls[walls <= k_gnd + 1].max(); hi = walls[walls >= k_patch - 1].min()
    out = dict(k_patch=k_patch, k_gnd=k_gnd, i_lo=int(ii.min()), i_hi=int(ii.max()),
               j_lo=int(jj.min()), j_hi=int(jj.max()),
               n_cells_x=int(ii.size), n_cells_y=int(jj.size),
               x_extent_real=float(ii.size * dx), y_extent_real=float(jj.size * dx),
               shape=[int(v) for v in cond.shape],
               walls_um=[float(z[k] * 1e6) for k in walls],
               cavity_lo_um=float(z[lo] * 1e6), cavity_hi_um=float(z[hi] * 1e6),
               cavity_um=float((z[hi] - z[lo]) * 1e6), cavity_cells=int(hi - lo),
               eps_in_cavity=[float(eps[ic, jc, lo:hi].min()), float(eps[ic, jc, lo:hi].max())],
               sum_d_over_eps_um=float(np.sum(dx / eps[ic, jc, lo:hi]) * 1e6),
               edge_mask_api=_edge_api)
    # #801-specific: what the SOLVED permittivity array holds in the lateral CPML pads.
    ncp = int(geom["cpml_layers"])
    # the substrate mid-plane node.  Grid indices carry the lo-side absorber offset on
    # EVERY axis, z included -- reading z/dx without it lands below the ground plane and
    # reports vacuum for a row that is not the substrate's.
    ksub = int(round(0.5 * (float(geom["z_sub_lo"]) + float(geom["z_sub_hi"])) / dx)) + ncp
    assert k_gnd < ksub < k_patch, (
        f"substrate mid node {ksub} not between ground {k_gnd} and patch {k_patch}")
    out["pad_eps"] = dict(
        cpml_layers=ncp, k_substrate_mid=ksub,
        xlo_row=[float(v) for v in eps[:ncp + 2, jc, ksub]],
        xhi_row=[float(v) for v in eps[-(ncp + 2):, jc, ksub]],
        ylo_row=[float(v) for v in eps[ic, :ncp + 2, ksub]],
        yhi_row=[float(v) for v in eps[ic, -(ncp + 2):, ksub]],
        z_column_in_xlo_pad=[float(v) for v in eps[1, jc, :]],
        conductor_xlo_row=[bool(v) for v in cond[:ncp + 2, jc, k_gnd]],
        conductor_xhi_row=[bool(v) for v in cond[-(ncp + 2):, jc, k_gnd]],
    )
    out["pec_realization"] = dict(
        n_pec_sheets=len(pec_sheets), n_pec_wires=len(pec_wires),
        has_cell_mask=cell_mask is not None,
        conductor_read_as=("primal cells (PEC volumes)" if cell_mask is not None
                           else "tangential E edges (PEC sheets, #931 1.3)"),
    )
    return out


def _provenance(expect_rfx_root, rfx_tree_sha):
    """Repo-root assert + dirty flag for BOTH the driver's tree and the imported rfx."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    assert os.path.isdir(os.path.join(root, "rfx")) and os.path.isfile(
        os.path.join(root, "pyproject.toml")), f"driver repo root not an rfx checkout: {root}"

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    import rfx
    rfx_root = os.path.abspath(os.path.join(os.path.dirname(rfx.__file__), os.pardir))
    if expect_rfx_root:
        assert os.path.abspath(expect_rfx_root) == rfx_root, (
            f"imported rfx from {rfx_root}, expected {expect_rfx_root} -- "
            "a stale editable install or a wrong PYTHONPATH would measure the wrong tree")
    porcelain = git("status", "--porcelain")
    try:
        import jax
        jax_info = dict(version=jax.__version__, backend=jax.default_backend(),
                        devices=[str(d) for d in jax.devices()],
                        x64_enabled=bool(jax.config.x64_enabled))
    except Exception as exc:  # pragma: no cover - reported, never swallowed silently
        jax_info = {"error": repr(exc)}
    return dict(
        jax=jax_info,
        driver_repo_root=root,
        driver_git_sha=git("rev-parse", "HEAD"),
        driver_dirty=bool(porcelain) if porcelain is not None else None,
        driver_dirty_paths=(porcelain or "").splitlines()[:40],
        rfx_package_file=rfx.__file__,
        rfx_root=rfx_root,
        rfx_version=getattr(rfx, "__version__", None),
        rfx_tree_sha=rfx_tree_sha,
        rfx_tree_is_driver_tree=(rfx_root == root),
        python=sys.version.split()[0],
        argv=sys.argv[1:],
    )


def envelope_report(ts, dt, decimate=64):
    """Per-probe running envelope + the growth #801 reads over the last 30 %.

    A diverging arm can reach inf/NaN.  ``settling_db`` is then reported as ``+inf`` for
    that probe rather than NaN: a NaN reaching a ``> -40`` comparison silently reads as
    "settled", which is the failure #885 closed.  ``nonfinite_first_step`` names the step.
    """
    raw = np.asarray(ts, dtype=float)
    finite_rows = np.isfinite(raw).all(axis=1)
    env = np.abs(np.where(np.isfinite(raw), raw, 0.0))
    nsteps = env.shape[0]
    peak = env.max(axis=0)
    tail = env[int(nsteps * 0.95):].max(axis=0)
    nonfinite_per_probe = (~np.isfinite(raw)).any(axis=0)

    def _db(t, pk, blew_up):
        if blew_up:
            return float("inf")
        return 20 * math.log10(max(float(t), 1e-300) / max(float(pk), 1e-300))

    settle = [_db(t, pk, bad) for t, pk, bad in zip(tail, peak, nonfinite_per_probe)]
    # last-30 % growth: peak envelope of the first vs the last half of that window
    i0 = int(nsteps * 0.70)
    w = env[i0:]
    half = w.shape[0] // 2
    g_first = w[:half].max(axis=0)
    g_last = w[half:].max(axis=0)
    ratio = [float(b) / float(a) if a > 0 else float("inf")
             for a, b in zip(g_first, g_last)]
    # exponential rate per step from a log-linear fit of the block maxima in that window
    nblk = 60
    blk = max(1, w.shape[0] // nblk)
    bm = np.array([w[i * blk:(i + 1) * blk].max(axis=0) for i in range(w.shape[0] // blk)])
    tt = (np.arange(bm.shape[0]) + 0.5) * blk
    rates = []
    for p in range(bm.shape[1]):
        y = bm[:, p]
        ok = y > 0
        rates.append(float(np.polyfit(tt[ok], np.log(y[ok]), 1)[0]) if ok.sum() > 2 else None)
    dec = env[::decimate]
    return dict(
        n_steps=int(nsteps), dt_s=float(dt),
        peak=[float(v) for v in peak], tail95=[float(v) for v in tail],
        nonfinite_per_probe=[bool(v) for v in nonfinite_per_probe],
        settling_db_per_probe=settle, settling_db=max(settle),
        settled=bool(max(settle) < BAR_DB),
        last30_growth_ratio=ratio, last30_growth_ratio_worst=max(ratio),
        last30_log_rate_per_step=rates,
        nonfinite_first_step=(int(np.argmax(~finite_rows)) if not bool(finite_rows.all())
                              else None),
        envelope_decimate=decimate,
        envelope_trace=[[float(v) for v in row] for row in dec],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--periods", type=float, default=150.0)
    p.add_argument("--pad", type=int, default=0, help="domain growth each side, in h")
    p.add_argument("--cpml", type=int, default=None)
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--swap-xy", action="store_true")
    p.add_argument("--gnd-cell", default="dielectric", choices=("dielectric", "vacuum"))
    p.add_argument("--pad-z", type=int, default=0)
    p.add_argument("--patch-plane", default="top", choices=("top", "inner"))
    p.add_argument("--sheet-conductors", action="store_true",
                   help="declare the ground and patch as zero-thickness Boxes (SHEETs) "
                        "instead of one-cell volumes; reconstructs the pre-#931 board")
    p.add_argument("--shrink-domain-ulp", type=int, default=0,
                   help="nudge the declared lateral domain lengths down by N ULPs "
                        "(falsifier for the one-extra-cell vacuum pad facet); 0 = the rig")
    p.add_argument("--subpixel", action="store_true",
                   help="run(subpixel_smoothing=True); OFF is the rig #801 measured")
    p.add_argument("--dry", action="store_true", help="build + raster + preflight, no solve")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rfx-tree-sha", default=None)
    p.add_argument("--expect-rfx-root", default=None)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    def persist(name, payload):
        tmp = os.path.join(a.out_dir, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, os.path.join(a.out_dir, name))

    prov = _provenance(a.expect_rfx_root, a.rfx_tree_sha)
    print(f"[{a.tag}] rfx from {prov['rfx_package_file']} (tree {prov['rfx_tree_sha']}); "
          f"driver {prov['driver_git_sha']} dirty={prov['driver_dirty']}; "
          f"jax {prov['jax']}", flush=True)

    sim, geom = build(a.n, a.shift_x, a.shift_y, a.pad, a.swap_xy, a.cpml, a.gnd_cell,
                      a.pad_z, a.patch_plane, a.shrink_domain_ulp,
                      a.sheet_conductors)
    ras = raster(sim, geom)
    rec = dict(tag=a.tag, provenance=prov, n=a.n, dx_um=geom["dx"] * 1e6, periods=a.periods,
               pad_h=a.pad, cpml_layers=geom["cpml_layers"], subpixel_smoothing=bool(a.subpixel),
               sheet_conductors=bool(a.sheet_conductors),
               L_mm=geom["L"] * 1e3, W_mm=geom["W"] * 1e3,
               raster=ras, geom={k: v for k, v in geom.items() if k != "quad"},
               status="built")
    exp_x, exp_y = (W_H, L_H) if a.swap_xy else (L_H, W_H)
    assert ras["n_cells_x"] == exp_x * a.n and ras["n_cells_y"] == exp_y * a.n, \
        f"raster {ras['n_cells_x']}x{ras['n_cells_y']} != {exp_x*a.n}x{exp_y*a.n}"
    print(f"[{a.tag}] grid {ras['shape']} patch {ras['n_cells_x']}x{ras['n_cells_y']} cells; "
          f"cpml {geom['cpml_layers']}; pad eps x-lo row {ras['pad_eps']['xlo_row']}", flush=True)
    persist(f"{a.tag}.json", rec)

    adv = [str(v) for v in sim.preflight()]
    rec["preflight"] = adv
    print(f"[{a.tag}] preflight ({len(adv)}), quoted verbatim:", flush=True)
    for v in adv:
        print(f"   ! {v}", flush=True)
    persist(f"{a.tag}.json", rec)
    if a.dry:
        rec["status"] = "dry"
        persist(f"{a.tag}.json", rec)
        print("[RESULT] " + json.dumps({"tag": a.tag, "status": "dry"}), flush=True)
        return 0

    t0 = time.time()
    kw = {"subpixel_smoothing": True} if a.subpixel else {}
    res = sim.run(num_periods=a.periods, **kw)
    ts = np.asarray(res.time_series)
    dt = float(res.dt)
    wall = time.time() - t0
    assert ts.ndim == 2 and ts.shape[1] == 4, f"unexpected time_series {ts.shape}"
    np.savez(os.path.join(a.out_dir, f"{a.tag}_ts.npz"), ts=ts, dt=dt)
    rep = envelope_report(ts, dt)
    rec.update(status="ran", wall_s=wall, **rep)
    persist(f"{a.tag}.json", rec)
    print(f"[{a.tag}] {rep['n_steps']} steps in {wall/60:.2f} min; settling per probe "
          f"{[round(s, 2) for s in rep['settling_db_per_probe']]} dB (bar {BAR_DB}) -> "
          f"{'SETTLED' if rep['settled'] else 'NOT SETTLED'}; last-30% growth "
          f"{[round(r, 3) for r in rep['last30_growth_ratio']]}", flush=True)
    print("[RESULT] " + json.dumps({k: v for k, v in rec.items()
                                    if k not in ("geom", "raster", "preflight", "provenance",
                                                 "envelope_trace")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
