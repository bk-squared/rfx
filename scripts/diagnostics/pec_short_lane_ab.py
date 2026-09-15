"""cv11 PEC short on the waveguide S-matrix lane: where did 0.0146 -> 0.0560 come from?

WHY THIS EXISTS. After #931 (lattice ownership) cv11's pec-short leg read
max||S11|-1| = 0.0560 against its 0.050 gate (VESSL 369367259194) where the
pre-change baseline read 0.0146 (369367259004). The aperture-trim A/B
(``cv11_aperture_trim_ab.py``, 369367259198) showed the trim is not the
cause and left "+0.057 belongs to the #931 core, on the waveguide S-matrix
lane" as the open attribution. This script adjudicates that under workspace
rule R5: per-bin |S11|(f) and the port time records are dumped for every
arm, and each hypothesis is run against an outcome written down BEFORE the
solve.

WHAT THE BUILD-ONLY PROBE FOUND (2026-09-07, no solve, both checkouts, the
cv11 fixture's own ``_build_sim``):

    grid (287, 24, 12) at dx = 1 mm: the domain is DECLARED 22.86 x 10.16 mm
    (WR-90) and REALIZED 23 x 11 mm (24 x 12 nodes, ceil), which cv11 has
    called QUOTE-REALIZED since #722 and uses for its cutoff.

    The short is drawn ``Box((x, 0, 0), (x + 2 mm, 22.86 mm, 10.16 mm))``
    -- to the DECLARED cross-section.

    baseline d990e18c  node mask, half-open [lo, hi): z nodes 0..10 of 12
                       -> 506 cells, sigma = 1e10 fill over the FULL 11 mm
                       height (node 10 at z = 10.000 mm < 10.16 is in).
    this branch        cell mask, centre-sampled (design note §1.1, nearest
                       plane): z cells 0..9 of 11 -> 460 cells. Cell 10's
                       centre is at 10.500 mm > 10.16 mm, so the plug's top
                       face realizes at z = 10.000 mm while the guide's top
                       wall is at 11.000 mm. Ez is LIVE at k = 10 in all 48
                       columns inside the plug (2 x-planes x 24 y-nodes).

    That is a 1 mm x 22.86 mm x 2 mm vacuum slot along the top broad wall,
    open at both ends: a parallel-plate section (no cutoff) in series with
    the TE10 field, which is Ez -- the component that lives in the slot.
    A capacitive slot of d/b = 1/11 in a 2 mm thick iris transmits a few
    percent of the incident power straight past the "short". In y the drawn
    22.86 mm rounds to the 23 mm wall (cell 22's centre 22.5 < 22.86), so
    there is no side slot; the defect is one face, one axis.

    Same mechanism, other fixtures: the broad-E5 live anchor on its AUTO
    mesh (dx = 2.1414 mm, DOMAIN (0.12, 0.04, 0.02)): z cell 9's centre is
    20.34 mm > 20 mm -> a 2.14 mm top slot (min |S11| 0.9572 measured in
    0b6f1239); at LIVE_DX = 2 mm the domain is on-lattice and the anchor
    passed. The validation battery's ``test_pec_short_s11_magnitude``
    (dx = None -> the same 2.1414 mm auto mesh) carries the same top slot
    (0.9670 "left RED on purpose"). The sheet-declared short in 0b6f1239
    (closed footprint [0, 0.04] x [0, 0.02] sampled at nodes on the auto
    mesh) misses the edge to the wall node on BOTH transverse axes -> an
    L-shaped zero-thickness slot, a resonator, which is what a 40-period
    record of ``[1.21, 0.71, 0.84, 0.96, 1.00, 1.48]`` looks like.

PRE-DECLARED OUTCOMES (written before any solve on this script):

    H0  cross-section under-coverage (the slot). Arm ``closed`` draws the
        plug to the REALIZED cross-section (A_WG_REALIZED, B_WG_REALIZED)
        with everything else identical. Expected: max||S11|-1| < 0.02 on
        this branch, i.e. back in the baseline's class and at least 2x
        below ``as_drawn``; the round-trip phase leg back under 15 deg;
        the right port's time record (behind the plug, left drive) drops by
        >= 20 dB relative to ``as_drawn``; single-run |S21| (the leak) falls
        from the 0.1-0.3 class to < 0.02. The baseline is INVARIANT to the
        redraw (node-half-open gives the same 11 nodes either way; checked
        build-only, not solved twice).
    H1  window / settling. ``as_drawn`` and ``closed`` at 100 and 200
        periods. Expected: max||S11|-1| moves by < 5e-3 between the two
        windows on both arms -- the deficit does NOT close with the window,
        and the reflecting plane is at 145.000 mm in every arm
        (``assert_realized_short`` before every solve).
    H2  reference-run contamination. cv11's pec-short leg is
        ``normalize=False`` (single run, no reference at all); the
        two-run lanes' vacuum reference receives no edge masks
        (``extract_waveguide_s_params_normalized`` passes none to the
        reference ``run``; the flux lane passes the per-port reference's
        OWN edges). cv11 passes no ``port_reference_sims``. Expected:
        nothing to contaminate -- reported as inspection, and the
        empty-guide build realizes no interior wall plane.
    H3  far-face closure. ``closed_t3`` moves the far face 147 -> 148 mm
        with the front face fixed. Expected: |S11| per bin within 5e-3 of
        ``closed`` and the round-trip phase leg unchanged; the region
        behind the plug ends in the x-hi CPML, not a wall, so there is no
        closed cavity to resonate.
    H4  the sheet path. ``sheet_closed`` is a zero-thickness Box at
        PEC_SHORT_X drawn to the realized cross-section (a §1.5 sheet
        declaration; the lane hands its footprint to
        ``realized_pec_edge_masks``). Expected: the lane realizes ONE wall
        plane at 145.000 mm with 23x12 + 24x11 tangential edges, and
        |S11| reads >= 0.99 on every bin -- the lane does apply sheets.
        ``sheet_as_drawn`` (footprint to the declared 22.86 x 10.16) is
        expected to be the non-passive nonsense class, with the realized
        edges showing the missing rim on both axes.
    Passivity: any |S11| > 1 must be explained by the record, not reported.

VERDICT (measured 2026-09-07 on this pod, JAX CPU, float32; JSON + port
records per arm in ``scripts/diagnostics/_artifacts/pec_short_lane_ab/``,
figure ``_scratch/pec_short_lane_ab/pec_short_lane_ab.png``):

    arm (cv11 fixture, 200 periods)  |S11| envelope      max dev  rt-phase  |S21|   right port
    baseline d990e18c, trim (shipped) [0.9854, 0.9940]   0.0146    9.99 deg  0.000  -366 dB
    baseline d990e18c, no trim        [0.9980, 1.0019]   0.0020    3.26 deg  0.000  -366 dB
    branch as_drawn (declared x-sec)  [0.9440, 0.9888]   0.0560   17.20 deg  0.22-0.33  -13.3 dB
    branch closed (realized x-sec)    [0.9980, 1.0019]   0.0020    3.26 deg  0 exactly  none
    branch closed_t3 (far face 148)   [0.9980, 1.0019]   0.0020    3.26 deg  0 exactly  none
    branch sheet_closed               [0.9980, 1.0019]   0.0020    3.26 deg  0 exactly  none
    branch sheet_as_drawn             [0.3553, 0.9996]   0.6447   67.89 deg  (resonant slot) -7.1 dB
    branch as_drawn, 100 periods      [0.9440, 0.9888]   0.0560   17.20 deg  (= 200 periods)
    branch closed, 100 periods        [0.9980, 1.0018]   0.0020    3.26 deg  (= 200 periods)

    H0 CONFIRMED: the slot is the whole step. Closing the plug's cross-section
       returns the leg to the pre-change baseline TO FOUR DECIMALS (the
       no-trim baseline and the closed branch arm print identical envelopes,
       max dev and phase legs); the single-run |S21| behind the plug and the
       right-port record go from a 5-11 % power leak to exactly zero.
    H1 REFUTED: 100 vs 200 periods changes nothing on either arm (< 1e-4).
    H2 MOOT by inspection: no reference run on this leg; the two-run lanes
       hand the vacuum reference no edge masks (per-port references carry
       their own). The empty-guide build realizes no interior wall.
    H3 REFUTED: far face 147 -> 148 mm changes nothing (< 1e-4).
    H4 REFUTED as a defect: the lane applies sheets. A sheet drawn to the
       realized walls reproduces the closed volume exactly; a sheet drawn to
       the declared cross-section misses the rim edge on both axes (230/264
       Ez, 242/276 Ey on the plane), is a resonant slot, and is what the
       0b6f1239 "non-passive" reading was (reproduced on the anchor's own
       auto mesh: [1.3405, 0.7135, 0.853, 0.9628, 1.0074, 1.5288]).
    Passivity: the closed arms' 3 bins at <= 1.0019 are the documented
       envelope of this fixture ("PEC short: passivity after the aperture
       trim" in cv11); the sheet_as_drawn arm's >1 readings on a 40-period
       anchor record are a resonator truncated by the window, not physics.

    Same mechanism on the broad-E5 live anchor's geometry (DOMAIN 0.12 x 0.04
    x 0.02, 40 periods): auto mesh 2.1414 mm, volume as drawn 180/200 Ez on
    the front plane, min |S11| 0.932; closed 200/200, [0.9907, 1.0184]; sheet
    closed identical to the closed volume. LIVE_DX = 2 mm (on-lattice): all
    four arms identical, [0.9938, 1.0006].

    Not a lane defect. The fixtures drew their shorts to the DECLARED domain
    extent; the contract realizes a volume's face at the nearest node and the
    domain at ceil. Fix: cv11, the validation battery and the live anchor now
    draw the plug to the grid's realized walls and assert a full-cross-section
    front wall at build time. Owed elsewhere: a preflight finding for "a
    conductor face drawn within a cell of a domain wall rounds AWAY from it"
    (the class this was), and the wr90_port/* diagnostics, which draw the plug
    to cv11's DOMAIN_Y x DOMAIN_Z and carry the same slot on this branch.

Usage (one solve per process; ~90 s for two drives at 200 periods on CPU):

    python scripts/diagnostics/pec_short_lane_ab.py --arm as_drawn --label branch \
        --num-periods 200 --out RUN_DIR
    python scripts/diagnostics/pec_short_lane_ab.py --checkout /path/to/baseline \
        --arm as_drawn --label baseline --num-periods 200 --out RUN_DIR
    python scripts/diagnostics/pec_short_lane_ab.py --build-only --arm closed
    python scripts/diagnostics/pec_short_lane_ab.py --plot RUN_DIR

``--checkout`` points the run at another rfx checkout (the pre-change
baseline); the script itself never edits it (set PYTHONDONTWRITEBYTECODE=1
to keep it byte-clean). The cv11 fixture is always the checkout's own
``validation/crossval/11_waveguide_port_wr90.py`` -- its constants, its
``_build_sim`` and ``_s_params`` -- so an arm cannot drift from the shipped
case; only the short's Box (and, for ``trim``, the port aperture) is
re-declared here, and that re-declaration is the one variable of the arm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ARMS = {
    # name: (thickness_m, cross_section, trim)   cross_section in {"declared", "realized"}
    "as_drawn": (0.002, "declared", False),
    "closed": (0.002, "realized", False),
    "closed_t3": (0.003, "realized", False),
    "sheet_closed": (0.0, "realized", False),
    "sheet_as_drawn": (0.0, "declared", False),
    "trim": (0.002, "declared", True),
}

GATE_MAG = 0.05          # cv11 pec-short |S11| gate
GATE_PHASE_DEG = 15.0    # cv11 pec-short round-trip phase gate
FLOOR_S11 = 0.99         # the live anchors' Meep-class floor


def _load_cv11(root: Path):
    path = root / "validation/crossval/11_waveguide_port_wr90.py"
    spec = importlib.util.spec_from_file_location("_cv11_pec_short_lane_ab", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def _build(cv11, arm: str):
    """cv11's simulation with the pec-short Box re-declared per arm.

    Ports are rebuilt explicitly in BOTH checkouts (the baseline's shipped
    ``_build_sim`` carries the aperture trim, this branch's does not), so
    every arm has the same port declaration except ``trim``.
    """
    import jax.numpy as jnp

    t_m, xsec, trim = ARMS[arm]
    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=None)
    x0 = cv11.PEC_SHORT_X
    if xsec == "declared":
        y_hi, z_hi = cv11.DOMAIN_Y, cv11.DOMAIN_Z          # 22.86 / 10.16 mm
    else:
        y_hi, z_hi = cv11.A_WG_REALIZED, cv11.B_WG_REALIZED  # 23 / 11 mm
    sim.add(cv11.Box((x0, 0.0, 0.0), (x0 + t_m, y_hi, z_hi)), material="pec")

    kw = {}
    if trim:
        kw = dict(y_range=(0.0, cv11.A_WG_REALIZED - cv11.DX_M),
                  z_range=(0.0, cv11.B_WG_REALIZED - cv11.DX_M))
    sim._waveguide_ports = []
    port_freqs = jnp.asarray(cv11.FREQS_HZ)
    sim.add_waveguide_port(
        cv11.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=cv11.F0_HZ, bandwidth=cv11.BANDWIDTH_REL,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        **kw)
    sim.add_waveguide_port(
        cv11.PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=cv11.F0_HZ, bandwidth=cv11.BANDWIDTH_REL,
        waveform="modulated_gaussian", reference_plane=0.150, name="right",
        **kw)
    return sim, dict(thickness_m=t_m, cross_section=xsec, trim=trim,
                     box_hi_m=[x0 + t_m, y_hi, z_hi])


def _build_only_report(cv11, sim, arm: str) -> dict:
    """What the lattice realizes for this arm, before any step.

    On this branch: the cell mask, the sheets, and the realized edge
    triple from the one owner (``realized_pec_edge_masks``). On the
    baseline: the node mask the sigma fold consumed.
    """
    grid = sim._build_grid()
    out = dict(grid_shape=[int(s) for s in grid.shape], dx_m=float(grid.dx))
    sheets: list = []
    wires: list = []
    try:
        assembled = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
        contract = True
    except TypeError:
        assembled = sim._assemble_materials(grid)
        contract = False
    pec = assembled[3]
    out["contract_branch"] = contract
    if pec is not None:
        pec = np.asarray(pec, dtype=bool)
        ii, jj, kk = np.nonzero(pec)
        out.update(
            mask_kind="cell" if contract else "node",
            mask_cells=int(pec.sum()),
            x_idx=sorted({int(v) for v in ii}),
            y_idx_range=[int(jj.min()), int(jj.max())],
            z_idx_range=[int(kk.min()), int(kk.max())],
            n_y_nodes=int(pec.shape[1]), n_z_nodes=int(pec.shape[2]),
        )
    else:
        out.update(mask_kind=None, mask_cells=0)
    out["n_sheets"] = len(sheets)
    if not contract:
        return out

    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    edges = realized_pec_edge_masks(pec, sheets=tuple(sheets), wires=tuple(wires),
                                    periodic=sim._periodic_flags())
    Mx, My, Mz = (np.asarray(m, dtype=bool) for m in edges)
    nodes = coords_from_uniform_grid(grid)
    planes_x = realized_wall_planes(edges, 0)
    out["wall_planes_x"] = [int(k) for k in planes_x]
    out["wall_planes_x_m"] = [float(np.asarray(nodes.x)[k]) for k in planes_x]
    if planes_x:
        i0, i1 = planes_x[0], planes_x[-1]
        sl = slice(i0, i1 + 1)
        # Tangential wall planes along z and y inside the short's x-span,
        # and the number of LIVE Ez / Ey edges on the top / side rows.
        out["wall_planes_z_in_short"] = realized_wall_planes(
            edges, 2, region=(sl, slice(None), slice(None)))
        out["wall_planes_y_in_short"] = realized_wall_planes(
            edges, 1, region=(sl, slice(None), slice(None)))
        nz, ny = Mz.shape[2], My.shape[1]
        out["ez_live_top_row"] = int((~Mz[sl, :, nz - 2]).sum())   # k = nz-2: last interior cell
        out["ey_live_side_row"] = int((~My[sl, ny - 2, :]).sum())  # j = ny-2: last interior cell
        out["ez_pec_edges_at_front_plane"] = int(Mz[i0].sum())
        out["ey_pec_edges_at_front_plane"] = int(My[i0].sum())
        out["ez_pec_edges_full_plane"] = int(ny * (nz - 1))
        out["ey_pec_edges_full_plane"] = int((ny - 1) * nz)
    return out


def _capture_port_records():
    """Monkeypatch ``rfx.simulation.run`` to keep each drive's final port cfgs."""
    import rfx.simulation as _rs
    captured: list = []
    orig = _rs.run

    def wrapped(*a, **k):
        res = orig(*a, **k)
        captured.append(res.waveguide_ports)
        return res

    _rs.run = wrapped
    return captured, lambda: setattr(_rs, "run", orig)


def _round_trip_phase(cv11, f_hz, s11):
    """cv11's own pec-short round-trip reference, verbatim from its main()."""
    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * cv11.F_CUTOFF_TE10 / cv11.C0
    beta = np.sqrt(np.maximum((omega / cv11.C0) ** 2 - kc ** 2, 0.0))
    d_pec = cv11.PEC_SHORT_X - 0.050
    ref = -np.exp(-1j * beta * 2.0 * d_pec)
    dphi = np.abs(np.angle(s11) - np.angle(ref))
    dphi = np.minimum(dphi, 2 * np.pi - dphi) * 180.0 / np.pi
    return dphi


def run_arm(root: Path, arm: str, label: str, num_periods: int, out_dir: Path | None,
            build_only: bool) -> dict:
    os.chdir(root)
    sys.path.insert(0, str(root))
    import rfx  # noqa: F401  (the checkout's own)
    cv11 = _load_cv11(root)
    sim, decl = _build(cv11, arm)
    rep = _build_only_report(cv11, sim, arm)
    head = _git_head(root)
    print(f"[{label}/{arm}] rfx from {Path(rfx.__file__).parent}  HEAD {head}")
    print(f"[{label}/{arm}] declared: {decl}")
    print(f"[{label}/{arm}] realized: {json.dumps(rep)}")
    result = dict(label=label, arm=arm, checkout=str(root), head=head,
                  declared=decl, realized=rep, num_periods=num_periods)
    if build_only:
        return result

    if rep.get("contract_branch") and decl["thickness_m"] > 0:
        # the same build-time gate cv11 runs before every solve (front face
        # at PEC_SHORT_X), with the arm's thickness in place of the fixture's
        exp_planes = [cv11.PEC_SHORT_X + n * cv11.DX_M
                      for n in range(int(round(decl["thickness_m"] / cv11.DX_M)) + 1)]
        got = rep["wall_planes_x_m"]
        assert len(got) == len(exp_planes) and all(
            abs(a - b) < 1e-9 for a, b in zip(got, exp_planes)), (got, exp_planes)

    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=num_periods))
    captured, restore = _capture_port_records()
    t0 = time.time()
    try:
        f_hz, s11, s21 = cv11._s_params(sim, num_periods=num_periods, normalize=False)
    finally:
        restore()
    dt_solve = time.time() - t0
    f_hz = np.asarray(f_hz); s11 = np.asarray(s11); s21 = np.asarray(s21)
    mag = np.abs(s11)
    dev = np.abs(mag - 1.0)
    dphi = _round_trip_phase(cv11, f_hz, s11)

    # Left-drive records: index 0 is the left drive (port order), cfgs in
    # port order. v_probe_t at the left port = incident + reflected; at the
    # right port (x = 160 mm, behind the plug) = whatever got through.
    rec = {}
    if captured:
        cfgs = captured[0]
        for name, cfg in zip(("left", "right"), cfgs):
            n_rec = int(np.asarray(cfg.n_steps_recorded))
            rec[name] = dict(
                v_probe_t=np.asarray(cfg.v_probe_t, dtype=np.float64)[:n_rec],
                i_probe_t=np.asarray(cfg.i_probe_t, dtype=np.float64)[:n_rec],
                dt=float(cfg.dt), n_recorded=n_rec)
    peak_l = float(np.abs(rec["left"]["v_probe_t"]).max()) if rec else float("nan")
    peak_r = float(np.abs(rec["right"]["v_probe_t"]).max()) if rec else float("nan")
    leak_db = 20.0 * np.log10(max(peak_r, 1e-300) / max(peak_l, 1e-300)) if rec else float("nan")

    result.update(
        n_steps=n_steps, dt_s=float(grid.dt), solve_s=dt_solve,
        freqs_hz=f_hz.tolist(),
        s11_re=s11.real.tolist(), s11_im=s11.imag.tolist(),
        s21_re=s21.real.tolist(), s21_im=s21.imag.tolist(),
        s11_mag=mag.tolist(), s21_mag=np.abs(s21).tolist(),
        max_dev=float(dev.max()), mean_dev=float(dev.mean()),
        min_mag=float(mag.min()), max_mag=float(mag.max()),
        n_over_unity=int((mag > 1.0).sum()),
        phase_rt_max_deg=float(dphi.max()), phase_rt_mean_deg=float(dphi.mean()),
        gates=dict(mag=bool(dev.max() < GATE_MAG), phase=bool(dphi.max() < GATE_PHASE_DEG),
                   floor=bool(mag.min() >= FLOOR_S11)),
        right_port_peak_over_left_db=float(leak_db),
    )
    print(f"[{label}/{arm}] {n_steps} steps, {dt_solve:.1f} s")
    print(f"[{label}/{arm}] |S11| per bin: {np.array2string(mag, precision=4)}")
    print(f"[{label}/{arm}] max||S11|-1| = {dev.max():.4f} (gate {GATE_MAG}); "
          f"min {mag.min():.4f} max {mag.max():.4f}; over-unity bins {int((mag > 1).sum())}")
    print(f"[{label}/{arm}] round-trip phase max {dphi.max():.2f} deg / mean "
          f"{dphi.mean():.2f} deg (gate {GATE_PHASE_DEG})")
    print(f"[{label}/{arm}] single-run |S21| per bin: {np.array2string(np.abs(s21), precision=4)}")
    print(f"[{label}/{arm}] right-port peak / left-port peak = {leak_db:+.1f} dB (left drive)")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{label}_{arm}_np{num_periods}"
        (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
        if rec:
            np.savez(out_dir / f"{stem}_records.npz",
                     **{f"{p}_{k}": v for p, d in rec.items() for k, v in d.items()
                        if isinstance(v, np.ndarray)},
                     dt=rec["left"]["dt"])
        print(f"[{label}/{arm}] wrote {out_dir / stem}.json")
    return result


def plot(run_dir: Path, out_png: Path | None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(run_dir.glob("*_np*.json"))
    res = [json.loads(p.read_text()) for p in runs]
    res = [r for r in res if "s11_mag" in r]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for r in res:
        f = np.asarray(r["freqs_hz"]) / 1e9
        ax.plot(f, r["s11_mag"], marker="o", ms=3,
                label=f"{r['label']}/{r['arm']} np{r['num_periods']} "
                      f"(max dev {r['max_dev']:.4f})")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axhspan(1 - GATE_MAG, 1 + GATE_MAG, color="0.9", zorder=0)
    ax.set_xlabel("f (GHz)"); ax.set_ylabel("|S11|  (left port, single run)")
    ax.legend(fontsize=6)
    ax.set_ylim(0.6, 1.25)

    ax = axes[1]
    for r in res:
        npz = run_dir / f"{r['label']}_{r['arm']}_np{r['num_periods']}_records.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        v = d["right_v_probe_t"]; t = np.arange(v.size) * float(d["dt"]) * 1e9
        vl = d["left_v_probe_t"]
        env = 20 * np.log10(np.maximum(np.abs(v), 1e-30) / np.abs(vl).max())
        ax.plot(t, env, lw=0.7, label=f"{r['label']}/{r['arm']} np{r['num_periods']}")
    ax.set_xlabel("t (ns)"); ax.set_ylabel("right-port |v(t)| re left-port peak (dB)")
    ax.set_ylim(-160, 5); ax.legend(fontsize=6)
    ax.set_title("left drive: what gets past the short", fontsize=9)

    ax = axes[2]
    for r in res:
        if r["num_periods"] != max(x["num_periods"] for x in res):
            continue
        npz = run_dir / f"{r['label']}_{r['arm']}_np{r['num_periods']}_records.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        v = d["left_v_probe_t"]; t = np.arange(v.size) * float(d["dt"]) * 1e9
        ax.plot(t, v / np.abs(v).max(), lw=0.6, label=f"{r['label']}/{r['arm']}")
    ax.set_xlim(0, 4.0)
    ax.set_xlabel("t (ns)"); ax.set_ylabel("left-port v(t) / peak")
    ax.set_title("incident pulse then the reflected one", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    out_png = out_png or (run_dir / "pec_short_lane_ab.png")
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkout", default=None,
                    help="rfx checkout to run against (default: this script's repo)")
    ap.add_argument("--arm", choices=sorted(ARMS), default="as_drawn")
    ap.add_argument("--label", default="branch")
    ap.add_argument("--num-periods", type=int, default=200)
    ap.add_argument("--out", default=None, help="directory for JSON + record dumps")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--plot", default=None, help="aggregate the JSONs in this dir into a PNG")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    if args.plot:
        plot(Path(args.plot), Path(args.png) if args.png else None)
        return 0
    root = Path(args.checkout).resolve() if args.checkout else Path(__file__).resolve().parents[2]
    run_arm(root, args.arm, args.label, args.num_periods,
            Path(args.out) if args.out else None, args.build_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
