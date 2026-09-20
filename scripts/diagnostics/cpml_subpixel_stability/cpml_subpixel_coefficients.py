"""#1043 Stage A — why CPML + subpixel smoothing diverges when a dielectric
reaches the absorber pad.

Pre-declaration: ``docs/design_notes/issue1043_cpml_subpixel_coefficient_predeclaration.md``
(hypotheses H1-H4 and gates G-A..G-D, frozen before this ran).

WHAT THIS MEASURES
------------------
The E half-step in a CPML cell is assembled from two halves that take their
permittivity from two different arrays:

    rfx/core/yee.py:827   update_e_aniso   cb = dt / (aniso_eps * EPS_0)
    rfx/boundaries/cpml.py:621-632  ce = dt / (materials.eps_r * EPS_0)

``kappa_max`` defaults to 1.0, so ``(1/kappa - 1) == 0`` and the whole CPML-E
correction is ``ce * psi`` with ``psi = b*psi_prev + c*curl`` and ``c < 0``.
The instantaneous curl coefficient of one pad cell is therefore proportional to

    K_eff_norm = 1/eps_a + c/eps_b        (eps_a = Yee half, eps_b = psi half)

which is non-negative for every ``c in (-1, 0]`` when ``eps_a == eps_b`` and
changes sign as soon as ``|c| * eps_a / eps_b > 1``.

Four arms on one small 2-D TMz rig (a dielectric guide spanning the full x
extent, so it reaches both x pads — the cv01 Run 1 topology at 1/4 the size):

  ========================  ==========================================
  arm                       what differs
  ========================  ==========================================
  cpml_baseline             subpixel ON, CPML, no pad replication (today)
  cpml_padrep               subpixel ON, CPML, pad replication on aniso_eps
  cpml_subpixel_off         subpixel OFF, CPML (materials pad ext, default)
  upml_padrep               subpixel ON, UPML, pad replication on aniso_eps
  ========================  ==========================================

For each arm: the static ``K_eff_norm`` map over every CPML pad cell (G-A),
the FDTD finite/non-finite verdict and first non-finite probe index (G-D),
and for the diverging arm the cells carrying the growth (G-B).
G-C (the amplification eigenvalue) lives in ``amplification.py``.

NOTHING under ``rfx/`` is monkeypatched except ``compute_smoothed_eps``, and
only in the two ``*_padrep`` arms, which stand in for the Stage-B assembly fix
that is not this branch's business.

Usage (2-D CPU, ~1 min)::

    PYTHONPATH=$(git rev-parse --show-toplevel) python3 \
        scripts/diagnostics/cpml_subpixel_stability/cpml_subpixel_coefficients.py \
        --output scripts/diagnostics/_artifacts/cpml_subpixel_stability/coefficients.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import subprocess
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]

C0 = 2.998e8

# --- rig: cv01 Run 1's topology, quarter size -------------------------------
a = 1.0e-6
eps_wg = 12.0
w_wg = 1.0 * a
dx = a / 10
cpml_n = 10
sx = 8.0 * a
sy = 8.0 * a
wg_y = sy / 2
fcen = 0.15 * C0 / a
fwidth = 0.1 * C0 / a
src_x = cpml_n * dx + dx
N_STEPS = 1200
# Far enough past the exponential onset to localise the seed, early enough
# that the diverging arm is still finite everywhere.
N_STEPS_LOCALISE = 300


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _assert_rfx_is_this_repo(allow_foreign: bool = False) -> dict:
    """Fail the run unless ``import rfx`` resolved inside THIS checkout.

    Running a diagnostic by path puts the script's directory on ``sys.path``,
    not the repo root, so ``import rfx`` can silently come from another tree
    while the artifact records this tree's sha.

    ``allow_foreign`` is for the deliberate before/after pair: the reference
    arm runs against a ``git archive`` of the pre-fix commit unpacked to a
    scratch directory, which is the point of pinning a verifier to an archive
    rather than to a checkout somebody can edit underneath it.
    """
    import rfx
    f = pathlib.Path(rfx.__file__).resolve()
    inside = True
    try:
        f.relative_to(REPO)
    except ValueError:
        inside = False
        if not allow_foreign:
            raise SystemExit(
                f"rfx provenance check FAILED: {f} is not under {REPO}")
    return {"rfx_file": str(f), "rfx_under_repo_root": inside,
            "repo_root": str(REPO),
            "driver_commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def install_pad_replication_patch():
    """Replicate the smoothed array into the pads (the Stage-B assembly fix).

    Same shape as #831's Arm F probe: wrap
    ``rfx.geometry.smoothing.compute_smoothed_eps``, which
    ``rfx/runners/uniform.py`` imports at call time, and extend each returned
    component with the one shared implementation (#627).
    """
    import jax.numpy as jnp
    import rfx.geometry.smoothing as _sm
    from rfx.geometry.rasterize_grid import extend_cpml_pad_materials

    orig = _sm.compute_smoothed_eps

    def patched(grid, shape_eps_pairs, background_eps=1.0, **kw):
        comps = orig(grid, shape_eps_pairs, background_eps=background_eps, **kw)
        pads = (int(grid.pad_x_lo), int(grid.pad_x_hi),
                int(grid.pad_y_lo), int(grid.pad_y_hi),
                int(grid.pad_z_lo), int(grid.pad_z_hi))
        out = []
        for c in comps:
            c = jnp.asarray(c)
            e2, _s, _m = extend_cpml_pad_materials(
                c, jnp.zeros_like(c), jnp.ones_like(c), *pads)
            out.append(e2)
        return tuple(out)

    _sm.compute_smoothed_eps = patched
    return lambda: setattr(_sm, "compute_smoothed_eps", orig)


def build_sim(boundary: str):
    import rfx
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources import GaussianPulse

    sim = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                     boundary=BoundarySpec.uniform(boundary),
                     cpml_layers=cpml_n, mode="2d_tmz")
    sim.add_material("wg", eps_r=eps_wg)
    sim.add(rfx.Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),
            material="wg")
    for i in range(10):
        y = wg_y - w_wg / 2 + (i + 0.5) * w_wg / 10
        sim.add_source(position=(src_x, y, 0), component="ez",
                       waveform=GaussianPulse(f0=fcen,
                                              bandwidth=fwidth / fcen,
                                              amplitude=1.0 / 10))
    sim.add_probe(position=(sx - cpml_n * dx - 5 * dx, wg_y, 0),
                  component="ez")
    return sim


# --- G-A: the static coefficient map ----------------------------------------

def pad_mask(grid) -> np.ndarray:
    """Boolean (nx, ny, nz) — True where ``apply_cpml_e`` writes.

    Mirrors ``apply_cpml_e``'s own face slices (cpml.py:625-630).
    """
    from rfx.boundaries.cpml import _axis_buffer_depths
    nx, ny, nz = grid.shape
    n_x, n_y, n_z = _axis_buffer_depths(grid, grid.cpml_layers)
    m = np.zeros((nx, ny, nz), dtype=bool)
    m[:n_x, :, :] = True
    m[-n_x:, :, :] = True
    m[:, :n_y, :] = True
    m[:, -n_y:, :] = True
    if nz > 1:
        m[:, :, :n_z] = True
        m[:, :, -n_z:] = True
    return m


def c_map(grid) -> np.ndarray:
    """Per-cell ``c`` coefficient, taking the WORST (most negative) face.

    ``apply_cpml_e`` adds one psi term per (component, face) pair; a cell in
    two pads receives two. The most negative ``c`` bounds the worst single
    contribution, which is what the H1 sign condition is written against.
    """
    from rfx.boundaries.cpml import init_cpml, _axis_buffer_depths, _clip_lo, _clip_hi
    params, _state = init_cpml(grid)
    nx, ny, nz = grid.shape
    n = grid.cpml_layers
    n_x, n_y, n_z = _axis_buffer_depths(grid, n)
    out = np.zeros((nx, ny, nz))

    extent = (nx, ny, nz)

    def _acc(face_c, axis, lo):
        arr = np.asarray(face_c, dtype=float)
        depth = arr.shape[0]
        shape = [1, 1, 1]
        shape[axis] = depth
        arr3 = arr.reshape(shape)
        sl = [slice(None)] * 3
        sl[axis] = slice(0, depth) if lo else slice(extent[axis] - depth, None)
        key = tuple(sl)
        out[key] = np.minimum(out[key], np.broadcast_to(arr3, out[key].shape))

    _acc(_clip_lo(params.x_lo.c, n_x, n), 0, True)
    _acc(_clip_hi(params.x_hi.c, n_x, n), 0, False)
    _acc(_clip_lo(params.y_lo.c, n_y, n), 1, True)
    _acc(_clip_hi(params.y_hi.c, n_y, n), 1, False)
    if nz > 1:
        _acc(_clip_lo(params.z_lo.c, n_z, n), 2, True)
        _acc(_clip_hi(params.z_hi.c, n_z, n), 2, False)
    return out


def coefficient_map(boundary: str, subpixel, padrep: bool) -> dict:
    """``K_eff_norm = 1/eps_a + c/eps_b`` over every CPML pad cell."""
    import rfx.geometry.smoothing as _sm

    undo = install_pad_replication_patch() if padrep else None
    try:
        sim = build_sim(boundary)
        grid = sim._build_grid()
        mats, *_ = sim._assemble_materials(grid)
        eps_b = np.asarray(mats.eps_r, dtype=float)
        if subpixel:
            pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                     for e in sim._geometry]
            comps = _sm.compute_smoothed_eps(grid, pairs, background_eps=1.0)
            eps_a = {k: np.asarray(v, dtype=float)
                     for k, v in zip(("ex", "ey", "ez"), comps)}
        else:
            eps_a = {k: eps_b for k in ("ex", "ey", "ez")}
        cm = c_map(grid)
        pm = pad_mask(grid)
        per = {}
        worst = np.inf
        worst_cells = []
        for comp, ea in eps_a.items():
            k = 1.0 / ea + cm / eps_b
            kp = k[pm]
            mn = float(kp.min())
            per[comp] = {"min": mn, "n_negative": int((kp < 0).sum()),
                         "n_pad_cells": int(pm.sum())}
            if mn < worst:
                worst = mn
            idx = np.argwhere(pm & (k < 0))
            worst_cells.extend([[comp] + [int(v) for v in row] for row in idx])
        return {
            "boundary": boundary, "subpixel": bool(subpixel), "padrep": padrep,
            # ``apply_cpml_e`` runs only on the CPML branch
            # (simulation.py:1467-1470). On UPML the whole absorber is folded
            # into ``apply_upml_e``, which builds its coefficients from
            # ``aniso_eps`` itself (upml.py:213-217) — one array, no second
            # epsilon, so K_eff_norm is not defined for that arm and the map
            # below is reported for reference only.
            "cpml_e_path_taken": boundary == "cpml",
            "grid_shape": [int(v) for v in grid.shape],
            "pad": [int(grid.pad_x_lo), int(grid.pad_x_hi),
                    int(grid.pad_y_lo), int(grid.pad_y_hi)],
            "c_min": float(cm.min()),
            "per_component": per,
            "K_eff_norm_min": worst,
            "n_negative_total": len(worst_cells),
            "negative_cells_sample": worst_cells[:40],
            "negative_cells": worst_cells,
        }
    finally:
        if undo is not None:
            undo()


# --- G-D / G-B: the FDTD witness --------------------------------------------

def run_arm(name: str, boundary: str, subpixel, padrep: bool,
            n_steps: int = N_STEPS, want_state: bool = False,
            negative_cells: list | None = None) -> dict:
    undo = install_pad_replication_patch() if padrep else None
    t0 = time.time()
    try:
        sim = build_sim(boundary)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = sim.run(n_steps=n_steps, subpixel_smoothing=subpixel)
        ts = np.asarray(res.time_series, dtype=float)
        finite = np.isfinite(ts)
        first_bad = int(np.argmin(finite.all(axis=1))) if not finite.all() else None
        out = {
            "arm": name, "boundary": boundary, "subpixel": bool(subpixel),
            "padrep": padrep, "n_steps": n_steps,
            "n_nonfinite": int((~finite).sum()),
            "first_nonfinite_index": first_bad,
            "max_abs_finite": float(np.abs(ts[finite]).max()) if finite.any() else None,
            "settling_db": (None if res.settling_db is None
                            else float(res.settling_db)),
            "stdout_tail": buf.getvalue()[-1500:],
            "wall_s": round(time.time() - t0, 1),
        }
        if want_state:
            ez = np.asarray(res.state.ez, dtype=float)
            ae = np.abs(ez)
            flat = np.argsort(ae, axis=None)[::-1][:12]
            loc = {
                "max_abs_ez": float(np.nanmax(ae)),
                "argmax_cell": [int(v) for v in
                                np.unravel_index(int(np.nanargmax(ae)), ae.shape)],
                "top_cells": [[int(v) for v in np.unravel_index(i, ae.shape)]
                              + [float(ae.flat[i])] for i in flat],
            }
            if negative_cells is not None:
                neg = {tuple(r[1:]) for r in negative_cells if r[0] == "ez"}
                # The K_eff_norm < 0 SEED set, and the one-cell neighbourhood
                # it can have reached by this step (the update is a
                # nearest-neighbour stencil, so growth spreads outward one
                # cell per step; scoring the seed alone would call a
                # neighbour's spill a miss).
                seed_ring = {(i + di, j + dj, k) for (i, j, k) in neg
                             for di in (-1, 0, 1) for dj in (-1, 0, 1)}
                thr = loc["max_abs_ez"] * 1e-3
                hot = [tuple(int(v) for v in np.unravel_index(i, ae.shape))
                       for i in np.argwhere(ae.ravel() >= thr).ravel()]
                loc["seed_membership"] = {
                    "argmax_in_seed": tuple(loc["argmax_cell"]) in neg,
                    "n_hot_cells": len(hot),
                    "frac_hot_in_seed": (sum(c in neg for c in hot) / len(hot)
                                         if hot else None),
                    "frac_hot_in_seed_ring": (
                        sum(c in seed_ring for c in hot) / len(hot)
                        if hot else None),
                    "hot_threshold_rel": 1e-3,
                }
            out["localise"] = loc
        return out
    finally:
        if undo is not None:
            undo()


def per_cell_dump(boundary, subpixel, padrep, cells) -> list:
    """eps_a, eps_b, c and K_eff_norm at named cells — the R5 intermediate.

    A headline "min K_eff_norm < 0" is not evidence on its own; this shows the
    two permittivities and the psi coefficient that produced it, per cell.
    """
    import rfx.geometry.smoothing as _sm

    undo = install_pad_replication_patch() if padrep else None
    try:
        sim = build_sim(boundary)
        grid = sim._build_grid()
        mats, *_ = sim._assemble_materials(grid)
        eps_b = np.asarray(mats.eps_r, dtype=float)
        if subpixel:
            pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                     for e in sim._geometry]
            comps = dict(zip(("ex", "ey", "ez"),
                             _sm.compute_smoothed_eps(grid, pairs,
                                                      background_eps=1.0)))
            eps_a = {k: np.asarray(v, dtype=float) for k, v in comps.items()}
        else:
            eps_a = {k: eps_b for k in ("ex", "ey", "ez")}
        cm = c_map(grid)
        rows = []
        for comp, i, j, k in cells:
            ea = float(eps_a[comp][i, j, k])
            eb = float(eps_b[i, j, k])
            cc = float(cm[i, j, k])
            rows.append({"component": comp, "cell": [i, j, k],
                         "eps_a_update": ea, "eps_b_psi": eb, "c": cc,
                         "K_eff_norm": 1.0 / ea + cc / eb,
                         "K_eff_norm_if_consistent": (1.0 + cc) / ea})
        return rows
    finally:
        if undo is not None:
            undo()


def _state_nonfinite_cells(boundary, subpixel, padrep, n_steps) -> list:
    """Cells of the FINAL field state that are non-finite after ``n_steps``."""
    undo = install_pad_replication_patch() if padrep else None
    try:
        sim = build_sim(boundary)
        with contextlib.redirect_stdout(io.StringIO()):
            res = sim.run(n_steps=n_steps, subpixel_smoothing=subpixel)
        out = []
        for comp in ("ex", "ey", "ez"):
            arr = np.asarray(getattr(res.state, comp))
            for idx in np.argwhere(~np.isfinite(arr)):
                out.append((comp, *(int(v) for v in idx)))
        return out
    finally:
        if undo is not None:
            undo()


def first_nonfinite_state(boundary, subpixel, padrep, hi: int) -> dict:
    """Bisect for the first step at which ANY field cell is non-finite.

    The probe time series only sees one point, so it reports divergence later
    than the field does; this finds the step the field itself first breaks and
    returns the cells that broke at it — which is what G-B is written against.
    """
    lo = 1
    if not _state_nonfinite_cells(boundary, subpixel, padrep, hi):
        return {"step": None, "cells": [], "reason": f"finite at {hi} steps"}
    while lo < hi:
        mid = (lo + hi) // 2
        if _state_nonfinite_cells(boundary, subpixel, padrep, mid):
            hi = mid
        else:
            lo = mid + 1
    return {"step": lo, "cells": _state_nonfinite_cells(
        boundary, subpixel, padrep, lo)}


ARMS = (
    # name,               boundary, subpixel, padrep
    ("cpml_baseline",     "cpml",   True,     False),
    ("cpml_padrep",       "cpml",   True,     True),
    ("cpml_subpixel_off", "cpml",   False,    False),
    ("upml_padrep",       "upml",   True,     True),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--skip-fdtd", action="store_true")
    ap.add_argument("--allow-foreign-rfx", action="store_true")
    ap.add_argument("--label", default="head")
    ap.add_argument("--expect", choices=("prefix", "fixed"), default="fixed",
                    help="which finite/non-finite table G-D is scored against")
    args = ap.parse_args()

    prov = _assert_rfx_is_this_repo(args.allow_foreign_rfx)
    rec = {"label": args.label, "provenance": prov,
           # ``K_eff_norm`` below is computed from ``materials.eps_r``, i.e.
           # the permittivity the PRE-FIX ``apply_cpml_e`` builds its psi
           # coefficient from. It is a property of the two arrays, not of the
           # running solver, so it reads the same on both trees; the FDTD
           # verdicts are what differ. ``K_eff_norm_if_consistent`` in the
           # per-cell dump is the post-fix value.
           "rig": {
        "a": a, "eps_wg": eps_wg, "w_wg": w_wg, "dx": dx,
        "cpml_layers": cpml_n, "sx": sx, "sy": sy, "n_steps": N_STEPS,
        "mode": "2d_tmz"}, "arms": {}}

    for name, boundary, subpixel, padrep in ARMS:
        entry = {"coefficients": coefficient_map(boundary, subpixel, padrep)}
        if not args.skip_fdtd:
            entry["fdtd"] = run_arm(name, boundary, subpixel, padrep,
                                    want_state=False)
        rec["arms"][name] = entry
        print(f"[{name}] K_eff_norm_min="
              f"{entry['coefficients']['K_eff_norm_min']:.4g} "
              f"n_neg={entry['coefficients']['n_negative_total']} "
              + ("" if args.skip_fdtd else
                 f"nonfinite={entry['fdtd']['n_nonfinite']} "
                 f"first={entry['fdtd']['first_nonfinite_index']}"))

    # --- verdicts, against the frozen gates ---
    # K_eff_norm is defined only where ``apply_cpml_e`` runs; the UPML arm
    # never reaches it (see ``cpml_e_path_taken``) and is excluded rather
    # than compared against a coefficient its solver never uses.
    cm = {k: v["coefficients"]["K_eff_norm_min"] for k, v in rec["arms"].items()
          if v["coefficients"]["cpml_e_path_taken"]}
    div = cm["cpml_padrep"]
    stable_worst = min(cm["cpml_baseline"], cm["cpml_subpixel_off"])
    rec["gates"] = {
        "G_A": {
            "scored_arms": sorted(cm),
            "excluded_no_cpml_e_path": sorted(
                k for k, v in rec["arms"].items()
                if not v["coefficients"]["cpml_e_path_taken"]),
            "diverging_min": div, "stable_worst_min": stable_worst,
            "r_A": max(0.0, div) + max(0.0, -stable_worst),
            "verdict": "H1 SUPPORTED" if (div < 0 <= stable_worst)
                       else "H1 NON-CLOSING",
        },
    }
    if not args.skip_fdtd:
        fv = {k: v["fdtd"]["n_nonfinite"] > 0 for k, v in rec["arms"].items()}
        # Pre-fix: the §0 table — only the pad-replicated CPML arm diverges.
        # Post-fix: nothing diverges, because the psi coefficient now reads
        # the same permittivity the Yee half did.
        expect = {"cpml_baseline": False,
                  "cpml_padrep": args.expect == "prefix",
                  "cpml_subpixel_off": False, "upml_padrep": False}
        rec["gates"]["G_D"] = {
            "expectation_set": args.expect,
            "diverged": fv, "expected": expect,
            "r_D": sum(1 for k in expect if fv[k] != expect[k]),
        }

        # --- G-B: which cells go non-finite FIRST ---
        neg = rec["arms"]["cpml_padrep"]["coefficients"]["negative_cells"]
        hi = rec["arms"]["cpml_padrep"]["fdtd"]["first_nonfinite_index"]
        if hi is None:
            # No divergence on this tree — G-B has nothing to localise, which
            # is itself the post-fix result rather than a missing measurement.
            rec["gates"]["G_B"] = {
                "first_nonfinite_state_step": None,
                "note": "arm cpml_padrep stayed finite; nothing to localise",
                "r_B": None,
            }
            out = pathlib.Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2))
            print(json.dumps(rec["gates"], indent=2))
            print(f"wrote {out}")
            return
        rec["first_nonfinite_state"] = fnf = first_nonfinite_state(
            "cpml", True, True, int(hi) + 2)
        neg_set = {(r[0], r[1], r[2], r[3]) for r in neg}
        neg_pos = {(r[1], r[2], r[3]) for r in neg}
        cells = [tuple(c) for c in fnf["cells"]]
        pos = sorted({c[1:] for c in cells})
        frac = (sum(c in neg_set for c in cells) / len(cells)) if cells else None
        frac_pos = (sum(p in neg_pos for p in pos) / len(pos)) if pos else None
        rec["gates"]["G_B"] = {
            "first_nonfinite_state_step": fnf["step"],
            "n_first_nonfinite_cells": len(cells),
            # As pre-declared: component-resolved. A cell whose Ez has
            # K_eff_norm < 0 drags its own Ex non-finite in the SAME step (one
            # Yee cell, shared Hy/Hz), so the component-resolved fraction
            # undercounts by construction; the position-resolved figure below
            # is reported alongside and neither is hidden.
            "frac_in_K_negative_set": frac,
            "r_B": (None if frac is None else 1.0 - frac),
            "n_first_nonfinite_positions": len(pos),
            "frac_positions_in_K_negative_set": frac_pos,
            "r_B_position_resolved": (None if frac_pos is None
                                      else 1.0 - frac_pos),
            "cells": cells[:40],
            "per_cell_dump": per_cell_dump("cpml", True, True, cells[:40]),
        }

        # Growth-envelope localisation, well before divergence.
        for n_loc in (200, N_STEPS_LOCALISE):
            rec[f"localise_cpml_padrep_{n_loc}"] = run_arm(
                "cpml_padrep", "cpml", True, True,
                n_steps=n_loc, want_state=True, negative_cells=neg)
        rec["localise_cpml_baseline"] = run_arm(
            "cpml_baseline", "cpml", True, False,
            n_steps=N_STEPS_LOCALISE, want_state=True, negative_cells=neg)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec["gates"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
