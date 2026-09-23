#!/usr/bin/env python
"""The Sheen low-pass filter in rfx with the substrate-top permittivity averaged.

Two 2.413 mm wide 50 ohm microstrip feeds on 0.794 mm of lossless RT/Duroid
(eps_r 2.2) are joined by one wide 20.320 x 2.540 mm low-impedance section, and
the section's transverse resonance puts a double transmission zero near 7 and
8 GHz into |S21|; the zeros are set by the fringing field at the section's
edges, which lives on the substrate-air interface.  Two FDTD codes that solve
the same discrete board with the same update should put those zeros at the
same frequencies.  What was seen: handed rfx's own lattice node for node,
openEMS reads the 8 GHz zero 2.22 / 1.64 / 1.35 % below rfx at h/3, h/5, h/7
(VESSL 369367263706), and the two codes give the tangential E on the
substrate-top node plane different permittivities -- rfx 1.0 there (one node
value, the substrate sampled half-open), openEMS 1.6 (its quarter-cell
average of two substrate and two air quarters).

ASSUMPTION this probe measures, stated as an assumption and not a finding:
that this one plane's permittivity is the whole code difference on one
lattice.  The rest of the two setups is not identical (rfx's CPML against
openEMS's UPML in the same pad cells, openEMS's strips continued through the
absorber, the two port models).

WHAT IT VARIES
--------------
The case's own board and run (``build``, ``assert_realized``, ``run_rung``:
``N_FREQS``, ``NUM_PERIODS``, the ring-down and passivity witnesses) at
dx = h_sub/n for n = 3, 5, 7, in these arms:

    baseline   the case's run as it is (h/3 only by default: the control
               must reproduce the ladder's 8.38113 GHz and 5.78586 GHz)
    null       the hook installed with the permittivity UNCHANGED -- the
               per-component E update fed three copies of the isotropic
               array (h/3 only by default).  It isolates the switch of
               update kernel from the permittivity change.
    averaged   Ex and Ey on the node plane k = n_sub given
               (eps[k-1] + eps[k]) / 2 = (2.2 + 1.0) / 2 = 1.6, every other
               component and every other plane unchanged.

Per arm and rung: both zeros (the |S21| minimum in 6.5-7.6 GHz and in
7.6-9.5 GHz, the case's ``stopband_null`` = ``refined_extremum`` in log), the
-3 dB corner, the passband mean, the ring-down settling, the passivity
excess, the wall time; a table against openEMS on rfx's lattice and against
rfx's ladder; one figure.  A diagnostic: nothing under rfx/ changes, no test,
no record.

THE HOOK, AND WHAT WAS READ TO CHOOSE IT (file:line on this branch)
--------------------------------------------------------------------
* ``run_uniform`` calls ``_simulation.run(grid, materials, n_steps, ...,
  aniso_eps=aniso_eps, ...)`` with ``_simulation`` the ``rfx.simulation``
  module (rfx/runners/uniform.py:8, the call at :782) and ``aniso_eps``
  None unless subpixel smoothing is asked for (:234, :272).  The hook
  replaces the MODULE ATTRIBUTE ``rfx.simulation.run`` for the duration of
  one arm with a wrapper that fills ``aniso_eps`` from the ``materials`` it
  is handed and calls the original.  ``subpixel_smoothing`` stays False.
* With ``aniso_eps`` set, the step calls ``update_e_aniso`` instead of
  ``update_e`` (rfx/simulation.py:452-457); both take ``sigma`` from
  ``materials`` and build the same Ca/Cb algebra per component
  (rfx/core/yee.py:346-360 vs :930-945).  The CPML's E psi coefficient then
  takes 1/eps per component from the same arrays (rfx/simulation.py:1599-1600,
  rfx/boundaries/cpml.py:668-684) instead of ``dt / (eps_r EPS_0)`` (:688).
* The GPU fused path is not in play: it needs ``not use_pec_edges``
  (rfx/simulation.py:2398) and this board has PEC sheets.
* PEC: after every E update the step multiplies E by (1 - mask) on the
  realized sheet edges (``apply_pec_edges``, rfx/simulation.py:1797,
  rfx/boundaries/pec.py:572-579), so the permittivity written on a PEC edge
  never reaches a field value.  The hook therefore writes 1.6 on the whole
  plane and the proof counts the metal edges among the changed ones.
* The absorber pad: by default (``--pad-plane keep``) the plane is changed
  over rfx's ``grid.interior`` only and the pad keeps rfx's own value;
  ``--pad-plane average`` changes the pad's part of the plane too.
* Proof, per run() call: the three arrays at k-1, k, k+1 at one (i, j) under
  the wide section and one beside it; the per-element count of Ca/Cb that
  differ from the isotropic update's, by component and by plane; the
  kernel the step actually traced (``update_e`` or ``update_e_aniso``,
  recorded by wrapping those two module attributes) and whether it received
  exactly the hook's arrays.

CLI: ``--rungs 3,5,7``, ``--arms baseline,null,averaged``, ``--pad-plane``,
``--out DIR``, ``--openems-json PATH``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ARMS = ("baseline", "null", "averaged")
DEFAULT_ARMS = {3: ("baseline", "null", "averaged"), 5: ("averaged",),
                7: ("averaged",)}
ZERO_BANDS_HZ = {"zero_7": (6.5e9, 7.6e9), "zero_8": (7.6e9, 9.5e9)}

# openEMS handed rfx's lattice node for node: VESSL 369367263706, commit
# 696267ba, artifacts on the lab share
# research/rfx/.omx/sheen-identical-grid-openems/20260923T033616Z-696267ba-h3-5-7/
# (sheen_identical_grid_openems.json, table_vs_rfx).  Every pass ended on its
# 1e-4 energy criterion (-44.49 / -40.76 / -40.06 dB).
OPENEMS_IDENTICAL = {
    3: {"zero_7": 7.108692651559819, "zero_8": 8.19484297681204,
        "corner": 5.661374122008719},
    5: {"zero_7": 7.113230966109308, "zero_8": 8.224029744719155,
        "corner": 5.658917177921425},
    7: {"zero_7": 7.049793413436163, "zero_8": 8.185513400154125,
        "corner": 5.650152987027912},
}
OPENEMS_JSON_DEFAULT = (
    Path.home() / "mnt" / "remilab-fs" / "personal-workspaces"
    / "byungkwan-workspace" / "research" / "rfx" / ".omx"
    / "sheen-identical-grid-openems" / "20260923T033616Z-696267ba-h3-5-7"
    / "sheen_identical_grid_openems.json")
# rfx's ladder (VESSL 369367263494, research/rfx/.omx/sheen-lpf-ladder/
# 20260922T160810Z-e2ba011e/ladder.log): the 8 GHz zero is the deepest 5-10 GHz
# minimum it printed; the 7 GHz zero is not in that log as a number.
RFX_LADDER = {3: {"zero_8": 8.38113, "corner": 5.78586},
              5: {"zero_8": 8.36112, "corner": 5.74375},
              7: {"zero_8": 8.29750, "corner": 5.71532}}
CONTROL_TOL = 1.0e-4                     # 0.01 %, the brief's control


# ------------------------------------------------------------------ modules
def repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def load_case():
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "tests" / "crossval" / "sheen_lpf" / "test_sheen_lpf.py"
    spec = importlib.util.spec_from_file_location("_sheen_interface_eps_case", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sheen_interface_eps_case"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- hook
class InterfaceHook:
    """Fill ``aniso_eps`` on the way into ``rfx.simulation.run``.

    ``mode``: "baseline" installs only the recorders (the case's run as it
    is), "null" hands the per-component update three copies of the isotropic
    array, "averaged" changes Ex and Ey on plane ``k``.
    """

    def __init__(self, mode: str, k: int, probe_ij: dict, *, pad_plane: str = "keep"):
        if mode not in ARMS:
            raise ValueError(mode)
        self.mode, self.k, self.probe_ij, self.pad_plane = mode, int(k), probe_ij, pad_plane
        self.calls: list = []
        self.kernel: list = []

    def __enter__(self):
        import rfx.simulation as S
        self._S = S
        self._orig = {"run": S.run, "update_e": S.update_e,
                      "update_e_aniso": S.update_e_aniso}
        hook = self

        def run(grid, materials, n_steps, *args, **kw):
            rec = {"mode": hook.mode, "n_steps": int(n_steps)}
            if kw.get("aniso_eps") is not None or kw.get("aniso_inv_eps") is not None:
                raise RuntimeError("the runner already carries anisotropic "
                                   "permittivity; this hook would overwrite it")
            if hook.mode != "baseline":
                aniso = hook._arrays(grid, materials)
                kw["aniso_eps"] = aniso
                rec["proof"] = hook._proof(grid, materials, aniso,
                                           kw.get("pec_edge_masks"))
                hook._last_aniso = aniso
            else:
                hook._last_aniso = None
            hook.calls.append(rec)
            return hook._orig["run"](grid, materials, n_steps, *args, **kw)

        def update_e(state, materials, *a, **k):
            hook.kernel.append({"kernel": "update_e",
                                "eps_is_materials": True})
            return hook._orig["update_e"](state, materials, *a, **k)

        def update_e_aniso(state, materials, eps_ex, eps_ey, eps_ez, *a, **k):
            same = None
            try:
                last = hook._last_aniso
                same = (last is not None and all(
                    np.array_equal(np.asarray(x), np.asarray(y))
                    for x, y in zip((eps_ex, eps_ey, eps_ez), last)))
            except Exception as exc:          # a tracer cannot be read here
                same = f"not readable at trace time: {type(exc).__name__}"
            hook.kernel.append({"kernel": "update_e_aniso",
                                "received_the_hook_arrays": same})
            return hook._orig["update_e_aniso"](state, materials, eps_ex, eps_ey,
                                                eps_ez, *a, **k)

        S.run = run
        S.update_e = update_e
        S.update_e_aniso = update_e_aniso
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(self._S, name, fn)
        return False

    def _plane_region(self, grid):
        if self.pad_plane == "average":
            return slice(None), slice(None)
        ix, iy, _iz = grid.interior
        return ix, iy

    def _arrays(self, grid, materials):
        import jax.numpy as jnp
        eps = materials.eps_r
        ex = jnp.array(eps)
        ey = jnp.array(eps)
        ez = jnp.array(eps)
        if self.mode == "averaged":
            k = self.k
            avg = 0.5 * (eps[:, :, k - 1] + eps[:, :, k])
            sx, sy = self._plane_region(grid)
            plane = eps[:, :, k].at[sx, sy].set(avg[sx, sy])
            ex = ex.at[:, :, k].set(plane)
            ey = ey.at[:, :, k].set(plane)
        return ex, ey, ez

    def _proof(self, grid, materials, aniso, pec_edge_masks) -> dict:
        from rfx.core.yee import EPS_0, e_update_coeffs
        k = self.k
        dt = float(grid.dt)
        eps = np.asarray(materials.eps_r)
        sig = np.asarray(materials.sigma)
        ca0, cb0 = (np.asarray(v) for v in e_update_coeffs(materials.eps_r,
                                                          materials.sigma, dt))
        out = {"plane_k": k, "dt_s": dt,
               "plane_region": "grid.interior" if self.pad_plane == "keep" else "whole plane"}
        masks = [None, None, None]
        if pec_edge_masks is not None:
            masks = [np.asarray(m, dtype=bool) for m in pec_edge_masks]
        dump = {}
        for name, (i, j) in self.probe_ij.items():
            dump[name] = {
                "ij": [int(i), int(j)],
                "isotropic_eps_r_k-1,k,k+1": [float(eps[i, j, kk]) for kk in (k - 1, k, k + 1)],
                **{f"eps_{c}_k-1,k,k+1": [float(np.asarray(a)[i, j, kk])
                                          for kk in (k - 1, k, k + 1)]
                   for c, a in zip(("ex", "ey", "ez"), aniso)},
                "pec_edge_Mx,My,Mz_on_k": ([bool(m[i, j, k]) for m in masks]
                                           if masks[0] is not None else None),
            }
        out["dump"] = dump
        counts = {}
        nx, ny, nz = eps.shape
        ix, iy, _ = grid.interior
        interior_plane = (len(range(nx)[ix]) * len(range(ny)[iy]))
        for c, a, m in zip(("ex", "ey", "ez"), aniso, masks):
            ca, cb = (np.asarray(v) for v in e_update_coeffs(a, materials.sigma, dt))
            dcb = cb != cb0
            dca = ca != ca0
            planes = sorted({int(v) for v in np.nonzero(dcb)[2]})
            counts[c] = {
                "cb_differ": int(dcb.sum()), "cb_differ_on_plane_k": int(dcb[:, :, k].sum()),
                "cb_differ_off_plane_k": int(dcb.sum() - dcb[:, :, k].sum()),
                "planes_with_a_difference": planes,
                "ca_differ": int(dca.sum()),
                "cb_differ_on_pec_edges": (int((dcb & m).sum()) if m is not None else None),
                "cb_ratio_on_plane_k_min_max": (
                    [float((cb / cb0)[:, :, k][dcb[:, :, k]].min()),
                     float((cb / cb0)[:, :, k][dcb[:, :, k]].max())]
                    if dcb[:, :, k].any() else None),
            }
        # The CPML psi coefficient each path builds: dt / (eps EPS_0) without
        # the hook (rfx/boundaries/cpml.py:688), dt * (1/eps) / EPS_0 with it
        # (:678-684).  Compared in float32, as the runner builds them.
        import jax.numpy as jnp
        ce0 = np.asarray(dt / (materials.eps_r * EPS_0))
        ce_diff = {}
        for c, a in zip(("ex", "ey", "ez"), aniso):
            inv = (1.0 / a).astype(materials.eps_r.dtype)
            ce = np.asarray(dt * jnp.asarray(inv) / EPS_0)
            d = ce != ce0
            rel = np.abs(ce - ce0) / np.abs(ce0)
            ce_diff[c] = {"differ": int(d.sum()),
                          "differ_on_plane_k": int(d[:, :, k].sum()),
                          "max_rel_off_plane_k": float(np.delete(rel, k, axis=2).max())}
        out["counts"] = counts
        out["expected_cb_differ_per_tangential_component"] = (
            interior_plane if self.mode == "averaged" and self.pad_plane == "keep"
            else (nx * ny if self.mode == "averaged" else 0))
        out["cpml_psi_coefficient_differ"] = ce_diff
        out["sigma_nonzero_on_plane_k"] = int((sig[:, :, k] != 0).sum())
        return out


# -------------------------------------------------------------------- run
def probe_points(g) -> dict:
    """(i, j) under the wide section's centre, and three rows beside it."""
    ci = (g["patch"]["cols"][0] + g["patch"]["cols"][1]) // 2
    cj = (g["patch"]["rows"][0] + g["patch"]["rows"][1]) // 2
    return {"under_the_section": (ci, cj),
            "beside_the_section": (ci, g["patch"]["rows"][0] - 3)}


def run_arm(m, n: int, arm: str, *, pad_plane: str) -> dict:
    dx = m.H_SUB / n
    sim = m.build(dx)
    g = m.realized_geometry(sim)
    k = int(g["sheet_plane_k"])
    pts = probe_points(g)
    print(f"\n{'=' * 78}\n=== h/{n} (dx = {dx * 1e6:.4f} um), arm {arm}, sheet plane k = {k}"
          f"{'' if arm != 'averaged' else ', pad plane ' + pad_plane}\n{'=' * 78}",
          flush=True)
    t0 = time.perf_counter()
    with InterfaceHook(arm, k, pts, pad_plane=pad_plane) as hook:
        r = m.run_rung(dx)
    wall = time.perf_counter() - t0
    f = np.asarray(r["freqs_hz"], dtype=float)
    s21 = np.abs(np.asarray(r["s21"]))
    zeros = {}
    for key, (lo, hi) in ZERO_BANDS_HZ.items():
        z = m.stopband_null(f, s21, lo=lo, hi=hi)
        band = np.flatnonzero((f >= lo) & (f <= hi))
        zeros[key] = {"f_ghz": z["f"] / 1e9, "bin_f_ghz": z["bin_f"] / 1e9,
                      "depth_db": z["depth_db"],
                      "at_window_edge": bool(z["index"] in (int(band[0]), int(band[-1])))}
    deep = m.stopband_null(f, s21)
    cut = m.passband_cutoff_3db(f, s21)
    out = {
        "n": n, "dx_m": dx, "arm": arm, "pad_plane": pad_plane if arm == "averaged" else None,
        "sheet_plane_k": k, "grid_shape": list(g["grid_shape"]), "n_cells": int(g["n_cells"]),
        "zeros": zeros, "deepest_5_10_ghz": deep["f"] / 1e9,
        "corner_ghz": None if cut["f_3db"] is None else cut["f_3db"] / 1e9,
        "passband_mean_db": cut["mean_db"],
        "settling_db": (None if r["settling_db"] is None
                        else np.asarray(r["settling_db"], dtype=float).tolist()),
        "max_excess_in_band": r["max_excess_in_band"],
        "solve_wall_s": float(r["wall_s"]), "arm_wall_s": wall,
        "hook_calls": hook.calls, "kernel_traced": hook.kernel,
        "freqs_ghz": (f / 1e9).tolist(),
        "s21_db": (20 * np.log10(np.maximum(s21, 1e-300))).tolist(),
    }
    print_arm(out)
    return out


def print_arm(o: dict) -> None:
    z7, z8 = o["zeros"]["zero_7"], o["zeros"]["zero_8"]
    print(f"  zero7 {z7['f_ghz']:.5f} GHz ({z7['depth_db']:.2f} dB, bin {z7['bin_f_ghz']:.5f}"
          f"{', AT THE WINDOW EDGE' if z7['at_window_edge'] else ''}); zero8 "
          f"{z8['f_ghz']:.5f} GHz ({z8['depth_db']:.2f} dB, bin {z8['bin_f_ghz']:.5f}"
          f"{', AT THE WINDOW EDGE' if z8['at_window_edge'] else ''}); deepest 5-10 GHz "
          f"{o['deepest_5_10_ghz']:.5f}")
    print(f"  corner {o['corner_ghz']} GHz; passband {o['passband_mean_db']:.3f} dB; settling "
          f"{o['settling_db']}; excess {o['max_excess_in_band']:.5f}; solve "
          f"{o['solve_wall_s']:.1f} s, arm {o['arm_wall_s']:.1f} s")
    kinds = sorted({(c['kernel'], str(c.get('received_the_hook_arrays'))) for c in o["kernel_traced"]})
    print(f"  run() calls through the hook: {len(o['hook_calls'])}; E kernels traced: {kinds}")
    for i, c in enumerate(o["hook_calls"]):
        p = c.get("proof")
        if not p:
            continue
        print(f"  proof, run() call {i + 1} (n_steps {c['n_steps']}), plane k = {p['plane_k']}, "
              f"region {p['plane_region']}, dt {p['dt_s']:.6e} s:")
        for name, d in p["dump"].items():
            print(f"    {name:20s} (i, j) = {tuple(d['ij'])}: eps_r {d['isotropic_eps_r_k-1,k,k+1']} "
                  f"| eps_ex {d['eps_ex_k-1,k,k+1']} | eps_ey {d['eps_ey_k-1,k,k+1']} "
                  f"| eps_ez {d['eps_ez_k-1,k,k+1']} | PEC Mx,My,Mz on k {d['pec_edge_Mx,My,Mz_on_k']}")
        for comp, cnt in p["counts"].items():
            print(f"    Cb_{comp}: {cnt['cb_differ']} elements differ from the isotropic update's "
                  f"({cnt['cb_differ_on_plane_k']} on k, {cnt['cb_differ_off_plane_k']} off k; planes "
                  f"{cnt['planes_with_a_difference']}; {cnt['cb_differ_on_pec_edges']} of them PEC "
                  f"edges; Cb ratio on k {cnt['cb_ratio_on_plane_k_min_max']}); Ca differ "
                  f"{cnt['ca_differ']}")
        print(f"    expected Cb differences per tangential component: "
              f"{p['expected_cb_differ_per_tangential_component']}; sigma nonzero on k: "
              f"{p['sigma_nonzero_on_plane_k']}")
        for comp, cd in p["cpml_psi_coefficient_differ"].items():
            print(f"    CPML psi coefficient {comp}: {cd['differ']} differ ({cd['differ_on_plane_k']} "
                  f"on k); max relative difference off k {cd['max_rel_off_plane_k']:.3e}")
        break


# ------------------------------------------------------------------ report
def write_figure(results, openems, directory: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  figure NOT written: {exc!r}")
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sheen_interface_eps.png"
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colours = {3: "tab:blue", 5: "tab:orange", 7: "tab:green"}
    for o in results:
        if o["arm"] == "averaged":
            ax.plot(o["freqs_ghz"], o["s21_db"], "-", lw=1.4, color=colours.get(o["n"]),
                    label=f"rfx h/{o['n']}, interface Ex/Ey eps 1.6")
        elif o["arm"] == "baseline":
            ax.plot(o["freqs_ghz"], o["s21_db"], ":", lw=1.4, color=colours.get(o["n"]),
                    label=f"rfx h/{o['n']}, as shipped (interface eps 1.0)")
    if openems is not None:
        for n, r in sorted(openems.items()):
            ax.plot(r["freqs_ghz"], r["s21_db"], "--", lw=1.0, color=colours.get(n),
                    label=f"openEMS on rfx's h/{n} lattice")
    ax.set_xlim(2, 12)
    ax.set_ylim(-70, 5)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def load_openems_curves(path: Path):
    if not path.is_file():
        print(f"  openEMS-identical curves NOT plotted: {path} is not reachable")
        return None
    with path.open() as fh:
        d = json.load(fh)
    out = {}
    for n_str, r in d.get("rungs", {}).items():
        rec = r["record"]
        s = np.asarray(rec["s21_mag"], dtype=float)
        out[int(n_str)] = {"freqs_ghz": rec["freqs_ghz"],
                           "s21_db": (20 * np.log10(np.maximum(s, 1e-300))).tolist()}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", default="3,5,7")
    p.add_argument("--arms", default=None,
                   help="comma separated subset of baseline,null,averaged applied at "
                        "every rung given; default: all three at h/3, averaged at h/5, h/7")
    p.add_argument("--pad-plane", choices=("keep", "average"), default="keep")
    p.add_argument("--out", default=None)
    p.add_argument("--openems-json", default=str(OPENEMS_JSON_DEFAULT))
    args = p.parse_args(argv)

    rungs = [int(v) for v in args.rungs.split(",") if v.strip()]
    if any(n not in RFX_LADDER for n in rungs):
        print(f"ERROR: rungs must be from {sorted(RFX_LADDER)}", file=sys.stderr)
        return 3
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        if any(a not in ARMS for a in arms):
            print(f"ERROR: arms must be from {ARMS}", file=sys.stderr)
            return 3
        plan = [(n, a) for n in rungs for a in arms]
    else:
        plan = [(n, a) for n in rungs for a in DEFAULT_ARMS[n]]
    out_dir = Path(args.out) if args.out else Path("sheen_interface_eps")
    out_dir.mkdir(parents=True, exist_ok=True)

    m = load_case()
    print("=" * 78)
    print("The Sheen low-pass filter -- rfx with the substrate-top Ex/Ey permittivity averaged")
    print("=" * 78)
    print(f"  plan {plan}; pad plane {args.pad_plane}; N_FREQS {m.N_FREQS}; "
          f"NUM_PERIODS {m.NUM_PERIODS}")

    results, failures = [], {}
    for n, arm in plan:
        try:
            results.append(run_arm(m, n, arm, pad_plane=args.pad_plane))
        except AssertionError as exc:
            print(f"  h/{n} {arm}: the case refused the run: {exc}")
            failures[f"h{n}_{arm}"] = str(exc)

    by = {(o["n"], o["arm"]): o for o in results}
    control = None
    if (3, "baseline") in by:
        b = by[(3, "baseline")]
        d8 = abs(b["zeros"]["zero_8"]["f_ghz"] - RFX_LADDER[3]["zero_8"]) / RFX_LADDER[3]["zero_8"]
        dc = abs(b["corner_ghz"] - RFX_LADDER[3]["corner"]) / RFX_LADDER[3]["corner"]
        control = {"zero_8_rel": d8, "corner_rel": dc,
                   "pass": bool(d8 <= CONTROL_TOL and dc <= CONTROL_TOL)}
        print(f"\nCONTROL: baseline h/3 zero8 {b['zeros']['zero_8']['f_ghz']:.5f} vs ladder "
              f"{RFX_LADDER[3]['zero_8']:.5f} GHz ({100 * d8:.4f} %); corner "
              f"{b['corner_ghz']:.5f} vs {RFX_LADDER[3]['corner']:.5f} GHz ({100 * dc:.4f} %) "
              f"-> {'PASS' if control['pass'] else 'FAIL'} (tolerance {100 * CONTROL_TOL:.2f} %)")

    print("\n" + "=" * 78)
    print("TABLE -- rfx arms against openEMS on rfx's lattice and against rfx's ladder")
    print("=" * 78)
    print("  rung arm        zero7     zero8     corner    | oE zero7  oE zero8  oE corner |"
          " d7 vs oE  d8 vs oE  dc vs oE | d8 vs ladder  dc vs ladder | settling (dB)     "
          "excess   solve (s)")
    table = []
    for o in results:
        n = o["n"]
        oe = OPENEMS_IDENTICAL[n]
        lad = RFX_LADDER[n]
        z7, z8, c = o["zeros"]["zero_7"]["f_ghz"], o["zeros"]["zero_8"]["f_ghz"], o["corner_ghz"]
        row = {"n": n, "arm": o["arm"], "zero_7": z7, "zero_8": z8, "corner": c,
               "openems_zero_7": oe["zero_7"], "openems_zero_8": oe["zero_8"],
               "openems_corner": oe["corner"],
               "d7_vs_openems_pct": 100 * (z7 - oe["zero_7"]) / oe["zero_7"],
               "d8_vs_openems_pct": 100 * (z8 - oe["zero_8"]) / oe["zero_8"],
               "dcorner_vs_openems_pct": 100 * (c - oe["corner"]) / oe["corner"],
               "d8_vs_ladder_pct": 100 * (z8 - lad["zero_8"]) / lad["zero_8"],
               "dcorner_vs_ladder_pct": 100 * (c - lad["corner"]) / lad["corner"],
               "settling_db": o["settling_db"], "excess": o["max_excess_in_band"],
               "solve_wall_s": o["solve_wall_s"], "arm_wall_s": o["arm_wall_s"]}
        table.append(row)
        print(f"  h/{n:<3d}{o['arm']:9s} {z7:9.5f} {z8:9.5f} {c:9.5f} | {oe['zero_7']:8.5f} "
              f"{oe['zero_8']:9.5f} {oe['corner']:9.5f} | {row['d7_vs_openems_pct']:+8.3f} "
              f"{row['d8_vs_openems_pct']:+9.3f} {row['dcorner_vs_openems_pct']:+9.3f} | "
              f"{row['d8_vs_ladder_pct']:+12.4f} {row['dcorner_vs_ladder_pct']:+13.4f} | "
              f"{'/'.join(f'{v:.2f}' for v in o['settling_db']):16s} "
              f"{o['max_excess_in_band']:.5f} {o['solve_wall_s']:9.1f}")

    fig = write_figure(results, load_openems_curves(Path(args.openems_json)), out_dir)
    if fig is not None:
        print(f"\n  figure: {fig}")
    payload = {"what": "rfx with the substrate-top Ex/Ey permittivity averaged -- a diagnostic",
               "pad_plane": args.pad_plane, "control": control, "table": table,
               "openems_identical": OPENEMS_IDENTICAL, "rfx_ladder": RFX_LADDER,
               "failures": failures, "arms": results,
               "figure": None if fig is None else str(fig)}
    path = out_dir / "sheen_interface_eps.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"  json:   {path}")
    if control is not None and not control["pass"]:
        return 1
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
