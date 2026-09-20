"""#831 / #1043 Arm F — is the assembly fix alone landable?

Pre-declaration: ``docs/design_notes/issue831_far_end_return_predeclaration.md``
section 11, frozen before this ran.

WHAT THIS PROBES.  Section 10 of that note classified the divergence seen when
cv01's guide Box is widened past the pad: not the Kottke half-cell, not the
record length, not x64, and not "a dielectric in a CPML pad" as such -- the
subpixel-OFF row puts eps 12 in the CPML pad through
``extend_cpml_pad_materials`` and is stable at 0.9887.  What is left is the
combination of the smoothed/anisotropic array with CPML.  But the widened Box
is a PROXY for the fix: it also trips the #61 geometry-in-absorber preflight
check, which a real pad-replication fix does not.  So the proxy diverging does
not establish that the fix diverges.  This runs the fix itself.

THE PATCH.  ``rfx/runners/uniform.py:267`` imports ``compute_smoothed_eps`` at
call time, so wrapping the function on its own module reaches the Stage-1 site
without touching any file.  The wrapper extends each returned component into
every pad face with ``extend_cpml_pad_materials``
(``rfx/geometry/rasterize_grid.py:877``) -- the shape a real fix would take,
reusing the one shared implementation rather than writing a second copy (#627).

**NOTHING UNDER rfx/ IS MODIFIED.**  The patch lives here, in the probe.

RIG.  cv01 Run 1, imported from
``scripts/diagnostics/cv01_cpml_flux_selfcheck.py`` (its own ``run_arm``),
CPML, 20 layers, 25000 steps, ``subpixel_smoothing=True``, the COMMITTED Box --
no declared geometry changes, so the #61 warning must not appear.

Run (~2 min, 2-D CPU):

    PYTHONPATH=<repo> python3 \
      scripts/diagnostics/cv03_seam_facet/probe_pad_replication.py \
      --output scripts/diagnostics/_artifacts/cv03_seam_facet/pad_replication_probe.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

# Before anything can pull JAX in -- cv01's fluxes underflow float32 (#304).
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]

# The reference this arm is gated against: section 10's subpixel-OFF CPML row,
# the same rig with eps 12 in the pad by the interior route.
REFERENCE_MEAN_SELF_SUBPIXEL_OFF = 0.9887
SETTLING_BAR_DB = -40.0
MEAN_SELF_TOL = 0.01


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _assert_rfx_is_this_repo() -> dict:
    import rfx
    f = pathlib.Path(rfx.__file__).resolve()
    try:
        f.relative_to(REPO)
    except ValueError:
        raise SystemExit(f"rfx provenance check FAILED: {f} is not under {REPO}")
    return {"rfx_file": str(f), "under_repo_root": True, "repo_root": str(REPO)}


def _cv01_module():
    path = REPO / "scripts" / "diagnostics" / "cv01_cpml_flux_selfcheck.py"
    spec = importlib.util.spec_from_file_location("cv01_selfcheck", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cv01_selfcheck"] = mod
    spec.loader.exec_module(mod)
    return mod


def install_pad_replication_patch():
    """Apply the pad replication to ``aniso_eps``. Returns an undo callable."""
    import jax.numpy as jnp
    import rfx.geometry.smoothing as _sm
    from rfx.geometry.rasterize_grid import extend_cpml_pad_materials

    orig = _sm.compute_smoothed_eps

    def patched(grid, shape_eps_pairs, background_eps=1.0, **kw):
        comps = orig(grid, shape_eps_pairs, background_eps=background_eps, **kw)
        plx, phx = int(grid.pad_x_lo), int(grid.pad_x_hi)
        ply, phy = int(grid.pad_y_lo), int(grid.pad_y_hi)
        plz, phz = int(grid.pad_z_lo), int(grid.pad_z_hi)
        out = []
        for c in comps:
            c = jnp.asarray(c)
            # sigma/mu stand-ins: a smoothed eps array carries neither, and
            # they are only read by the #627a hi-face vacuum test.
            e2, _s, _m = extend_cpml_pad_materials(
                c, jnp.zeros_like(c), jnp.ones_like(c),
                plx, phx, ply, phy, plz, phz)
            out.append(e2)
        return tuple(out)

    _sm.compute_smoothed_eps = patched

    def undo():
        _sm.compute_smoothed_eps = orig

    return undo


def solved_centre_row(cv01, patched: bool) -> dict:
    """Centre-row eps the solver receives for cv01's COMMITTED build."""
    import rfx
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    import rfx.geometry.smoothing as _sm

    undo = install_pad_replication_patch() if patched else None
    try:
        sim = Simulation(freq_max=0.25 * cv01.C0 / cv01.a,
                         domain=(cv01.sx, cv01.sy, cv01.dx), dx=cv01.dx,
                         boundary=BoundarySpec.uniform("cpml"),
                         cpml_layers=20, mode="2d_tmz")
        sim.add_material("wg", eps_r=cv01.eps_wg)
        sim.add(rfx.Box((0, cv01.wg_y - cv01.w_wg / 2, 0),
                        (cv01.sx, cv01.wg_y + cv01.w_wg / 2, cv01.dx)),
                material="wg")
        grid = sim._build_grid()
        mats, *_ = sim._assemble_materials(grid)
        pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                 for e in sim._geometry]
        _, _, az = _sm.compute_smoothed_eps(grid, pairs, background_eps=1.0)
        jy = int(round(cv01.wg_y / cv01.dx)) + int(grid.pad_y_lo)
        built = np.asarray(mats.eps_r)[:, jy, 0]
        solved = np.asarray(az)[:, jy, 0]
        return {
            "patched": patched,
            "nx": int(grid.shape[0]),
            "pad_x_lo": int(grid.pad_x_lo), "pad_x_hi": int(grid.pad_x_hi),
            "built_n_at_eps_wg": int(np.sum(np.isclose(built, cv01.eps_wg))),
            "solved_n_at_eps_wg": int(np.sum(np.isclose(solved, cv01.eps_wg,
                                                        rtol=1e-3))),
            "solved_first3": [float(v) for v in solved[:3]],
            "solved_last3": [float(v) for v in solved[-3:]],
        }
    finally:
        if undo is not None:
            undo()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prov = _assert_rfx_is_this_repo()
    cv01 = _cv01_module()

    out = {
        "schema": "issue831-arm-f-pad-replication-probe-v1",
        "issue": 831, "landing_issue": 1043,
        "predeclaration":
            "docs/design_notes/issue831_far_end_return_predeclaration.md "
            "section 11",
        "rfx_modified": False,
        "patch": ("rfx.geometry.smoothing.compute_smoothed_eps wrapped in this "
                  "probe; each returned component extended into every pad face "
                  "with rfx.geometry.rasterize_grid.extend_cpml_pad_materials"),
        "provenance": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
        },
        "rfx_provenance_check": prov,
        "rig": {
            "case": "cv01 Run 1 (run_arm), imported not copied",
            "boundary": "cpml", "cpml_layers": 20,
            "n_steps": int(cv01.n_steps),
            "subpixel_smoothing": True,
            "box": "COMMITTED (0 .. sx) -- no declared geometry change",
        },
        "gate": {
            "reference_mean_self_subpixel_off":
                REFERENCE_MEAN_SELF_SUBPIXEL_OFF,
            "settling_bar_db": SETTLING_BAR_DB,
            "mean_self_tol": MEAN_SELF_TOL,
        },
        "instrument_check": {
            "unpatched": solved_centre_row(cv01, False),
            "patched": solved_centre_row(cv01, True),
        },
    }
    ic = out["instrument_check"]
    print(f"[instrument] unpatched solved eps=12 cells "
          f"{ic['unpatched']['solved_n_at_eps_wg']}/{ic['unpatched']['nx']}; "
          f"patched {ic['patched']['solved_n_at_eps_wg']}/{ic['patched']['nx']}",
          flush=True)

    import rfx
    from rfx.api import Simulation as _SimCls
    captured: dict = {}

    class _Instrumented(_SimCls):
        def run(self, *a, **kw):
            self.add_probe(position=(9.0 * cv01.a, cv01.wg_y, cv01.dx / 2),
                           component="ez")
            res = super().run(*a, **kw)
            captured["res"] = res
            return res

    undo = install_pad_replication_patch()
    _sim_orig = rfx.Simulation
    rfx.Simulation = _Instrumented
    t0 = time.time()
    try:
        arm = cv01.run_arm("cpml", None, cpml_layers=20)
    finally:
        rfx.Simulation = _sim_orig
        undo()

    res = captured.get("res")
    ts = np.asarray(res.time_series, dtype=float).ravel()
    finite = np.isfinite(ts)
    first_bad = int(np.argmin(finite)) if not finite.all() else None
    n = len(ts)
    deciles = [float(np.nanmax(np.abs(ts[i * n // 10:(i + 1) * n // 10])))
               for i in range(10)]
    mean_self = float(arm["mean_self"])
    settling = (None if getattr(res, "settling_db", None) is None
                else float(res.settling_db))

    r_f = (0.0 if finite.all() else 1.0)
    if settling is None:
        r_f += 1.0
    else:
        r_f += max(0.0, settling - SETTLING_BAR_DB)
    if np.isfinite(mean_self):
        r_f += max(0.0, abs(mean_self - REFERENCE_MEAN_SELF_SUBPIXEL_OFF)
                   - MEAN_SELF_TOL)
    else:
        r_f += 1.0

    arm.pop("preflight_report", None)
    arm.update({
        "all_finite": bool(finite.all()),
        "n_nonfinite": int((~finite).sum()),
        "first_nonfinite_index": first_bad,
        "amplitude_deciles": deciles,
        "settling_db": settling,
        "settling_witness": getattr(res, "settling_witness", None),
        "residual_r_F": float(r_f),
        "verdict": ("ASSEMBLY FIX ALONE IS LANDABLE" if r_f == 0.0
                    else ("BLOCKED -- diverged" if not finite.all()
                          else "INCONCLUSIVE -- finite but a leg missed")),
        "preflight_mentions_geometry_in_absorber": bool(
            "absorb" in arm.get("preflight_stdout", "").lower()
            or any("absorb" in w.lower()
                   for w in arm.get("preflight_warnings", []))),
        "wall_s": round(time.time() - t0, 1),
    })
    out["arm"] = arm

    print(f"[arm F] finite={arm['all_finite']} "
          f"n_nonfinite={arm['n_nonfinite']} "
          f"first_nonfinite_index={arm['first_nonfinite_index']} "
          f"mean_self={mean_self:.6f} settling={settling} "
          f"r_F={r_f:.4f} -> {arm['verdict']} ({arm['wall_s']}s)", flush=True)

    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
        fh.write("\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
