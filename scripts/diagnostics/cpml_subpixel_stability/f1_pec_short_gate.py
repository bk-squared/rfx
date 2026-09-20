"""#1043 review round 1, F1 — why the PEC-short conformal |S11| gate moved.

Pre-declaration: ``docs/design_notes/issue1043_f1_pec_short_gate_predeclaration.md``
(arms A-D and the decision rule, frozen before this ran).

The rig is ``tests/unit/geometry/test_subpixel_pec.py::_pec_short_sim`` with
``conformal=True``: x CPML, y/z PEC with conformal faces, a one-cell PEC short,
``normalize=False``, 40 periods, 6 bins over 5-7 GHz.

It touches #1043's code because ``rfx/runners/uniform.py:279-303`` sets
``aniso_eps`` straight from ``conformal_eps_correction`` -- ``eps_eff = eps/w``
with ``w < 1`` at wall cells, so ``eps_a > eps_b = materials.eps_r`` there --
and the x CPML pad spans every y and z, so the y/z conformal wall cells at x
inside the pads sit in the absorber.

Run once per tree::

    PYTHONPATH=$(git rev-parse --show-toplevel) python3 \
        scripts/diagnostics/cpml_subpixel_stability/f1_pec_short_gate.py \
        --label head --output .../f1_head.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import subprocess
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
GATE_MIN = 0.99


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a],
                          capture_output=True, text=True).stdout.strip()


def _prov(allow_foreign: bool) -> dict:
    import rfx
    f = pathlib.Path(rfx.__file__).resolve()
    inside = True
    try:
        f.relative_to(REPO)
    except ValueError:
        inside = False
        if not allow_foreign:
            raise SystemExit(f"rfx provenance check FAILED: {f}")
    return {"rfx_file": str(f), "rfx_under_repo_root": inside,
            "driver_commit": _git("rev-parse", "HEAD")}



def _scalar(v):
    """``settling_db`` is a float on some lanes and an array on others."""
    if v is None:
        return None
    a = np.asarray(v, dtype=float)
    return float(a) if a.size == 1 else [float(x) for x in a.ravel()]


def build(boundary_x: str = "cpml", conformal: bool = True):
    """``_pec_short_sim`` hand-copied, with the x absorber family a parameter."""
    import jax.numpy as jnp
    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9, domain=(0.12, 0.04, 0.02), dx=0.003,
        boundary=BoundarySpec(
            x=boundary_x,
            y=Boundary(lo="pec", hi="pec", conformal=conformal),
            z=Boundary(lo="pec", hi="pec", conformal=conformal),
        ),
        cpml_layers=10,
    )
    sim.add(Box((0.084, 0, 0), (0.087, 0.04, 0.02)), material="pec")
    freqs = jnp.linspace(5e9, 7e9, 6)
    sim.add_waveguide_port(0.010, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="left")
    sim.add_waveguide_port(0.090, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="right")
    return sim


def restrict_conformal_to_interior(n_pad: int):
    """Arm B: keep ``eps_base`` at conformal wall cells INSIDE the x CPML pads.

    Wraps ``conformal_eps_correction`` on its own module, which
    ``rfx/runners/uniform.py`` imports at call time. Nothing under ``rfx/`` is
    edited. The count of cells actually restored is recorded, so "the pads were
    excluded" is evidence rather than an assumption.
    """
    import jax.numpy as jnp
    import rfx.geometry.conformal as _cf

    orig = _cf.conformal_eps_correction
    seen: dict = {"n_pad": int(n_pad)}

    def patched(eps_base, w_ex, w_ey, w_ez):
        out = orig(eps_base, w_ex, w_ey, w_ez)
        eb = jnp.asarray(eps_base)
        nx = eb.shape[0]
        pad = np.zeros(nx, dtype=bool)
        pad[:n_pad] = True
        pad[nx - n_pad:] = True
        mask = jnp.asarray(pad)[:, None, None]
        restricted = tuple(jnp.where(mask, eb, c) for c in out)
        seen["n_restricted"] = int(sum(
            int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
            for a, b in zip(out, restricted)))
        return restricted

    _cf.conformal_eps_correction = patched

    def undo():
        _cf.conformal_eps_correction = orig
    return undo, seen


def measure(label: str, boundary_x: str, restrict_pads: bool = False,
            num_periods: int = 40) -> dict:
    undo = None
    seen: dict = {}
    try:
        if restrict_pads:
            probe = build(boundary_x=boundary_x)
            n_pad = int(probe._build_grid().pad_x_lo)
            undo, seen = restrict_conformal_to_interior(n_pad)
        sim = build(boundary_x=boundary_x)
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(buf):
                res = sim.compute_waveguide_s_matrix(
                    num_periods=num_periods, normalize=False)
        s = np.asarray(res.s_params)
        s11 = np.abs(s[0, 0, :])
        # Arm D: the self-check's own inputs -- per-column power per bin, and
        # which entry carries the largest magnitude.
        mag = np.abs(s)
        col_power = (mag ** 2).sum(axis=0)          # (n_ports, n_freqs)
        imax = np.unravel_index(int(np.argmax(mag)), mag.shape)
        dev = np.abs(s11 - 1.0)
        return {
            "label": label, "boundary_x": boundary_x,
            "num_periods": num_periods,
            # A' -- the analytic oracle. A lossless PEC short has
            # |S11| = 1 at every bin, so the physical score is two-sided
            # distance from 1; the committed gate (min >= 0.99) is
            # one-sided and cannot see an over-unity excursion at all.
            "dev_from_unity_per_bin": [float(v) for v in dev],
            "D_max_dev_from_unity": float(dev.max()),
            "D_rms_dev_from_unity": float(np.sqrt((dev ** 2).mean())),
            "restrict_pads": restrict_pads,
            "n_pad_cells": seen.get("n_pad"),
            "restricted_cells": seen.get("n_restricted"),
            "port_names": list(res.port_names),
            "s11_per_bin": [float(v) for v in s11],
            "s11_min": float(s11.min()), "s11_max": float(s11.max()),
            "s11_mean": float(s11.mean()),
            "gate_min_0p99_passes": bool(s11.min() >= GATE_MIN),
            "col_power_per_port_per_bin": [[float(v) for v in row]
                                           for row in col_power],
            "max_col_power": float(col_power.max()),
            "max_abs_S": float(mag.max()),
            "max_abs_S_at": {"row": int(imax[0]), "col": int(imax[1]),
                             "bin": int(imax[2])},
            "abs_S_all": [[[float(v) for v in b] for b in row] for row in mag],
            "settling_db": _scalar(getattr(res, "settling_db", None)),
            "warnings": [str(w.message)[:400] for w in rec][:12],
            "stdout_tail": buf.getvalue()[-1200:],
        }
    except ValueError as exc:
        # An arm the API refuses is a property of the rig, recorded as
        # such -- not a measurement that happened to fail.
        return {"label": label, "boundary_x": boundary_x,
                "num_periods": num_periods,
                "not_constructible": str(exc)}
    finally:
        if undo is not None:
            undo()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="head")
    ap.add_argument("--allow-foreign-rfx", action="store_true")
    args = ap.parse_args()

    rec = {"label": args.label, "provenance": _prov(args.allow_foreign_rfx),
           "gate": GATE_MIN, "arms": {}}
    rec["arms"]["cpml"] = measure("cpml", "cpml")
    # Arm A as pre-declared: struck, and why, recorded in the artifact.
    rec["arms"]["upml"] = measure("upml", "upml")
    # Arm B.
    rec["arms"]["cpml_pads_excluded"] = measure(
        "cpml_pads_excluded", "cpml", restrict_pads=True)
    # Arm A'' -- record length, the independent axis.
    for n in (80, 160):
        rec["arms"][f"cpml_{n}periods"] = measure(
            f"cpml_{n}periods", "cpml", num_periods=n)

    for k, v in rec["arms"].items():
        if "not_constructible" in v:
            print(f"[{args.label}/{k}] NOT CONSTRUCTIBLE: {v['not_constructible']}")
            continue
        print(f"[{args.label}/{k}] n={v['num_periods']} |S11| "
              f"[{v['s11_min']:.4f}, {v['s11_max']:.4f}] "
              f"mean={v['s11_mean']:.4f} D={v['D_max_dev_from_unity']:.4f} "
              f"gate={'PASS' if v['gate_min_0p99_passes'] else 'FAIL'} "
              f"maxcol={v['max_col_power']:.4f} max|S|={v['max_abs_S']:.4f} "
              f"restricted={v['restricted_cells']}")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
