"""#1043 Stage A — what the CPML psi-coefficient fix does to cv01 Run 1.

cv01 Run 1 (20 CPML layers, 25000 steps, ``subpixel_smoothing=True``, the
committed ``Box``) is the number the fix was gated against: ``mean_self =
0.9195301017439319``. It is EXPECTED to move, and this measures by how much
rather than asserting it does not.

Why it moves, measured on the same build and not argued from the source: in
the committed geometry the guide's centre row inside the x pads reads
``materials.eps_r = 12`` (the pad extension put it there) while the smoothed
``aniso_eps`` reads ``1`` (the declared ``Box`` stops at the interior edge and
no replication step touches the rebuilt array — that gap IS #1043). Before the
fix the psi coefficient used 12 and the Yee half used 1, so the absorber in
those cells was twelve times weaker than the medium it was absorbing into.
After the fix both read 1.

Run it once per tree and diff::

    REF=$(mktemp -d); git archive origin/main | tar -x -C "$REF"
    PYTHONPATH=$REF python3 scripts/diagnostics/cpml_subpixel_stability/cv01_control.py \
        --allow-foreign-rfx --label main --output .../cv01_main.json
    PYTHONPATH=$(git rev-parse --show-toplevel) python3 \
        scripts/diagnostics/cpml_subpixel_stability/cv01_control.py \
        --label head --output .../cv01_head.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]

# The committed 20-layer CPML value this arm reproduces on the pre-fix tree.
COMMITTED_CPML_MEAN_SELF_20 = 0.9195301017439319


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _cv01_module(root: pathlib.Path):
    path = root / "scripts" / "diagnostics" / "cv01_cpml_flux_selfcheck.py"
    spec = importlib.util.spec_from_file_location("cv01_selfcheck", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cv01_selfcheck"] = mod
    spec.loader.exec_module(mod)
    return mod



def _band_trace(arm: dict) -> dict:
    """min/max/first/last of ``T_self_smooth`` over cv01's own band mask."""
    ts = np.asarray(arm.get("T_self_smooth", []), dtype=float)
    mask = np.asarray(arm.get("above_mask", []), dtype=bool)
    if ts.size == 0 or mask.size != ts.size or not mask.any():
        return {"band_trace": None}
    band = ts[mask]
    return {"band_trace": {
        "n_bins": int(band.size),
        "min": float(band.min()), "max": float(band.max()),
        "first": float(band[0]), "last": float(band[-1]),
        "values": [float(v) for v in band],
    }}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="head")
    ap.add_argument("--allow-foreign-rfx", action="store_true")
    ap.add_argument("--layers", type=int, default=20)
    args = ap.parse_args()

    import rfx
    rfx_file = pathlib.Path(rfx.__file__).resolve()
    inside = True
    try:
        rfx_file.relative_to(REPO)
    except ValueError:
        inside = False
        if not args.allow_foreign_rfx:
            raise SystemExit(f"rfx provenance check FAILED: {rfx_file}")

    # The driver is read from the SAME tree ``rfx`` came from, so the pre-fix
    # arm runs the pre-fix rig too.
    root = REPO if inside else rfx_file.parents[1]
    cv01 = _cv01_module(root)

    arm = cv01.run_arm("cpml", None, cpml_layers=args.layers)
    rec = {
        "label": args.label,
        "provenance": {"rfx_file": str(rfx_file), "rfx_under_repo_root": inside,
                       "driver_root": str(root),
                       "driver_commit": _git("rev-parse", "HEAD"),
                       "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
        "cpml_layers": args.layers,
        "mean_self": arm["mean_self"],
        # R5: a band mean is not a result. The smoothed per-bin trace
        # over cv01's own `above` mask says whether the change is a
        # uniform offset or an interference change.
        **_band_trace(arm),
        "settling_db": arm.get("settling_db"),
        "committed_reference": COMMITTED_CPML_MEAN_SELF_20,
        "delta_vs_committed": (None if arm["mean_self"] is None
                               else arm["mean_self"] - COMMITTED_CPML_MEAN_SELF_20),
        "preflight_warnings": arm.get("preflight_warnings"),
        "run_warnings": arm.get("run_warnings"),
        "wall_s": arm.get("wall_s"),
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str))
    bt = rec.get("band_trace")
    print(f"[{args.label}] layers={args.layers} mean_self={rec['mean_self']!r} "
          f"delta={rec['delta_vs_committed']!r} settling_db={rec['settling_db']!r}")
    if bt:
        print(f"[{args.label}] band T_self_smooth n={bt['n_bins']} "
              f"min={bt['min']:.4f} max={bt['max']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
