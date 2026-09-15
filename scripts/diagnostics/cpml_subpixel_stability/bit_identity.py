"""#1043 Stage A — bit-identity table for the CPML psi-coefficient fix.

Hashes the raw final field bytes of a set of configurations and compares this
tree against a pristine ``git archive`` of a reference commit. Run it twice,
once per tree, and diff the two artifacts:

    # reference (origin/main), unpacked to a scratch dir
    REF=$(mktemp -d); git archive origin/main | tar -x -C "$REF"
    PYTHONPATH=$REF python3 scripts/diagnostics/cpml_subpixel_stability/bit_identity.py \
        --output .../bit_identity_main.json --allow-foreign-rfx
    # this tree
    PYTHONPATH=$(git rev-parse --show-toplevel) python3 \
        scripts/diagnostics/cpml_subpixel_stability/bit_identity.py \
        --output .../bit_identity_head.json

Pinning the reference to a ``git archive`` rather than to a checkout is
deliberate: a builder editing the same worktree moves the review target
silently.

The claim under test: ``inv_eps_r_update=None`` on every path that has no
anisotropic array, and the anisotropic array equalling ``materials.eps_r``
wherever ``apply_cpml_e`` writes, must leave the field bytes untouched. The
configurations where it does NOT are listed in ``expect_change`` and are the
point of the change, not a regression.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import subprocess

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
C0 = 2.998e8
A = 1.0e-6
DX = A / 10
N_STEPS = 400


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _field_hash(state) -> str:
    h = hashlib.sha256()
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        arr = np.asarray(getattr(state, comp))
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _base(boundary: str, cpml_layers: int = 10, nu: bool = False):
    import rfx
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources import GaussianPulse

    sx = sy = 8.0 * A
    kw = {}
    if nu:
        kw["dz_profile"] = None
    sim = Simulation(freq_max=0.25 * C0 / A, domain=(sx, sy, DX), dx=DX,
                     boundary=BoundarySpec.uniform(boundary),
                     cpml_layers=cpml_layers, mode="2d_tmz")
    return sim, sx, sy


def _finish(sim, sx, sy):
    from rfx.sources import GaussianPulse
    fcen = 0.15 * C0 / A
    sim.add_source(position=(11 * DX, sy / 2, 0), component="ez",
                   waveform=GaussianPulse(f0=fcen, bandwidth=0.667,
                                          amplitude=1.0))
    sim.add_probe(position=(sx - 15 * DX, sy / 2, 0), component="ez")
    return sim


def cfg_touching(boundary: str, lossy: bool = False):
    """Guide spanning the full x extent — reaches both x pads."""
    import rfx
    sim, sx, sy = _base(boundary)
    if lossy:
        sim.add_material("wg", eps_r=12.0, sigma=0.05)
    else:
        sim.add_material("wg", eps_r=12.0)
    sim.add(rfx.Box((0, sy / 2 - 0.5 * A, 0), (sx, sy / 2 + 0.5 * A, DX)),
            material="wg")
    return _finish(sim, sx, sy)


def cfg_interior(boundary: str):
    """Slab well clear of every pad — no interface cell inside an absorber."""
    import rfx
    sim, sx, sy = _base(boundary)
    sim.add_material("wg", eps_r=12.0)
    sim.add(rfx.Box((3.0 * A, 3.0 * A, 0), (5.0 * A, 5.0 * A, DX)),
            material="wg")
    return _finish(sim, sx, sy)


def cfg_vacuum(boundary: str):
    sim, sx, sy = _base(boundary)
    return _finish(sim, sx, sy)


CASES = (
    # name,                         builder,               subpixel
    ("cpml_vacuum_nosub",           lambda: cfg_vacuum("cpml"),         False),
    ("cpml_vacuum_sub",             lambda: cfg_vacuum("cpml"),         True),
    ("cpml_interior_nosub",         lambda: cfg_interior("cpml"),       False),
    ("cpml_interior_sub",           lambda: cfg_interior("cpml"),       True),
    ("cpml_touching_nosub",         lambda: cfg_touching("cpml"),       False),
    ("cpml_touching_lossy_nosub",   lambda: cfg_touching("cpml", True), False),
    ("cpml_touching_sub",           lambda: cfg_touching("cpml"),       True),
    ("cpml_touching_lossy_sub",     lambda: cfg_touching("cpml", True), True),
    ("upml_touching_sub",           lambda: cfg_touching("upml"),       True),
    ("upml_touching_nosub",         lambda: cfg_touching("upml"),       False),
    ("pec_touching_sub",            lambda: cfg_touching("pec"),        True),
)

# Where the fix is INTENDED to move numbers: subpixel smoothing on, CPML on,
# and a smoothed interface cell inside a CPML pad. Everything else must hash
# identically.
EXPECT_CHANGE = {"cpml_touching_sub", "cpml_touching_lossy_sub"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow-foreign-rfx", action="store_true",
                    help="running against a git-archive reference tree")
    args = ap.parse_args()

    import rfx
    rfx_file = str(pathlib.Path(rfx.__file__).resolve())
    if not args.allow_foreign_rfx:
        try:
            pathlib.Path(rfx_file).relative_to(REPO)
        except ValueError:
            raise SystemExit(f"rfx provenance check FAILED: {rfx_file}")

    rec = {"provenance": {"rfx_file": rfx_file, "commit": _git("rev-parse", "HEAD"),
                          "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
           "n_steps": N_STEPS, "expect_change": sorted(EXPECT_CHANGE),
           "hashes": {}}
    for name, build, subpixel in CASES:
        sim = build()
        with contextlib.redirect_stdout(io.StringIO()):
            res = sim.run(n_steps=N_STEPS, subpixel_smoothing=subpixel)
        ez = np.asarray(res.state.ez)
        rec["hashes"][name] = {
            "sha256": _field_hash(res.state),
            "finite": bool(np.all(np.isfinite(ez))),
            "max_abs_ez": (float(np.nanmax(np.abs(ez)))
                           if np.any(np.isfinite(ez)) else None),
        }
        print(f"{name:28s} {rec['hashes'][name]['sha256'][:16]} "
              f"finite={rec['hashes'][name]['finite']}")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
