#!/usr/bin/env python
"""Issue #1075 measurement: a soft source placed ON a whole-boundary PEC face.

WHAT IS MEASURED
----------------
#1075 reports that an ``ez`` soft source sitting exactly on the x-lo domain
wall of a ``boundary="pec"`` simulation is KEPT by the single-device lane but
zeroed by all three distributed lanes. The single-device E half-step in
``rfx/simulation.py`` applies ``apply_pec``/``apply_pec_faces`` BEFORE source
injection, so the injected tangential E survives to the probe read of that same
step; the distributed lanes apply the PEC face AFTER injection (#1041/#1055), so
the same cell reads back exactly ``0.0``.

Three arms:

A. PREFLIGHT COVERAGE. Which advisory codes ``sim.preflight()`` emits for the
   on-face placement, under the two ways of asking for a PEC wall:

     * ``boundary="pec"``            -- whole-boundary PEC via ``apply_pec(axes)``
     * ``boundary=BoundarySpec(...)`` with ``lo="pec"`` on one axis

   Before the #1075 fix the first emits NOTHING: P1.6
   (``_validate_cfg_source_on_reflector_plane``, code ``source_decoupled``)
   iterates ``self._pec_faces``, which the legacy scalar path leaves EMPTY --
   only an explicit ``pec_faces=`` kwarg or a ``BoundarySpec`` populates it --
   so the loop body never runs. The per-face spec arm is the control that shows
   the check itself works; it is the placement, not the rule, that was uncovered.

B. SINGLE-DEVICE RETENTION. Probe peak and post-source domain E-energy for the
   on-face source against the SAME source moved one cell inside, over a sweep of
   probe distances. The point of the sweep is that the ratio is NOT a physical
   constant: this is a closed PEC cavity, the two placements excite different
   mode amplitudes, and the probe-peak ratio measured here runs from ~0.07 to
   ~3.7 purely as a function of where the probe sits. What is invariant, and
   what the advisory is about, is that the on-face field is LARGE AND FINITE on
   this lane -- the source is not discarded, it drives a component the PEC
   mirror is supposed to hold at zero.

C. DISTRIBUTED LANE (optional, ``--devices N`` with N >= 2). The same on-face
   fixture through ``sim.run(devices=...)``. Expected: probe time series
   identically zero, which is the disagreement #1075 is about.

USAGE
-----
    python scripts/diagnostics/issue1075_source_on_face.py
    python scripts/diagnostics/issue1075_source_on_face.py --devices 2

Arm C forces 2 host CPU devices via ``XLA_FLAGS``; that must be set before jax
is imported, which is why the flag is read from ``sys.argv`` at module import
time rather than after ``argparse``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

# ---- host-device count must be set before jax is imported ------------------
_N_DEV = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--devices" and _i + 1 < len(sys.argv):
        _N_DEV = int(sys.argv[_i + 1])
    elif _a.startswith("--devices="):
        _N_DEV = int(_a.split("=", 1)[1])
if _N_DEV > 1:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={_N_DEV}"
    ).strip()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here.parent, text=True)
    return Path(out.strip())


# Import THIS checkout's rfx, not the editable install's -- same trap as
# scripts/diagnostics/issue1041_v2_step_order.py: running a script by path puts
# the SCRIPT's directory on sys.path[0], not the cwd, so ``import rfx`` inside a
# git worktree of an editable install silently resolves to the install's tree
# and the measurement is made against code that was never edited.
_REPO = _repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _assert_rfx_is_this_checkout():
    import rfx
    import rfx.preflight.sources as pfs
    for mod in (rfx, pfs):
        got = Path(mod.__file__).resolve()
        if not str(got).startswith(str(_REPO) + os.sep):
            raise SystemExit(
                f"{mod.__name__} imported from {got}, which is OUTSIDE this "
                f"checkout ({_REPO}). The measurement would be run against "
                "another tree. Run from the checkout root, or set "
                f"PYTHONPATH={_REPO}.")
    return Path(pfs.__file__).resolve()


import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

DX = 1e-3
NX_CELLS = 20
NYZ_CELLS = 15
CENTER = NYZ_CELLS * DX / 2.0
FREQ_MAX = 10e9
N_STEPS = 400
# amplitude_kind='current' so the injected amplitude has a boundary-independent
# meaning (#571); the ratios below are unchanged under 'field' (verified: the
# two kinds differ by a constant factor at every probe distance).
AMP_KIND = "current"


def build_sim(source_x: float, probe_x: float, *, spec_face: bool = False):
    """``ez`` soft source at ``source_x`` on the x axis, probe at ``probe_x``.

    ``spec_face=False`` asks for the wall the legacy way (``boundary="pec"``,
    all six faces, realized as ``apply_pec(axes="xyz")``); ``spec_face=True``
    asks for the SAME x-lo wall through a per-face ``BoundarySpec``, which is
    what populates ``Simulation._pec_faces``.
    """
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    if spec_face:
        boundary = BoundarySpec(x=Boundary(lo="pec", hi="pec"), y="pec", z="pec")
    else:
        boundary = "pec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(
            freq_max=FREQ_MAX,
            domain=(NX_CELLS * DX, NYZ_CELLS * DX, NYZ_CELLS * DX),
            dx=DX, boundary=boundary, cpml_layers=0,
        )
        sim.add_source((source_x, CENTER, CENTER), "ez",
                       amplitude_kind=AMP_KIND)
        sim.add_probe((probe_x, CENTER, CENTER), "ez")
    return sim


# ---------------------------------------------------------------------------
# Arm A -- preflight coverage
# ---------------------------------------------------------------------------

def _codes(sim) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return sorted({i.code for i in report.issues if i.code})


def arm_a() -> None:
    print("## A. preflight codes for an on-face ez source")
    print(f"   fixture: {NX_CELLS}x{NYZ_CELLS}x{NYZ_CELLS} cells at "
          f"dx={DX*1e3:g} mm, ez at x=0 (the x_lo wall)")
    for label, kw in (
        ('boundary="pec"            ', dict(spec_face=False)),
        ('BoundarySpec(x.lo="pec")   ', dict(spec_face=True)),
    ):
        on_face = _codes(build_sim(0.0, 6 * DX, **kw))
        inside = _codes(build_sim(DX, 6 * DX, **kw))
        print(f"   {label} on-face  : {on_face or '[]'}")
        print(f"   {label} one cell : {inside or '[]'}")
    print()


# ---------------------------------------------------------------------------
# Arm B -- single-device retention
# ---------------------------------------------------------------------------

def _peak_and_energy(sim) -> tuple[float, float]:
    res = sim.run(n_steps=N_STEPS, skip_preflight=True)
    ts = np.asarray(res.time_series, dtype=np.float64)
    peak = float(np.max(np.abs(ts)))
    energy = 0.0
    for f in ("ex", "ey", "ez"):
        arr = getattr(res.state, f, None)
        if arr is not None:
            energy += float(np.sum(np.asarray(arr, dtype=np.float64) ** 2))
    return peak, energy


def arm_b() -> None:
    print("## B. single-device probe peak / end-state E-energy, on-face vs "
          "one cell inside")
    print("   probe_x |    on-face peak |   one-cell peak | peak ratio |"
          "   on-face U_E |  one-cell U_E | U ratio")
    for k in (4, 6, 8, 10, 12):
        probe_x = k * DX
        pa, ea = _peak_and_energy(build_sim(0.0, probe_x))
        pb, eb = _peak_and_energy(build_sim(DX, probe_x))
        print(f"   {k:2d} mm   | {pa:15.6g} | {pb:15.6g} | "
              f"{pa/pb:10.3f} | {ea:13.6g} | {eb:13.6g} | {ea/eb:7.3f}")
    print("   (the peak ratio is fixture-dependent -- it is a cavity mode "
          "amplitude, not a retention fraction.")
    print("    The invariant is that the on-face column is FINITE AND LARGE: "
          "the source is not discarded.)")
    print()


# ---------------------------------------------------------------------------
# Arm C -- distributed lane
# ---------------------------------------------------------------------------

def arm_c(n_devices: int) -> None:
    import jax
    devices = list(jax.devices())[:n_devices]
    print(f"## C. distributed lane, devices={len(devices)}")
    if len(devices) < 2:
        print("   SKIPPED: fewer than 2 devices visible.")
        print()
        return
    for label, source_x in (("on-face  x=0   ", 0.0),
                            ("one cell x=dx  ", DX)):
        sim = build_sim(source_x, 6 * DX)
        res = sim.run(n_steps=N_STEPS, skip_preflight=True, devices=devices)
        ts = np.asarray(res.time_series, dtype=np.float64)
        print(f"   {label} peak = {float(np.max(np.abs(ts))):.6g}")
    print()


def main() -> int:
    pfs_path = _assert_rfx_is_this_checkout()
    import jax
    rev = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True).strip()
    print("# issue #1075 -- source on a whole-boundary PEC face")
    print(f"# repo={_REPO} rev={rev} jax={jax.__version__}")
    print(f"# rfx.preflight.sources={pfs_path}")
    print()
    arm_a()
    arm_b()
    if _N_DEV > 1:
        arm_c(_N_DEV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
