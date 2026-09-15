#!/usr/bin/env python
"""Issue #1041 measurement: does ``distributed_v2``'s step order corrupt the
rank seam, and does ``distributed_nu``'s post-``ac782d4f`` order fix it?

WHAT IS MEASURED
----------------
``rfx/runners/distributed_v2.py`` has two scan bodies. Before the #1041 fix
they read::

    step_fn_cpml: H -> CPML-H -> exch H -> PMC -> E -> CPML-E -> exch E -> src -> probes
    step_fn_pec : H -> exch H -> PMC -> E -> exch E -> PEC face -> src -> probes

``distributed_nu.py`` was changed by ``ac782d4f`` (#931 T3) to exchange the E
ghosts LAST::

    ... E -> CPML-E -> src -> PEC face -> PEC mask -> occupancy -> exch E -> probes

Two orderings therefore differ between the lanes:

1. ``exch E`` before ``src`` (both v2 bodies). A source injected into rank d's
   FIRST real cell is not in rank d-1's right ghost until the NEXT step's
   exchange, so rank d-1's H update reads a pre-injection E. That is the lag
   ``tests/unit/runners/test_distributed_nu_smoke.py`` documents in its
   docstring -- and that test reaches v2, not ``distributed_nu``.
2. ``exch E`` before ``PEC face`` (``step_fn_pec``). #1041 assessed this
   "probably unreachable for v2"; this script measures it instead of assuming.

REFERENCE LANE
--------------
The SAME model on ONE device through the uniform lane (``sim.run(n_steps=...)``
-> ``rfx.simulation.run``), whose E half-step is
``E -> CPML-E -> PEC faces -> ... -> sources -> probes``. ``v1``
(``rfx.runners.distributed.run_distributed``) is run as a second witness: v1
and v2 are NOT bit-identical (the #1038 lock records 2.794e-09 on a 9.4145e-03
peak for the same model), so |v1 - reference| on the interior-source control
is what "as good as a distributed lane gets" looks like here.

FIXTURES
--------
S  uniform 31x15x15 mm at dx = 1 mm -> nx = 32, ny = nz = 16, 2 ranks,
   nx_per_rank = 16 (pad_x = 0). Seam plane is global x-index 16 = rank 1's
   FIRST real cell.
   S-seam      : ez source AT x = 16 mm (rank 1's first real cell);
                 probes at x = 12 mm (rank 0) and x = 20 mm (rank 1).
   S-interior  : the same probes with the source moved to x = 8 mm
                 (8 cells inside rank 0) -- the control that shows the effect
                 is seam-specific, not "any signal crossing the seam".
   Both run on boundary="pec" (reaches ``step_fn_pec``) and boundary="cpml"
   (reaches ``step_fn_cpml``).

P  the PEC-body question. ``sim.add(Box(...), material="pec")`` is REFUSED on
   both distributed lanes (``NotImplementedError``, see
   ``tests/unit/runners/test_distributed_pec_geometry_refusal.py``), so a PEC
   Box straddling the seam cannot be built on v2 at all; the script asserts
   that refusal and records its message. The only PEC v2 ever applies is the
   DOMAIN FACE (``_distributed_common.apply_pec_face_shmap``), so fixture P is
   the domain-face PEC with an INTERIOR source: with no source within one cell
   of the seam, the ``src``/``exch E`` swap cannot matter and any difference
   between the two orderings is attributable to the ``PEC face``/``exch E``
   swap alone.

HOW THE TWO ORDERINGS ARE RUN (mechanism)
-----------------------------------------
No test seam was added to production code. Each arm loads
``rfx/runners/distributed_v2.py`` FROM A GIT REVISION as a private module
(``git show <rev>:rfx/runners/distributed_v2.py`` -> ``importlib``), and calls
its ``run_distributed`` directly. ``--legacy-rev`` selects the pre-fix file
(default: the commit this script was written against) and the nu ordering
comes from the working tree. So the table is reproducible from ONE checkout at
any later commit.

USE
---
    XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    python scripts/diagnostics/issue1041_v2_step_order.py \
        --out scripts/diagnostics/_artifacts/issue1041

Writes ``<out>/issue1041_v2_step_order.npz`` (every probe trace and every
final field array for every arm) and prints the table below.

MEASURED (2026-09-15, this pod: linux x86_64, 2 virtual CPU devices,
python 3.11.16, jax 0.10.2, float32; 300 steps; legacy rev 541f703f)
--------------------------------------------------------------------
See ``docs/research_notes/2026-09-15_issue1041_v2_step_order.md`` and the
issue thread for the committed table.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

# The last commit that touched rfx/runners/distributed_v2.py before the #1041
# step-order change. The legacy arm is this file, loaded as its own module.
LEGACY_REV_DEFAULT = "541f703f"

_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
_E_FIELDS = ("ex", "ey", "ez")
_H_FIELDS = ("hx", "hy", "hz")

# float32 has ~1.2e-7 eps; a relative deviation at or under this is not
# distinguishable from a re-association of the same arithmetic.
F32_NOISE = 1e-7


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here.parent, text=True)
    return Path(out.strip())


# Import THIS checkout's rfx, not the editable install's.
#
# Running a script by path puts the SCRIPT's directory on sys.path[0], not the
# cwd, so ``import rfx`` in a git worktree of an editable install silently
# resolves to the install's source tree instead. That is not academic: the
# first run of this script measured the primary checkout on every arm and
# reported the two orderings as bit-identical, because the file the "working
# tree" arm imported was never the edited one. The checkout that owns this
# script goes on the front of sys.path, and _assert_rfx_is_this_checkout()
# below refuses to run if that did not take.
_REPO = _repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _assert_rfx_is_this_checkout():
    import rfx
    import rfx.runners.distributed_v2 as v2
    for mod in (rfx, v2):
        got = Path(mod.__file__).resolve()
        if not str(got).startswith(str(_REPO) + os.sep):
            raise SystemExit(
                f"{mod.__name__} imported from {got}, which is OUTSIDE this "
                f"checkout ({_REPO}). The measurement would be run against "
                "another tree. Run from the checkout root, or set "
                f"PYTHONPATH={_REPO}.")
    return Path(v2.__file__).resolve()


# ---------------------------------------------------------------------------
# Loading distributed_v2 from a git revision, as its own module
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, object] = {}


def load_v2(rev: str | None, repo: Path, tmpdir: Path):
    """Return ``distributed_v2``'s ``run_distributed`` from ``rev``.

    ``rev=None`` means the working tree (imported normally).
    """
    key = rev or "__worktree__"
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    if rev is None:
        from rfx.runners.distributed_v2 import run_distributed
        _MODULE_CACHE[key] = run_distributed
        return run_distributed
    blob = subprocess.check_output(
        ["git", "show", f"{rev}:rfx/runners/distributed_v2.py"], cwd=repo)
    safe = rev.replace("/", "_").replace("~", "_").replace("^", "_")
    path = tmpdir / f"distributed_v2_{safe}.py"
    path.write_bytes(blob)
    name = f"_rfx_distributed_v2_at_{safe}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod.run_distributed
    return mod.run_distributed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DX = 1e-3
NX_CELLS = 31          # -> nx = 32 nodes, EVEN, so pad_x = 0 at 2 devices
NYZ_CELLS = 15         # -> ny = nz = 16
SEAM_X = 16e-3         # global node 16 == rank 1's first real cell
PROBE_LO_X = 12e-3     # 4 cells INSIDE rank 0 (signal must cross the seam)
PROBE_HI_X = 20e-3     # 4 cells inside rank 1
INTERIOR_X = 8e-3      # 8 cells from the seam, inside rank 0
CENTER = 8e-3


def build_sim(boundary: str, source_x: float, *, pec_box=False):
    from rfx import Box, Simulation
    kw = dict(freq_max=15e9,
              domain=(NX_CELLS * DX, NYZ_CELLS * DX, NYZ_CELLS * DX),
              dx=DX, boundary=boundary)
    if boundary == "cpml":
        kw["cpml_layers"] = 6
    else:
        kw["cpml_layers"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(**kw)
        if pec_box:
            # x-extent 14..18 mm straddles the seam at 16 mm.
            sim.add(Box((14e-3, 4e-3, 4e-3), (18e-3, 11e-3, 11e-3)),
                    material="pec")
        sim.add_source((source_x, CENTER, CENTER), "ez")
        sim.add_probe((PROBE_LO_X, CENTER, CENTER), "ez")
        sim.add_probe((PROBE_HI_X, CENTER, CENTER), "ez")
    return sim


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _snapshot(res):
    ts = np.asarray(res.time_series, dtype=np.float64)
    st = {f: np.asarray(getattr(res.state, f), dtype=np.float64)
          for f in _FIELDS}
    return {"time_series": ts, **st}


def run_reference(sim_builder, n_steps):
    sim = sim_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _snapshot(sim.run(n_steps=n_steps, skip_preflight=True))


def run_v2(sim_builder, n_steps, rev, repo, tmpdir, devices):
    run_distributed = load_v2(rev, repo, tmpdir)
    sim = sim_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _snapshot(run_distributed(sim, n_steps=n_steps,
                                         devices=devices))


def run_v1(sim_builder, n_steps, devices):
    from rfx.runners.distributed import run_distributed as v1
    sim = sim_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _snapshot(v1(sim, n_steps=n_steps, devices=devices))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(snap, ref, *, tol=F32_NOISE):
    """Absolute / relative deviation of ``snap`` from ``ref``."""
    out = {}
    ts_a = snap["time_series"]
    ts_r = ref["time_series"]
    n_prb = ts_r.shape[1]
    for p in range(n_prb):
        d = np.abs(ts_a[:, p] - ts_r[:, p])
        peak = float(np.max(np.abs(ts_r[:, p])))
        denom = peak if peak > 0 else 1.0
        first = np.nonzero(d / denom > tol)[0]
        out[f"ts{p}_absmax"] = float(np.max(d))
        out[f"ts{p}_peak"] = peak
        out[f"ts{p}_rel"] = float(np.max(d)) / denom
        out[f"ts{p}_first"] = int(first[0]) if first.size else -1
    for group, name in ((_E_FIELDS, "E"), (_H_FIELDS, "H")):
        amax = 0.0
        peak = 0.0
        for f in group:
            amax = max(amax, float(np.max(np.abs(snap[f] - ref[f]))))
            peak = max(peak, float(np.max(np.abs(ref[f]))))
        out[f"{name}_absmax"] = amax
        out[f"{name}_peak"] = peak
        out[f"{name}_rel"] = amax / (peak if peak > 0 else 1.0)
    return out


def _fmt(row):
    return (f"{row['ts0_rel']:10.3e} {row['ts0_absmax']:10.3e} "
            f"{row['ts0_first']:>6d} "
            f"{row['ts1_rel']:10.3e} {row['ts1_absmax']:10.3e} "
            f"{row['ts1_first']:>6d} "
            f"{row['E_rel']:10.3e} {row['E_absmax']:10.3e} "
            f"{row['H_rel']:10.3e} {row['H_absmax']:10.3e}")


HEADER = (f"{'arm':<26}{'rel@p0':>10} {'abs@p0':>10} {'1stdiv':>6} "
          f"{'rel@p1':>10} {'abs@p1':>10} {'1stdiv':>6} "
          f"{'relE':>10} {'absE':>10} {'relH':>10} {'absH':>10}")


def _peaks_line(ref_row):
    return ("#   reference peaks: "
            f"p0={ref_row['ts0_peak']:.4e}  p1={ref_row['ts1_peak']:.4e}  "
            f"E={ref_row['E_peak']:.4e}  H={ref_row['H_peak']:.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-rev", default=LEGACY_REV_DEFAULT,
                    help="git rev holding the PRE-#1041 distributed_v2.py")
    ap.add_argument("--nu-rev", default=None,
                    help="git rev holding the nu-ordered distributed_v2.py "
                         "(default: the working tree)")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="directory for the .npz (default: no file written)")
    args = ap.parse_args()

    if jax.device_count() < 2:
        raise SystemExit(
            "need >=2 JAX devices; run with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2")
    devices = jax.devices()[:2]
    repo = _REPO
    v2_path = _assert_rfx_is_this_checkout()
    tmp = Path(tempfile.mkdtemp(prefix="issue1041_"))
    n_steps = args.n_steps

    print("# issue #1041 -- distributed_v2 step order vs single device")
    print(f"# devices={devices}  n_steps={n_steps}  jax={jax.__version__}")
    print(f"# legacy rev = {args.legacy_rev}   nu ordering = "
          f"{args.nu_rev or 'working tree'}")
    print(f"# working-tree distributed_v2 = {v2_path}")
    print(f"# float32 noise threshold for 'first divergent step' = {F32_NOISE:g}")

    # ---- preflight, quoted verbatim (repo rule) ----
    print("\n## preflight (verbatim)")
    for label, bnd, sx in (("S-seam/pec", "pec", SEAM_X),
                           ("S-seam/cpml", "cpml", SEAM_X),
                           ("S-interior/pec", "pec", INTERIOR_X),
                           ("S-interior/cpml", "cpml", INTERIOR_X)):
        sim = build_sim(bnd, sx)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            issues = sim.preflight(check_ntff="advisory")
        print(f"[{label}] {len(issues)} issue(s)")
        for it in issues:
            print(f"    {it}")
        for rec in w:
            print(f"    (warning) {rec.message}")

    # ---- fixture P premise: the PEC volume refusal ----
    print("\n## fixture P premise: PEC Box straddling the seam on v2")
    run_v2_legacy = load_v2(args.legacy_rev, repo, tmp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_v2_legacy(build_sim("pec", INTERIOR_X, pec_box=True),
                          n_steps=4, devices=devices)
        print("    NOT REFUSED -- the fixture is buildable, re-read this script")
    except NotImplementedError as exc:
        print("    NotImplementedError (verbatim):")
        for line in str(exc).splitlines():
            print(f"      {line}")

    store: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    cases = [
        ("S-seam/pec", "pec", SEAM_X),
        ("S-seam/cpml", "cpml", SEAM_X),
        ("S-interior/pec", "pec", INTERIOR_X),
        ("S-interior/cpml", "cpml", INTERIOR_X),
        # Fixture P: domain-face PEC, source 8 cells from the seam. The only
        # PEC v2 applies is the domain face; with no source near the seam the
        # src/exch swap is inert, so legacy vs nu isolates PEC-face/exch-E.
        ("P-facePEC/pec", "pec", INTERIOR_X),
    ]

    for label, bnd, sx in cases:
        def builder(bnd=bnd, sx=sx):
            return build_sim(bnd, sx)

        ref = run_reference(builder, n_steps)
        arms = {
            "v2_asis (exch E -> src)": run_v2(builder, n_steps,
                                              args.legacy_rev, repo, tmp,
                                              devices),
            "v2_nu   (src -> exch E)": run_v2(builder, n_steps,
                                              args.nu_rev, repo, tmp,
                                              devices),
            "v1 (pmap)": run_v1(builder, n_steps, devices),
        }
        print(f"\n## {label}   (source x = {sx * 1e3:g} mm, "
              f"boundary={bnd}, reference = single-device uniform lane)")
        print(_peaks_line(compare(ref, ref)))
        print(HEADER)
        for name, snap in arms.items():
            row = compare(snap, ref)
            results[f"{label}|{name}"] = row
            print(f"{name:<26}{_fmt(row)}")
        # the two orderings against EACH OTHER (bit identity question)
        row = compare(arms["v2_nu   (src -> exch E)"],
                      arms["v2_asis (exch E -> src)"])
        results[f"{label}|nu_vs_asis"] = row
        print(f"{'v2_nu vs v2_asis':<26}{_fmt(row)}")

        key = label.replace("/", "_").replace("-", "_")
        store[f"{key}__ref__time_series"] = ref["time_series"]
        for f in _FIELDS:
            store[f"{key}__ref__{f}"] = ref[f]
        for name, snap in arms.items():
            tag = name.split()[0]
            store[f"{key}__{tag}__time_series"] = snap["time_series"]
            for f in _FIELDS:
                store[f"{key}__{tag}__{f}"] = snap[f]

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        npz = outdir / "issue1041_v2_step_order.npz"
        np.savez_compressed(npz, **store)
        # The .npz is ~9 MB of raw traces and is NOT committed; the summary
        # below is, so the table can be diffed without re-running.
        summary = outdir / "issue1041_v2_step_order.summary.json"
        summary.write_text(json.dumps({
            "generated": "scripts/diagnostics/issue1041_v2_step_order.py",
            "legacy_rev": args.legacy_rev,
            "nu_rev": args.nu_rev or "working tree",
            "n_steps": n_steps,
            "jax": jax.__version__,
            "devices": [str(d) for d in devices],
            "f32_noise_threshold": F32_NOISE,
            "rows": results,
        }, indent=2, sort_keys=True) + "\n")
        print(f"\n# arrays  -> {npz}")
        print(f"# summary -> {summary}")

    print("\n# columns: rel = max|arm - reference| / peak(|reference|); "
          "1stdiv = first step index where that ratio exceeds "
          f"{F32_NOISE:g} (-1 = never)")


if __name__ == "__main__":
    main()
