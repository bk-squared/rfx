"""Capture actual cv02/cv24 estimator inputs without replacing repo evidence.

Run against a disposable checkout of the baseline commit. cv02 executes its
existing Meep-absent record ladder; cv24 runs the chosen full-length arm and
captures its nine channel/window inputs, skipping estimator/energy work.
cv24 capture is NOT a crossval verdict. Every NPZ is reusable by both old
and new estimators, so a comparison never conflates changes in FDTD fields
with changes in extraction. ``--out`` must name a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import runpy
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True, help="disposable baseline checkout")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--case", choices=["cv02", "cv24-uniform", "cv24-single_band"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.is_relative_to(repo):
        ap.error("capture output must be outside the disposable source checkout")
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    # cv02 pins x64 before importing JAX; cv24 uses the default float32 lane.
    os.environ["JAX_ENABLE_X64"] = "1" if args.case == "cv02" else "0"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

    import numpy as np
    import scipy
    import jax
    import rfx
    assert Path(rfx.__file__).resolve().is_relative_to(repo)
    module = importlib.import_module("rfx.harminv")
    original = module.harminv
    records = []

    def capture(signal, dt, f_min, f_max, **kwargs):
        signal = np.asarray(signal)
        name = f"input-{len(records):02d}.npz"
        with (out / name).open("xb") as stream:
            np.savez_compressed(stream, signal=signal)
        record = dict(file=name, dt=float(dt), f_min=float(f_min), f_max=float(f_max),
                      kwargs=kwargs, n_samples=int(signal.size), dtype=str(signal.dtype),
                      sha256=hashlib.sha256((out / name).read_bytes()).hexdigest())
        records.append(record)
        print(f"captured {name}: {signal.size} samples", flush=True)
        if args.case == "cv02":
            result = original(signal, dt, f_min, f_max, **kwargs)
            record["baseline_modes"] = [m._asdict() for m in result]
            return result
        return []  # Capture only; no numerical verdict is constructed here.

    module.harminv = capture
    exit_code, arm = None, None
    try:
        if args.case == "cv02":
            # Explicitly select the rfx-only ladder; never launch Meep by
            # accident on a workstation where it happens to be installed.
            sys.modules["meep"] = None
            try:
                runpy.run_path(str(repo / "validation/crossval/02_ring_resonator.py"),
                               run_name="__main__")
            except SystemExit as exc:
                exit_code = exc.code
        else:
            path = repo / "validation/crossval/24_nu_rect_cavity_pozar.py"
            spec = importlib.util.spec_from_file_location("_cv24_capture", path)
            case = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(case)
            name = args.case.removeprefix("cv24-")
            rig = case.G.ARMS[name]
            arm = case.run_arm(name, rig["lane"], rig["dx"], case.G.PROFILES[rig["profile"]],
                               smoke=False, search_band_hz=case.G.BAND_HZ, with_energy=False)
    finally:
        module.harminv = original
        summary = dict(case=args.case, source_commit=args.commit,
                       source_harminv_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                       jax=jax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64),
                       records=records, case_exit_code=exit_code, arm=arm,
                       note="cv24 measured fields are empty because this is input capture, not a verdict")
        (out / "capture.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
