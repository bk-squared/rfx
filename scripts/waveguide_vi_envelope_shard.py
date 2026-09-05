"""Split a VI-envelope case list into K balanced VESSL jobs by estimated cost.

    python scripts/waveguide_vi_envelope_shard.py <cases.json> <out_dir> \
        --template <job.yaml> [--shards K] [--max-min M] [--run-dir DIR]

Why this exists: the 2026-09-04/05 campaign ran ~750 solve-minutes SERIALLY in five jobs,
and every job's wall time was set by one 60-97 min N=72 case at the end of its list. Split
per case the whole campaign would have taken about as long as its largest single case.

Cost model: wall = max(18 s, 7.35e-10 s * cells * steps), calibrated to 1.04x on the sweep
(largest case 1.00x). Cells and steps are read the same way the harness computes them, so
the estimate sees the record rule, the absorber rule and the layout the case will actually
use. Greedy longest-first bin packing; a case never splits across shards.

Each shard gets its own case file and its own YAML (the template with the stage name and
job name substituted), so `vessl run create -f` on each is the whole launch. The job names
carry the shard index and the largest case id so the queue is readable.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = os.environ.get("RFX_WT") or str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import waveguide_vi_envelope_sweep as H  # noqa: E402

SLOPE = 7.35e-10
FLOOR = 18.0


def estimate_s(case: dict) -> float:
    N = int(case["N"])
    r_lo = float(case.get("r_lo") or 0.0)
    lay = H.layout(float(case.get("layout_r_lo") or r_lo), float(case.get("domain_mult", 1.0)))
    fr = H.band_freqs(case, N)
    fc = H.FC_CONTINUOUS_HZ * H.sinc_te10(N)
    ab = H.absorber_layers(dict(case, r_lo=float(case.get("layout_r_lo") or r_lo)), lay, fc)
    npp, _ = H.num_periods_for(case, lay, fc, float(fr[0]), float(fr[-1]),
                               pad_m=ab["cpml_layers"] * H.DX_BY_N[N])
    sim, _ = H._build(case, fr, lay, ab["cpml_layers"])
    g = sim._build_grid()
    steps = int(g.num_timesteps(float(case.get("num_periods") or math.ceil(npp))))
    w = max(FLOOR, SLOPE * g.shape[0] * g.shape[1] * g.shape[2] * steps)
    if case.get("precision") == "float64":
        w *= 3.0
    return w


def pack(cases: list[dict], k: int | None, max_min: float | None) -> list[list[dict]]:
    costed = sorted(((estimate_s(c), c) for c in cases), key=lambda t: -t[0])
    if k is None:
        # enough shards that no shard exceeds max_min, never more than one per case
        total = sum(w for w, _ in costed)
        k = max(1, min(len(costed), math.ceil(total / (max_min * 60.0))))
    shards = [[] for _ in range(k)]
    load = [0.0] * k
    for w, c in costed:
        i = min(range(k), key=lambda j: load[j])
        shards[i].append(dict(c, _est_s=w))
        load[i] += w
    return [s for s in shards if s]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases")
    ap.add_argument("out_dir")
    ap.add_argument("--template", required=True, help="a working VI-envelope VESSL YAML to clone")
    ap.add_argument("--shards", type=int, default=None, help="fixed shard count (default: sized by --max-min)")
    ap.add_argument("--max-min", type=float, default=30.0, help="target max shard solve time, minutes")
    ap.add_argument("--stage-name", default=None, help="stage name inside the YAML to replace (default: auto-detect)")
    ap.add_argument("--dry", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args(argv)

    cases = json.load(open(a.cases))
    stem = Path(a.cases).stem
    shards = pack(cases, a.shards, a.max_min)
    tpl = open(a.template).read()
    stage = a.stage_name or re.search(r"for stage in (\S+); do", tpl).group(1)
    name = re.search(r"^name:\s*(\S+)", tpl, re.M).group(1)

    print(f"{len(cases)} cases -> {len(shards)} shards (target <= {a.max_min:.0f} min each)")
    out = Path(a.out_dir)
    for i, s in enumerate(shards):
        est = sum(c["_est_s"] for c in s) / 60
        big = max(s, key=lambda c: c["_est_s"])["case_id"]
        sname = f"{stem}_shard{i:02d}"
        print(f"  shard {i:02d}: {len(s):2d} cases, ~{est:5.0f} min, largest {big}")
        if a.dry:
            continue
        out.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in c.items() if k != "_est_s"} for c in s]
        json.dump(clean, open(out / f"{sname}.json", "w"), indent=1)
        y = tpl.replace(stage, sname).replace(f"name: {name}", f"name: {name}-s{i:02d}")
        open(out / f"{sname}.yaml", "w").write(y)
    if not a.dry:
        print(f"\nwrote {len(shards)} case files + YAMLs under {out}/")
        print("launch: for y in " + str(out) + "/*.yaml; do (cd /tmp && vessl run create -f $y); done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
