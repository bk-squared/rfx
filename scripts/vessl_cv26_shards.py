#!/usr/bin/env python3
"""Emit one VESSL YAML per cv26 shard, and the submission order.

cv26 round 1 ran the whole case in ONE job and took 21 hours; the wall clock was the
sum of thirty arms rather than the slowest one. This emits the same work as independent
jobs -- one per arm or small arm group -- so a failure names one group and the wall
clock is the largest shard. Every shard writes into ONE shared results directory on
NFS, and ``scripts/crossval/merge_cv26_arm_shards.py`` assembles the baseline arms into
the single ``rfx.json`` the gate replay reads.

    python scripts/vessl_cv26_shards.py --lane <label> --checkout <nfs path> --out-dir <dir>

Waves (the emitter prints them; the submitter runs them in order):
  1  meep-*     the six Meep primaries and the wrong-2pi falsifier leg
  2  base-*     the ten baseline arms, in seven shards, reading the Meep JSONs
     fals-*     the six pre-declared rfx falsifiers (each MUST exit 1)
     ladder-*   the absorber depth rungs and the dx rungs (evidence, rc 0/1/2 admissible)

Rules obeyed here, from _configs/.claude/rules/vessl-jobs.md: sh (no bashisms), no
heredocs in the run block, provenance that aborts rather than writing a placeholder,
and submission through ``scripts/vessl_submit.sh`` so the run id lands beside the
artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys

MEEP_ARMS = ("te_00", "te_30", "te_45", "te_60", "tm_45", "tm_60")

# (shard, wave, kind, cpu, memory Gi, [commands])
# a command is a list: ["rfx", <name>, <args...>] or ["meep", <name>, <args...>]
SHARDS = [
    # ---- wave 1: the Meep legs. Each builds its own conda env; one arm per job so a
    #      leg that rejects its own output names itself.
    *[(f"meep-{a.replace('_', '')}", 1, "meep", 8, 16, [["meep", a, "--arm", a]]) for a in MEEP_ARMS],
    ("meep-k2pi", 1, "meep", 8, 16, [["meep", "te_45__falsifier_k_2pi", "--arm", "te_45", "--falsifier", "k_2pi"]]),

    # ---- wave 2: the ten baseline arms at their declared recipes. te_00 and tm_00 run
    #      at dx (four times cheaper) so they share a shard; the three compact-box arms
    #      are a small grid and share one.
    ("base-te00tm00", 2, "rfx", 8, 16, [["rfx", "shard_te00tm00", "--arm", "te_00,tm_00", "--tag", "shard_te00tm00"]]),
    ("base-te30", 2, "rfx", 8, 16, [["rfx", "shard_te30", "--arm", "te_30", "--tag", "shard_te30"]]),
    ("base-te45", 2, "rfx", 8, 16, [["rfx", "shard_te45", "--arm", "te_45", "--tag", "shard_te45"]]),
    ("base-te60", 2, "rfx", 8, 16, [["rfx", "shard_te60", "--arm", "te_60", "--tag", "shard_te60"]]),
    ("base-tm45", 2, "rfx", 8, 16, [["rfx", "shard_tm45", "--arm", "tm_45", "--tag", "shard_tm45"]]),
    ("base-tm60", 2, "rfx", 8, 16, [["rfx", "shard_tm60", "--arm", "tm_60", "--tag", "shard_tm60"]]),
    ("base-graze", 2, "rfx", 8, 16, [["rfx", "shard_graze", "--arm", "graze_vac,graze_pec,graze_te",
                                      "--tag", "shard_graze"]]),

    # ---- wave 2: the pre-declared falsifiers. Each must exit 1; the shard says so.
    ("fals-te60-angle", 2, "rfx-f1", 8, 16, [["rfx", "falsifier_te_60_angle_m5", "--falsifier", "te_60_angle_m5"]]),
    ("fals-te45-swap", 2, "rfx-f1", 8, 16, [["rfx", "falsifier_te_45_swap_tm", "--falsifier", "te_45_swap_tm"]]),
    ("fals-tm60-swap", 2, "rfx-f1", 8, 16, [["rfx", "falsifier_tm_60_swap_te", "--falsifier", "tm_60_swap_te"]]),
    ("fals-te45-eps", 2, "rfx-f1", 8, 16, [["rfx", "falsifier_te_45_eps_x1p2", "--falsifier", "te_45_eps_x1p2"]]),
    ("fals-graze", 2, "rfx-f1", 8, 16, [
        ["rfx", "falsifier_graze_pec_depth_half", "--falsifier", "graze_pec_depth_half"],
        ["rfx", "falsifier_graze_pec_sigma_half", "--falsifier", "graze_pec_sigma_half"]]),
    ("fals-meep-k2pi", 2, "rfx-f1", 8, 16, [["rfx", "falsifier_meep_te_45_k_2pi", "--falsifier", "meep_te_45_k_2pi"]]),

    # ---- wave 2: evidence rungs. rc 0/1/2 are all admissible here (note section 13.5:
    #      the dx rungs carry the absorber-echo measurement that PICKS dx/2, so several
    #      are pre-declared over W_bin), so the shard does not gate them.
    ("ladder-depth", 2, "rfx-e", 8, 16, [
        ["rfx", f"graze_pec_d{d}", "--arm", "graze_pec", "--n-cpml", str(d), "--tag", f"graze_pec_d{d}"]
        for d in (8, 16, 32)]),
    ("ladder-dx", 2, "rfx-e", 8, 16, [
        ["rfx", f"{a}_dx1", "--arm", a, "--dx-div", "1", "--tag", f"{a}_dx1"]
        for a in ("te_30", "te_45", "te_60", "tm_45", "tm_60")]),
]

HEADER = """name: {name}
description: "cv26 round 3 (r2 rebase), shard {shard} of the sharded oblique-Fresnel lane. {what} Lane {lane}: every shard writes into the ONE shared results directory {res}, and scripts/crossval/merge_cv26_arm_shards.py assembles the baseline arms into rfx.json. Round 1 ran this case as a single serial job and took 21 h; the wall clock here is the largest shard. Pre-declaration docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md (section 18 carries the PI decision of 2026-09-13 on issue #905: G3_passivity and G3_closure are N/A on the two compact grazing arms, which are judged on G6/G7 and the tail witness). Submit: scripts/vessl_submit.sh <this yaml> {prefix}"
tags: [rfx, crossval, cv26, oblique, fresnel, cpml, round-3, shard-{shard}]
resources:
  cluster: remilab-c0
  cpu: {cpu}
  memory: {mem}Gi
image: ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8
env:
  DEBIAN_FRONTEND: noninteractive
  PYTHONUNBUFFERED: "1"
  OMP_NUM_THREADS: "4"
  HDF5_USE_FILE_LOCKING: "FALSE"
  LANG: "C.UTF-8"
  JAX_PLATFORMS: cpu
  RFX_CHECKOUT_NFS: {checkout}
  RFX_RUNS_ROOT: /root/workspace/claude-workspace/rfx/runs
  CV26_RES: {res}
mount:
  /root/workspace/: volume://remilab-fs/personal-workspaces/
run: |-
  set -eux
  echo "=== cv26 round 3 shard {shard} ==="
  date
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  OUT="$RFX_RUNS_ROOT/{prefix}$STAMP"
  mkdir -p "$OUT"
  trap 'chmod -R a+rX "$OUT" 2>/dev/null || true' EXIT

  # -- stage the checkout; provenance ABORTS rather than writing a placeholder --
  test -d "$RFX_CHECKOUT_NFS" || {{ echo "ERROR: checkout missing at $RFX_CHECKOUT_NFS"; exit 3; }}
  WORK=/tmp/rfx-cv26-{shard}
  rm -rf "$WORK"; mkdir -p "$WORK"
  cp -a "$RFX_CHECKOUT_NFS/." "$WORK/"
  cd "$WORK" || exit 3
  test -s "$WORK/.staged_commit" || {{ echo "ERROR: no .staged_commit in the staged checkout"; exit 3; }}
  cat "$WORK/.staged_commit" > "$OUT/commit.txt"
  echo "commit: $(cat "$OUT/commit.txt")"
  echo "{shard}" > "$OUT/shard.txt"
  RES="$CV26_RES"
  mkdir -p "$RES"
  chmod a+rwX "$RES" 2>/dev/null || true
"""

RFX_ENV = """
  # -- rfx environment --
  PY=python
  command -v "$PY" >/dev/null 2>&1 || PY=/opt/conda/bin/python
  "$PY" -m pip install -q "jax[cpu]==0.6.2" "numpy<2" "scipy>=1.11" "h5py>=3.8" "matplotlib>=3.7" "pytest>=7"
  export RFX_REPO_ROOT="$WORK"
  export PYTHONPATH="$WORK"
  "$PY" -c "import jax, rfx; print('rfx probe ok', jax.__version__, rfx.__file__)"

  run_rfx() {
    NAME=$1; shift
    set +e
    "$PY" validation/crossval/26_oblique_slab_fresnel.py "$@" --out-dir "$RES" --meep-dir "$RES" --no-plots \\
      > "$OUT/rfx_$NAME.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$OUT/rfx_$NAME.rc"
    grep -E "^\\[|  record|  aux-echo|  E2 |  gates|  verdict rule|  E4|  G6|  G7|  leakage|  Brewster|  lattice|FDTD|artifact|PASSED|FAIL|SKIP" "$OUT/rfx_$NAME.log" | tail -n 40 || true
    echo "rfx_${NAME}_rc=$rc"
  }
"""

MEEP_ENV = """
  # -- Meep environment (conda-forge pymeep, its own env; the recipe cv22/cv23/cv26 use) --
  CONDA=/opt/conda/bin/conda
  if [ ! -x "$CONDA" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    sh /tmp/miniconda.sh -b -p /opt/miniconda-cv26
    CONDA=/opt/miniconda-cv26/bin/conda
  fi
  "$CONDA" create -y -q -n cv26meep --override-channels -c conda-forge python=3.11 pymeep numpy scipy
  MEEP_PY="$(dirname "$CONDA")/../envs/cv26meep/bin/python"
  test -x "$MEEP_PY" || MEEP_PY="$("$CONDA" run -n cv26meep which python)"
  "$MEEP_PY" -c "import meep, numpy; print('meep', meep.__version__, 'numpy', numpy.__version__)"

  run_meep() {
    NAME=$1; shift
    set +e
    "$MEEP_PY" scripts/crossval/meep_cv26_oblique_slab.py "$@" --out-dir "$RES" > "$OUT/meep_$NAME.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$OUT/meep_$NAME.rc"; tail -n 8 "$OUT/meep_$NAME.log" || true
    echo "meep_${NAME}_rc=$rc"
  }
"""

VERDICTS = {
    # kind -> (per-command expected rc test, closing text)
    "meep": ('[ "$rc" -eq 0 ] || {{ echo "ERROR: Meep leg {name} exited $rc"; fail=1; }}',
             "the leg must exit 0: a rejected reference writes a rejection record and exits 1"),
    "rfx": ('[ "$rc" -eq 0 ] || {{ echo "ERROR: baseline shard command {name} exited $rc"; fail=1; }}',
            "a baseline shard must exit 0 on every arm it owns"),
    "rfx-f1": ('[ "$rc" -eq 1 ] || {{ echo "ERROR: falsifier {name} exited $rc, expected 1"; fail=1; }}',
               "a pre-declared falsifier MUST exit 1"),
    "rfx-e": ('case "$rc" in 0|1|2) ;; *) echo "ERROR: rung {name} exited $rc"; fail=1;; esac',
              "evidence rungs: rc 0/1/2 admissible, anything else is a crash"),
}


def emit(shard, wave, kind, cpu, mem, cmds, *, lane, checkout, res):
    prefix = f"cv26r3-{shard}-"
    what = VERDICTS[kind][1]
    body = HEADER.format(name=f"rfx-cv26r3-{shard}", shard=shard, what=what, lane=lane, res=res,
                         prefix=prefix, cpu=cpu, mem=mem, checkout=checkout)
    body += MEEP_ENV if kind == "meep" else RFX_ENV
    body += "\n  fail=0\n"
    for c in cmds:
        runner = "run_meep" if c[0] == "meep" else "run_rfx"
        body += f'  {runner} {c[1]} {" ".join(c[2:])}\n'
        body += f'  rc=$(cat "$OUT/{"meep" if c[0] == "meep" else "rfx"}_{c[1]}.rc")\n'
        body += "  " + VERDICTS[kind][0].format(name=c[1]) + "\n"
    body += '''
  ls -la "$RES" | tail -n 40
  chmod -R a+rX "$OUT" 2>/dev/null || true
  chmod -R a+rwX "$RES" 2>/dev/null || true
  echo "=== shard verdict ==="
  echo "shard=''' + shard + '''  fail=$fail  results=$RES"
  exit "$fail"
'''
    return prefix, body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True, help="lane label, e.g. 20260913a")
    ap.add_argument("--checkout", required=True, help="NFS path of the staged checkout")
    ap.add_argument("--out-dir", required=True, help="where to write the YAMLs")
    ap.add_argument("--runs-root", default="/root/workspace/claude-workspace/rfx/runs")
    a = ap.parse_args(argv)
    res = f"{a.runs_root}/cv26r3res-{a.lane}"
    os.makedirs(a.out_dir, exist_ok=True)
    plan = []
    for shard, wave, kind, cpu, mem, cmds in SHARDS:
        prefix, body = emit(shard, wave, kind, cpu, mem, cmds, lane=a.lane, checkout=a.checkout, res=res)
        path = os.path.join(a.out_dir, f"cv26r3_{shard}.yaml")
        with open(path, "w") as fh:
            fh.write(body)
        plan.append((wave, shard, path, prefix))
    print(f"shared results directory: {res}")
    for wave in sorted({w for w, *_ in plan}):
        print(f"\n--- wave {wave} ---")
        for w, shard, path, prefix in plan:
            if w == wave:
                print(f"scripts/vessl_submit.sh {path} {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
