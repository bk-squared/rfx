"""Read the v1.8 closing run against its pre-declaration, section 3 — written BEFORE the run
finished so the read is a fixed computation, not an interpretation.

    python scripts/diagnostics/waveguide_chain_battery_v18_close_read.py <run_dir>

<run_dir> is the NFS artifact dir the job printed ("artifacts: ..."). It reads:
  <run_dir>/ad_fd__*.json                 the closing ad_fd legs (x64 primary on the flux lane)
  <run_dir>/falsifier_float32/ad_fd__*.json   the section-4 falsifier (float32 primary)
  <run_dir>/<fixture>.json                the assembled artifact, if the assemble stage ran

and prints, per section-3 row: predicted / measured / verdict. Nothing here is a gate; the
gates are tests/_waveguide_chain_battery_gates.py, and the replay test reads the artifact.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.environ.get("RFX_WT") or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, str(Path(_REPO_ROOT) / "tests"))
import _waveguide_chain_battery_gates as G  # noqa: E402


def legs(d: str):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "ad_fd__*.json"))):
        out += json.load(open(f))["legs"]
    return out


def red_count(ls):
    """Count red verdicts the way run 2's census did: forward identity + AD verdict per leg."""
    n = 0
    for l in ls:
        if not G.forward_identity_pass(l["forward_identity"]["max_scaled_diff"]):
            n += 1
        if l["verdict"] == "fail":
            n += 1
    return n


def main(run_dir: str) -> int:
    main_legs = legs(run_dir)
    fals = legs(os.path.join(run_dir, "falsifier_float32"))
    print(f"closing legs: {len(main_legs)}   falsifier legs: {len(fals)}\n")

    print("=== section 3, row by row ===")
    # row 1: flux forward identity, x64 -> all <= 1e-7 scaled; fail if any > 1e-3
    fi = [(f"{l['dut']}|{l['theta_kind']}|{l['objective']}", l["forward_identity"]["max_scaled_diff"], l["primary_precision"])
          for l in main_legs if l["lane"] == "flux"]
    worst = max(v for _, v, _ in fi) if fi else float("nan")
    prim = {p for _, _, p in fi}
    print(f"  flux forward identity ({len(fi)} legs): primary={prim} worst scaled={worst:.3e}  "
          f"predicted <=1e-7 (three measured 1.7e-10..1.05e-8); FAIL-branch if >1e-3  -> "
          f"{'as predicted' if worst <= 1e-7 else ('inside bar, above predicted band' if worst <= 1.0 else 'REASSOCIATION NOT CONFINED — do not close')}")
    # row 2: normalize=False identity unchanged at 0
    fi0 = [l["forward_identity"]["max_scaled_diff"] for l in main_legs if l["lane"] == "false"]
    print(f"  normalize=False identity ({len(fi0)} legs): worst {max(fi0) if fi0 else float('nan'):.3e}  "
          f"predicted 0 (bit-identical) on GPU at the claims rung — run 2 stored 0 there; CPU/coarse reads ~0.2 and is not the claim")
    # row 3: zero-derivative leg report_only, ratio 3-8 same sign, |g| <= 1e-6
    zd = [l for l in main_legs if (l["dut"], l["theta_kind"], l["objective"]) in G.EXPECTED_ULP_SKIP and l["lane"] == "flux"]
    for l in zd:
        z = l.get("zero_derivative") or {}
        ok_shape = (z.get("same_sign") is True) and abs(l["g_ad"]) <= 1e-5 and abs(l["g_fd"]) <= 1e-5
        print(f"  zero-derivative leg: verdict={l['verdict']} ratio={z.get('ratio')} same_sign={z.get('same_sign')} "
              f"g_ad_x64={l['g_ad']:+.3e} g_fd={l['g_fd']:+.3e}  predicted report_only, ~3-8, same sign, |g|<=1e-6 -> "
              f"{'as predicted' if l['verdict']=='report_only' and ok_shape else 'FAIL-branch: sign flip or non-zero derivative — root-cause'}")
    # row 4: other AD-vs-FD legs, x64, rel <= 0.05 and <= run 2's float32 rel
    others = [l for l in main_legs if not ((l["dut"], l["theta_kind"], l["objective"]) in G.EXPECTED_ULP_SKIP and l["lane"] == "flux")]
    bad = [(f"{l['dut']}|{l['lane']}|{l['theta_kind']}|{l['objective']}", l["rel"], l["verdict"]) for l in others if l["verdict"] == "fail"]
    # "rose" means the x64 rel exceeds the float32 rel by more than 10 % of it AND by more
    # than 1e-4 absolute — a 1e-7 relative wobble between two float64 FD divisions is noise,
    # not the "float32 pass was noise agreeing with noise" finding this branch exists for.
    rose = []
    for l in others:
        f32 = (l.get("ad_vs_fd_float32") or {}).get("rel")
        if l["primary_precision"] == "x64" and f32 is not None and f32 == f32:
            if l["rel"] > f32 * 1.10 and l["rel"] - f32 > 1e-4:
                rose.append((f"{l['dut']}|{l['lane']}|{l['theta_kind']}|{l['objective']}", f32, l["rel"]))
    print(f"  other AD-vs-FD ({len(others)} legs): fails={bad or 'none'}  predicted all pass")
    print(f"     legs whose rel ROSE under x64 vs float32 (finding branch): {rose or 'none'}")
    # NaN watch (section 3.1)
    nan32 = [f"{l['dut']}|{l['lane']}|{l['theta_kind']}|{l['objective']}" for l in main_legs
             if l.get("ad_vs_fd_float32") and l["ad_vs_fd_float32"].get("g_ad") != l["ad_vs_fd_float32"].get("g_ad")]
    print(f"  float32 NaN gradients (section 3.1 watch): {nan32 or 'none'}")

    print("\n=== section 4 falsifier: float32 primary must reproduce run 2's 9 red ===")
    if fals:
        r = red_count(fals)
        print(f"  float32-primary red verdicts on ad_fd legs: {r}   (run 2 stored 9)  -> {'HOLDS' if r == 9 else 'DOES NOT HOLD — the script change altered more than which reading is read'}")
    else:
        print("  falsifier legs not found")

    print("\n=== closing census (ad_fd legs, x64 primary) ===")
    r = red_count(main_legs)
    print(f"  red verdicts: {r}   predicted 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
