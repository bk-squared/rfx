"""P2, the secondary lane of study P: the paper-geometry square spiral.

Study H's 3-turn square spiral (``r_out`` 218 um, W 30 um, S 14 um, M2 3 um,
``validation/fdfd/rfic_spiral.py``'s stack, vacuum dielectrics) on the SAME
level-invariant fixture as the small spiral (walls at ``wall_w`` widths,
the lid the same distance above the stack, the physical short standard and
port gap, nested joint refinement), with a VALID uniform-current protocol:
the earlier lane used 3e6 S/m at 100 MHz, a 29 um skin depth on a 30 um
strip. Here (10 MHz, 2e6 S/m): delta = 112.5 um = 3.75 W = 37.5 t.

Blocks: ``plans`` (cuDSS plan-only estimates at ``RFX_P_PAPER_PLANS``),
``plateau`` (the sigma sweep at 10 MHz at level ``RFX_P_PAPER_PLATEAU``,
which has 4 cells across W, plus the 1 MHz / 2e7 twin with the same delta),
``levels`` (forward L_dut at ``RFX_P_PAPER_LEVELS``) and ``walls`` (level 1
at 5 / 10 / 20 W).
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _gpu_lane_lib as lib                                        # noqa: E402
from lane_p0_probe import forward, host_memory, load_il             # noqa: E402


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    rec = lib.Recorder("p2_paper.json", "P2")
    rec["env"] = lib.env_record()
    rec["host_memory_gb"] = host_memory()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    blocks = os.environ.get("RFX_P_BLOCKS", "plans walls plateau levels").split()
    deadline = rec.t0 + float(os.environ.get("RFX_P_DEADLINE_S", "4200"))
    ls.factor_cache_size(int(os.environ.get("RFX_P_FACTOR_CACHE", "1")))
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend, "reason": _cudss.unavailable_reason(backend)}
        return 3
    il = load_il()
    f0, s0 = il.PAPER_FREQ, il.PAPER_SIGMA
    rec["fixture"] = dict(il.PAPER, freq=f0, sigma=s0, skin_depth=il.skin_depth(f0, s0),
                          metal_cells_level1=1)

    if "plans" in blocks:
        plans: dict[str, Any] = {}
        for m in [int(v) for v in os.environ.get("RFX_P_PAPER_PLANS", "2 3").split()]:
            if time.time() > deadline:
                break
            try:
                t0 = time.time()
                model = il.build(il.spec_paper(m))
                d, r, c, b = lib.dut_system_at(None, model, f0, s0)
                p = _cudss.plan_estimate(d, r, c, int(b.shape[0]),
                                         m=int(b.shape[1]) if b.ndim > 1 else 1)
                p["grid"] = il.model_record(model)
                p["assembly_seconds"] = time.time() - t0
                plans[str(m)] = p
                print(f"  paper plan m={m}: N={p['grid']['n_unknowns']} perm "
                      f"{p.get('permanent_device_memory_gb', float('nan')):.2f} GB", flush=True)
                del model, d, r, c, b
            except Exception as exc:
                plans[str(m)] = {"error": f"{type(exc).__name__}: {exc}"}
            rec["plans"] = plans

    if "walls" in blocks and time.time() < deadline:
        walls: dict[str, Any] = {"level": 1, "runs": {}}
        for w in (5.0, 10.0, 20.0):
            try:
                model = il.build(il.spec_paper(1, wall_w=w))
                r = forward(il, model, backend, sigma=s0, freq=f0)
                r["grid"] = il.model_record(model)
                walls["runs"][f"{w:g}"] = r
                print(f"  paper walls {w:g} W: L={r['L_dut'] * 1e12:.2f} pH", flush=True)
            except Exception as exc:
                walls["runs"][f"{w:g}"] = {"error": f"{type(exc).__name__}: {exc}"}
            rec["walls"] = walls

    if "plateau" in blocks and time.time() < deadline:
        lvl = int(os.environ.get("RFX_P_PAPER_PLATEAU", "2"))
        pl: dict[str, Any] = {"level": lvl, "runs": []}
        rec["plateau"] = pl
        try:
            model = il.build(il.spec_paper(lvl))
            pl["grid"] = il.model_record(model)
            cases = [(f0, s) for s in il.PAPER_PLATEAU_SIGMAS] + [
                (il.PAPER_FREQ_RC, il.PAPER_SIGMA_RC)]
            for f, s in cases:
                if time.time() > deadline:
                    break
                r = forward(il, model, backend, sigma=s, freq=f)
                r["skin_depth"] = il.skin_depth(f, s)
                r["skin_over_width"] = r["skin_depth"] / il.PAPER["width"]
                pl["runs"].append(r)
                print(f"  paper plateau f={f:.0e} sigma={s:.0e} delta/W="
                      f"{r['skin_over_width']:.2f} L={r['L_dut'] * 1e12:.2f} pH", flush=True)
                rec["plateau"] = pl
            del model
        except Exception as exc:
            pl["error"] = f"{type(exc).__name__}: {exc}"
        rec["plateau"] = pl

    if "levels" in blocks:
        lv: dict[str, Any] = {}
        rec["levels"] = lv
        for m in [int(v) for v in os.environ.get("RFX_P_PAPER_LEVELS", "1 2").split()]:
            if time.time() > deadline:
                break
            try:
                model = il.build(il.spec_paper(m))
                r = forward(il, model, backend, sigma=s0, freq=f0)
                r["grid"] = il.model_record(model)
                r["m"] = m
                lv[str(m)] = r
                print(f"  paper level {m}: N={r['grid']['n_unknowns']} "
                      f"L={r['L_dut'] * 1e12:.2f} pH ({r['seconds']:.0f} s)", flush=True)
                del model
            except Exception as exc:
                lv[str(m)] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  paper level {m} FAILED: {exc}", flush=True)
            rec["levels"] = lv
    rec.dump()
    print(f"P2 done in {rec['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
