"""Compare the paper builders' observables and AD with and without probes.

Same geometry, fields, extraction and input per arm. The only mutation is
removing the user probes added by #918. This tests diagnostic passivity and
AD preservation, not RF accuracy or completion of the inverse-design search.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from validation.tmtt_paper._settling_record import retain_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["beam", "msl"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.case == "beam":
        from validation.tmtt_paper import beam_steering_superstrate as paper
        sim, region, grid, plate, _, _ = paper.build_problem()
        lo, hi, shape = paper.resolve_design_indices(region, grid)
        n_steps = int(grid.num_timesteps(num_periods=paper.NUM_PERIODS))
        ramp = np.linspace(2., 9., shape[0], dtype=np.float32)
        parameter = jnp.asarray(np.broadcast_to(ramp[:, None, None], shape).copy())

        def factory(candidate):
            return paper.make_pattern_fn(candidate, grid, plate, lo, hi, n_steps)[0]

        def objective(value):
            return jnp.sum(value * paper._W)

    else:
        from validation.tmtt_paper import msl_stub_notch_tuning as paper
        freqs = jnp.asarray([paper.F_TARGET], dtype=jnp.float32)
        sim, _, trace_y_hi, driven, passive = paper.build_sim(freqs)
        grid = sim._build_grid()
        raw_steps = math.ceil(10. / (sim._freq_max * float(grid.dt)))
        segments = max(8, math.isqrt(raw_steps))
        n_steps = math.ceil(raw_steps / segments) * segments
        parameter = jnp.asarray(6.5e-3, dtype=jnp.float32)

        def factory(candidate):
            return paper.make_s21_fn(candidate, grid, trace_y_hi, driven, passive,
                                     freqs, n_steps=n_steps, checkpoint_segments=segments)

        def objective(value):
            return jnp.abs(value)**2

    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "case": args.case, "jax": jax.__version__, "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "smoke": os.environ.get("SMOKE", "1") if args.case == "beam" else None,
        "grid_shape": list(grid.shape), "n_steps": n_steps,
        "parameter": np.asarray(parameter).tolist(),
        "ad_invocation": "jax.value_and_grad, matching both paper scripts; outer jit is not used",
        "scope": "point-probe instrumentation identity on the real paper forward/extractor; no optimization or RF accuracy claim",
    }
    (args.out / "plan.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "parameter"}), flush=True)
    if args.plan:
        return 0
    arms = []
    for label, include_probes in (("without_probes", False), ("with_probes", True)):
        candidate = copy.copy(sim)
        candidate._probes = list(sim._probes) if include_probes else []
        evaluate = factory(candidate)
        print(f"{args.case} {label}: forward", flush=True)
        value, result = evaluate(parameter, return_result=True)
        value = np.asarray(value)
        witness = retain_observation(args.out, label, result, value)
        del result
        print(f"{args.case} {label}: value_and_grad (paper invocation)", flush=True)
        def cost_fn(p, evaluate=evaluate):
            return objective(evaluate(p))

        cost, gradient = jax.value_and_grad(cost_fn)(parameter)
        cost, gradient = np.asarray(cost), np.asarray(gradient)
        np.savez_compressed(args.out / f"{label}_ad.npz", cost=cost, gradient=gradient)
        arms.append((value, cost, gradient, witness))
        print(f"{args.case} {label}: cost={cost}, |grad|max={np.max(np.abs(gradient))}", flush=True)
        del cost_fn, evaluate, candidate, gradient, cost
        jax.clear_caches()
        gc.collect()
    comparisons = {}
    for i, key in enumerate(("observable", "cost", "gradient")):
        before, after = arms[0][i], arms[1][i]
        comparisons[key] = {
            "bit_identical": bool(np.array_equal(before, after)),
            "finite": bool(np.isfinite(before).all() and np.isfinite(after).all()),
            "max_abs_difference": float(np.max(np.abs(before - after))),
        }
    comparisons["gradient_nonzero"] = bool(np.any(arms[1][2] != 0))
    comparisons["witness"] = arms[1][3]
    comparisons["probe_coverage_verified"] = (
        arms[0][3]["settling_witness"]["status"] == "absent"
        and arms[1][3]["settling_witness"]["status"] == "measured")
    passed = all(comparisons[k]["bit_identical"] and comparisons[k]["finite"]
                 for k in ("observable", "cost", "gradient"))
    passed = passed and comparisons["gradient_nonzero"] and comparisons["probe_coverage_verified"]
    comparisons["identity_passed"] = passed
    (args.out / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(comparisons), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
