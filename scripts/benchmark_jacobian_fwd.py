#!/usr/bin/env python3
"""Regenerable cost-characterization benchmark for
`rfx.observables.jacobian_fwd` (issue #577).

The repo forbids publishing prose cost numbers (they rot) and forbids
calling a compiler estimate a "profile" -- see
`tests/test_estimate_ad_memory.py`'s forbidden-terms list, mirrored here.
This script is the regenerable source of truth: run it, quote its printed
table (with the grid size / n_steps / backend / dtype it prints in its own
header) in the PR body and CHANGELOG, and re-run it before trusting a
number that is more than a few commits old.

ALL numbers from this script are CPU unless you explicitly run it on a
GPU-backed JAX install -- it prints `jax.default_backend()` in its header
precisely so a copied number carries its own backend label. No GPU claim
is made or implied by anything in this file.

Measures, for `n_t` in `--n-t-values` (default 1, 4, 10) against one plain
`sim.forward()` call:
    - wall time (median of `--reps` timed calls on an already-compiled
      program)
    - flops (`compiled.cost_analysis()` -- a COMPILER ESTIMATE of the scan
      BODY's per-step cost, not a measured runtime profile)
    - XLA temp bytes (`compiled.memory_analysis().temp_size_in_bytes` -- a
      COMPILER ESTIMATE, not a measured/observed/certified runtime peak;
      see rfx/api/__init__.py's `ad_memory_compiled_certificate` for the
      one sanctioned way to attach a measured-memory number to a compiled
      object, which this script does not attempt)

and fits `cost(n_t) = a + b * n_t` through the first and last requested
`n_t` -- the INTERCEPT `a` is a DIAGNOSTIC primal-sharing witness (it
should land near one plain solve on CPU, where it reads ~0.98-1.10) and
the SLOPE `b` is the marginal per-tangent cost. The printed
intercept/plain ratio is CPU-calibrated: a two-point fit pushes any fixed
per-call overhead that does not scale with `n_t` (kernel launch, tiling)
into the intercept, and that effect is largest where the marginal cost
per tangent is smallest -- on an RTX 4090 the same ratio reads 1.4-1.9x
across grid sizes (issue #632) with sharing intact. Do not treat this
ratio as a regression signal on an accelerator; see G3 in
tests/test_jacobian_fwd.py for the authoritative, backend-independent
structural check (it inspects the jaxpr directly for one unbatched
primal scan rather than fitting wall-clock/flops numbers).

Fixture note (issue #577 cost-measurement discipline): the fixture registers
NO point probes (no `add_probe`). The uniform forward lane cannot drop the
probe time-series tape (`emit_time_series=False` raises there), and under
`vmap(jvp)` that tape would batch to `(n_t, n_steps, n_probes)` -- with
even one point probe registered, that batched time-series tape would
dominate the measured memory and turn the number into a tape artifact
instead of a tangent-memory measurement. Only a DFT-plane probe is
registered (a fixed-size frequency-domain accumulator, not a per-step
time series).

Run:
    python scripts/benchmark_jacobian_fwd.py
    python scripts/benchmark_jacobian_fwd.py --n-p 10 --n-t-values 1 4 10 \
        --n-steps 120 --grid-scale 24   # the "large" fixture, slower
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from rfx import Simulation
from rfx.observables import dft_field, jacobian_fwd

_DX = 1.0e-3
# FORBIDDEN honesty-label terms (tests/test_estimate_ad_memory.py's
# _FORBIDDEN_RECOMMENDATION_TERMS / _FORBIDDEN_CURRENT_EVIDENCE_FIELDS):
# never call this a "profile", "runtime peak", "guaranteed", or
# "certified" number, and never name a field observed_peak_gb /
# profile_peak_gb / peak_bound_gb. This script's field names below are
# deliberately "flops" / "temp_bytes" / "t_median_s" -- the compiler's own
# vocabulary, not a claim about measured/observed/certified behaviour.


def build_sim(*, nx: int = 24, ny: int = 20, nz: int = 20,
              cpml_layers: int = 5, freq_max: float = 8.0e9) -> Simulation:
    """No point probes registered -- see the module docstring's Fixture
    note. Only a source and one DFT-plane probe."""
    sim = Simulation(
        freq_max=freq_max, domain=(nx * _DX, ny * _DX, nz * _DX), dx=_DX,
        boundary="cpml", cpml_layers=cpml_layers,
    )
    sim.add_source(
        position=(4 * _DX, (ny / 2) * _DX, (nz / 2) * _DX),
        component="ez", amplitude_kind="current",
    )
    sim.add_dft_plane_probe(
        axis="x", coordinate=(nx - 5) * _DX, component="ez",
        freqs=jnp.asarray([5.0e9]), name="leak",
    )
    return sim


def make_design_fn(sim: Simulation, n_p: int, *, n_steps: int):
    """params: flat (n_p,) float32 array, one design x-slice per entry ->
    dft_field tensor. Matches the #577 evidence-base fixture convention
    (a flat array, not a pytree-of-scalars) so `tangents=jnp.eye(n_p)`
    works directly as the explicit-tangent escape hatch."""
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    x0 = grid.position_to_index((8 * _DX, 0.0, 0.0))[0]

    def sim_fn(p):
        eps = eps_base
        for i in range(n_p):
            eps = eps.at[x0 + i, :, :].set(1.0 + p[i])
        result = sim.forward(
            eps_override=eps, n_steps=n_steps, checkpoint=False,
            skip_preflight=True,
        )
        return dft_field("leak")(result)

    return grid, sim_fn


def _compile_and_measure(fn, args, *, reps: int) -> dict[str, Any]:
    jitted = jax.jit(fn)
    compiled = jitted.lower(*args).compile()

    info: dict[str, Any] = {}
    try:
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        info["flops"] = ca.get("flops") if ca else None
    except Exception:  # pragma: no cover -- backend-dependent availability
        info["flops"] = None
    try:
        ma = compiled.memory_analysis()
        info["temp_bytes"] = getattr(ma, "temp_size_in_bytes", None)
    except Exception:  # pragma: no cover -- backend-dependent availability
        info["temp_bytes"] = None

    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = compiled(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    info["t_median_s"] = times[len(times) // 2]
    info["t_min_s"] = times[0]
    return info


def _linear_fit_endpoints(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Fit cost(x) = a + b*x through the FIRST and LAST (x, y) points --
    matches the evidence base's fit convention (bench.py/tables.py),
    robust to a possibly-noisy interior point without needing a full
    least-squares dependency."""
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    b = (y1 - y0) / (x1 - x0)
    a = y0 - b * x0
    return a, b


def build_benchmark_table(
    *,
    n_p: int = 10,
    n_t_values: tuple[int, ...] = (1, 4, 10),
    n_steps: int = 120,
    grid_scale: int = 0,
    reps: int = 5,
) -> dict[str, Any]:
    """Return a JSON-serializable table: plain baseline, jvp (batched) at
    each n_t, jvp (sequential, batch_tangents=False) at the largest n_t,
    the linear fit, and a second n_steps run for the memory-independence
    witness. All CPU/GPU-labelled by `jax.default_backend()`."""
    nx = 24 + grid_scale
    ny = 20 + grid_scale
    nz = 20 + grid_scale
    sim = build_sim(nx=nx, ny=ny, nz=nz)
    grid, sim_fn = make_design_fn(sim, n_p, n_steps=n_steps)
    params = jnp.full((n_p,), 1.0, dtype=jnp.float32)

    def plain_fn(p, t):
        return sim_fn(p)

    def jvp_fn(p, t):
        return jacobian_fwd(sim_fn, p, tangents=t, batch_tangents=True)

    def jvp_seq_fn(p, t):
        return jacobian_fwd(sim_fn, p, tangents=t, batch_tangents=False)

    rows = []
    T_full = jnp.eye(n_p, dtype=jnp.float32)
    plain_info = _compile_and_measure(plain_fn, (params, T_full[:1]), reps=reps)
    rows.append({"mode": "plain", "n_t": 1, **plain_info})

    for n_t in n_t_values:
        info = _compile_and_measure(jvp_fn, (params, T_full[:n_t]), reps=reps)
        rows.append({"mode": "jvp_batched", "n_t": n_t, **info})

    seq_n_t = n_t_values[-1]
    seq_info = _compile_and_measure(jvp_seq_fn, (params, T_full[:seq_n_t]), reps=reps)
    rows.append({"mode": "jvp_sequential", "n_t": seq_n_t, **seq_info})

    # Fit through the jvp_batched SERIES ONLY (n_t = n_t_values[0]..[-1]) --
    # NOT mixed with the separate "plain" baseline point, which sits on a
    # DIFFERENT curve (jvp has fixed per-call overhead vs plain even at
    # n_t=1: measured ~2.6x flops). Mixing them as one series with a
    # duplicate x=1 point produces a meaningless (even negative) intercept.
    # The primal-sharing witness instead COMPARES the fitted intercept `a`
    # against the separately-measured plain baseline (see
    # `intercept_vs_plain_ratio` below): the two should be close if the
    # primal sweep really is shared.
    batched_xs = list(n_t_values)
    batched_flops = [r["flops"] for r in rows if r["mode"] == "jvp_batched"]
    batched_temp = [r["temp_bytes"] for r in rows if r["mode"] == "jvp_batched"]
    batched_wall = [r["t_median_s"] for r in rows if r["mode"] == "jvp_batched"]
    fit = {}
    if all(v is not None for v in batched_flops):
        fit["flops_a_plus_b_nt"] = _linear_fit_endpoints(batched_xs, batched_flops)
    if all(v is not None for v in batched_temp):
        fit["temp_bytes_a_plus_b_nt"] = _linear_fit_endpoints(batched_xs, batched_temp)
    fit["wall_s_a_plus_b_nt"] = _linear_fit_endpoints(batched_xs, batched_wall)

    intercept_vs_plain_ratio = {}
    if "flops_a_plus_b_nt" in fit and plain_info.get("flops"):
        intercept_vs_plain_ratio["flops"] = fit["flops_a_plus_b_nt"][0] / plain_info["flops"]
    if "temp_bytes_a_plus_b_nt" in fit and plain_info.get("temp_bytes"):
        intercept_vs_plain_ratio["temp_bytes"] = fit["temp_bytes_a_plus_b_nt"][0] / plain_info["temp_bytes"]
    if plain_info.get("t_median_s"):
        intercept_vs_plain_ratio["wall_s"] = fit["wall_s_a_plus_b_nt"][0] / plain_info["t_median_s"]

    # n_steps-independence witness (memory only; wall time DOES scale with
    # n_steps as expected -- only the forward-mode MEMORY claim is
    # n_steps-independent).
    n_steps_2x = 2 * n_steps
    grid2, sim_fn_2x = make_design_fn(sim, n_p, n_steps=n_steps_2x)
    n_t_witness = n_t_values[len(n_t_values) // 2]

    def jvp_fn_2x(p, t):
        return jacobian_fwd(sim_fn_2x, p, tangents=t, batch_tangents=True)

    info_2x = _compile_and_measure(
        jvp_fn_2x, (params, T_full[:n_t_witness]), reps=1,
    )
    info_1x = next(r for r in rows if r["mode"] == "jvp_batched" and r["n_t"] == n_t_witness)

    return {
        "backend": jax.default_backend(),
        "dtype": "float32",
        "grid_shape": list(grid.shape),
        "grid_cells": int(grid.shape[0] * grid.shape[1] * grid.shape[2]),
        "n_steps": n_steps,
        "n_p": n_p,
        "rows": rows,
        "fit": fit,
        "intercept_vs_plain_ratio": intercept_vs_plain_ratio,
        "n_steps_independence_witness": {
            "n_t": n_t_witness,
            "n_steps_1x": n_steps,
            "n_steps_2x": n_steps_2x,
            "temp_bytes_1x": info_1x.get("temp_bytes"),
            "temp_bytes_2x": info_2x.get("temp_bytes"),
        },
    }


def _print_table(table: dict[str, Any]) -> None:
    print(
        f"jacobian_fwd cost table -- backend={table['backend']} "
        f"dtype={table['dtype']} grid={table['grid_shape']} "
        f"({table['grid_cells']} cells) n_steps={table['n_steps']} "
        f"n_p={table['n_p']} (CPU unless backend says otherwise)"
    )
    print(f"{'mode':16s} {'n_t':>4s} | {'t_median_s':>11s} {'x':>6s} | "
          f"{'flops':>12s} {'x':>6s} | {'temp_bytes':>11s} {'x':>6s}")
    plain = table["rows"][0]
    for r in table["rows"]:
        t_x = r["t_median_s"] / plain["t_median_s"] if plain["t_median_s"] else float("nan")
        f_x = (r["flops"] / plain["flops"]) if (r.get("flops") and plain.get("flops")) else float("nan")
        m_x = (r["temp_bytes"] / plain["temp_bytes"]) if (r.get("temp_bytes") and plain.get("temp_bytes")) else float("nan")
        print(f"{r['mode']:16s} {r['n_t']:>4d} | {r['t_median_s']:>11.4f} {t_x:>6.2f} | "
              f"{str(r.get('flops')):>12s} {f_x:>6.2f} | "
              f"{str(r.get('temp_bytes')):>11s} {m_x:>6.2f}")
    print()
    for name, (a, b) in table["fit"].items():
        print(f"  fit {name}: a={a:.4g} (intercept) b={b:.4g} (marginal cost per tangent)")
    for name, ratio in table["intercept_vs_plain_ratio"].items():
        print(f"  intercept/plain ratio ({name}): {ratio:.3f} "
              f"(primal-sharing witness, DIAGNOSTIC not authoritative -- "
              f"CPU-calibrated expectation ~1.0; on an accelerator this "
              f"reads higher, e.g. 1.4-1.9x measured on an RTX 4090, "
              f"because the two-point endpoint fit pushes fixed per-call "
              f"overhead [kernel launch, tiling] that does not scale with "
              f"n_t into the intercept, and that effect is largest where "
              f"per-tangent marginal cost is smallest [issue #632]. Do not "
              f"read a high ratio here as a sharing regression by itself "
              f"-- G3 in tests/test_jacobian_fwd.py is the authoritative, "
              f"backend-independent structural check: it inspects the "
              f"jaxpr directly for one unbatched primal scan.)")
    w = table["n_steps_independence_witness"]
    print(
        f"  n_steps independence (n_t={w['n_t']}): temp_bytes "
        f"n_steps={w['n_steps_1x']} -> {w['temp_bytes_1x']}, "
        f"n_steps={w['n_steps_2x']} -> {w['temp_bytes_2x']} "
        f"(compiler ESTIMATE, not a measured/observed/certified peak)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-p", type=int, default=10)
    parser.add_argument("--n-t-values", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--grid-scale", type=int, default=0,
                         help="0 = small fixture; e.g. 24 for a ~5x-cell 'large' fixture")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None,
                         help="optional path to write the table as JSON")
    args = parser.parse_args(argv)

    table = build_benchmark_table(
        n_p=args.n_p, n_t_values=tuple(args.n_t_values), n_steps=args.n_steps,
        grid_scale=args.grid_scale, reps=args.reps,
    )
    _print_table(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
