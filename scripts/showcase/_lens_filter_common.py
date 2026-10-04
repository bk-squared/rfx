"""Stage orchestration for the 20261004 lens/filter pre-declaration.

No solver or JAX import at module scope. Full stages require a GPU; --smoke
uses coarse geometry, 200/240 steps and two updates, with no scientific gates.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np

import _design_common as dc
import _record


def integral_cells(length, dx):
    n = int(round(length / dx))
    if not np.isclose(n * dx, length, rtol=0, atol=1e-12):
        raise ValueError(f"{length} m does not divide into {dx} m cells")
    return n


def assert_node(grid, axis, coordinate):
    n = integral_cells(coordinate, grid.dx)
    p = [0., 0., 0.]
    p[axis] = float(coordinate)
    actual = grid.position_to_index(tuple(p))[axis]
    if actual != n + grid.axis_pads[axis]:
        raise ValueError(f"axis {axis}, face {coordinate}: realized node {actual}")


def sqrt_segments(n):
    return next(k for k in range(math.isqrt(n), 0, -1) if n % k == 0)


def segmented_steps(n0):
    """Round a record up to a multiple of isqrt(n0), so sqrt_segments keeps
    about sqrt(n) checkpoint segments (pre-declaration amendment 2)."""
    k = max(math.isqrt(int(n0)), 1)
    return ((int(n0) + k - 1) // k) * k


def cosine_schedule(step, initial, final, iterations, xp=np):
    """Updates 0..N-1 use the two declared endpoint rates; trials truncate."""
    fraction = xp.minimum(xp.maximum(step / (iterations - 1), 0), 1)
    return final + .5 * (initial - final) * (1 + xp.cos(xp.pi * fraction))


def best_iterate(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("non-finite objective history")
    return int(np.argmin(values))  # numpy chooses the first occurrence


class ModelBase:
    def preflight(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            issues = self.sim.preflight()
        messages = [str(i) for i in issues] + [f"warning: {w.message}" for w in caught]
        print("PREFLIGHT BEGIN", flush=True)
        for message in messages:
            print(message, flush=True)
        print("PREFLIGHT END", flush=True)
        return messages

    def describe(self):
        g = self.grid
        return {"dx_m": self.dx, "grid_shape": g.shape, "grid_points": int(np.prod(g.shape)),
                "cells": int(np.prod(np.asarray(g.shape) - 1)), "dt_s": g.dt,
                "cpml_layers": g.pad_x_lo, "cpml_thickness_m": g.pad_x_lo * self.dx,
                "n_steps": self.default_steps, "checkpoint_segments": sqrt_segments(self.default_steps),
                "design_lo": self.lo, "design_hi_exclusive": self.hi,
                "design_lo_m": (np.asarray(self.lo) - np.asarray(g.axis_pads)) * self.dx,
                "design_hi_m": (np.asarray(self.hi) - np.asarray(g.axis_pads)) * self.dx,
                "freqs_hz": self.freqs, "precision": self.precision, "smoke": self.smoke}


def record(out, case, model, stage, details):
    """Rewrite a schema-validated result after every iterate and every stage."""
    dc.save_json(out / "stage.json", {"stage": stage, **details})
    source_path = out / f"source_{model.precision}.json"
    if not source_path.exists():
        dc.save_json(source_path, _record.source_block(case.REPO, model.precision))
    rec = {"schema": _record.SCHEMA, "id": f"design-{case.CASE}",
           "question": "Boresight directivity" if case.CASE == "lens" else "WR-90 specification mask",
           "source": dc.load_json(source_path),
           "run": {"platform": "local CPU smoke" if model.smoke else "VESSL", "run_id": None,
                   "preset": os.environ.get("RFX_SHOWCASE_PRESET"), "stage": stage},
           "model": model.describe(), "claims": [
               _record.claim("record length", details.get("n_steps", model.default_steps), "steps", "stage.json"),
               _record.claim("stage measurements", details, "1", "stage.json")],
           "derived": [], "out_of_scope": ["realized gain"] if case.CASE == "lens" else []}
    if model.smoke:
        rec["out_of_scope"].append("scientific or promotional numbers: CPU smoke only")
    _record.write_result(out, rec, _record.data_files(out))


def setup(case, out, dx, precision, smoke):
    model = case.Model(dx, precision, smoke)
    description = model.describe()
    print(f"MESH {case.CASE}: {description}", flush=True)
    messages = model.preflight()
    dc.save_json(out / f"preflight_{dx:.9g}.json", messages)
    dc.save_json(out / f"model_{dx:.9g}.json", description)
    return model


def fd(case, model, e, n, out, precision):
    import jax
    import jax.numpy as jnp
    response = model.response_fn(n)  # built eagerly, outside the trace
    objective = jax.jit(lambda eps: case.loss(response(eps)))
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    e = np.asarray(e, dtype=np.float64 if precision == "float64" else np.float32)
    g = np.asarray(jax.jit(jax.grad(objective))(jnp.asarray(e, dtype=dtype)))
    steps = (.1, .05, .025)
    ladder = dc.fd_ladder(objective, e, case.FD_PIXELS, steps)
    ad = {name: float(g[index]) for name, index in case.FD_PIXELS.items()}
    judgement = dc.judge_fd(ad, ladder, steps, .05, .05, .1)
    data = {"precision": precision, "n_steps": n, "ladder": ladder, "judgement": judgement,
            "grad_eps": g}
    dc.save_json(out / f"fd_{precision}.json", data)
    if precision == "float32" and judgement["roundoff"]:
        dc.save_json(out / "x64_needed.json", {"pixels": judgement["roundoff"]})
    return data


def rlw(case, model, e, n, out, label):
    import jax
    import jax.numpy as jnp
    from rfx import gradient_record_length_witness
    functions = {}
    def objective(eps, steps):
        if steps not in functions:
            # Build the response (NTFF box, frequencies) eagerly, outside the
            # trace; only the solve and the transform are jitted.
            response = model.response_fn(steps)
            functions[steps] = jax.jit(lambda x: case.loss(response(x)))
        return functions[steps](eps)
    w = gradient_record_length_witness(objective, jnp.asarray(e), n, factor=1.5, tol=.05)
    short, long = next(iter(w.grad.values()))[0], next(iter(w.grad_long.values()))[0]
    ratios = {}
    for name, index in case.FD_PIXELS.items():
        delta = abs(long[index] - short[index])
        ratios[name] = float(delta / abs(long[index])) if long[index] else (0. if delta == 0 else float("inf"))
    data = {**dataclasses.asdict(w), "per_fd_pixel": ratios,
            "all_passed": bool(w.passed and all(v <= .05 for v in ratios.values()))}
    dc.save_json(out / f"rlw_{label}.json", data)
    print(w.summary(), flush=True)
    return data


def evaluate(case, model, designs, n, out, label, normalize="flux"):
    rows, arrays = {}, {}
    for name, e in designs.items():
        t0 = time.perf_counter()
        response, row = model.evaluate(e, n, normalize)
        rows[name] = {**row, "objective": float(case.loss(response)), "n_steps": n,
                      "wall_s": time.perf_counter() - t0}
        arrays[name] = response
        dc.save_json(out / f"{label}.json", rows)
        dc.save_npz(out / f"{label}.npz", **arrays)
    if case.CASE == "lens" and "uniform_1" in rows:
        names = [k for k in rows if k.startswith("uniform_")]
        winner = min(names, key=lambda k: rows[k]["objective"])
        rows["best_uniform"] = winner
        dc.save_json(out / f"{label}.json", rows)
    return rows


def prepare(case, model, names, n, out, allow_extension=True):
    """One common record before descent, including A's baseline settling rule."""
    designs = {**case.baselines(), **{name: case.starts()[name] for name in names}}
    for attempt, factor in enumerate((1., 1.5) if allow_extension else (1.,)):
        steps = math.ceil(n * factor)
        rows = evaluate(case, model, designs, steps, out, f"settling_start_attempt{attempt}")
        required = designs if case.CASE == "lens" else names
        if all(rows[name]["settled"] for name in required):
            dc.save_json(out / "record_length.json", {"n_steps": steps, "dt_s": model.grid.dt,
                                                       "attempt": attempt, "passed": True})
            return steps
    dc.save_json(out / "record_length.json", {"n_steps": steps, "dt_s": model.grid.dt,
                                               "attempt": attempt, "passed": False})
    raise RuntimeError("start/baseline settling failed at the permitted record; descent stopped")


def optimize(case, model, e, n, count, out, stage):
    import jax
    import jax.numpy as jnp
    import optax
    out.mkdir(parents=True, exist_ok=True)
    response = model.response_fn(n)
    def objective(eps):
        r = response(eps)
        return case.loss(r), r
    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    # AD is explicitly in free-pixel eps units; chain it to psi for Adam.
    fraction0 = (np.asarray(e, dtype=float) - 1) / (case.UPPER - 1)
    # A start value on a bound (the textbook GRIN clips to eps = 1) begins
    # 1e-3 of the range inside it. At the old latent -20 the sigmoid slope is
    # ~2e-9 and Adam never moved those pixels (PR 1477 review: 420 of 2250).
    p = np.clip(fraction0, 1e-3, 1 - 1e-3)
    latent = np.log(p / (1 - p))
    psi = jnp.asarray(latent, dtype=jnp.float32)
    schedule = lambda i: cosine_schedule(i, *case.LR, case.ITERATIONS, xp=jnp)
    opt = optax.adam(schedule)
    state = opt.init(psi)
    store = dc.IterateStore(out / "iterations.npz", {"freqs_hz": case.FREQS})
    for it in range(count + 1):
        t0 = time.perf_counter()
        fraction = jax.nn.sigmoid(psi)
        eps = 1 + (case.UPPER - 1) * fraction
        (value, r), g = vg(eps)
        jax.block_until_ready(g)
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(g)).all() or not np.isfinite(np.asarray(r)).all():
            dc.save_json(out / "nonfinite.json", {"iteration": it})
            raise RuntimeError(f"non-finite iterate {it}; descent stopped")
        store.append(iteration=it, eps=eps, psi=psi, response=r, grad_eps=g, objective=value,
                     lr=schedule(it), wall_s=time.perf_counter() - t0, **dc.adam_state_arrays(state))
        store.persist()
        best = best_iterate(store.rows["objective"])
        details = {"n_steps": n, "iteration": it, "best_iteration": best, "last_iteration": it,
                   "best_objective": float(store.rows["objective"][best]), "last_objective": float(value)}
        dc.save_json(out / "selection.json", details)
        record(out, case, model, stage, details)
        print(f"ITERATE {it}: objective={float(value):.9g} best={best} wall_s={store.rows['wall_s'][-1]}", flush=True)
        if it != count:
            updates, state = opt.update(g * (case.UPPER - 1) * fraction * (1 - fraction), state)
            psi = optax.apply_updates(psi, updates)
    return store


def trial_pass(case, store):
    objective = np.asarray(store.rows["objective"])
    if case.CASE == "lens":
        first = case.summary(store.rows["response"][0])["mean_boresight_dbi"]
        last = case.summary(store.rows["response"][-1])["mean_boresight_dbi"]
        return {"passed": bool(last - first >= 2), "improvement_db": last - first}
    return {"passed": bool(objective[-1] <= .5 * objective[0]),
            "initial_objective": float(objective[0]), "iteration30_objective": float(objective[-1])}


def selected_design(directory):
    with np.load(directory / "iterations.npz") as data:
        index = best_iterate(data["objective"])
        return np.asarray(data["eps"][index]), index


def resolved(case, args, source):
    e, best = selected_design(source)
    length_path = source / "reported_record_length.json"
    length = dc.load_json(length_path if length_path.exists() else source / "record_length.json")
    duration = length["n_steps"] * length["dt_s"]
    designs = {"reported": e}
    if case.CASE == "lens":
        bas = dc.load_json(source / "baselines.json")
        designs.update(no_lens=case.baselines()["no_lens"], grin=case.baselines()["grin"],
                       best_uniform=case.baselines()[bas["best_uniform"]])
    all_rows = []
    for dx in case.MESHES:
        directory = args.out / f"mesh_{dx:.9g}"
        directory.mkdir(parents=True, exist_ok=True)
        model = setup(case, directory, dx, args.precision, False)
        n = math.ceil(duration / model.grid.dt)
        rows = evaluate(case, model, designs, n, directory, "resolve")
        if case.CASE == "filter":
            initial_n = n
            for factor in (1.5, 2.):
                if rows["reported"]["settled"]:
                    break
                n = math.ceil(initial_n * factor)
                rows = evaluate(case, model, designs, n, directory, f"resolve_{factor:g}x")
            evaluate(case, model, designs, n, directory, "normalization_crosscheck", normalize=False)
        all_rows.append(rows)
        record(directory, case, model, "resolve", {"n_steps": n, "best_iteration": best, "measurements": rows})
    trend = {}
    if case.CASE == "lens":
        for name in designs:
            values = np.asarray([row[name]["boresight_dbi"] for row in all_rows])
            changes = np.abs(np.diff(values, axis=0))
            trend[name] = {"boresight_dbi": values, "signed_changes_db": np.diff(values, axis=0),
                           "absolute_changes_db": changes,
                           "passed_per_bin": (changes[1] <= .3) & (changes[1] <= changes[0]),
                           "settled_each_mesh": [r[name]["settled"] for r in all_rows],
                           "fine_value_dbi": values[-1]}
    else:
        edges = [row["reported"]["passband_edges_hz"] for row in all_rows]
        if all(edge is not None for edge in edges):
            changes = np.abs(np.diff(np.asarray(edges), axis=0))
            edge_pass = bool(np.all(changes[1] / np.asarray(edges[-1]) <= .01)
                             and np.all(changes[1] <= changes[0]))
        else:
            changes, edge_pass = None, False
        trend = {"edges_hz": edges, "edge_changes_hz": changes, "edge_trend_passed": edge_pass,
                 "mask_each_mesh": [r["reported"]["inside_mask"] for r in all_rows],
                 "power_balance_each_mesh": [r["reported"]["power_balance_passed"] for r in all_rows],
                 "settled_each_mesh": [r["reported"]["settled"] for r in all_rows],
                 "worst_passband_S11_db": [r["reported"]["worst_passband_S11_db"] for r in all_rows],
                 "worst_stopband_S21_db": [r["reported"]["worst_stopband_S21_db"] for r in all_rows]}
    dc.save_json(args.out / "mesh_trend.json", trend)
    fd_path = source / ("fd_float64.json" if (source / "x64_needed.json").exists() else "fd_float32.json")
    fd_passed = fd_path.exists() and dc.load_json(fd_path)["judgement"]["all_judged_passed"]
    if fd_path.name == "fd_float64.json" and fd_path.exists():
        seen = dc.load_json(fd_path).get("field_dtypes", [])
        fd_passed = fd_passed and bool(seen) and all(
            row["observed"] in ("float64", "complex128") for row in seen)
    witness_passes = {}
    for label in ("start", "reported"):
        path = source / f"rlw_{label}.json"
        witness_passes[label] = path.exists() and dc.load_json(path)["all_passed"]
    if case.CASE == "lens":
        mesh_passed = all(np.all(row["passed_per_bin"]) and all(row["settled_each_mesh"])
                          for row in trend.values())
    else:
        mesh_passed = (trend["edge_trend_passed"] and all(trend["power_balance_each_mesh"])
                       and all(trend["settled_each_mesh"]) and trend["mask_each_mesh"][-1])
    eligibility = {"fd_passed": bool(fd_passed), "rlw_passed": witness_passes,
                   "mesh_witnesses_passed": bool(mesh_passed),
                   "promotional_number_eligible": bool(fd_passed and all(witness_passes.values()) and mesh_passed)}
    dc.save_json(args.out / "eligibility.json", eligibility)
    record(args.out, case, model, "resolve", {"best_iteration": best, "trend": trend, **eligibility})


def main(case, argv=None):
    parser = argparse.ArgumentParser(description=case.__doc__)
    parser.add_argument("--stage", choices=("timing", "fd", "rlw", "trial", "main", "resolve", "baselines", "describe"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="source trial/main record; never modified")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--start", help="lens: primary or grin; filter: S1 or S2 for standalone checks")
    parser.add_argument("--reported", action="store_true", help="FD/RLW: use best stored design")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    if args.precision == "float64" and (args.stage != "fd" or not jax.config.x64_enabled):
        raise SystemExit("float64 is an FD-only separate process: set JAX_ENABLE_X64=1")
    if args.precision == "float32" and jax.config.x64_enabled:
        raise SystemExit("float32 stages require JAX_ENABLE_X64=0")
    if not args.smoke and args.stage != "describe" and jax.default_backend() == "cpu":
        raise SystemExit("full solves are VESSL GPU only; use --smoke on the Mac")
    if args.smoke and args.stage not in ("timing", "fd"):
        raise SystemExit("--smoke supports --stage timing only; it cannot create trial admission")
    if args.stage == "describe":
        for dx in case.MESHES:
            model = setup(case, args.out, dx, args.precision, False)
        record(args.out, case, model, args.stage, {"build_only": True})
        return 0
    source = args.record or args.out
    if args.stage in ("main", "resolve") and source.resolve() == args.out.resolve():
        raise SystemExit("--main/--resolve require --record and a distinct --out to preserve the source run")
    if args.stage == "trial" and case.CASE == "lens" and args.start not in (None, "primary"):
        raise SystemExit("the admission trial uses the primary start; GRIN is a reported secondary main run")
    if args.stage == "resolve":
        resolved(case, args, source)
        return 0
    model = setup(case, args.out, case.SMOKE_DX if args.smoke else case.DESIGN_DX,
                  args.precision, args.smoke)
    n = model.default_steps
    if (source / "record_length.json").exists():
        length = dc.load_json(source / "record_length.json")
        n = math.ceil(length["n_steps"] * length["dt_s"] / model.grid.dt)
    name = args.start or ("primary" if case.CASE == "lens" else "S1")
    if name not in case.starts():
        raise SystemExit(f"unknown start {name}")
    if case.CASE == "filter" and (source / "trial_selection.json").exists():
        name = dc.load_json(source / "trial_selection.json")["chosen_start"]
    e = case.starts()[name]
    if args.reported:
        e, _ = selected_design(source)
    details = {"n_steps": n}
    if args.stage == "timing":
        t0 = time.perf_counter()
        optimize(case, model, e, n, 2, args.out, "timing")
        best, _ = selected_design(args.out)
        details["eager_check"] = evaluate(case, model, {"reported": best}, n, args.out, "timing_eager")
        details.update(wall_s=time.perf_counter() - t0, smoke=args.smoke)
    elif args.stage == "fd":
        # Observe all actual solver field dtypes, not just design-array dtype.
        import rfx.simulation as simulation
        original, seen = simulation.run, []
        def observed(*a, **kw):
            result = original(*a, **kw)
            state = getattr(result, "state", None)
            ntff = getattr(result, "ntff_data", None)
            actual = str(state.ex.dtype) if state is not None else (str(ntff.x_lo.dtype) if ntff is not None else None)
            seen.append({"requested": str(kw.get("field_dtype")), "observed": actual})
            return result
        simulation.run = observed
        try:
            details = fd(case, model, e, n, args.out, args.precision)
        finally:
            simulation.run = original
        details["field_dtypes"] = seen
        dc.save_json(args.out / f"fd_{args.precision}.json", details)
        if args.precision == "float64" and (not seen or any(
                row["observed"] not in ("float64", "complex128") for row in seen)):
            raise RuntimeError(f"float64 FD refused: observed field dtypes {seen}")
    elif args.stage == "rlw":
        details = rlw(case, model, e, n, args.out, "reported" if args.reported else "start")
    elif args.stage == "baselines":
        details = evaluate(case, model, case.baselines(), n, args.out, "baselines")
    elif args.stage in ("trial", "main"):
        if args.precision != "float32":
            raise RuntimeError("Adam is float32")
        if args.stage == "main":
            admission = dc.load_json(source / "trial_selection.json")
            if not admission["passed"]:
                raise RuntimeError("trial did not meet the declared continuation rule; amendment required")
            name = admission["chosen_start"] if case.CASE == "filter" else name
        names = list(case.starts()) if args.stage == "trial" and case.CASE == "filter" else [name]
        if args.stage == "main" and case.CASE == "lens" and args.start is None:
            names = ["primary", "grin"]
        chosen_name = name
        # The trial already fixed the case record. A main job may not apply
        # another 1.5x start extension to that admitted record.
        n = prepare(case, model, names, n, args.out, allow_extension=args.stage == "trial")
        results = {}
        for name in names:
            directory = (args.out / name if len(names) > 1 and
                         (case.CASE == "filter" or name == "grin") else args.out)
            directory.mkdir(parents=True, exist_ok=True)
            dc.save_json(directory / "record_length.json", {"n_steps": n, "dt_s": model.grid.dt})
            e = case.starts()[name]
            check = rlw(case, model, e, n, directory, "start")
            if not check["all_passed"]:
                record(directory, case, model, args.stage, {"stopped": "start RLW", "witness": check})
                raise RuntimeError("start gradient record-length witness failed; descent stopped")
            fd_data = (fd(case, model, e, n, directory, "float32")
                       if case.CASE == "filter" or name == "primary" else None)
            if fd_data is not None and fd_data["judgement"]["roundoff"]:
                # Release compiled tapes before the independent x64 process
                # uses this job's GPU. The model rebuilds its functions later.
                import gc
                model._functions.clear()
                jax.clear_caches()
                gc.collect()
                env = {**os.environ, "JAX_ENABLE_X64": "1"}
                subprocess.run([sys.executable, str(Path(case.__file__)), "--stage", "fd", "--out", str(directory),
                                "--start", name, "--precision", "float64", "--record", str(directory)]
                               + (["--smoke"] if args.smoke else []), env=env, check=True)
            store = optimize(case, model, e, n, 30 if args.stage == "trial" else case.ITERATIONS,
                             directory, args.stage)
            results[name] = trial_pass(case, store)
            best, _ = selected_design(directory)
            evaluate(case, model, case.baselines(), n, directory, "baselines")
            end = evaluate(case, model, {"reported": best}, n, directory, "reported")
            if case.CASE == "filter":
                for factor in (1.5, 2.):
                    if end["reported"]["settled"]:
                        break
                    end = evaluate(case, model, {"reported": best}, math.ceil(n * factor), directory,
                                   f"reported_{factor:g}x")
                evaluate(case, model, {"reported": best}, end["reported"]["n_steps"], directory,
                         "normalization_crosscheck", normalize=False)
            dc.save_json(directory / "reported_record_length.json",
                         {"n_steps": end["reported"]["n_steps"], "dt_s": model.grid.dt})
            witness = rlw(case, model, best, n, directory, "reported")
            dc.save_json(directory / "gradient_panel.json", {"colour_scale": witness["all_passed"],
                         "label": "gradient" if witness["all_passed"] else "not witnessed"})
            record(directory, case, model, args.stage, {"n_steps": n, "reported": end,
                                                        "reported_rlw_passed": witness["all_passed"]})
        if args.stage == "trial":
            chosen = min(names, key=lambda k: results[k].get("iteration30_objective", 0))
            details = {"chosen_start": chosen, "passed": results[chosen]["passed"], "starts": results,
                       "n_steps": n}
            dc.save_json(args.out / "trial_selection.json", details)
        else:
            details = {"chosen_start": chosen_name, "starts_run": names, "n_steps": n}
    record(args.out, case, model, args.stage, details)
    return 3 if args.stage == "trial" and not details["passed"] else 0
