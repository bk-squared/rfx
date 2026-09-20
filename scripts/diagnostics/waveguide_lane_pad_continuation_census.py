#!/usr/bin/env python3
"""Issue #1066: who opts into smoothing on the waveguide lane, and what the pad holds.

Two measurements, both cheap and neither solving:

A1, the CALLER CENSUS. ``subpixel_smoothing`` defaults to False on
``compute_waveguide_s_matrix`` (``rfx/sparams/waveguide.py``), so the block that
builds ``shape_eps_pairs`` is only entered when a caller opts in. Counting those
callers by grep is wrong and was wrong: the calls span lines, and a line-oriented
grep found one where there are four. This walks the AST instead.

WHAT THE CENSUS CANNOT SEE (review of PR #1131, F3). It matches on the callee
NAME, so a call that reaches the lane by forwarding ``**kwargs`` carries no
visible ``subpixel_smoothing`` and cannot be classified -- those sites are
counted and listed separately as UNKNOWN rather than folded into "takes the
default". ``compute_s_matrix`` (``rfx/sparams/dispatch.py:452``) is one such
route and is in the entry-point set for that reason. An indirection the walker
also cannot follow is ``getattr``-style dispatch and any call through an alias;
neither exists in the tree today, and neither would show up here if it did.

A3, the PAD. For a dielectric that runs into a port-side CPML face, it prints
three permittivity columns down the x-hi pad on one geometry: what the LANE
hands the solver, what the RUNNER hands it, and the pre-fold local construction
the fold removed. The first two are captured from the real calls by wrapping
``rfx.geometry.smoothing.compute_smoothed_eps`` and stopping there, so they are
the arrays themselves and not a rebuild. The third is reconstructed BY THIS
SCRIPT -- that code is no longer in the tree -- and is the historical column,
not a measurement of anything running today. Build-only: nothing time-steps, so
this is a statement about what the solver would be handed, not about any
S-parameter.

Run::

    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py
    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py --census
    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py --pad
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The lane's public entry points. ``compute_waveguide_s_matrix`` is the one
#: that takes ``subpixel_smoothing``; the others are listed so a future rename
#: or split shows up here as a count change rather than as silence.
LANE_ENTRY_POINTS = frozenset({
    "compute_waveguide_s_matrix",
    "compute_waveguide_s_params",
    "compute_waveguide_s_matrix_flux",
    # The dispatch route: ``compute_s_matrix(**kwargs)`` forwards to
    # ``self.compute_waveguide_s_matrix(**kwargs)`` when the ports are
    # waveguide ports (``rfx/sparams/dispatch.py:452``). Without this name in
    # the set a caller reaching the lane that way is invisible here.
    "compute_s_matrix",
})

_SKIP_PREFIXES = (".venv", "build/", "dist/")


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def census() -> dict:
    """Classify every call into the lane by what it says about smoothing.

    Three buckets, because "takes the default" is only honest about calls whose
    arguments are visible: ``explicit`` passes ``subpixel_smoothing``,
    ``kwargs`` forwards ``**something`` and could be passing it, ``default``
    passes neither.
    """
    explicit: list[tuple[str, int, str, str, bool]] = []
    forwarded: list[tuple[str, int, str, str]] = []
    total = 0
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_SKIP_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node)
            if callee not in LANE_ENTRY_POINTS:
                continue
            total += 1
            named = {kw.arg for kw in node.keywords if kw.arg is not None}
            if "subpixel_smoothing" in named:
                for kw in node.keywords:
                    if kw.arg != "subpixel_smoothing":
                        continue
                    if isinstance(kw.value, ast.Constant):
                        shown, truthy = repr(kw.value.value), bool(kw.value.value)
                    else:
                        # A name or expression: it CAN be truthy, so it counts.
                        shown, truthy = ast.unparse(kw.value), True
                    explicit.append((rel, node.lineno, callee, shown, truthy))
                continue
            starred = [kw for kw in node.keywords if kw.arg is None]
            if starred:
                forwarded.append((rel, node.lineno, callee,
                                  "**" + ast.unparse(starred[0].value)))
    return {
        "explicit": explicit,
        "forwarded": forwarded,
        "total_calls": total,
        "default_calls": total - len(explicit) - len(forwarded),
    }


class _StopAfterCapture(BaseException):
    """Ends a captured call once the permittivity exists, before the scan."""


def _capture(call, attr: str = "compute_smoothed_eps") -> dict:
    """What ``call`` handed to ``rfx.geometry.smoothing.<attr>``.

    Both the lane and the runners import the function INSIDE their smoothing
    block, so patching the module attribute reaches the real call site rather
    than a re-export. Same hook as
    ``tests/unit/sparams/test_waveguide_lane_pad_continuation.py``.
    """
    import numpy as np

    from rfx.geometry import smoothing as _smoothing

    captured: dict = {}
    real = getattr(_smoothing, attr)

    def _spy(grid, shapes, background_eps=1.0):
        out = real(grid, shapes, background_eps=background_eps)
        captured["eps"] = tuple(np.asarray(c) for c in out)
        captured["shapes"] = shapes
        captured["grid"] = grid
        raise _StopAfterCapture

    setattr(_smoothing, attr, _spy)
    try:
        call()
    except _StopAfterCapture:
        captured["stopped"] = True
    finally:
        setattr(_smoothing, attr, real)
    return captured


def _build_guide():
    import jax.numpy as jnp

    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    a_wg, b_wg, dx = 22.86e-3, 10.16e-3, 1.27e-3
    lx = 60 * dx
    freqs = jnp.asarray([8.5e9, 10.0e9, 11.5e9])

    sim = Simulation(
        freq_max=12e9, domain=(lx, a_wg, b_wg), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    sim.add_material("slab", eps_r=4.0)
    # Runs INTO the x-hi face, i.e. into the port-side absorber.
    sim.add(Box((lx / 2.0, 0.0, 0.0), (lx, a_wg, b_wg)), material="slab")
    for pos, direction, nm in ((6 * dx, "+x", "left"),
                               (lx - 6 * dx, "-x", "right")):
        sim.add_waveguide_port(pos, direction=direction, freqs=freqs,
                               f0=1e10, name=nm)
    return sim


def pad_comparison() -> dict:
    """Lane, runner and pre-fold columns down the x-hi pad."""
    import contextlib
    import io
    import warnings

    import numpy as np

    import rfx
    from rfx.geometry.smoothing import compute_smoothed_eps

    n_steps = 10  # never reached: every capture aborts at the smoothing site
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lane = _capture(
            lambda: _build_guide().compute_waveguide_s_matrix(
                subpixel_smoothing=True, n_steps=n_steps))
        runner = _capture(
            lambda: _build_guide().run(n_steps=n_steps, subpixel_smoothing=True,
                                       skip_preflight=True))
        # The removed code, rebuilt here to keep the historical column
        # readable. Nothing in the tree builds pairs this way any more.
        sim = _build_guide()
        grid = sim._build_grid()
        prefold_pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                         for e in sim._geometry]
        prefold = np.asarray(
            compute_smoothed_eps(grid, prefold_pairs, background_eps=1.0)[0])

    if not (lane.get("stopped") and runner.get("stopped")):
        raise SystemExit(
            "a capture did not reach compute_smoothed_eps -- lane="
            f"{sorted(lane)}, runner={sorted(runner)}")

    lane_eps = lane["eps"][0]
    runner_eps = runner["eps"][0]
    pad = int(lane["grid"].face_pads[1])
    jm, km = lane_eps.shape[1] // 2, lane_eps.shape[2] // 2
    diff = np.abs(lane_eps - runner_eps)
    hist = np.abs(prefold - runner_eps)
    return {
        "rfx_file": rfx.__file__,
        "grid_shape": tuple(int(v) for v in lane_eps.shape),
        "boundary": getattr(sim, "_boundary", None),
        "cpml_layers": int(getattr(sim, "_cpml_layers", 0)),
        "x_hi_pad_cells": pad,
        "column": [(int(i), float(lane_eps[i, jm, km]),
                    float(runner_eps[i, jm, km]), float(prefold[i, jm, km]))
                   for i in range(lane_eps.shape[0] - pad - 2,
                                  lane_eps.shape[0])],
        "max_abs_diff": float(diff.max()),
        "cells_differing": int((diff > 1e-12).sum()),
        "prefold_max_abs_diff": float(hist.max()),
        "prefold_cells_differing": int((hist > 1e-12).sum()),
        "cells_total": int(diff.size),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", action="store_true", help="A1 only")
    parser.add_argument("--pad", action="store_true", help="A3 only")
    args = parser.parse_args(argv)
    both = not (args.census or args.pad)

    if both or args.census:
        result = census()
        print("A1 -- calls into the lane, by what they say about smoothing")
        print(f"{'file':60s} {'line':>5}  {'value':16s} truthy")
        print("-" * 96)
        for rel, line, _callee_name, shown, truthy in result["explicit"]:
            print(f"{rel:60s} {line:5d}  {shown:16s} {truthy}")
        print(f"\n{len(result['explicit'])} explicit call site(s); "
              f"{sum(1 for h in result['explicit'] if h[4])} truthy.")

        print("\nUNKNOWN -- reach the lane forwarding **kwargs, so this "
              "script cannot say what they pass")
        print(f"{'file':60s} {'line':>5}  callee")
        print("-" * 96)
        for rel, line, callee_name, shown in result["forwarded"]:
            print(f"{rel:60s} {line:5d}  {callee_name}({shown})")
        print(f"\n{result['total_calls']} call(s) into "
              f"{sorted(LANE_ENTRY_POINTS)}: "
              f"{len(result['explicit'])} explicit, "
              f"{len(result['forwarded'])} forwarding **kwargs (unknown), "
              f"{result['default_calls']} passing neither.\n")

    if both or args.pad:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        result = pad_comparison()
        print("A3 -- eps_xx down the x-hi pad, mid y/z (build-only, no solve)")
        print(f"rfx.__file__ = {result['rfx_file']}")
        print(f"grid {result['grid_shape']}  boundary={result['boundary']!r}  "
              f"cpml_layers={result['cpml_layers']}  "
              f"x-hi pad = {result['x_hi_pad_cells']} cells")
        print("'lane' and 'runner' are CAPTURED from the real calls; "
              "'pre-fold' is the removed code rebuilt by this script.")
        print(f"{'i':>5} {'lane':>9} {'runner':>9} {'pre-fold':>9}  differ")
        for i, lane_v, runner_v, prefold_v in result["column"]:
            mark = "YES" if abs(lane_v - runner_v) > 1e-12 else ""
            print(f"{i:5d} {lane_v:9.4f} {runner_v:9.4f} {prefold_v:9.4f}"
                  f"  {mark}")
        print(f"\nmax |lane - runner| = {result['max_abs_diff']:.6f}; "
              f"{result['cells_differing']} of {result['cells_total']} cells "
              f"differ.")
        print(f"max |pre-fold - runner| = "
              f"{result['prefold_max_abs_diff']:.6f}; "
              f"{result['prefold_cells_differing']} of "
              f"{result['cells_total']} cells differ -- that is the gap the "
              f"fold closed, measured against code no longer in the tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
