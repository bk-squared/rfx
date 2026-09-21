#!/usr/bin/env python3
"""A lumped port on a line whose input impedance is known in closed form.

The known-load decision run for the single-cell lumped port. ``rfx`` has no
other fixture in which a lumped port faces a load whose reflection is an exact
number at every frequency, so there has been nothing to decide the lumped
lane's V/I convention against. This is that line.

What it measures
----------------
A four-cell air parallel-plate channel: PEC plates on z, magnetic walls on y
and on the back of the port (x_lo), which makes the field an exact TEM wave and
the port's input impedance the closed form of a terminated line. The port
bridges the one-cell gap at node 1; a lumped resistor ``R`` sits at node 3, and
the magnetic wall beyond it carries no current, so the line is terminated in
``R`` and nothing follows it.

``h`` and ``w`` are both one cell, so ``Zc = eta0 * h / w = eta0``. The port's
reference impedance is set to that same ``Zc``, which collapses the closed form
to ``S11 = Gamma_L * exp(-2 j beta L)``: the magnitude is ``|Gamma_L|`` at every
frequency, independent of the line's length and of ``beta``. For
``R = Zc/2, Zc, 2 Zc`` that is ``1/3, 0, 1/3``.

The only difference between the two rows of each block is ``extent=dx`` on
``add_port`` — a WIRE port spanning exactly the one cell the lumped port
occupies. Same cell, same load, same reference impedance, same drive, same
record.

Why the port is not AT the wall
-------------------------------
``rfx``'s preflight refuses that arrangement and names the mechanism:
``apply_pmc_faces`` zeroes ``hy[0]``, the half-cell that carries the wave off
the port, so with the port on node 0 every load returns ``|S11| = 1.000000``.
On node 1 the same zeroed half-cell is a zero-length open behind the port
rather than a one-cell stub, so it costs nothing electrically.

Using it
--------
No arguments. It prints three blocks — one per load — and writes the same
numbers to ``lumped_port_known_load_line.json`` beside this file. It takes a
few seconds; the largest array in it is 5 x 2 x 2 nodes.

    python scripts/diagnostics/lumped_port_known_load_line.py

The CLI is stable on purpose: something that fixes the lumped lane can run this
before and after and compare two JSON records.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import rfx  # noqa: E402
from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402

OUT_JSON = Path(__file__).with_suffix(".json")

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5                       # port on node 1, load on node 3 -> L = 2 mm
I_PORT = 1
I_LOAD = N_NODES - 2
FREQS = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
NUM_PERIODS = 20.0
LOADS = {"half_zc": 0.5, "matched": 1.0, "double_zc": 2.0}


def git_sha() -> str:
    """The commit this tree is at, or None. A record that cannot name its
    commit still says so rather than inventing one."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                      text=True, stderr=subprocess.PIPE).strip()
        return out or None
    except Exception:  # noqa: BLE001 — provenance is recorded, never fatal
        return None


def build(kind: str, r_over_zc: float) -> Simulation:
    """The channel, with the port declared one way or the other."""
    sim = Simulation(
        freq_max=10e9, domain=((N_NODES - 1) * DX, DX, DX), dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    extent = {} if kind == "lumped" else {"extent": DX}
    sim.add_port(position=(I_PORT * DX, 0.0, 0.0), component="ez",
                 impedance=ETA0, waveform=GaussianPulse(f0=5e9, bandwidth=1.6),
                 **extent)
    sim.add_lumped_rlc(position=(I_LOAD * DX, 0.0, 0.0), component="ez",
                       R=r_over_zc * ETA0, topology="parallel")
    return sim


def measure(kind: str, r_over_zc: float) -> dict:
    """One solve. Returns S11, the sampled V/I pair, and every warning."""
    sim = build(kind, r_over_zc)
    grid = sim._build_grid()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.forward(port_s11_freqs=jnp.asarray(FREQS),
                          num_periods=NUM_PERIODS, skip_preflight=True)
    s11 = np.asarray(res.s_params).reshape(-1)
    accs = (res.lumped_port_sparams or res.wire_port_sparams)[0][1]
    v, i = np.asarray(accs[0]), np.asarray(accs[1])
    z = v / (ETA0 * i)
    seen: dict[str, int] = {}
    for w in caught:
        key = f"{w.category.__name__}: {w.message}"
        seen[key] = seen.get(key, 0) + 1
    return {
        "port_kind": kind,
        "grid_shape_nodes": [int(s) for s in grid.shape],
        "dt_s": float(grid.dt),
        "s11_real": np.real(s11).astype(float).tolist(),
        "s11_imag": np.imag(s11).astype(float).tolist(),
        "abs_s11": np.abs(s11).astype(float).tolist(),
        "v_over_zc_i_real": np.real(z).astype(float).tolist(),
        "v_over_zc_i_imag": np.imag(z).astype(float).tolist(),
        "warnings": [{"warning": k, "count": n} for k, n in seen.items()],
        "nonpassive_warning": any("non-passive" in k for k in seen),
    }


def main() -> int:
    sha = git_sha()
    record = {
        "schema": "rfx.lumped_port_known_load_line",
        "schema_version": 1,
        "what": ("a lumped port and a wire port on the same single cell of a "
                 "line whose reflection is an exact number at every frequency"),
        "script": "scripts/diagnostics/lumped_port_known_load_line.py",
        "commit": sha,
        "rfx_import_tail": "/".join(Path(rfx.__file__).resolve().parts[-2:]),
        "rfx_resolved_inside_this_tree": str(Path(rfx.__file__).resolve()).startswith(
            str(REPO) + "/"),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "channel": {
            "dx_m": DX, "n_nodes_x": N_NODES, "port_node": I_PORT,
            "load_node": I_LOAD, "length_m": (I_LOAD - I_PORT) * DX,
            "gap_cells": 1, "width_cells": 1,
            "zc_ohm": ETA0, "zref_ohm": ETA0,
            "boundaries": {"x": ["pmc", "pmc"], "y": ["pmc", "pmc"],
                           "z": ["pec", "pec"]},
            "num_periods": NUM_PERIODS,
            "drive": {"f0_hz": 5e9, "bandwidth": 1.6},
        },
        "freqs_hz": FREQS.astype(float).tolist(),
        "loads": {},
    }

    print(f"rfx  {rfx.__file__}")
    print(f"jax  {jax.__version__}   numpy {np.__version__}")
    print(f"commit {sha}")
    print(f"Zc = {ETA0:.6f} ohm, L = {(I_LOAD - I_PORT) * DX * 1e3:.1f} mm, "
          f"dx = {DX * 1e3:.1f} mm, port node {I_PORT}, load node {I_LOAD}")
    print(f"freqs (GHz) {FREQS / 1e9}")

    for name, r in LOADS.items():
        gamma = abs((r - 1.0) / (r + 1.0))
        entry = {"r_over_zc": r, "r_ohm": r * ETA0,
                 "closed_form_abs_s11": gamma,
                 "closed_form_note": ("|S11| = |Gamma_L| at every bin, because "
                                      "Zref is the line's own Zc"),
                 "ports": {}}
        print(f"\nR = {r} Zc   closed form |S11| = {gamma:.6f} at every bin")
        for kind in ("lumped", "wire"):
            m = measure(kind, r)
            entry["ports"][kind] = m
            print(f"  {kind:6s} |S11| "
                  f"{np.array2string(np.asarray(m['abs_s11']), precision=5)}")
            print(f"  {kind:6s} V/(Zc I) re "
                  f"{np.array2string(np.asarray(m['v_over_zc_i_real']), precision=4)}"
                  f"  im "
                  f"{np.array2string(np.asarray(m['v_over_zc_i_imag']), precision=4)}")
            for w in m["warnings"]:
                if "non-passive" in w["warning"]:
                    print(f"  {kind:6s} WARN {w['warning'][:150]}")
        record["loads"][name] = entry

    OUT_JSON.write_text(json.dumps(record, indent=1) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
