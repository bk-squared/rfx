#!/usr/bin/env python3
"""Known-load decision run for the one-cell lumped port.

The same single cell is declared two ways — as a LUMPED port
(``add_port(..., extent=None)``) and as a one-cell WIRE port
(``add_port(..., extent=dx)``) — on a fixture whose |S11| is known in closed
form at every frequency, so the two extraction lanes can be compared against
an answer neither of them produced.

The fixture is a 4-cell air-filled parallel-plate channel: PEC plates on z,
magnetic (PMC) walls on y and on the back of the port at x_lo, so the exact
1-D TEM line applies. The port bridges the 1-cell gap at node 1; a lumped
resistor R sits at node 3, and the magnetic wall beyond it carries no
current, so the line is terminated in R and nothing else.

Because the cell is as wide as it is high, the line impedance is

    Zc = eta0 * h / w = eta0 = 376.730313668 ohm,

and the port's reference impedance is set to that same Zc. Then

    S11 = Gamma_L * exp(-2 j beta L),   Gamma_L = (R - Zc)/(R + Zc)

so |S11| = |Gamma_L| at EVERY frequency — 1/3, 0, 1/3 for R = Zc/2, Zc,
2 Zc. The magnitude is the oracle; it does not depend on the line length,
on beta, or on the numerical dispersion of the mesh.

The companion witness printed with each block is V/(Zc*I) at the port: on a
lossless line the port sees the load transformed by the two-cell section, and
at low frequency (beta*L << 1) that is the load itself — 0.5, 1.0, 2.0 in
units of Zc. A lane that reads a different number is not measuring the
terminal V/I pair of this circuit.

Run with no arguments. Prints the three load blocks and writes
``lumped_port_known_load_line_results.json`` beside this file.

Two records are committed beside it: ``..._results_before_fix.json``, run on
b4cf8f29 where the lumped lane read |S11| 0.714 / 1.248 / 4.757 against a
closed form of 0.333 / 0 / 0.333, and ``..._results.json`` from the commit
that fixed it, where the lumped lane is bit-identical to the wire lane.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import rfx  # noqa: E402
from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5  # port at node 1, load at node 3 -> L = 2 mm
FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
LOADS = (0.5, 1.0, 2.0)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        ).strip()
    except Exception:
        return "unknown"


def build(kind: str, r_over_zc: float) -> Simulation:
    """The fixture, with the port declared as ``kind`` ('lumped' or 'wire')."""
    sim = Simulation(
        freq_max=10e9,
        domain=((N_NODES - 1) * DX, DX, DX),
        dx=DX,
        boundary=BoundarySpec(
            x=Boundary(lo="pmc", hi="pmc"),
            y=Boundary(lo="pmc", hi="pmc"),
            z=Boundary(lo="pec", hi="pec"),
        ),
    )
    extra = {} if kind == "lumped" else {"extent": DX}
    sim.add_port(
        position=(1.0 * DX, 0.0, 0.0),
        component="ez",
        impedance=ETA0,
        waveform=GaussianPulse(f0=5e9, bandwidth=1.6),
        **extra,
    )
    sim.add_lumped_rlc(
        position=((N_NODES - 2) * DX, 0.0, 0.0),
        component="ez",
        R=r_over_zc * ETA0,
        topology="parallel",
    )
    return sim


def measure(kind: str, r_over_zc: float) -> dict:
    """|S11| and the terminal V/(Zc I) for one lane and one load."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = build(kind, r_over_zc).forward(
            port_s11_freqs=jnp.asarray(FREQS_HZ),
            num_periods=20.0,
            skip_preflight=True,
        )
    s = np.asarray(res.s_params).reshape(-1)
    accs = (res.lumped_port_sparams or res.wire_port_sparams)[0][1]
    v, i = np.asarray(accs[0]), np.asarray(accs[1])
    z = v / (ETA0 * i)
    return {
        "abs_s11": np.abs(s).tolist(),
        "v_over_zc_i_re": z.real.tolist(),
        "v_over_zc_i_im": z.imag.tolist(),
        "nonpassive_warnings": [
            str(w.message) for w in caught if "non-passive" in str(w.message)
        ],
    }


def main() -> None:
    record: dict = {
        "rfx_file": rfx.__file__,
        "git_commit": _git_commit(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "zc_ohm": ETA0,
        "dx_m": DX,
        "line_length_m": 2 * DX,
        "grid_shape": list(build("lumped", 1.0)._build_grid().shape),
        "freqs_hz": FREQS_HZ.tolist(),
        "blocks": [],
    }

    print(f"rfx  {rfx.__file__}  ({record['git_commit']})")
    print(f"jax  {jax.__version__}   numpy {np.__version__}")
    print(
        f"Zc = {ETA0:.6f} ohm, L = {2 * DX * 1e3:.1f} mm, dx = {DX * 1e3:.1f} mm, "
        f"grid {tuple(record['grid_shape'])}"
    )
    print(f"freqs (GHz) {FREQS_HZ / 1e9}")

    for rr in LOADS:
        gamma = abs((rr - 1.0) / (rr + 1.0))
        block: dict = {"r_over_zc": rr, "closed_form_abs_s11": gamma, "lanes": {}}
        print(f"\nR = {rr} Zc   closed form |S11| = {gamma:.6f} at every bin")
        for kind in ("lumped", "wire"):
            out = measure(kind, rr)
            block["lanes"][kind] = out
            print(
                f"  {kind:6s} |S11| "
                f"{np.array2string(np.asarray(out['abs_s11']), precision=5)}"
            )
            print(
                f"  {kind:6s} V/(Zc I) re "
                f"{np.array2string(np.asarray(out['v_over_zc_i_re']), precision=4)}"
                f"  im "
                f"{np.array2string(np.asarray(out['v_over_zc_i_im']), precision=4)}"
            )
            for msg in out["nonpassive_warnings"]:
                print(f"  {kind:6s} WARN {msg[:150]}")
        record["blocks"].append(block)

    out_path = Path(__file__).with_name("lumped_port_known_load_line_results.json")
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path.name}")


if __name__ == "__main__":
    main()
