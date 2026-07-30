"""One-cell vs two-cell PEC trace: the step is a MODEL change, not a thickness.

Backs the numbers quoted in ``Simulation.add_thin_conductor``'s docstring
(issue #504). A reviewer correctly objected that those figures sat in permanent
public text with no committed artifact behind them; this is the artifact.

WHAT IS BEING SHOWN
-------------------
``rfx/boundaries/pec.py:88-118`` zeroes a tangential E component only where the
PEC mask has a neighbour along that component's own axis, and says so:
"For thin PEC sheets (1 cell thick), only tangential E-components are zeroed;
the normal component is preserved (represents surface charge)."

So a ONE-cell trace is a surface and a TWO-cell trace is a volume conductor.
Going from one cell to two does not make the strip "thicker" within one model —
it swaps the model. The giveaway is the direction of the impedance shift: a
genuinely thicker strip has MORE fringing capacitance and therefore LOWER Z0
(Wheeler / Hammerstad thickness correction), so a *higher* Z0 for the thicker
stamp cannot be a thickness effect.

MEASURED (2026-07-29, this script, CPU)
---------------------------------------
Fixture: MSL thru, RO4350B-like eps_r=3.66, h_sub=254 um, w=600 um, L=10 mm,
dx=84.67 um (= h_sub/3), ports at both ends, band 3.0-4.5 GHz,
``compute_msl_s_matrix(n_freqs=30, num_periods=12, enforce_passivity=False)``.

    trace stamp        PEC z-layers   mean|S11|   mean|S21|   Re(Z0)
    1 cell (surface)        1           0.2233      1.0195     38.77 ohm
    2 cells (volume)        2           0.2017      1.0100     41.86 ohm

The thicker stamp reads the HIGHER impedance (+3.09 ohm, +8.0%), i.e. backwards
for a thickness effect — consistent with the boundary condition changing.

Two further observations recorded rather than chased here:
  * Re(Z0) = 38.8 ohm sits 19% below the analytic Hammerstad-Jensen anchor of
    47.89 ohm that the extractor normalises S11 against, which alone accounts
    for |Gamma| = 0.105, about 47% of the measured 0.2233 floor.
  * mean|S21| > 1 on a lossless passive thru at both stamps. Non-physical;
    an extraction/normalisation matter, not geometry.

RUNTIME
-------
Two FDTD runs, ~6 min each on one CPU core. Deliberately not a pytest gate:
nothing in the library depends on these exact values, they exist to make a
docstring claim checkable.

    python scripts/diagnostics/thin_conductor_cell_thickness_probe.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Import the rfx in THIS checkout, not whatever is pip-installed. Running a
# script file puts the script's own directory on sys.path, never the cwd, so
# `import rfx` silently resolved to the installed package — which cost a
# session's worth of confusion during issue #511 when this probe reported
# pre-fix numbers from a post-fix tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

H_SUB = 254e-6
W_TRACE = 600e-6
DX = 84.67e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
EPS_R = 3.66
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9


def run_one(n_cells: int) -> dict:
    """Stamp the trace `n_cells` cells thick and extract the MSL S-matrix."""
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
    lz = H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(
        Box((0.0, y_c - W_TRACE / 2.0, H_SUB),
            (lx, y_c + W_TRACE / 2.0, H_SUB + n_cells * DX)),
        material="pec",
    )
    for x, direction in ((PORT_MARGIN, "+x"), (PORT_MARGIN + L_LINE, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=W_TRACE,
                         height=H_SUB, direction=direction, impedance=50.0)

    grid = sim._build_grid()
    pec = np.asarray(sim._assemble_materials(grid)[3])
    k0 = int(round(H_SUB / DX))
    layers = int(pec[pec.shape[0] // 2, pec.shape[1] // 2, k0:k0 + 6].sum())

    res = sim.compute_msl_s_matrix(n_freqs=30, num_periods=12,
                                   enforce_passivity=False)
    f = np.asarray(res.freqs)
    band = (f >= GATE_F_LO) & (f <= GATE_F_HI)
    S = np.asarray(res.S)
    return {
        "n_cells": n_cells,
        "pec_layers": layers,
        "mean_s11": float(np.abs(S[0, 0, band]).mean()),
        "mean_s21": float(np.abs(S[1, 0, band]).mean()),
        "re_z0": float(np.asarray(res.Z0)[0, band].real.mean()),
        "settling_db": None if res.settling_db is None
        else [round(float(v), 1) for v in np.asarray(res.settling_db)],
    }


def main() -> None:
    # Preflight output is part of the result — do not suppress it.
    rows = []
    for n in (1, 2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = run_one(n)
        rows.append(row)
        print(f"\n--- n_cells={n} preflight/warnings ---")
        for w in caught:
            print(f"  [{w.category.__name__}] {w.message}")
        print(
            f"n_cells={row['n_cells']} pec_layers={row['pec_layers']} "
            f"mean|S11|={row['mean_s11']:.4f} mean|S21|={row['mean_s21']:.4f} "
            f"Re(Z0)={row['re_z0']:.2f} settling_db={row['settling_db']}",
            flush=True,
        )

    d_z0 = rows[1]["re_z0"] - rows[0]["re_z0"]
    print(
        f"\nVERDICT: two cells read {d_z0:+.2f} ohm "
        f"({100 * d_z0 / rows[0]['re_z0']:+.1f}%) vs one cell. A genuinely "
        "thicker strip would read LOWER Z0 (more fringing capacitance), so a "
        "positive shift is the signature of the PEC boundary condition "
        "changing from surface to volume — not of thickness."
    )


if __name__ == "__main__":
    main()
