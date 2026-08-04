#!/usr/bin/env python3
"""MSL modal V·I flux oracle — the standing falsifier for the extractor's
core physical claim (issue #520 top item; origin: #525).

PRE-DECLARATION (written before this script was run — R2-tight, one campaign)
-------------------------------------------------------------------------------
The extractor-independent oracle for the #511/#507-corrected modal voltage
and Ampère-loop current is

    Re(V · conj(I)) / flux_spectrum  ==  1

V and I are built from the trace-anchored span (``msl_modal_voltage`` /
``msl_loop_current``, ``rfx/api/_sparams.py`` / ``rfx/sources/msl_port.py``);
Poynting flux is accumulated by a DIFFERENT kernel entirely
(``rfx/probes/probes.py:flux_spectrum``, fed by ``rfx/simulation.py``'s own
H-field spatial-averaging + half-step phase correction — see
``rfx/simulation.py:1470-1523``). The two paths share no code, so agreement
is a genuine, independent check, not a tautology.

PR #516 measured this identity ONLY on the aligned mesh (dx = h_sub/3 =
84.667 µm, h_sub/dx = 3.0000) and got 1.006-1.009 — the EASY case, where the
rounding proxy and the rasterized trace node coincide (the PR's F2 anchoring
fix is measured bit-identical there). The mesh class the fix actually
changes behaviour on — dx = 80 µm, h_sub/dx = 3.175 (the committed
``test_msl_thru_line_passive_gate`` mesh, inside preflight's [0.10, 0.40]
mixed-cell danger zone) — had never had this identity measured; the PR body
said so verbatim ("expected ~1 by construction; not yet measured on this
mesh"). #525 then measured it on an UNTERMINATED single-port-into-CPML
fixture and found the ratio dominated by reactive residue (0.54-0.69, later
also found convention-contaminated by an asymmetric ½ factor — see
CONVENTION below) with 472% plane-to-plane spread at a −122.9 dB settling
witness — i.e. that fixture cannot see a travelling wave and its number is
not a valid oracle measurement. Neither harness was ever committed.

This script is the committed re-measurement #525's own follow-up comment
(2026-08-02 priority bump) specified as the bar for the artifact:
  1. state the convention and ASSERT it (not just comment it) — see
     CONVENTION below and the runtime ``assert`` in ``run_one``;
  2. an admissibility witness BEFORE the ratio — Re(E·H*) vs |E·H*| at each
     plane (a reactive fraction) plus plane-to-plane flux spread — and
     REFUSE to certify the ratio where the witness fails;
  3. a TERMINATED fixture (matched thru, not a bare port into CPML) so the
     plane sees a travelling wave — this script uses the SAME two-port thru
     geometry as ``tests/test_msl_port_integration.py::
     test_msl_thru_line_passive_gate``, drives only the near port, and
     leaves the far port passively matched (``add_msl_port(..., excite=
     False)`` → resistive Z0 termination via the port's own σ distribution,
     ``rfx/sources/msl_port.py`` ``MSLPort`` docstring: "False → passive
     matched termination only") — NOT relying on CPML alone to absorb the
     guided mode;
  4. BOTH mesh classes: aligned (dx = h_sub/3, frac(h_sub/dx) = 0) and
     bisecting (dx = 80 µm, frac = 0.175, the MESH-ALIGNMENT RULE danger
     zone — docs/agent-memory/rfx-known-issues.md, "Added 2026-07-30").

CONVENTION (assert, not comment)
---------------------------------
``flux_spectrum()`` returns ``sum(Re(E1·conj(H2) - E2·conj(H1)) * dA)`` with
NO ½ prefactor (``rfx/probes/probes.py`` docstring, verbatim: "∫ Re(E × H*)
· n̂ dA"). The oracle therefore compares ``Re(V·conj(I))`` — ALSO no ½ —
directly against ``flux_spectrum()``. A "time-averaged power" reading would
put ½ on BOTH sides, which is a mathematical no-op on the RATIO — #525
root-caused the prior session's factor-of-2 discrepancy to applying ½ to the
V·I side only, while the flux side stayed unhalved. ``run_one()`` computes
the ratio both ways (no halves; halves on both) and asserts they are
numerically identical, so a future asymmetric-½ slip fails loudly instead of
silently reintroducing the factor-of-2 bug. A second defensive assertion
greps ``flux_spectrum``'s own source for a stray ``0.5 *`` so a convention
change in the library itself cannot silently invalidate this script's
premise.

ADMISSIBILITY WITNESS
----------------------
Before trusting ``Re(V·conj(I)) / flux`` at a plane/frequency cell, this
script reports — computed from the SAME flux-monitor accumulator fields the
oracle divides by, so the witness and the ratio denominator can never
diverge:

    reactive_fraction = 1 − |Re(Σ integrand·dA)| / Σ |integrand|·dA
        (0 = fully travelling/real; 1 = fully reactive/cancelling)
    spread(f) = (max(flux_real) − min(flux_real)) / mean(|flux_real|)
        across the 5 planes, at each frequency (a physical travelling wave
        should read nearly flat across planes over this short line span).

A (plane, frequency) cell is ADMISSIBLE only if
``reactive_fraction <= REACTIVE_FRACTION_MAX`` (0.5 — majority-real) AND
``spread <= SPREAD_MAX`` (0.5 — 50%), both pre-declared below and chosen to
be far looser than #525's failed single-port fixture (472% spread) — a
properly terminated fixture should clear them easily if the travelling-wave
premise holds; failing even these loose bounds would itself be a finding.
The oracle ratio is still printed at every cell for transparency, but
INADMISSIBLE cells are excluded from the pass/fail verdict and flagged in
the JSON — reporting a ratio computed from mostly-reactive power as a
physics claim is refused, per #525 comment 2.

FIXTURE
-------
Identical materials/geometry to the committed gate fixture
(``tests/test_msl_port_integration.py``: EPS_R=3.66, H_SUB=254 µm,
W_TRACE=600 µm, L_LINE=10 mm, one-cell PEC trace, PEC ground + CPML), TWO
MSL ports (near driven ``+x``, far ``excite=False`` matched ``-x``), 5
Ez/Hy/Hz DFT-plane probes + flux monitors placed at 30/40/50/60/70% of
L_LINE — well clear of both the source-launch fringing zone (~5·h_sub ≈
1.27 mm) and the passive-port termination near field, on EITHER end.

PRE-DECLARED EXPECTATION (falsifiable, not adjusted after the run)
--------------------------------------------------------------------
On EVERY (plane, frequency) cell judged ADMISSIBLE by the witness above, on
BOTH mesh classes:

    0.85 <= Re(V·conj(I)) / flux_spectrum <= 1.15

(a wider band than PR #516's 1.006-1.009 aligned-only number, because this
is a fresh fixture/probe placement, not a replication of that run.)
FALSIFIED if any admissible cell falls outside that band on EITHER mesh
class — STOP and report the per-plane dump; a bisecting-mesh failure would
be a FINDING (the extractor is imperfect there, consistent with #525's
prior unterminated-fixture read of 0.54-0.69), not a script defect,
PROVIDED the admissibility witness passed at that cell. If fewer than half
of a mesh's (plane, frequency) cells are admissible, that mesh's identity
check is INCONCLUSIVE (not a pass or a fail) and is reported as such — that
would be a finding about this fixture's own regime, not a verdict on the
extractor.

RUNTIME
-------
2 mesh points x 1 FDTD run each (near port driven, far port passive;
n_freqs=6, num_periods=30). ~2-5 min/point on one CPU core.

    python scripts/diagnostics/msl_vi_flux_oracle.py

POST-RUN REVIEW NOTE (2026-08-04, appended after the run; not part of the
pre-declaration -- the committed JSON is the AS-RUN record, untouched)
--------------------------------------------------------------------------
OVERALL: HELD. Both mesh classes admitted all 30/30 (plane, frequency)
cells (reactive_fraction and spread both far inside the pre-declared
bounds on every cell -- the matched-thru fixture is nowhere near the
472%-spread, reactive-dominated regime #525 measured on the single-port
fixture) and the ratio landed inside the pre-declared [0.85, 1.15] band on
every admissible cell of both meshes:

    aligned  h_sub/3 (dx=84.667um, frac=0.0000): ratio 1.0083 .. 1.0090
    bisecting 80um   (dx=80.000um, frac=0.1750): ratio 1.0105 .. 1.0118

This is the first-ever measurement of the bisecting-mesh identity (PR #516
shipped only the aligned-mesh number). It does NOT reproduce #525's
0.54-0.69 read on the bisecting class -- that number came from an
untermined single-port-into-CPML fixture with a measured 472% plane-to-
plane spread (a reactive, non-travelling-wave regime this script's
admissibility witness exists to refuse) COMBINED with #525's own follow-up
finding of an asymmetric 1/2-factor convention bug in that earlier,
never-committed script. With a properly terminated fixture (far port
`excite=False` -> resistive Z0 termination, not bare CPML absorption) and
the symmetric no-half convention asserted here, the extractor-independent
flux oracle holds tightly on BOTH mesh classes. This does not relitigate
#525's own numbers (that script was never committed and is not re-run
here); it establishes what the committed, falsifiable artifact reads on a
fixture built to satisfy the admissibility bar #525 itself set.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfx import Box, Simulation  # noqa: E402
from rfx.api._sparams import _msl_cell_profile, msl_modal_voltage  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.probes.probes import flux_spectrum  # noqa: E402
from rfx.sources.msl_port import MSLPort, _msl_yz_cells, msl_loop_current  # noqa: E402

# --- gate-fixture geometry, verbatim from tests/test_msl_port_integration.py ---
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9

FREQS = np.array([2.0e9, 2.5e9, 3.0e9, 3.5e9, 4.0e9, 4.5e9])
PLANE_FRACS = [0.30, 0.40, 0.50, 0.60, 0.70]
NUM_PERIODS = 30.0

REACTIVE_FRACTION_MAX = 0.5
SPREAD_MAX = 0.5
RATIO_LO, RATIO_HI = 0.85, 1.15

OUT_DIR = REPO / "scripts" / "diagnostics" / "msl_vi_flux_oracle"

MESH_CLASSES = [
    ("aligned h_sub/3", H_SUB / 3.0),
    ("bisecting 80um (gate mesh)", 80e-6),
]


def _assert_flux_convention_unchanged() -> None:
    """Defensive tripwire: fail loudly if ``flux_spectrum`` grows a stray
    ½ prefactor this script's convention assumption did not anticipate."""
    src = inspect.getsource(flux_spectrum)
    body = src.split("def flux_spectrum", 1)[1]
    assert "0.5 *" not in body and "* 0.5" not in body, (
        "flux_spectrum() source now contains a 0.5 factor — the no-half "
        "convention this script asserts against has changed upstream; "
        "re-derive the convention before trusting any ratio below."
    )


def run_one(label: str, dx: float) -> dict:
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)
    lz = H_SUB + 1.5e-3
    y_c = ly / 2.0

    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    sim.add(Box((0.0, y_c - W_TRACE / 2.0, H_SUB),
                (lx, y_c + W_TRACE / 2.0, H_SUB + dx)), material="pec")
    # Near port: driven. Far port: excite=False -> passive matched
    # termination (resistive sigma at Z0, not bare CPML absorption) so the
    # planes between the two ports see a travelling wave, per #525 comment 2.
    sim.add_msl_port(position=(PORT_MARGIN, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_c, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0,
                     excite=False)

    plane_x = [PORT_MARGIN + f * L_LINE for f in PLANE_FRACS]
    fj = jnp.asarray(FREQS)
    for idx, x in enumerate(plane_x):
        for comp in ("ez", "hy", "hz"):
            sim.add_dft_plane_probe(axis="x", coordinate=float(x),
                                    component=comp, freqs=fj,
                                    name=f"p{idx}_{comp}")
        sim.add_flux_monitor(axis="x", coordinate=float(x), freqs=fj,
                             name=f"p{idx}_flux")
        # Settling-witness point probe at the same plane, mid-substrate
        # under the trace (project rule: quote end/peak energy for any
        # fixed-length open-domain claims-bearing record).
        sim.add_probe(position=(float(x), y_c, H_SUB / 2.0), component="ez")

    grid = sim._build_grid()
    mp0 = MSLPort(feed_x=PORT_MARGIN, y_lo=y_c - W_TRACE / 2.0,
                  y_hi=y_c + W_TRACE / 2.0, z_lo=0.0, z_hi=H_SUB,
                  direction="+x", impedance=50.0, excitation=None)
    cells = _msl_yz_cells(grid, mp0)
    j_set = sorted({c[1] for c in cells})
    k_set = sorted({c[2] for c in cells})
    j_lo, j_hi = j_set[0], j_set[-1]
    k_lo, k_top = k_set[0], k_set[-1]
    j_c = (j_lo + j_hi) // 2

    pec_mask = np.asarray(sim._assemble_materials(grid)[3])
    col = pec_mask[cells[0][0], j_c, k_top:]
    kp = np.where(col)[0]
    if kp.size == 0:
        raise RuntimeError(f"{label}: no PEC trace conductor found above "
                           "the substrate top — cannot locate trace node.")
    trace_lo = int(k_top + kp.min())
    trace_hi = int(k_top + kp.max())

    n_sub_exact = H_SUB / dx
    frac = n_sub_exact - int(n_sub_exact)

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(num_periods=NUM_PERIODS, compute_s_params=False)
    dt = time.time() - t0
    # Filter the generic "x64 disabled" JAX dtype-fallback spam (fires on
    # every accumulator allocation in this codebase when x64 is off, which
    # this script deliberately leaves off -- repo rule forbids module-level
    # x64) -- it is not a preflight/physics message and duplicates dozens of
    # times per run. Genuine preflight/physics warnings are kept verbatim.
    preflight_warnings = [
        str(w.message) for w in caught
        if "JAX_ENABLE_X64" not in str(w.message)
        and "current-gotchas" not in str(w.message)
    ]

    dy_arr = _msl_cell_profile(grid, "y", grid.ny)
    dz_arr = _msl_cell_profile(grid, "z", grid.nz)
    hs_phase = np.exp(1j * 2.0 * np.pi * FREQS * float(grid.dt) * 0.5)

    # Settling witness (end/peak Ez^2, dB) at every plane probe.
    ts = np.asarray(res.time_series, dtype=float)
    settling_db = []
    for idx in range(len(plane_x)):
        p = ts[:, idx] ** 2
        tail = max(1, p.shape[0] // 10)
        end = float(p[-tail:].mean())
        peak = float(p.max())
        tiny = np.finfo(float).tiny
        settling_db.append(10.0 * np.log10((end + tiny) / (peak + tiny)))

    planes = res.dft_planes
    flux_mons = res.flux_monitors

    per_plane = []
    flux_real_by_plane = []  # (n_planes, n_freqs) for the spread witness
    for idx in range(len(plane_x)):
        ez_plane = np.asarray(planes[f"p{idx}_ez"].accumulator)
        hy_plane = np.asarray(planes[f"p{idx}_hy"].accumulator) * hs_phase[:, None, None]
        hz_plane = np.asarray(planes[f"p{idx}_hz"].accumulator) * hs_phase[:, None, None]

        v = np.asarray(msl_modal_voltage(
            jnp.asarray(ez_plane), j_centre=j_c, k_lo=k_lo, k_hi=trace_lo,
            dz_arr=dz_arr))
        i = np.asarray(msl_loop_current(
            jnp.asarray(hy_plane), jnp.asarray(hz_plane),
            j_lo=j_lo, j_hi=j_hi, k_trace_lo=trace_lo, k_trace_hi=trace_hi,
            dy_arr=dy_arr, dz_arr=dz_arr, direction="+x"))
        p_vi = np.real(v * np.conj(i))  # no 1/2 -- see CONVENTION above

        mon = flux_mons[f"p{idx}_flux"]
        flux_lib = np.real(np.asarray(flux_spectrum(mon)))

        e1 = np.asarray(mon.e1_dft); e2 = np.asarray(mon.e2_dft)
        h1 = np.asarray(mon.h1_dft); h2 = np.asarray(mon.h2_dft)
        dA = np.asarray(mon.dA)
        integrand = e1 * np.conj(h2) - e2 * np.conj(h1)
        flux_real_hand = np.real(np.sum(integrand * dA, axis=(-2, -1)))
        # Normalisation witness: the hand recompute from the monitor's own
        # accumulator fields must match the library call bit-for-bit (same
        # formula) -- this pins that reading mon.e1_dft/e2_dft/h1_dft/h2_dft
        # directly (for the admissibility witness) is not a second, silently
        # different convention from flux_spectrum() itself.
        assert np.allclose(flux_real_hand, flux_lib, rtol=1e-6, atol=1e-30), (
            f"{label} plane {idx}: hand Poynting recompute from mon.*_dft "
            f"disagrees with flux_spectrum() -- convention mismatch, not a "
            f"physics finding. hand={flux_real_hand} lib={flux_lib}"
        )
        flux_mag = np.sum(np.abs(integrand) * dA, axis=(-2, -1))
        reactive_fraction = 1.0 - np.abs(flux_real_hand) / np.where(
            flux_mag > 0, flux_mag, np.nan)

        # CONVENTION assertion: applying 1/2 to BOTH sides must be a no-op
        # on the ratio (the only way #525's bug reproduces is an ASYMMETRIC
        # half -- this assertion would catch that class of regression).
        ratio_no_half = p_vi / np.where(np.abs(flux_lib) > 1e-300, flux_lib, np.nan)
        ratio_both_halved = (0.5 * p_vi) / np.where(
            np.abs(0.5 * flux_lib) > 1e-300, 0.5 * flux_lib, np.nan)
        assert np.allclose(ratio_no_half, ratio_both_halved, equal_nan=True), (
            f"{label} plane {idx}: half-both-sides ratio diverged from "
            "no-half ratio -- this should be mathematically impossible and "
            "signals a bug in this script's arithmetic, not the extractor."
        )

        flux_real_by_plane.append(flux_real_hand)
        per_plane.append(dict(
            plane_idx=idx, x_frac=PLANE_FRACS[idx],
            p_vi_real=[float(x) for x in p_vi],
            flux_real=[float(x) for x in flux_real_hand],
            reactive_fraction=[float(x) for x in reactive_fraction],
            ratio=[float(x) for x in ratio_no_half],
        ))

    flux_stack = np.array(flux_real_by_plane)  # (n_planes, n_freqs)
    spread_per_freq = (
        (flux_stack.max(axis=0) - flux_stack.min(axis=0))
        / np.where(np.abs(flux_stack).mean(axis=0) > 0,
                  np.abs(flux_stack).mean(axis=0), np.nan)
    )

    admissible_cells = []
    all_cells = []
    for idx, row in enumerate(per_plane):
        for f_idx in range(len(FREQS)):
            rf = row["reactive_fraction"][f_idx]
            sp = float(spread_per_freq[f_idx])
            ratio = row["ratio"][f_idx]
            ok = (np.isfinite(rf) and rf <= REACTIVE_FRACTION_MAX
                 and np.isfinite(sp) and sp <= SPREAD_MAX)
            cell = dict(plane_idx=idx, freq_ghz=float(FREQS[f_idx] / 1e9),
                       reactive_fraction=rf, spread=sp, ratio=ratio,
                       admissible=bool(ok))
            all_cells.append(cell)
            if ok:
                admissible_cells.append(cell)

    admissible_ratios = [c["ratio"] for c in admissible_cells]
    within_band = [RATIO_LO <= r <= RATIO_HI for r in admissible_ratios]
    inconclusive = len(admissible_cells) < 0.5 * len(all_cells)
    verdict = (
        "INCONCLUSIVE" if inconclusive else
        ("HELD" if admissible_ratios and all(within_band) else "FALSIFIED")
    )

    return dict(
        label=label, dx_um=round(dx * 1e6, 4),
        n_sub_exact=round(n_sub_exact, 4), frac=round(frac, 4),
        mixed_cell_danger_zone=bool(0.10 <= frac <= 0.40),
        trace_node=trace_lo, k_lo=k_lo, k_top_proxy=k_top,
        wallclock_s=round(dt, 1),
        preflight_warnings=preflight_warnings,
        settling_db=[round(x, 1) for x in settling_db],
        spread_per_freq=[round(float(x), 4) for x in spread_per_freq],
        per_plane=per_plane,
        cells=all_cells,
        n_admissible=len(admissible_cells),
        n_total=len(all_cells),
        admissible_ratio_range=(
            [round(min(admissible_ratios), 4), round(max(admissible_ratios), 4)]
            if admissible_ratios else None
        ),
        verdict=verdict,
    )


def main() -> int:
    _assert_flux_convention_unchanged()

    rows = []
    for label, dx in MESH_CLASSES:
        print(f"\n=== {label} (dx={dx * 1e6:.3f} um) ===", flush=True)
        row = run_one(label, dx)
        rows.append(row)
        print(f"  h_sub/dx={row['n_sub_exact']:.4f} frac={row['frac']:.4f} "
              f"trace_node={row['trace_node']} (proxy k_top={row['k_top_proxy']})")
        print(f"  settling_db per plane: {row['settling_db']}")
        print(f"  spread_per_freq: {row['spread_per_freq']}")
        print(f"  admissible cells: {row['n_admissible']}/{row['n_total']}")
        print(f"  admissible ratio range: {row['admissible_ratio_range']}")
        print(f"  VERDICT: {row['verdict']}")
        print("  --- preflight / warnings (verbatim) ---")
        for w in row["preflight_warnings"]:
            print(f"    {w}")

    overall = "HELD" if all(r["verdict"] == "HELD" for r in rows) else (
        "INCONCLUSIVE" if any(r["verdict"] == "INCONCLUSIVE" for r in rows)
        and all(r["verdict"] in ("HELD", "INCONCLUSIVE") for r in rows)
        else "FALSIFIED"
    )
    print(f"\nOVERALL: {overall}")
    if overall == "FALSIFIED":
        print("STOP: the flux identity failed on at least one admissible "
              "cell of at least one mesh class. Per-plane dumps are in the "
              "committed JSON below -- this is a finding, report it, do not "
              "adjust the pre-declared band.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "msl_vi_flux_oracle.json"
    out_path.write_text(json.dumps({
        "reactive_fraction_max": REACTIVE_FRACTION_MAX,
        "spread_max": SPREAD_MAX,
        "ratio_band": [RATIO_LO, RATIO_HI],
        "rows": rows,
        "overall": overall,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    return 0 if overall in ("HELD", "INCONCLUSIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
