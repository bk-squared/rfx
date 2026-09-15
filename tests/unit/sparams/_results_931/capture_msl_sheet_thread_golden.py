"""Recapture the MSL sheet-threading golden under the ownership contract (#931).

This is the module's OWN documented capture procedure, nothing else:
``build_msl_thru(sheet=None)`` then
``compute_msl_s_matrix(freqs=FREQS, num_periods=12.0)``, exactly as
``tests/unit/sparams/test_msl_sheet_threading.py`` describes it.  It is a
file rather than a heredoc inside the VESSL yaml because VESSL's run
wrapper mangles an embedded heredoc (measured: run 369367259166 died with
``syntax error: unexpected end of file (expecting ")")`` before a single
line of the job ran), and because a capture procedure that produces a
committed golden should be reviewable on the branch, not buried in job
YAML.

Why the golden has to be recaptured at all: ``golden_msl_sheet_thread_s_13de212.npy``
records a board this tree no longer builds.  Its trace was a one-cell PEC
Box on a mesh that bisects the laminate face (dx = 80 um against
h_sub = 254 um, so h_sub/dx = 3.175 and the single realized wall landed at
node 4 = 320 um, 26 % above the declared board).  Under the contract the
foil is a SHEET on an on-lattice board (dx = h_sub/3, one wall at node 3 =
254 um exactly).  Both the realization and the mesh moved, so byte identity
against the old file cannot hold and must not be relaxed into a tolerance.

Two guards, both refusals rather than warnings:

* the trace must actually be declared as a sheet before anything is
  captured — a golden taken from a slab would be silently wrong;
* the capture is run TWICE and nothing is written unless the two are
  byte-equal, because a golden that is not reproducible on its own machine
  is not a golden.

Run it from the repository root with ``JAX_PLATFORMS=cpu``.  It writes
``tests/fixtures/golden_msl_sheet_thread_{s,freqs}_931.npy``; the pre-#931
files stay beside them as history.  Committing the result is the ingest
phase's job, not this script's.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

import tests.unit.sparams.test_msl_sheet_threading as M
from tests._realized_geometry import realized

_N_CAPTURES = 2
_OUT = Path("tests/fixtures")
_OLD = _OUT / "golden_msl_sheet_thread_s_13de212.npy"


def _witness() -> None:
    """State the geometry the golden is about to record, before recording it."""
    sim = M.build_msl_thru(sheet=None)
    rz = realized(sim)
    print("dx             =", rz.grid.dx)
    print("h_sub / dx     =", M.H_SUB / rz.grid.dx)
    print("sheet planes   =", rz.sheet_planes)
    print("wall planes z  =", rz.wall_planes(2))
    print("owns any cell  =",
          None if rz.pec_mask is None else bool(np.asarray(rz.pec_mask).any()))
    if not rz.sheet_planes.get(2):
        raise SystemExit(
            "REFUSED: the trace is not declared as a z-normal sheet, so this "
            "capture would freeze a volume conductor into the golden. Fix the "
            "fixture, not this script.")
    walls = rz.wall_planes(2)
    if len(walls) != 1:
        raise SystemExit(
            f"REFUSED: a sheet realizes ONE wall plane; got {walls}.")


def _capture(i: int):
    sim = M.build_msl_thru(sheet=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=M.FREQS, num_periods=12.0)
    for w in caught:
        print(f"[capture {i}] WARN {w.message}")
    print(f"[capture {i}] settling_db = {np.asarray(res.settling_db)}")
    return np.asarray(res.S), np.asarray(res.freqs)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _witness()

    caps = [_capture(i) for i in range(_N_CAPTURES)]
    s0, f0 = caps[0]
    s1, _ = caps[1]
    same = bool(np.array_equal(s0, s1))
    print("deterministic (two captures byte-equal):", same)
    print("max |cap0 - cap1| =", float(np.max(np.abs(s0 - s1))))
    if not same:
        raise SystemExit(
            "REFUSED: the capture is not deterministic on this platform; a "
            "golden taken from it would pin noise. Do not commit.")

    np.save(_OUT / "golden_msl_sheet_thread_s_931.npy", s0)
    np.save(_OUT / "golden_msl_sheet_thread_freqs_931.npy", f0)
    print("wrote golden_msl_sheet_thread_{s,freqs}_931.npy", s0.dtype, s0.shape)

    if _OLD.exists():
        s_old = np.load(_OLD)
        print("max |new - pre931 golden| =", float(np.max(np.abs(s0 - s_old))))
        print("|S11| pre-931 :", np.abs(s_old[0, 0]).round(5))
        print("|S11| post    :", np.abs(s0[0, 0]).round(5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
