"""The coaxial line's open end is passive and settled.

An open-circuited lossless coaxial line reflects what it receives and no more,
|Gamma| <= 1, and once the line has rung down its reflection no longer changes
when the record is lengthened. The one-port lane as shipped before issue 1218
absorbed on z only, which left the line inside a closed PEC can: the field
leaving the open end was held between the shell and the can's walls and coupled
back, so the open read |Gamma| above 1 and moved with the record. Absorbing on
all three axes puts the line in open space.

This solves the open on the cheapest board where the shipped lane fails the
check: the coax chain battery's PTFE-filled SMA line at 4 annulus cells
(0.355 mm cells) on an 8 x 8 x 25 mm board, at 12 and 24 line traversals, and
holds it to the battery's bounds — max |Gamma| <= 1.02 at both records, and the
largest per-bin change of |Gamma| between them below 0.0259 (one tenth of the
2 dB magnitude gate at |Gamma| = 1). The sweep that chose it, the shipped lane
against the fixed one at 4 and 6 annulus cells on 40 mm and 25 mm boards, is
``rfx/records/20260924-coax-closed-can/open_check_config.json`` in rfx-archive:

  annulus  board   cell-steps   shipped lane (z only)          fixed lane (x, y, z)
  cells            (2 records)  max|Gamma| 12/24     shift     max|Gamma| 12/24   shift
  4        25 mm   2.2e9        1.0141 / 1.0151   0.0313 red   1.0051 / 1.0052   7e-5
  4        40 mm   4.8e9        1.0116 / 1.0153   0.0190       1.0002 / 1.0002   4e-5
  6        25 mm   6.2e9        1.0126 / 1.0086   0.0243       1.0008 / 1.0007   9e-5
  6        40 mm   1.4e10       1.0068 / 1.0146   0.0136       1.0006 / 1.0006   4e-5

The 25 mm board at 4 annulus cells is both the cheapest and the only one of
the four on which the shipped lane fails; it fails on the shift, not on the
peak. The case is built by the battery's own driver, so the live board is the
swept one, and the record lengths are asserted before the solve.

Lane: the pull-request lane, unmarked. It pins a number a user receives (the
open's |Gamma| <= 1.02), and a cheap physics lock runs before merge (PI,
2026-09-24). It takes 70 s on four pinned VESSL CPU cores (run 369367264656),
which ``.test_durations`` carries so the shards stay balanced. With both coax
lanes forced back to ``cpml_axes="z"`` inside the runner call, every helper
call kept, the same test fails on the shift, 0.0313 at 9.2 GHz (run
369367264657).
"""
from __future__ import annotations

import numpy as np

from tests import _chain_battery_drift as drift

DRIVER = "scripts/diagnostics/coax_chain_battery_measure.py"
RUNG = 4
BOARD_M = (0.008, 0.008, 0.025)
RECORD_UNITS = (12.0, 24.0)
RECORD_STEPS = (2200, 4400)           # the swept records' own step counts
MAX_ABS_GAMMA = 1.02
DOUBLING_SHIFT_MAX = (10 ** (2.0 / 20.0) - 1.0) / 10.0          # 0.025893


def test_the_open_end_is_passive_and_settled():
    driver = drift.load_driver(DRIVER)
    gammas = {}
    for units, want_steps in zip(RECORD_UNITS, RECORD_STEPS):
        sim = driver.build_sim(RUNG, "open", domain=BOARD_M)
        n_steps = driver.record_steps(sim._build_grid(), units)
        assert n_steps == want_steps, (
            f"{units:g} traversals of this board are {n_steps} steps today, "
            f"{want_steps} when the configuration was chosen: the board moved")
        res = driver.solve_one_port(sim, "open", n_steps=n_steps)
        g = np.asarray(res.s11)
        assert np.all(np.isfinite(g)), f"the {units:g}-traversal record is not finite"
        gammas[units] = g

    freqs = np.asarray(res.freqs, dtype=float)
    for units, g in gammas.items():
        k = int(np.argmax(np.abs(g)))
        assert np.abs(g[k]) <= MAX_ABS_GAMMA, (
            f"the open end reflects |Gamma| = {np.abs(g[k]):.5f} at {freqs[k] / 1e9:.2f} GHz "
            f"on the {units:g}-traversal record, above {MAX_ABS_GAMMA}: a lossless open "
            "returned more than it received")
    d = np.abs(np.abs(gammas[RECORD_UNITS[1]]) - np.abs(gammas[RECORD_UNITS[0]]))
    k = int(np.argmax(d))
    assert d[k] < DOUBLING_SHIFT_MAX, (
        f"doubling the record moved |Gamma| by {d[k]:.5f} at {freqs[k] / 1e9:.2f} GHz "
        f"(bound {DOUBLING_SHIFT_MAX:.5f}): the reflection had not settled")
