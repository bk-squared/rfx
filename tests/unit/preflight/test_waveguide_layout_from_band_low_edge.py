"""Falsifiers for the near-cutoff layout note (post-v1.8 plan item 5, section 2-2).

``docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md``
section 2-2. The note says, in input units, how many guide wavelengths at the
band's LOWEST bin this layout puts between the port and the far wall, and how
many of those sit in the absorber. It carries no threshold: the only thing a
reviewer can check is that the arithmetic is right and that it fires in the
regime it was measured in, so every number it prints is recomputed here from
the fixture's declared dimensions and the grid the runner builds.

No FDTD anywhere in this file. The independent oracles are the same two the
sibling record-length audits use
(``tests/unit/preflight/test_waveguide_setup_audits.py``): the analytic
discrete TE10 cutoff of an ``N``-cell PEC guide, and ``far_path`` / the pad
recomputed from the fixture constants and the grid's own pad counts.
"""

from __future__ import annotations

import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F


C0 = 299_792_458.0
RUNG_M = F.DX_LADDER[2]          # a / 36, the fine rung of the battery ladder
RUNG_N = F.N_LADDER[2]
CODE = "layout_measured_from_band_low_edge"


def _sinc(x: float) -> float:
    return 1.0 if x == 0.0 else math.sin(x) / x


def _analytic_te10_cutoff_hz(n_cells: int) -> float:
    """Discrete TE10 cutoff of an ``n_cells``-wide PEC guide of width ``A_M``."""
    return (C0 / (2.0 * F.A_M)) * _sinc(math.pi / (2 * n_cells))


def _lambda_g_m(f_hz: float, fc_hz: float) -> float:
    """lambda_0 / sqrt(1 - (fc/f)^2), the note's own definition."""
    return (C0 / f_hz) / math.sqrt(1.0 - (fc_hz / f_hz) ** 2)


def _built(dx: float = RUNG_M, num_periods: float = 20.0):
    """``(sim, grid, cfgs, n_steps)`` through the RUNNER's own builders."""
    sim = F.build_simulation("thru", dx)
    grid = sim._build_grid()
    n_steps = int(grid.num_timesteps(num_periods))
    cfgs = [sim._build_waveguide_port_config(e, grid, jnp.asarray(F.FREQS), n_steps)
            for e in sim._waveguide_ports]
    return sim, grid, cfgs, n_steps


def _collect(sim, freqs, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim._validate_cfg_layout_from_band_low_edge(
            warnings, freqs=freqs, **kwargs)
    return [w.message for w in rec]


def _number_after(text: str, marker: str) -> float:
    tail = text.split(marker, 1)[1]
    return float(re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tail).group(0))


# A band whose lowest bin sits 3 % above this guide's own discrete cutoff --
# inside the near-cutoff regime the WR-90 validity-envelope sweep measured, and
# the regime the note is gated to.
NEAR_CUTOFF_FREQS = np.linspace(6.75e9, 8.0e9, 9)


def test_it_is_silent_at_the_batterys_own_band():
    """f_min / f_c = 8.4 / 6.555 = 1.28 -- far from cutoff, so no note.

    This is the half of the gate that matters: a note that fired on every
    healthy WR-90 setup would be noise, and the #470 advisory-flooding lesson
    says noise buries the genuine findings.
    """
    sim, grid, cfgs, n_steps = _built()
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    ratio = float(min(F.FREQS)) / f_c
    assert ratio == pytest.approx(1.28, abs=0.01), ratio      # the regime claim

    msgs = _collect(sim, F.FREQS, grid=grid, cfgs=cfgs, n_steps=n_steps)
    assert [getattr(m, "code", None) for m in msgs] == []


def test_it_fires_near_cutoff_and_every_number_is_a_hand_computation():
    sim, grid, cfgs, n_steps = _built()

    # --- hand computation, from the fixture's declared dimensions ---------
    f_min = float(NEAR_CUTOFF_FREQS.min())
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    ratio = f_min / f_c
    assert ratio < 1.06, ratio            # the regime this test is about
    lam_g = _lambda_g_m(f_min, f_c)
    pad_m = int(grid.pad_x_hi) * float(grid.dx)
    far_path = (F.DOMAIN_X_M - F.PORT_LEFT_X_M) + pad_m

    msgs = _collect(sim, NEAR_CUTOFF_FREQS,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    hits = [m for m in msgs if getattr(m, "code", None) == CODE]
    assert len(hits) == 2, [getattr(m, "code", None) for m in msgs]  # one per port
    assert all(m.severity == "info" for m in hits)

    left = next(m for m in hits if m.loc == "waveguide_port[0]")
    text = str(left)
    assert _number_after(text, "f_min/f_c = ") == pytest.approx(ratio, rel=1e-6)
    assert _number_after(text, "f_min = ") == pytest.approx(f_min / 1e9, rel=1e-6)
    assert _number_after(text, "cutoff f_c = ") == pytest.approx(f_c / 1e9, rel=1e-6)
    assert _number_after(text, "lambda_g(f_min) = ") == pytest.approx(
        lam_g * 1e3, rel=1e-6)
    assert _number_after(text, "gives ") == pytest.approx(
        far_path / lam_g, rel=1e-6)
    assert _number_after(text, "far_path = ") == pytest.approx(
        far_path * 1e3, rel=1e-6)
    assert _number_after(text, "in the absorber (pad = ") == pytest.approx(
        pad_m * 1e3, rel=1e-6)
    # The absorber's own share, printed just before "in the absorber".
    pad_share = float(re.search(
        r"and (-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?) in the absorber",
        text).group(1))
    assert pad_share == pytest.approx(pad_m / lam_g, rel=1e-6)
    # The measured lesson travels with the numbers, not in a separate doc.
    assert "size the layout from below the band" in text
    assert "T/tau 16 (+16 %)" in text
    assert "+3 %" in text


def test_the_right_port_reports_its_own_far_path():
    """The note is per port, and the two ports of this fixture are mirror
    images, so the '-x' port's far_path is measured to the LO wall."""
    sim, grid, cfgs, n_steps = _built()
    f_min = float(NEAR_CUTOFF_FREQS.min())
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    lam_g = _lambda_g_m(f_min, f_c)
    pad_m = int(grid.pad_x_lo) * float(grid.dx)
    far_path = F.PORT_RIGHT_X_M + pad_m

    msgs = _collect(sim, NEAR_CUTOFF_FREQS,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    right = next(m for m in msgs
                 if getattr(m, "code", None) == CODE
                 and m.loc == "waveguide_port[1]")
    text = str(right)
    assert "(-x)" in text
    assert _number_after(text, "far_path = ") == pytest.approx(
        far_path * 1e3, rel=1e-6)
    assert _number_after(text, "gives ") == pytest.approx(
        far_path / lam_g, rel=1e-6)


def test_a_band_at_or_below_the_ports_own_cutoff_is_silent():
    """lambda_g is undefined there; the record-length audit already says so,
    and two checks describing one unusable band is the flooding class."""
    sim, grid, cfgs, n_steps = _built()
    f_c = _analytic_te10_cutoff_hz(RUNG_N)
    below = np.linspace(0.9 * f_c, 1.1 * f_c, 5)
    msgs = _collect(sim, below, grid=grid, cfgs=cfgs, n_steps=n_steps)
    assert [m for m in msgs if getattr(m, "code", None) == CODE] == []


def test_the_gate_is_the_ratio_not_the_geometry():
    """One rung finer, same layout in cells: the note still fires near cutoff
    and stays silent at the battery's band. The gate reads f_min/f_c only."""
    sim, grid, cfgs, n_steps = _built(dx=F.DX_LADDER[1])
    near = _collect(sim, NEAR_CUTOFF_FREQS,
                    grid=grid, cfgs=cfgs, n_steps=n_steps)
    far = _collect(sim, F.FREQS, grid=grid, cfgs=cfgs, n_steps=n_steps)
    assert len([m for m in near if getattr(m, "code", None) == CODE]) == 2
    assert [m for m in far if getattr(m, "code", None) == CODE] == []


def test_the_hook_runs_it_alongside_the_other_two_audits():
    """``_preflight_waveguide_setup`` is the single hook both waveguide
    S-parameter entry points call; a check wired only into its own method
    would never reach a user."""
    sim, grid, cfgs, n_steps = _built()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim._preflight_waveguide_setup(
            warnings, freqs=NEAR_CUTOFF_FREQS, num_periods=20.0,
            grid=grid, cfgs=cfgs, n_steps=n_steps,
        )
    codes = [getattr(w.message, "code", None) for w in rec]
    assert codes.count(CODE) == 2, codes
