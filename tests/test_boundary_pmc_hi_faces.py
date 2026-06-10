"""Regression lock: PMC enforcement on every ``_hi`` face (fix 2026-04).

Background — discovered 2026-04-19 while building
``examples/crossval/09_half_symmetric_waveguide.py``:
``apply_pmc_faces`` zeroed the ghost half-cell ``0.5·dx`` OUTSIDE the
wall on ``_hi`` faces (array index ``-1``). In a Yee grid with forward-
difference update, that cell does not drive the interior E curl, so the
write was a silent no-op and the wall behaved as PEC. The pre-existing
oracle (:mod:`tests.test_boundary_pmc_oracle`) only exercised PMC on
``z_lo`` / ``y_lo`` / ``x_lo``, hiding the bug.

Fix (commit pending): both single-device (``rfx/boundaries/pmc.py``) and
distributed (``rfx/runners/distributed.py`` / ``distributed_v2.py`` /
``distributed_nu.py``) variants now zero ``H_tan`` at index ``-2``
(``last_real - 1`` for x-sharded code), mirroring the ``_lo`` pattern
at the half-cell ``0.5·dx`` INSIDE the wall.

This test pins the fix with a 1-D PMC-PEC cavity mirrored onto each of
the six faces. The quarter-wave mode ``f_0 = c/(4L)`` is the unambiguous
PMC-PEC signature — a PEC-PEC cavity would land at ``c/(2L) = 2·f_0``
and fail the 10 % gate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


_C0 = 299_792_458.0
_L = 0.02
_DX = 0.5e-3
_N_STEPS = 2048
_F0_QW = _C0 / (4.0 * _L)
_TRANS_WIDTH = 0.002  # transverse thickness for the 1-D-ish cavity


def _axis_kwargs(axis: str, side: str):
    """Return (domain, source_pos, probe_pos, spec, component) for a 1-D
    cavity along ``axis`` with PMC on the ``side`` face and PEC on the
    opposite face. Other two axes are periodic."""
    long_len = _L
    trans = _TRANS_WIDTH
    source_frac = 0.3 if side == "lo" else 0.7
    probe_frac = 0.7 if side == "lo" else 0.3

    if axis == "x":
        domain = (long_len, trans, trans)
        src = (source_frac * long_len, 0.5 * trans, 0.5 * trans)
        prb = (probe_frac * long_len, 0.5 * trans, 0.5 * trans)
        comp = "ey"
        other1, other2 = "y", "z"
    elif axis == "y":
        domain = (trans, long_len, trans)
        src = (0.5 * trans, source_frac * long_len, 0.5 * trans)
        prb = (0.5 * trans, probe_frac * long_len, 0.5 * trans)
        comp = "ex"
        other1, other2 = "x", "z"
    else:  # z
        domain = (trans, trans, long_len)
        src = (0.5 * trans, 0.5 * trans, source_frac * long_len)
        prb = (0.5 * trans, 0.5 * trans, probe_frac * long_len)
        comp = "ex"
        other1, other2 = "x", "y"

    if side == "lo":
        axis_bd = Boundary(lo="pmc", hi="pec")
    else:
        axis_bd = Boundary(lo="pec", hi="pmc")

    spec = BoundarySpec(**{axis: axis_bd, other1: "periodic", other2: "periodic"})
    return domain, src, prb, spec, comp


def _dominant_mode_freq(axis: str, side: str) -> float:
    domain, src_pos, prb_pos, spec, comp = _axis_kwargs(axis, side)
    sim = Simulation(
        freq_max=40e9, domain=domain, dx=_DX,
        boundary=spec, cpml_layers=0,
    )
    sim.add_source(src_pos, comp)
    sim.add_probe(prb_pos, comp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=_N_STEPS)
    ts = np.asarray(res.time_series)[:, 0]
    dt = float(res.dt)
    window = np.hanning(len(ts))
    spec_fft = np.abs(np.fft.rfft(ts * window))
    freqs = np.fft.rfftfreq(len(ts), dt)
    mask = (freqs > 0.3 * _F0_QW) & (freqs < 2.5 * _F0_QW)
    return float(freqs[mask][np.argmax(spec_fft[mask])])


@pytest.mark.parametrize(
    "axis,side",
    [
        ("x", "lo"), ("x", "hi"),
        ("y", "lo"), ("y", "hi"),
        ("z", "lo"), ("z", "hi"),
    ],
)
def test_pmc_quarter_wave_on_every_face(axis: str, side: str):
    """PMC-PEC cavity mirrored onto each of the six faces must land
    within 10 % of the quarter-wave analytic ``c/(4L)``. Before the
    Before the 2026-04 fix, ``_hi`` faces landed at ``c/(2L)`` (twice f_0,
    PEC-PEC behaviour) because PMC was a silent no-op on hi-side
    faces.
    """
    peak = _dominant_mode_freq(axis, side)
    rel_err_qw = abs(peak - _F0_QW) / _F0_QW
    rel_err_hw = abs(peak - 2.0 * _F0_QW) / (2.0 * _F0_QW)
    assert rel_err_qw < 0.10, (
        f"PMC on {axis}_{side}: peak at {peak/1e9:.3f} GHz vs "
        f"analytic quarter-wave {_F0_QW/1e9:.3f} GHz "
        f"(rel err {rel_err_qw:.3%}). rel err to PEC-PEC half-wave "
        f"{2.0*_F0_QW/1e9:.3f} GHz: {rel_err_hw:.3%}. The hi-face "
        f"silent-drop bug (pre-2026-04 fix) gives PEC-PEC behaviour here."
    )
