"""A port's load reaches the dispersive E update on the graded-mesh lane (#1257).

A 16 mm PEC box on 1 mm cells, graded to 0.5 mm in z through its middle,
holds a port at the centre and an Ez probe 3 mm away. A lumped port is a
resistor across one cell edge: its conductance ``sigma = d_par / (Z * d_perp1
* d_perp2)`` is stamped into the material arrays, and the E update loses the
cavity's energy into it. A 50 ohm port therefore damps the cavity ring-down
and a 5000 ohm port barely does: the late-time (steps 300-600) probe
amplitude of the 50 ohm run is about 0.007 of the 5000 ohm run.

A Debye or Lorentz material anywhere in the model -- here a 2 mm cube in a
far corner -- puts the whole grid on the dispersive E update, which reads
only the coefficients ``init_debye`` / ``init_lorentz`` build from the
material arrays. The graded lane built them BEFORE it stamped the port loads,
so the loads never reached the update: every port was an open circuit. Since
#1256 the drive is built with the port's own load, so the two runs were no
longer bit-identical (as the issue measured before that) but proportional --
the 50 ohm trace was the 5000 ohm trace times the ratio of the two drive
coefficients, ``(1 + s_5000*dt/2e) / (1 + s_50*dt/2e)`` = 0.572 for this
lumped port, 0.255 for the 2 mm wire port, and the late-time ratio read that
instead of ~0.007. The uniform lane builds the coefficients after the stamps.

What is checked:

1. graded lane, lumped / wire / MSL port, Debye / Lorentz / both: the 50/5000
   ohm late-time ratio stays within a factor 1.5 of the same board without
   the dispersive block. The block is a 2 mm cube in a far corner; it moves
   the cavity's modes a little but cannot change how strongly a port absorbs.
   Measured after the fix: 0.999-1.003 (lumped), 1.000 (wire), 0.88 (MSL);
   with the load missing: 77, 40 and 4.2. On the MSL board the corner lies
   under the ground plane, where the field does not reach (the Debye and
   Lorentz boards agree to six digits), so its 0.88 comes from switching the
   whole grid to the dispersive E update, not from the block.
2. lane parity: the graded lane on uniform-valued cells (``dz_profile`` all
   1 mm) and the uniform lane, same board with the Debye block. The two lanes
   read a lumped port's waveform in units a factor ``d_perp1 * d_perp2`` =
   dx**2 apart (``test_wire_port_drive_cb_1256`` records why) and their dt
   differs in the last bits, so the checks are the unit-free 50/5000 ratio
   (1 % bar; measured 2.9e-4, missing load 1900x) and the 50 ohm trace times
   dx**2 against the uniform trace, relative to its peak (1e-4 bar: float32
   fields through two different stepping codes; measured 2.3e-6).
3. the defect restored with every call kept: ``init_debye`` / ``init_lorentz``
   are still called where they now are, but handed the materials as
   assembled, before any port stamp -- what the old order gave them. Both
   dispersive builds restored (a) turn every check above red; the Debye build
   alone (b) turns the Debye boards red, the Lorentz build alone the Lorentz
   boards.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from unittest import mock

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.runners import nonuniform as _nu_runner

DX = 1e-3
L = 16 * DX
N_STEPS = 600
DZ_GRADED = np.concatenate([np.full(6, DX), np.full(8, DX / 2), np.full(6, DX)])
DZ_FLAT = np.full(16, DX)
PULSE = GaussianPulse(f0=5e9, bandwidth=0.8)
Z_LOW, Z_HIGH = 50.0, 5000.0

# Pre-declared bars (module docstring).
BLOCK_BAND = 1.5
LANE_RATIO_RTOL = 1e-2
LANE_TRACE_RTOL = 1e-4


def _add_dispersive(sim, disp):
    if disp in ("debye", "both"):
        sim.add_material("debye", eps_r=2.0,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box((1 * DX, 1 * DX, 1 * DX), (3 * DX, 3 * DX, 3 * DX)),
                material="debye")
    if disp in ("lorentz", "both"):
        # resonance at 20 GHz, above the 0-10 GHz band of the pulse
        sim.add_material("lorentz", eps_r=2.0, lorentz_poles=[lorentz_pole(
            delta_eps=1.0, omega_0=2 * np.pi * 20e9, delta=2 * np.pi * 1e9)])
        sim.add(Box((13 * DX, 1 * DX, 1 * DX), (15 * DX, 3 * DX, 3 * DX)),
                material="lorentz")


def _board(port, disp, impedance, mesh):
    kw = {"graded": {"dz_profile": DZ_GRADED},
          "flat": {"dz_profile": DZ_FLAT},
          "uniform": {}}[mesh]
    sim = Simulation(freq_max=10e9, domain=(L, L, L), dx=DX, boundary="pec",
                     **kw)
    _add_dispersive(sim, disp)
    if port == "lumped":
        sim.add_port((8 * DX, 8 * DX, 8 * DX), "ez", impedance=impedance,
                     waveform=PULSE)
        sim.add_probe((11 * DX, 8 * DX, 8 * DX), "ez")
    elif port == "wire":
        # four 0.5 mm Ez edges on the graded board
        sim.add_port((8 * DX, 8 * DX, 7 * DX), "ez", impedance=impedance,
                     extent=2 * DX, waveform=PULSE)
        sim.add_probe((11 * DX, 8 * DX, 8 * DX), "ez")
    elif port == "msl":
        # a 2 mm trace on a 2 mm, eps_r 3 substrate over a ground plane
        sim.add_material("sub", eps_r=3.0)
        sim.add(Box((0, 0, 3 * DX), (L, L, 5 * DX)), material="sub")
        sim.add(Box((0, 0, 3 * DX), (L, L, 3 * DX)), material="pec")
        sim.add(Box((0, 7 * DX, 5 * DX), (L, 9 * DX, 5 * DX)), material="pec")
        sim.add_msl_port((4 * DX, 8 * DX, 3 * DX), width=2 * DX,
                         height=2 * DX, direction="+x", impedance=impedance,
                         waveform=PULSE)
        sim.add_probe((11 * DX, 8 * DX, 4 * DX), "ez")
    else:
        raise ValueError(port)
    return sim


@contextlib.contextmanager
def _dispersion_built_before_the_stamps(*kinds):
    """Mutation: ``init_<kind>`` is still called, once, where the runner now
    calls it, but handed the materials as ``assemble_materials_nu`` returned
    them -- before any port stamp, which is what the old order built them
    from (these boards carry no override and no RLC fold in between)."""
    assembled = {}
    real_assemble = _nu_runner.assemble_materials_nu

    def _assemble(*a, **kw):
        out = real_assemble(*a, **kw)
        assembled["materials"] = out[0]
        return out

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(_nu_runner, "assemble_materials_nu", _assemble))
        for kind in kinds:
            real = getattr(_nu_runner, f"init_{kind}")

            def _pre_stamp(poles, materials, dt, *a, _real=real, **kw):
                return _real(poles, assembled["materials"], dt, *a, **kw)

            stack.enter_context(
                mock.patch.object(_nu_runner, f"init_{kind}", _pre_stamp))
        yield


_KINDS = {None: (), "debye": ("debye",), "lorentz": ("lorentz",),
          "both": ("debye", "lorentz")}


def _run(port, disp, impedance, mesh, mutation=()):
    """``(late-time max |Ez|, probe trace)`` of one run. A mutation of a
    build the board does not use is the unmutated run."""
    return _run_cached(port, disp, impedance, mesh,
                       tuple(k for k in mutation if k in _KINDS[disp]))


@functools.lru_cache(maxsize=None)
def _run_cached(port, disp, impedance, mesh, mutation):
    sim = _board(port, disp, impedance, mesh)
    with _dispersion_built_before_the_stamps(*mutation):
        r = sim.run(n_steps=N_STEPS, skip_preflight=True)
    ts = np.asarray(r.time_series, dtype=np.float64).reshape(N_STEPS, -1)[:, 0]
    return float(np.max(np.abs(ts[N_STEPS // 2:]))), ts


def _split(port, disp, mesh="graded", mutation=()):
    """The 50/5000 ohm late-time probe ratio."""
    return (_run(port, disp, Z_LOW, mesh, mutation)[0]
            / _run(port, disp, Z_HIGH, mesh, mutation)[0])


def _block_effect(port, disp, mutation=()):
    """How far the dispersive block moves the port's split, as a factor."""
    return _split(port, disp, mutation=mutation) / _split(port, None)


def _inside_band(factor):
    return 1.0 / BLOCK_BAND <= factor <= BLOCK_BAND


# --------------------------------------------------------------------------
# 1. the port absorbs on the graded lane with a dispersive block present
# --------------------------------------------------------------------------

_SPLIT_CASES = [("lumped", "debye"), ("lumped", "lorentz"),
                ("lumped", "both"), ("wire", "debye"), ("wire", "lorentz"),
                ("msl", "debye"), ("msl", "lorentz")]


@pytest.mark.parametrize("port,disp", _SPLIT_CASES)
def test_graded_port_load_reaches_the_dispersive_update(port, disp):
    factor = _block_effect(port, disp)
    print(f"[split] {port}+{disp}: 50/5000 late ratio "
          f"{_split(port, disp):.6f} (no block {_split(port, None):.6f}), "
          f"factor {factor:.4f}", file=sys.stderr)
    assert _inside_band(factor), (
        f"{port} port with a {disp} block: the 50/5000 ohm late-time ratio "
        f"is {factor:.3f}x the ratio without the block -- the port's load "
        f"is missing from the dispersive E update")


# --------------------------------------------------------------------------
# 2. lane parity on uniform-valued cells
# --------------------------------------------------------------------------

def _lane_parity(mutation=()):
    ratio_u = _split("lumped", "debye", "uniform")
    ratio_g = _split("lumped", "debye", "flat", mutation)
    trace_u = _run("lumped", "debye", Z_LOW, "uniform")[1]
    trace_g = _run("lumped", "debye", Z_LOW, "flat", mutation)[1]
    trace = np.max(np.abs(trace_g * DX * DX - trace_u)) / np.max(np.abs(trace_u))
    return ratio_g / ratio_u - 1.0, trace, ratio_u, ratio_g


def test_graded_lane_on_uniform_cells_matches_the_uniform_lane():
    d_ratio, d_trace, ratio_u, ratio_g = _lane_parity()
    print(f"[lane] 50/5000 ratio uniform {ratio_u:.6e} graded {ratio_g:.6e} "
          f"(rel {d_ratio:.2e}); 50 ohm trace rel {d_trace:.2e}", file=sys.stderr)
    assert abs(d_ratio) <= LANE_RATIO_RTOL, (ratio_u, ratio_g)
    assert d_trace <= LANE_TRACE_RTOL, d_trace


# --------------------------------------------------------------------------
# 3. the defect restored, every call kept
# --------------------------------------------------------------------------

def test_mutation_both_builds_before_the_stamps_sends_everything_red():
    """(a): both dispersive builds handed the pre-stamp materials."""
    both = ("debye", "lorentz")
    factors = {case: _block_effect(*case, mutation=both)
               for case in _SPLIT_CASES}
    d_ratio, d_trace, _, _ = _lane_parity(mutation=both)
    print("[mutation a] " + ", ".join(f"{p}+{d} {f:.3f}"
                                      for (p, d), f in factors.items())
          + f"; lane ratio rel {d_ratio:.3e}, trace rel {d_trace:.3e}", file=sys.stderr)
    assert not any(_inside_band(f) for f in factors.values()), factors
    assert abs(d_ratio) > LANE_RATIO_RTOL
    assert d_trace > LANE_TRACE_RTOL


@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_mutation_one_build_before_the_stamps_sends_its_boards_red(kind):
    """(b): only one of the two builds handed the pre-stamp materials. With
    both kinds in one model the E update takes the load through the Lorentz
    coefficients (``rfx.nonuniform._update_e_nu_dispersive``, mixed branch),
    so the mixed board goes red with the Lorentz build and not with the
    Debye one."""
    cases = [c for c in _SPLIT_CASES
             if c[1] == kind or (c[1] == "both" and kind == "lorentz")]
    factors = {case: _block_effect(*case, mutation=(kind,)) for case in cases}
    print(f"[mutation b {kind}] " + ", ".join(
        f"{p}+{d} {f:.3f}" for (p, d), f in factors.items()), file=sys.stderr)
    assert not any(_inside_band(f) for f in factors.values()), factors
