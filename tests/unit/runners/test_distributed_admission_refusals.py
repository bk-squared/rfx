"""The distributed lane refuses five silent-wrong classes instead of running them.

Five features reached ``run(devices=...)`` with **no refusal and no warning**
and came back wrong. The map is
``rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md``
S3-S4 and its survey ``decomposition-survey.md`` SC1; this file is the
first layer of that note's distributed preflight -- the position-independent
admission check plus the one position-dependent slab check.

Every number below was MEASURED on ``origin/main`` before the refusals
existed, on 2 virtual CPU devices (the root ``conftest.py`` default), with the
harness these tests still use -- ``_build()`` for classes 1-4, ``_asym()`` for
class 6 and ``_asym_deep()`` for class 5, all of which carry the source
position and the probe row explicitly, because none of the digits below
survive a change to either. Classes 1-4 and 6 were measured on d56f68eb;
round 3 re-measured the class-5 band on 7b511591 (identical to d56f68eb in
every runtime file this lane touches) and the commit is named beside each
round-3 number. None of the classes raised, and none of them warned -- except
``_asym_deep()``, which raised a broadcasting error naming no feature.

Two conventions, stated rather than left implicit:

* "energy" is ``np.sum(trace.astype(np.float64) ** 2)`` over the WHOLE probe
  trace. Summing in float32 moves the 7th digit (9.953717e-04 instead of
  9.953718e-04), and dropping the second probe moves the 4th
  (9.951558e-04) -- which is why ``_build`` pins the two-probe row.
* the RED assertions in this file are the REFUSALS. No test here re-measures
  a RED number; they are recorded in docstrings as the provenance of the
  refusals, measured out-of-tree on a pristine d56f68eb tree with these
  fixtures.

1. **Periodic / Bloch.** ``set_periodic_axes('y')`` (and the BoundarySpec
   ``periodic`` token, which sets the same attribute), 24x12x12 mm PEC box at
   dx=1 mm, 40 steps, ``amplitude_kind='field'`` source at (6, 6, 6) mm,
   probes at x=12/22 mm -- i.e. ``_build(periodic='y')`` exactly::

       max|Ez_distributed - Ez_native| = 1.963798e-04
       native peak |Ez|                = 1.090241e-03   (18.0% of peak)
       warnings raised                 = 0
       errors raised                   = 0

   Through the public ``sim.run(devices=...)`` with the source at the domain
   centre the same disagreement is ``3.261566e-04`` on a ``7.734966e-01`` peak.
   Cause: the distributed local kernels are unconditionally non-periodic
   (``rfx/runners/distributed.py:351``, and again at :380/:415/:440) and
   ``sim._periodic_axes`` occurred ZERO times in
   ``rfx/runners/distributed_v2.py`` **on d56f68eb** (grep, 2026-09-14).
   On HEAD ``grep -c`` gives 4 lines, all inside the admission gate --
   pinned by ``test_the_runner_still_never_reads_periodic_axes``, which
   compares the file count against the gate count rather than asserting
   either number.

2. **Extended lumped port** (``impedance=50``, ``extent=3e-3``)::

       distributed probe energy sum(Ez^2) = 0.000000e+00  (max|Ez| = 0.0)
       native     probe energy sum(Ez^2) = 9.953718e-04  (max|Ez| = 1.412484e-02)
       warnings raised                    = 0

   Cause: the lane forks ``impedance > 0 and extent is None`` then
   ``elif impedance == 0`` (the source/termination fork in
   ``distributed_v2.py``, and its twin at the same two
   ``if pe.impedance > 0.0 and pe.extent is None:`` / ``elif pe.impedance
   == 0.0:`` lines of ``distributed.py``).
   A wire port satisfies neither, so it gets no source and no resistive
   termination.

3. **Passive port** (``excite=False``), with an explicit waveform::

       distributed energy = 1.983971e-03   (max|Ez| = 1.910045e-02)
       native     energy = 0.000000e+00   (max|Ez| = 0.0)
       same port with excite=True, distributed = 1.983971e-03  (bit-identical)
       warnings raised                    = 0

   With the documented ``waveform=None`` default the lane instead died inside
   ``make_port_source`` with ``TypeError: Expected a callable value, got
   None`` -- a crash that names no feature. Cause: ``excite`` occurred ZERO
   times in either distributed runner **on d56f68eb**; on HEAD ``grep -c``
   gives 15 lines in ``distributed_v2.py``, all inside the admission gate.
   ``rfx/runners/uniform.py:421,440`` honours it (the ``setup_wire_port``
   call itself starts at :419 and runs to :420).

4. **Flux monitor / NTFF box**::

       distributed result.flux_monitors = None   (native: 1 monitor, 'flux_x_0')
       distributed result.ntff_data     = None   (native: NTFFData)
       distributed result.ntff_box      = None
       warnings raised                  = 0

   Same class as the #579 DFT-plane refusal: no accumulator, no cross-rank
   reduce, so a registered monitor is dropped and the field comes back None.

5. **x-absorber overflowing a rank's slab by more than one cell.** Round 3
   of the review narrowed this class by one cell on each face, by
   measurement, and re-attributed the fixture below.

   ``_asym(8, 6)`` -- ``x_lo='cpml'``/``x_hi='pec'`` with y/z CPML,
   6x8x8 mm at dx=1 mm, ``cpml_layers=8``, 2 devices, 60 steps, field
   source at (3, 4, 4) mm, Ez probe row at x = 1, 3, 5, 6 mm -> ``nx=15``,
   ``pad_x=1``, ``nx_per=8`` -- overflows the x-hi window by exactly ONE
   cell, and on origin/main (7b511591) it **RUNS**::

       native peak abs(Ez)             = 4.416633e+00  (probe x=3 mm)
       max|Ez_distributed - Ez_native| = 2.207244e+00  (49.98% of peak)
       probe at x=5 mm, one cell inside
         the x-hi PEC face             = 1.646758e-01 wrong on its own
                                         1.646768e-01 peak = 99.9994%
       errors raised                   = 0
       warnings raised                 = 0

   That 49.98% is **100% class 6**, not class 5: the innermost CPML layer
   has ``sigma=0`` and ``kappa=1`` (``rfx/boundaries/cpml.py``,
   ``rho = 1 - arange(n)/(n-1)``), so its correction ``-ce*psi -
   ce*(1/kappa - 1)*curl`` is identically zero, and both windows are
   anchored -- one cell of overflow moves only that no-op layer into the
   halo. Measured on origin/main with a SYMMETRIC absorber (so class 6 is
   silent), ``cpml_layers=8``, dx=1 mm, y/z 8 mm, field source at x=3 mm,
   probes x=1/3/5/6 mm, 120 steps, XLA host device count 4::

       11x8x8 mm, nx=28, 4 dev, pad_x=0, nx_per=7 (BOTH faces over by 1)
         rel 1.032290e-07 of a 9.238437e+00 peak, 0 warnings, trace
         BYTE-IDENTICAL to the fitting 2-device run (nx_per=14)
       8x8x8 mm, nx=25, 3 dev, pad_x=2, nx_per=9 (x-hi over by 1)
         rel 1.548455e-07, 0 warnings, same digits as the fitting run

   So the refused band is TWO or more cells of overflow, and the class-5
   fixture is ``_asym_deep()``: the same ASYM spec at 4x8x8 mm (``nx=13``,
   ``pad_x=1``, ``nx_per=7``), which on origin/main does not run at all --
   ``TypeError: mul got incompatible shapes for broadcasting: (8, 1, 1),
   (7, 25, 25)``, an error naming no feature, no face and no remedy. The
   clip that produces it happens past ``ghost`` cells of overflow
   (``n > nx_per + ghost`` / ``n > nx_per + ghost - pad_x``), so at
   ``ghost > 1`` the 2..ghost band would be silent instead of fatal --
   which is why the bound is ``> 1`` and not the clip.

   **The bound is NECESSARY, NOT SUFFICIENT** (round-2 review). The case
   that satisfies it exactly -- same ASYM spec at 7x8x8 mm,
   ``cpml_layers=8`` (nx=16, pad_x=0, nx_per=8), same source and probe row
   -- is wrong by the same order::

       native peak abs(Ez) (x=3 mm)    = 4.420052e+00
       max|Ez_distributed - Ez_native| = 2.040125e+00   (46.16%)
       probe x=5 mm  native 1.800406e-01 vs dist 5.905863e-05   (99.97%)
       probe x=6 mm  native 4.441055e-02 vs dist 1.295477e-07   (100.0%)
       warnings raised                 = 0

   The mechanism there is the PHANTOM x window -- the lane applies both
   x-face CPML windows whenever ``boundary='cpml'`` and
   ``cpml_layers > 0``, never reading ``grid.face_pads`` -- and that is
   class 6 below, refused separately. The decisive measurement: the
   asymmetric 24 mm model is wrong by 1.610351e-04 on a 3.165971e-03 peak
   (5.086e-02) at ``n_devices=2`` AND by the identical figure at
   ``n_devices=1``, where there is no slab to overflow.

   The ``(5, 25, 25)`` / ``(6, 25, 25)`` broadcasting shapes quoted by an
   earlier draft belong to ``BoundarySpec(x=('pec','pec'), y=cpml,
   z=cpml)`` at 6 mm and 8 mm, **not** to any 8-layer ASYM fixture: the
   ASYM 8-layer 8 mm model (nx=17, pad_x=1, nx_per=9) RUNS on origin/main
   at max|dEz| 1.309694e+00 on a 4.421298e+00 peak = 2.962240e-01, with 0
   warnings -- a 29.6% silent case that class 5 admits and only class 6
   catches. A 20-layer ASYM fixture at 7x12x12 mm (nx=28, pad_x=0,
   nx_per=14, ny=nz=53) is the one that gives ``(20, 1, 1),
   (15, 53, 53)``.

6. **Phantom CPML window at a non-absorbing face, on all six faces.** The
   lane applies EVERY CPML face window whenever ``boundary='cpml'`` and
   ``cpml_layers>0``, never reads ``grid.face_pads``, and uses a non-zero
   per-face ``dt/(eps_r*EPS_0)`` at the face -- so it ABSORBS at a face the
   caller asked to reflect. Found in round 2 of the review on the two x
   faces, because the class-5 arithmetic admits them; **round 4 found the
   same defect ADMITTED on the y and z faces** and widened the check from
   ``pad_x_lo``/``pad_x_hi`` to all six entries of ``grid.face_pads``.
   ``_init_cpml_distributed`` (``rfx/runners/distributed.py``) builds ONE
   scalar ``_cpml_profile`` and the kernel drives it at y-lo/y-hi/z-lo/z-hi
   unconditionally too. The y/z rows are measured on ``origin/main``
   883615c6 in the block comment above the y/z tests; the x rows are
   MEASURED on pristine d56f68eb, 2 devices, 0 warnings every time, and
   re-derived on 883615c6 to the printed digit::

       x_lo='cpml'/x_hi='pec', y/z cpml, cpml_layers=8, 24x8x8 mm,
       source (6,6,6) mm, probes x=12/22 mm, 60 steps:
           max|dEz| = 1.610351e-04 on a 3.165971e-03 row peak  (5.086e-02)
           the x=22 mm probe alone: 1.610351e-04 on its own 1.619078e-04
                                    peak  ->  99.46% wrong
           at 100 steps the row figure grows to 7.890e-02
           at n_devices=1: the IDENTICAL 1.610351e-04 / 5.086e-02
       x_lo='cpml'/x_hi='pec', y/z cpml, cpml_layers=8, 8x8x8 mm,
       source (3,4,4) mm, probes x=1/3/5/6 mm, 60 steps:
           max|dEz| = 1.309694e+00 on a 4.421298e+00 peak  (2.962240e-01)
           -- the row an earlier draft hid by attributing the XLA
           broadcasting shapes (5,25,25)/(6,25,25) to this spec; those
           belong to x='pec'/'pec'. class 5 ADMITS this one (nx=17,
           pad_x=1, nx_per=9, n=8 <= 9)
       x='pec'/'pec', y/z cpml, cpml_layers=8, with the fixture STATED.
       An earlier draft gave 2.74%/3.06%/3.07% "at the source" with no
       source position and no probe row; round 4 measured the fixture the
       branch itself states and got source-probe figures of 1.7610% /
       0.0225% / 0.00002%, so that row was not re-derivable and is
       replaced. The source-probe figure DECAYS as the domain grows (the
       source moves away from a window that stays 8 cells deep), so the
       figure that does not depend on where the source sits -- the one on
       the FACE probes -- is what is shipped. Both are given here, with
       which probe the row max sits on, so the labels cannot cross again:
           15 mm (nx=16), source x=7 mm, probes x=2/7/13 mm:
               source probe 7.786655e-02 of its own 4.421772e+00 peak
               = 1.7610%; row max 1.760981e-02 of the row peak, ON THE
               SOURCE PROBE; the x=2/13 mm face probes 99.9986% /
               99.9319% on their own peaks
           19 mm (nx=20), source x=9 mm, probes 2/9/17 mm:
               source probe 9.970665e-04 of 4.422243e+00 = 0.0225%;
               row max 6.308181e-04, ON THE x=2 mm FACE PROBE;
               99.9917% / 99.8366% on the face probes
           39 mm (nx=40), source x=19 mm, probes 2/19/37 mm:
               source probe 9.536743e-07 of 4.422517e+00 = 0.00002%;
               row max 3.307773e-05, ON THE x=2 mm FACE PROBE;
               99.9424% / 99.4168% on the face probes
           39 mm again, CENTRE PROBE ONLY, 200 steps (the shape of the
               shipped test_distributed.py parity tests): 8.356664e-04 on
               a 9.237972e+00 peak = 9.045994e-05 of peak -- INSIDE the
               shipped 1e-3 CPML tolerance, while the face probes on the
               same run are 99.4-99.9% wrong. That is why class 6 is a
               refusal and not a warning: a caller who probes only far
               from the faces cannot see it at any tolerance this suite
               uses.
       x_lo='pmc'/x_hi='cpml', y/z cpml, dx=5 mm, 16x8x8 cells:
           54.1% of peak at 30 steps, 93.5% at 80 steps

       y/z faces (round 4, ADMITTED until then, 24x8x8 mm, dx=1 mm,
       cpml_layers=8, field Ez source (6,4,4) mm, probes x=6/12/20 mm,
       60 steps, identical digits on 883615c6 / HEAD 461cfe53 / base
       7b511591):
           z=(pec,cpml), x/y cpml: 90.5782% of the source probe's own
               peak (4.003862e+00 on 4.420338e+00), 409.4824% / 281.3395%
               at x=12/20 mm; bit-for-bit the same through the v1 pmap
               runner at ONE device
           z=(pec,pec):  max|dEz| 2.615769e+00 on a 5.171041e-03 probe
               peak; 81.2831% at the source probe
           y=(pmc,cpml): 521.9300% of the x=12 mm probe's own peak
           y=(pec,pec):  373.3991% of the source probe's own peak

   Negative control (all six faces absorbing): ``boundary='cpml'``,
   ``cpml_layers=8``, 24x8x8 mm, probes 12/22 mm, 60 steps ->
   5.820766e-09 on a 3.170117e-03 peak = 1.836e-06 of peak at 2 devices
   (1.395e-06 at 1 device); on the round-4 y/z fixture 4.768372e-07 on
   4.422548e+00 = 1.078195e-07. Parity. The refusal costs the fully
   absorbing model nothing.

7. **The exported v1 pmap runner was ungated for all of 1-6.**
   ``rfx/runners/__init__.py`` re-exports ``run_distributed`` from
   ``rfx.runners.distributed``, NOT from ``distributed_v2``, and the first
   round of this change gated only ``distributed_v2`` and the
   ``run(devices=...)`` dispatch. MEASURED on that first-round tree with
   the committed ``_build`` fixture at 23x12x12 mm (nx=24, evenly
   divisible so the "not evenly divisible" ValueError could not bounce
   it), 40 steps, 2 devices, 0 warnings every time::

       extent port    pmap energy 0.000000e+00  vs native 9.951533e-04
       excite=False   pmap energy 1.982867e-03  vs native 0.0
       flux monitor   result.flux_monitors is None
       periodic 'y'   max|dEz| 1.963800e-04 on a 1.090239e-03 peak (18%)
       x absorber     x_lo='cpml'/x_hi='pec', cpml_layers=8, 5x8x8 mm
                      (nx=14, nx_per=7): 2.207114e+00 on 4.416774e+00 (50%)

8. **Ghost-width formula.** ``rfx/api/_execute.py`` computed
   ``floor(K/2)+1`` while ``rfx/runners/distributed_nu.build_sharded_nu_grid``
   uses ``ghost = K``::

       K:                1   2   3   4
       floor(K/2)+1:     1   2   2   3
       builder (g=K):    1   2   3   4

   The preflight was SHORT by one cell from K=3 up, i.e. it would clear a
   configuration the builder cannot shard. There is now one source of truth,
   :func:`rfx.runners.distributed_nu.nu_ghost_width`.

NOT closed here, and recorded so the "position-independent" scope of this
file is not read as "no other silent class remains": a source whose x cell
is the FIRST REAL CELL of rank 1 (global index == nx_per). MEASURED on
pristine d56f68eb, symmetric ``boundary='cpml'``, ``cpml_layers=8``, 2
devices, 60 steps, 0 warnings -- 4x8x8 mm (nx=21, nx_per=11) with the
source at x=3 mm (global 11) gives native 4.422122e+00 vs distributed
4.557471e+00 at the source probe and 8.203998e-01 vs 6.845230e-01 one cell
before (3.07e-02 of peak); 24x8x8 mm (nx=41, nx_per=21) with the source at
x=13 mm (global 21) gives 4.422516e+00 vs 4.558136e+00, the same 3.07e-02.
Moving the source ONE cell earlier, onto the last cell of rank 0, restores
1.08e-07 parity. That is 30x the shipped CPML tolerance of 1e-3
(``test_distributed.py``), it is position-DEPENDENT in the source rather
than in the absorber, and it belongs to the cut-plane census (lane B2), not
to this file.

The GREEN assertions here are the refusals. The negative controls at the
bottom reuse the fixtures and tolerances of the shipped parity tests in
``tests/unit/runners/test_distributed.py`` unchanged -- no tolerance is
weakened anywhere in this file.
"""
# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

import warnings  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import GaussianPulse, Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402

DOMAIN = (24e-3, 12e-3, 12e-3)
DX = 1e-3
N_STEPS = 40

# The asymmetric absorber fixture for class 5: an x-lo CPML face with an
# x-hi PEC reflector is the composition that makes nx small relative to
# cpml_layers (a symmetric absorber pads BOTH faces outside the domain, so
# nx_per > cpml_layers holds by construction at 2 devices).
ASYM_SPEC = BoundarySpec(
    x=Boundary(lo="cpml", hi="pec"),
    y=Boundary(lo="cpml", hi="cpml"),
    z=Boundary(lo="cpml", hi="cpml"),
)


def _devices():
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip("need 2 virtual devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    return devs[:2]


def _build(*, periodic="", boundary="pec", cpml_layers=None, port=None,
           flux=False, ntff=False, domain=DOMAIN, source=(6e-3, 6e-3, 6e-3),
           probes=None):
    """THE measuring fixture for classes 1-4, spelled out.

    ``source`` and ``probes`` are parameters and not constants because the
    RED digits in the module docstring do not survive a change to either:
    the class 2/3 energies are sums over the WHOLE trace, so the probe row
    is part of the measurement (one probe at x=12 mm alone gives
    9.951558e-04 and 1.982875e-03 instead of 9.953718e-04 and
    1.983971e-03).  The defaults here are exactly what was measured.
    """
    if probes is None:
        # The two-probe row of the measurement: x=12 mm and x=22 mm.  On a
        # domain too short for 22 mm, fall back to the centre probe only.
        probes = ((12e-3, 22e-3) if domain[0] >= 24e-3
                  else (min(12e-3, domain[0] / 2),))
    kwargs = dict(freq_max=15e9, domain=domain, dx=DX, boundary=boundary)
    if cpml_layers is not None:
        kwargs["cpml_layers"] = cpml_layers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(**kwargs)
        if periodic:
            sim.set_periodic_axes(periodic)
        if port is None:
            sim.add_source(position=source, component="ez",
                           amplitude_kind="field")
        else:
            sim.add_port(position=source, component="ez", **port)
        for _x in probes:
            sim.add_probe(position=(_x, domain[1] / 2, domain[2] / 2),
                          component="ez")
        if flux:
            sim.add_flux_monitor(axis="x", coordinate=domain[0] / 2, n_freqs=3)
        if ntff:
            sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 8e-3, 8e-3),
                             n_freqs=3)
    return sim


def _run_distributed(sim, n_steps=N_STEPS):
    """Call the runner directly (the entry the research note measured)."""
    from rfx.runners.distributed_v2 import run_distributed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_distributed(sim, n_steps=n_steps, devices=_devices())


def _run_api(sim, n_steps=N_STEPS):
    """Call the public dispatch (``run(devices=...)``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.run(n_steps=n_steps, devices=_devices(),
                       skip_preflight=True)


# ---------------------------------------------------------------------------
# 1. Periodic / Bloch
# ---------------------------------------------------------------------------

def test_periodic_axes_are_refused_through_the_public_dispatch():
    """RED: 1.963798e-04 on a 1.090241e-03 peak, 0 warnings (see docstring)."""
    with pytest.raises(NotImplementedError, match="periodic"):
        _run_api(_build(periodic="y"))


def test_periodic_axes_are_refused_inside_the_runner():
    with pytest.raises(NotImplementedError, match="periodic"):
        _run_distributed(_build(periodic="y"))


def test_the_periodic_refusal_names_the_axes_and_the_way_out():
    with pytest.raises(NotImplementedError) as excinfo:
        _run_distributed(_build(periodic="yz"))
    msg = str(excinfo.value)
    assert "'y'" in msg and "'z'" in msg, msg
    assert "devices=" in msg, "the message must say how to proceed"
    # The claim the refusal rests on, cited in the message itself.
    assert "_periodic_axes" in msg, msg


def test_the_boundaryspec_periodic_token_is_refused_too():
    """``set_periodic_axes()`` is deprecated; the spec token is the new route
    and sets the same attribute, so it must hit the same refusal."""
    spec = BoundarySpec(
        x=Boundary(lo="pec", hi="pec"),
        y=Boundary(lo="periodic", hi="periodic"),
        z=Boundary(lo="pec", hi="pec"),
    )
    sim = _build(boundary=spec)
    assert sim._periodic_axes == "y", "fixture no longer sets _periodic_axes"
    with pytest.raises(NotImplementedError, match="periodic"):
        _run_distributed(sim)


def test_the_runner_still_never_reads_periodic_axes():
    """The refusal's premise, checked instead of trusted.

    If a future change teaches the runner about ``_periodic_axes``, this
    fails and the refusal above must be re-derived rather than kept out of
    habit.
    """
    import inspect
    from pathlib import Path

    import rfx.runners.distributed_v2 as dv2
    src = Path(dv2.__file__).read_text()
    gate = inspect.getsource(dv2.refuse_unsupported_distributed_features)
    # Every mention of the attribute must live inside the gate itself (it
    # reads the attribute there, and quotes this claim in its docstring and
    # in the refusal message). The gate's source is a substring of the
    # file, so equal counts IS "nothing outside the gate" -- asserted that
    # way rather than against a magic total, which only pinned how many
    # times the docstring happens to say the word.
    assert gate.count("_periodic_axes") >= 1, (
        "the admission gate no longer reads sim._periodic_axes at all")
    assert src.count("_periodic_axes") == gate.count("_periodic_axes"), (
        "rfx/runners/distributed_v2.py now mentions _periodic_axes outside "
        "the admission gate; the periodic refusal rests on the runner NOT "
        "reading it, so re-derive the refusal rather than keeping it out of "
        f"habit (file: {src.count('_periodic_axes')}, "
        f"gate: {gate.count('_periodic_axes')})")


# ---------------------------------------------------------------------------
# 2. Extended (wire) lumped port
# ---------------------------------------------------------------------------

def test_extended_lumped_port_is_refused_through_the_public_dispatch():
    """RED: distributed energy 0.0 vs native 9.953718e-04 (see docstring)."""
    with pytest.raises(NotImplementedError, match="extent"):
        _run_api(_build(port=dict(impedance=50.0, extent=3e-3)))


def test_extended_lumped_port_is_refused_inside_the_runner():
    with pytest.raises(NotImplementedError, match="extent"):
        _run_distributed(_build(port=dict(impedance=50.0, extent=3e-3)))


def test_the_extended_port_refusal_names_both_branches_it_falls_between():
    with pytest.raises(NotImplementedError) as excinfo:
        _run_distributed(_build(port=dict(impedance=50.0, extent=3e-3)))
    msg = str(excinfo.value)
    assert "extent is None" in msg and "impedance == 0.0" in msg, msg
    assert "no source" in msg and "no resistive termination" in msg, msg


# ---------------------------------------------------------------------------
# 3. Passive port (excite=False)
# ---------------------------------------------------------------------------

def test_passive_port_is_refused_through_the_public_dispatch():
    """RED: distributed energy 1.983971e-03 -- bit-identical to excite=True --
    where the native run reads exactly 0.0 (see docstring)."""
    port = dict(impedance=50.0, excite=False,
                waveform=GaussianPulse(f0=7.5e9, bandwidth=0.8))
    with pytest.raises(NotImplementedError, match="excite=False"):
        _run_api(_build(port=port))


def test_passive_port_is_refused_inside_the_runner():
    port = dict(impedance=50.0, excite=False,
                waveform=GaussianPulse(f0=7.5e9, bandwidth=0.8))
    with pytest.raises(NotImplementedError, match="excite=False"):
        _run_distributed(_build(port=port))


def test_the_documented_waveform_none_default_is_refused_not_crashed():
    """``excite=False`` with no waveform is the documented spelling. On main
    it reached ``make_port_source`` and died with "Expected a callable value,
    got None"; the refusal must come first and name the feature."""
    with pytest.raises(NotImplementedError, match="excite=False"):
        _run_distributed(_build(port=dict(impedance=50.0, excite=False)))


# ---------------------------------------------------------------------------
# 4. Flux monitors / NTFF box -- the #579 DFT treatment
# ---------------------------------------------------------------------------

def test_flux_monitor_is_refused_through_the_public_dispatch():
    """RED: result.flux_monitors was None, 0 warnings (see docstring)."""
    with pytest.raises(NotImplementedError, match="add_flux_monitor"):
        _run_api(_build(flux=True))


def test_flux_monitor_is_refused_inside_the_runner():
    with pytest.raises(NotImplementedError, match="add_flux_monitor"):
        _run_distributed(_build(flux=True))


def test_ntff_box_is_refused_through_the_public_dispatch():
    """RED: result.ntff_data and result.ntff_box were None, 0 warnings."""
    with pytest.raises(NotImplementedError, match="add_ntff_box"):
        _run_api(_build(ntff=True))


def test_ntff_box_is_refused_inside_the_runner():
    with pytest.raises(NotImplementedError, match="add_ntff_box"):
        _run_distributed(_build(ntff=True))


def test_the_monitor_refusal_is_in_the_579_style():
    with pytest.raises(NotImplementedError) as excinfo:
        _run_distributed(_build(flux=True, ntff=True))
    msg = str(excinfo.value)
    assert "#579" in msg, "same class as the DFT-plane refusal, say so"
    assert "silently dropped" in msg, msg
    assert "devices=" in msg, msg


def test_the_579_dft_plane_refusal_is_untouched():
    """The refusals that already existed keep their own messages."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec")
        sim.add_source(position=(6e-3, 6e-3, 6e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(12e-3, 6e-3, 6e-3), component="ez")
        sim.add_dft_plane_probe(axis="x", coordinate=12e-3, n_freqs=3)
    with pytest.raises(NotImplementedError, match="add_dft_plane_probe"):
        _run_api(sim)


def test_the_dual_average_refusal_is_untouched():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec",
                         interface_eps="dual_average")
        sim.add_source(position=(6e-3, 6e-3, 6e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(12e-3, 6e-3, 6e-3), component="ez")
    with pytest.raises(ValueError, match="dual_average"):
        _run_api(sim)


# ---------------------------------------------------------------------------
# 5. x-absorber spanning ranks
# ---------------------------------------------------------------------------

def _asym(cpml_layers, domain_x_mm):
    """THE class-5 measuring fixture, source and probe row included.

    Both are stated explicitly because neither the RED number nor the
    control reproduces without them:

    * the source sits at x=3 mm, NOT at the ``_build`` default of 6 mm --
      on the 6 mm domain 6 mm IS the x-hi PEC plane, where the preflight
      reports the source as silently discarded and the trace collapses to
      5.55e-03 (a different, smaller silent-wrong);
    * the probe ROW is x = 1, 3, 5, 6 mm.  The probe at 6 mm reads exactly
      0.0 on both lanes (it is on the PEC face), so the "99.9994% of its
      own peak" probe in the docstring is the one at x=5 mm, a cell
      inside it.

    Measured on pristine main (d56f68eb): with this fixture ``_asym(8, 6)``
    gives native peak 4.416633e+00 and max|dEz| 2.207244e+00 (49.98%), and
    ``_asym(8, 24)`` gives 1.005828e-06 on 4.422700e+00 = 2.274240e-07.
    """
    return _build(boundary=ASYM_SPEC, cpml_layers=cpml_layers,
                  domain=(domain_x_mm * 1e-3, 8e-3, 8e-3),
                  source=(3e-3, 4e-3, 4e-3),
                  probes=(1e-3, 3e-3, 5e-3, 6e-3))


def _asym_deep():
    """THE class-5 measuring fixture after round 3 narrowed the bound.

    ``_asym(8, 6)`` is NOT it any more: at nx=15 / nx_per=8 / pad_x=1 its
    x-hi window overflows by exactly ONE cell, which is a measured no-op
    (see ``check_x_absorber_fits_ranks``) and is now admitted by class 5
    and refused by class 6 instead.  This fixture overflows by two, which
    is the band that is really broken: same ASYM spec, ``cpml_layers=8``,
    4x8x8 mm at dx = 1 mm -> ``nx=13``, ``pad_x=1``, ``nx_per=7``, so the
    x-hi limit ``nx_per - pad_x + 1 = 7`` is exceeded by one.

    MEASURED on pristine origin/main (7b511591), 2 CPU devices, 60 steps:
    this configuration does not run at all -- it dies inside XLA with
    ``TypeError: mul got incompatible shapes for broadcasting:
    (8, 1, 1), (7, 25, 25)``, an error that names no feature, no face and
    no remedy.  That is what class 5 replaces with a named refusal.
    """
    return _build(boundary=ASYM_SPEC, cpml_layers=8,
                  domain=(4e-3, 8e-3, 8e-3),
                  source=(2e-3, 4e-3, 4e-3),
                  probes=(1e-3, 2e-3, 3e-3))


def test_x_absorber_spanning_ranks_is_refused():
    """RED: on origin/main this died inside XLA with ``mul got incompatible
    shapes for broadcasting: (8, 1, 1), (7, 25, 25)`` -- see
    ``_asym_deep``."""
    with pytest.raises(ValueError, match="x CPML absorber"):
        _run_distributed(_asym_deep(), n_steps=8)


def test_one_cell_of_overflow_is_admitted_by_class_5():
    """Round 3, BLOCKING 1: ``_asym(8, 6)`` overflows the x-hi window by
    exactly one cell and class 5 must NOT be what refuses it.

    MEASURED on pristine origin/main (7b511591), 2 devices, 60 steps:
    ``_asym(8, 6)`` RUNS -- 0 warnings, no error, max|dEz| 2.207244e+00 on
    a 4.416633e+00 peak (49.98%). The one-cell overflow is not the cause:
    the innermost CPML layer has sigma=0 and kappa=1, so its correction is
    identically zero, and the window is anchored, so one cell of overflow
    moves only that no-op layer into the halo. Decisive measurement, same
    tree, symmetric ``boundary='cpml'`` (both x faces absorbing, so class 6
    is silent), ``cpml_layers=8``, dx=1 mm, field source at x=3 mm, probe
    row x=1/3/5/6 mm, 120 steps, XLA host device count 4:

        11x8x8 mm, nx=28, 4 devices, pad_x=0, nx_per=7 (n=8 overflows BOTH
          faces by one): rel 1.032290e-07 of a 9.238437e+00 peak, 0
          warnings, and the trace BYTE-IDENTICAL to the 2-device run that
          fits (nx_per=14)
        8x8x8 mm, nx=25, 3 devices, pad_x=2, nx_per=9 (x-hi only):
          rel 1.548455e-07, 0 warnings, same digits as the fitting 2-device
          run

    So class 5 admits the one-cell band, and ``_asym(8, 6)`` is refused by
    class 6 -- which is the real defect in it (49.98% is 100% phantom
    window: the same model at n_devices=1, with no slab to overflow, is
    wrong by the same amount).
    """
    from rfx.runners.distributed_v2 import check_x_absorber_fits_ranks

    # _asym(8, 6): nx=15, pad_x=1, nx_per=8 -> x-hi overflows by one cell
    assert check_x_absorber_fits_ranks(
        nx=15, n_devices=2, nx_per=8, pad_x=1, ghost=1, cpml_layers=8,
        pad_x_lo=8, pad_x_hi=0) is None
    # the measured symmetric cases: one cell over on both faces, and on
    # x-hi only
    assert check_x_absorber_fits_ranks(
        nx=28, n_devices=4, nx_per=7, pad_x=0, ghost=1, cpml_layers=8,
        pad_x_lo=8, pad_x_hi=8) is None
    assert check_x_absorber_fits_ranks(
        nx=25, n_devices=3, nx_per=9, pad_x=2, ghost=1, cpml_layers=8,
        pad_x_lo=8, pad_x_hi=8) is None
    # and end to end: _asym(8, 6) is refused by class 6, not class 5
    with pytest.raises(ValueError, match="no CPML absorber"):
        _run_distributed(_asym(8, 6), n_steps=8)


def test_x_absorber_refusal_reports_the_slab_arithmetic():
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(_asym_deep(), n_steps=8)
    msg = str(excinfo.value)
    for token in ("cpml_layers=8", "nx=13", "n_devices=2", "nx_per=7",
                  "pad_x=1"):
        assert token in msg, f"{token!r} missing from: {msg}"
    # At n_devices=2 there IS no smaller multi-device count, so the message
    # must not recommend "n_devices=1" (that is "omit devices=" said twice);
    # it names the depth that would fit and the single-device way out.
    assert "n_devices=1" not in msg, msg
    assert "omit devices=" in msg, msg
    assert "reduce cpml_layers" in msg, msg
    # Round 2: the message must not attribute the 49.98% to the overflow.
    # Measured: the case that satisfies this bound EXACTLY is 46.16% wrong,
    # so the overflow is not the dominant term and the message says so.
    assert "NECESSARY, NOT SUFFICIENT" in msg, msg
    assert "46.16%" in msg, msg
    assert "check_absorber_faces_are_absorbing" in msg, msg
    # Round 3, BLOCKING 2: the "died inside XLA" claim is asserted only
    # when it is true of the caller's own configuration. It is here (the
    # window clips at ghost=1), and it was NOT for _asym(8, 6), which ran.
    assert "died inside XLA" in msg, msg
    # Round 3, BLOCKING 1: the one-cell band is named as admitted.
    assert "ONE cell of overflow is harmless" in msg, msg


def test_x_absorber_refusal_also_fires_through_the_public_dispatch():
    with pytest.raises(ValueError, match="x CPML absorber"):
        _run_api(_asym_deep(), n_steps=8)


def test_the_x_absorber_condition_is_the_window_arithmetic():
    """The exact boundary, unit-tested without running a simulation.

    The x-hi window is ``[nx_per + ghost - pad_x - n, nx_per + ghost -
    pad_x)`` on a slab whose owned cells are ``[ghost, ghost + nx_per)``, so
    ``n == nx_per - pad_x`` is the last depth that stays entirely INSIDE the
    owned cells.  That is not where this check draws the line, and round 4
    deleted the sentence that said it was: the paragraph below, and the body,
    have pinned ``n + 1`` admitted / ``n + 2`` refused since round 3.

    Round 3 narrowed the bound by one cell on each face, by measurement: the
    innermost CPML layer has ``sigma=0``/``kappa=1`` so its correction is
    identically zero, and both windows are anchored, so ONE cell of
    overflow moves only that no-op layer into the halo -- byte-identical
    to the fitting run on origin/main. The last depth this check admits is
    therefore ``nx_per - pad_x + 1`` (x-hi) and ``nx_per + 1`` (x-lo), and
    one deeper is the first it refuses.

    "Admits" means this check alone, and the bound is a NECESSARY
    condition only -- passing it does not make the configuration right.
    Measured on pristine d56f68eb at exactly ``n == nx_per - pad_x``
    (ASYM spec, cpml_layers=8, 7x8x8 mm -> nx=16, pad_x=0, nx_per=8,
    source (3, 4, 4) mm, probes 1/3/5/6 mm, 2 devices, 60 steps): max|dEz|
    2.040125e+00 on a 4.420052e+00 peak = 46.16% of peak, with the x=5 and
    x=6 mm probes 99.97% and 100.0% wrong on their own peaks and 0
    warnings -- the same order as the refused 6 mm case. What makes it
    wrong is the phantom x-hi window at the PEC face, which
    :func:`check_absorber_faces_are_absorbing` refuses separately
    (class 6). This test pins the arithmetic; the class-6 tests below pin
    the sufficiency.
    """
    from rfx.runners.distributed_v2 import check_x_absorber_fits_ranks
    for pad_x in (0, 1, 3):
        nx_per = 12
        kw = dict(nx=nx_per * 2, n_devices=2, nx_per=nx_per, pad_x=pad_x,
                  ghost=1)
        # inside the owned cells: silent, returns None
        assert check_x_absorber_fits_ranks(
            cpml_layers=nx_per - pad_x, **kw) is None
        # one cell of overflow: a measured no-op, so still admitted
        assert check_x_absorber_fits_ranks(
            cpml_layers=nx_per - pad_x + 1, **kw) is None
        # two cells: refused
        with pytest.raises(ValueError, match="x CPML absorber"):
            check_x_absorber_fits_ranks(cpml_layers=nx_per - pad_x + 2, **kw)
        # The x-lo face has its own, looser bound (no pad_x eats into it),
        # so one cell over is admitted there too -- separable from the x-hi
        # bound only at pad_x == 0, since at pad_x > 0 a depth of
        # nx_per + 1 has already tripped x-hi.
        if pad_x == 0:
            assert check_x_absorber_fits_ranks(
                cpml_layers=nx_per + 1, **kw) is None
        else:
            with pytest.raises(ValueError, match="x-hi"):
                check_x_absorber_fits_ranks(cpml_layers=nx_per + 1, **kw)
        with pytest.raises(ValueError, match="x-lo"):
            check_x_absorber_fits_ranks(cpml_layers=nx_per + 2, **kw)


def _sym_cpml(cpml_layers=8, domain_x_mm=24):
    """A SYMMETRIC absorber: both x faces CPML, so classes 5 and 6 admit.

    ``boundary='cpml'`` pads the absorber OUTSIDE the requested domain on
    both x faces (``rfx/grid.py``), so ``pad_x_lo == pad_x_hi ==
    cpml_layers > 0`` and ``nx_per > cpml_layers`` holds by construction
    at 2 devices. This is the negative control for BOTH slab checks.
    """
    return _build(boundary="cpml", cpml_layers=cpml_layers,
                  domain=(domain_x_mm * 1e-3, 8e-3, 8e-3))


def test_a_fitting_x_absorber_is_not_refused():
    """Negative control for classes 5 AND 6: a symmetric 8-layer absorber
    on a 24 mm domain (nx=41, pad_x=1, nx_per=21) both fits the slab and
    has an absorber on every x face it drives a window at.

    Measured on pristine d56f68eb with this exact fixture (probes at
    x=12/22 mm, 60 steps): max|dEz| 5.820766e-09 on a 3.170117e-03 peak =
    1.836e-06 of peak at 2 devices, 1.395e-06 at 1 device -- parity, three
    orders inside the shipped CPML tolerance of 1e-3. Asserted here at
    that shipped tolerance, unweakened.

    The previous version of this control was ``_asym(8, 24)``, which fits
    the slab but leaves ``x_hi='pec'`` with a phantom absorber -- and is
    5.086e-02 of peak wrong. It is now a class-6 RED case; see
    ``test_the_fitting_asymmetric_absorber_is_refused_by_class_6``.
    """
    single = _sym_cpml().run(n_steps=60, skip_preflight=True)
    multi = _run_distributed(_sym_cpml(), n_steps=60)
    ts_s = np.asarray(single.time_series)
    ts_m = np.asarray(multi.time_series)
    assert ts_s.shape == ts_m.shape
    peak = np.max(np.abs(ts_s)) + 1e-30
    rel = np.max(np.abs(ts_s - ts_m)) / peak
    assert rel < 1e-3, f"symmetric CPML distributed error {rel:.2e}"


# ---------------------------------------------------------------------------
# 6. Phantom CPML window at a non-absorbing face (all six faces)
# ---------------------------------------------------------------------------

def test_the_fitting_asymmetric_absorber_is_refused_by_class_6():
    """RED: ``_asym(8, 24)`` PASSES the class-5 arithmetic (nx=33, pad_x=1,
    nx_per=17, so an 8-layer window fits) and is still wrong, because
    ``x_hi='pec'`` gets an absorber it never asked for.

    Measured on pristine d56f68eb, 2 devices, 60 steps, 0 warnings, with
    ``_build``'s default centre source (6, 6, 6) mm and default probe row
    x=12/22 mm on the same 24x8x8 mm ASYM model: max|dEz| 1.610351e-04 on
    a 3.165971e-03 row peak = 5.086e-02 of peak, and the x=22 mm probe --
    2 mm inside the PEC face -- 99.46% wrong on its own 1.619078e-04 peak.
    At 100 steps the row figure is 7.890e-02 (the x=22 mm probe 99.48%).
    At n_devices=1 it is the identical 1.610351e-04 / 5.086e-02, which is
    how we know the window and not the decomposition is the mechanism.
    Round 4 re-derived every figure on ``origin/main`` 883615c6 to the
    printed digit.
    """
    with pytest.raises(ValueError, match="no CPML absorber"):
        _run_distributed(_asym(8, 24), n_steps=8)


def test_the_phantom_window_refusal_names_the_face_and_the_way_out():
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(_asym(8, 24), n_steps=8)
    msg = str(excinfo.value)
    assert "face x-hi declare(s) no CPML absorber" in msg, msg
    # all six pads are reported, not only the two the caller broke
    for token in ("grid.pad_x_lo=8", "grid.pad_x_hi=0", "grid.pad_y_lo=8",
                  "grid.pad_y_hi=8", "grid.pad_z_lo=8", "grid.pad_z_hi=8"):
        assert token in msg, f"{token!r} missing from: {msg}"
    assert "never reads grid.face_pads" in msg, msg
    # the ways out, each nameable
    assert "make ALL SIX faces absorbing" in msg, msg
    assert "cpml_layers=0" in msg, msg
    assert "omit devices=" in msg, msg


def test_the_remedy_never_points_at_the_x_only_composition():
    """ROUND-4 (i): the round-3 remedy said 'make BOTH x faces absorbing
    (BoundarySpec(x="cpml") ...)', and a caller with y/z reflectors who
    followed it landed in the 90.58%-wrong y/z gap with 0 warnings.

    The message must not offer an x-only remedy, and must say so in the
    words a reader will act on.
    """
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(_asym(8, 24), n_steps=8)
    msg = str(excinfo.value)
    assert "BOTH x faces" not in msg, msg
    assert "BoundarySpec(x='cpml')" not in msg, msg
    assert "NOT just the x faces" in msg, msg


def test_the_phantom_window_refusal_fires_through_the_public_dispatch():
    with pytest.raises(ValueError, match="no CPML absorber"):
        _run_api(_asym(8, 24), n_steps=8)


def test_a_pmc_x_face_is_refused_too():
    """The same defect with a PMC reflector instead of a PEC one.

    This is the configuration
    ``tests/unit/boundaries/test_boundary_pmc_distributed.py``'s
    ``test_pmc_distributed_v2_x_lo_owner_and_non_owner`` used to run: on
    pristine d56f68eb, dx=5 mm, 16x8x8 cells, ``x=Boundary(lo='pmc',
    hi='cpml')`` with y/z CPML, it was 54.1% of peak wrong at that test's
    own 30 steps (1.350925e-01 on 2.498208e-01) and 93.5% at 80, with the
    probes 1 and 2 cells off the PMC face 100% wrong on their own peaks
    and 0 warnings. Round 4 re-derived 54.08% / 93.48% on ``origin/main``
    883615c6 with that test's own fixture (default ``cpml_layers``, i.e.
    16, not 8). That test now composes the PMC face with PEC.
    """
    spec = BoundarySpec(x=Boundary(lo="pmc", hi="cpml"),
                        y=Boundary(lo="cpml", hi="cpml"),
                        z=Boundary(lo="cpml", hi="cpml"))
    sim = _build(boundary=spec, cpml_layers=8, domain=(24e-3, 8e-3, 8e-3))
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(sim, n_steps=8)
    assert "face x-lo declare(s) no CPML absorber" in str(excinfo.value)


# --- round 4: the same class on the y and z faces --------------------------
#
# Measured on ``origin/main`` 883615c6 (identical digits on this branch's
# base 7b511591 and on its HEAD before this widening), 24x8x8 mm at
# dx=1 mm, ``cpml_layers=8``, ``amplitude_kind='field'`` Ez source at
# (6, 4, 4) mm, Ez probes at x = 6/12/20 mm, 60 steps, 2 virtual CPU
# devices, 0 warnings every time -- ADMITTED on HEAD until now:
#
#   z=(pec,cpml), x/y cpml   90.5782% at the source probe
#                            (max|dEz| 4.003862e+00 on 4.420338e+00),
#                            409.4824% / 281.3395% of their own peaks at
#                            x=12/20 mm.  Through the v1 pmap runner at
#                            ONE device: bit-for-bit the same three
#                            figures.
#   z=(pec,pec),  x/y cpml   max|dEz| 2.615769e+00 against a 5.171041e-03
#                            probe peak at x=12 mm; 81.2831% at the source
#   y=(pmc,cpml), x/z cpml   521.9300% of the x=12 mm probe's own peak
#                            (3.988669e-02 on 7.642152e-03); 48.6214% at
#                            the source
#   y=(pec,pec),  x/z cpml   373.3991% of the source probe's own peak
#                            (1.649596e+01 on 4.417783e+00)
#   SYMMETRIC control        1.078195e-07 of peak (4.768372e-07 on
#                            4.422548e+00)
#
# The z=(pec,cpml) row is the exact shape that
# tests/unit/boundaries/test_boundary_pmc_composition.py's OQ9 test was
# running at devices=devices[:2]: it asserted only a zero-pattern on the
# PEC face and stayed green on the 90.58%-wrong run -- the same false green
# the x face had in test_boundary_pmc_distributed.py. Round 4 split it into
# test_oq9_uniform_cpml_path_enforces_pec_face_via_cpml_init (the
# structural claim, on the lane that implements it, assertions unchanged)
# and test_oq9_distributed_v2_refuses_the_pec_face_composition (what this
# lane owes the fixture).

_C = "cpml"

YZ_RED_SPECS = {
    "z_lo_pec": (BoundarySpec(x=_C, y=_C, z=Boundary(lo="pec", hi=_C)),
                 "z-lo"),
    "z_both_pec": (BoundarySpec(x=_C, y=_C, z=Boundary(lo="pec", hi="pec")),
                   "z-lo and z-hi"),
    "y_lo_pmc": (BoundarySpec(x=_C, y=Boundary(lo="pmc", hi=_C), z=_C),
                 "y-lo"),
    "y_both_pec": (BoundarySpec(x=_C, y=Boundary(lo="pec", hi="pec"), z=_C),
                   "y-lo and y-hi"),
}


def _yz(spec):
    """The round-4 y/z measuring fixture, spelled out.

    Source at (6, 4, 4) mm and probes at x = 6/12/20 mm are part of the
    measurement: the source probe carries the 90.58% figure and the two
    downstream probes the 409% / 281% ones.
    """
    return _build(boundary=spec, cpml_layers=8, domain=(24e-3, 8e-3, 8e-3),
                  source=(6e-3, 4e-3, 4e-3), probes=(6e-3, 12e-3, 20e-3))


@pytest.mark.parametrize("key", sorted(YZ_RED_SPECS))
def test_a_reflector_on_a_y_or_z_face_is_refused_too(key):
    """RED: admitted on this branch's HEAD before round 4, 0 warnings, and
    90.58% wrong at the source probe for ``z_lo_pec`` -- see the block
    comment above for all four rows."""
    spec, faces = YZ_RED_SPECS[key]
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(_yz(spec), n_steps=8)
    msg = str(excinfo.value)
    _plural = "faces" if " and " in faces else "face"
    assert f"{_plural} {faces} declare(s) no CPML absorber" in msg, msg
    assert "90.58% wrong at the source probe" in msg, msg


@pytest.mark.parametrize("key", sorted(YZ_RED_SPECS))
def test_the_y_z_refusal_fires_through_the_public_dispatch_too(key):
    spec, _faces = YZ_RED_SPECS[key]
    with pytest.raises(ValueError, match="no CPML absorber"):
        _run_api(_yz(spec), n_steps=8)


@pytest.mark.parametrize("key", sorted(YZ_RED_SPECS))
def test_the_y_z_refusal_fires_in_the_v1_pmap_runner_too(key):
    """The exported ``rfx.runners.run_distributed`` is the v1 pmap runner,
    and it is the runner that BUILDS the offending profile
    (``_init_cpml_distributed``), so its copy of the check must be widened
    with the other.

    At 23 mm rather than the measurement's 24: this runner raises
    ``Grid nx=... is not evenly divisible`` BEFORE the slab checks, and
    the 24 mm fixture is nx=41. 23 mm gives nx=40. The refusal under test
    does not depend on nx (the pads do not change with the domain).
    """
    from rfx.runners import run_distributed as exported
    spec, _faces = YZ_RED_SPECS[key]
    sim = _build(boundary=spec, cpml_layers=8, domain=(23e-3, 8e-3, 8e-3),
                 source=(6e-3, 4e-3, 4e-3), probes=(6e-3, 12e-3, 20e-3))
    assert sim._build_grid().nx % 2 == 0, "fixture no longer evenly divisible"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no CPML absorber"):
            exported(sim, n_steps=8, devices=_devices())


def test_the_y_z_refusal_fires_at_one_device_too():
    """The defect needs no decomposition: through the v1 pmap runner at ONE
    device on 883615c6 the ``z=(pec,cpml)`` fixture is wrong by bit-for-bit
    the 2-device figures (90.5782% / 409.4823% / 281.3395%), so the check
    must fire there as well."""
    from rfx.runners.distributed import run_distributed as run_v1
    spec, _faces = YZ_RED_SPECS["z_lo_pec"]
    # nx must be divisible by the device count; at one device any nx is.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no CPML absorber"):
            run_v1(_yz(spec), n_steps=8, devices=_devices()[:1])


def test_the_phantom_window_check_reads_all_six_pads():
    """Unit boundary of class 6, simulation-free: it fires exactly when any
    one of ``grid.face_pads``' six entries is 0, and never otherwise.

    Round 4: before this, the check took ``pad_x_lo``/``pad_x_hi`` and the
    four y/z entries were invisible to it -- which is precisely how the
    90.58%-wrong ``z=(pec,cpml)`` model was admitted.
    """
    from rfx.runners.distributed_v2 import (
        check_absorber_faces_are_absorbing,
    )
    assert check_absorber_faces_are_absorbing(
        cpml_layers=8, face_pads=(8,) * 6, n_devices=2) is None
    # per-face thicknesses differ but every face absorbs: still admitted
    assert check_absorber_faces_are_absorbing(
        cpml_layers=8, face_pads=(6, 10, 8, 8, 7, 9), n_devices=2) is None
    faces = ("x-lo", "x-hi", "y-lo", "y-hi", "z-lo", "z-hi")
    for i, face in enumerate(faces):
        pads = [8] * 6
        pads[i] = 0
        with pytest.raises(ValueError) as excinfo:
            check_absorber_faces_are_absorbing(
                cpml_layers=8, face_pads=tuple(pads), n_devices=2)
        assert f"face {face} declare(s) no CPML absorber" in str(
            excinfo.value), (face, str(excinfo.value))
    # every face off at once: all six named, joined as a list
    with pytest.raises(ValueError) as excinfo:
        check_absorber_faces_are_absorbing(
            cpml_layers=8, face_pads=(0,) * 6, n_devices=2)
    msg = str(excinfo.value)
    assert "faces x-lo, x-hi, y-lo, y-hi, z-lo and z-hi" in msg, msg
    # the six-tuple is required, not a two-tuple silently accepted
    with pytest.raises(ValueError, match="six-tuple"):
        check_absorber_faces_are_absorbing(
            cpml_layers=8, face_pads=(8, 8), n_devices=2)


def test_a_2d_cpml_model_gets_a_named_refusal_instead_of_an_xla_crash():
    """``rfx/grid.py`` drops z from ``cpml_axes`` in 2-D, so a 2-D CPML
    model has ``pad_z_lo == pad_z_hi == 0`` and class 6 now refuses it.

    That takes nothing away. MEASURED on ``origin/main`` 883615c6,
    ``mode='2d_tmz'`` and ``'2d_tez'``, 24x8x8 mm at dx=1 mm,
    ``boundary='cpml'``, ``cpml_layers=8``, 2 devices: the run died inside
    XLA with ``ValueError: Incompatible types for broadcasting: input
    type=float32[23,25,8] and requested type=float32[23,25,1]``, an error
    that names no feature, no face and no remedy. This refusal replaces
    that with a named one -- and the message's only honest way forward for
    a 2-D model is to omit ``devices=``, which it says.
    """
    for mode in ("2d_tmz", "2d_tez"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=15e9, domain=(24e-3, 8e-3, 8e-3),
                             dx=DX, boundary="cpml", cpml_layers=8, mode=mode)
            sim.add_source(position=(6e-3, 4e-3, 0.0), component="ez",
                           amplitude_kind="field")
            sim.add_probe(position=(12e-3, 4e-3, 0.0), component="ez")
        assert sim._build_grid().face_pads[4:] == (0, 0), mode
        with pytest.raises(ValueError) as excinfo:
            _run_distributed(sim, n_steps=8)
        msg = str(excinfo.value)
        assert "faces z-lo and z-hi declare(s) no CPML absorber" in msg, msg
        assert "every 2-D model" in msg, msg
        assert "Incompatible types for broadcasting" in msg, msg


def test_the_symmetric_absorber_is_still_admitted_on_all_six_faces():
    """The negative control for the WIDENED check, at the shipped tolerance.

    ``boundary='cpml'`` pads all six faces outside the requested domain by
    construction, so the common case is untouched. Measured on 883615c6
    with the round-4 y/z fixture (24x8x8 mm, source (6, 4, 4) mm, probes
    x=6/12/20 mm, 60 steps, 2 devices): 4.768372e-07 on a 4.422548e+00
    peak = 1.078195e-07 of peak. Asserted at the shipped 1e-3, unweakened.
    """
    sim_n = _yz("cpml")
    sim_d = _yz("cpml")
    assert sim_n._build_grid().face_pads == (8,) * 6
    single = sim_n.run(n_steps=60, skip_preflight=True)
    multi = _run_distributed(sim_d, n_steps=60)
    ts_s = np.asarray(single.time_series)
    ts_m = np.asarray(multi.time_series)
    assert ts_s.shape == ts_m.shape
    peak = np.max(np.abs(ts_s)) + 1e-30
    rel = np.max(np.abs(ts_s - ts_m)) / peak
    assert rel < 1e-3, f"symmetric six-face CPML distributed error {rel:.2e}"


def test_the_runners_still_never_read_the_y_z_face_pads():
    """The refusal's premise, checked instead of trusted.

    Class 6 exists because ``_init_cpml_distributed`` builds ONE scalar
    ``_cpml_profile`` and the kernel applies it at every face without
    consulting the per-face pads. If a future change teaches either runner
    about ``pad_y_*`` / ``pad_z_*`` / ``face_layers``, this fails and the
    refusal must be re-derived rather than kept out of habit.
    """
    from pathlib import Path

    import rfx.runners.distributed as d1
    import rfx.runners.distributed_v2 as d2

    for mod in (d1, d2):
        src = Path(mod.__file__).read_text()
        # ATTRIBUTE READS, not prose: the gate's own docstring and message
        # must name the attributes the kernel ignores, and do.
        for attr in ("grid.pad_y_lo", "grid.pad_y_hi",
                     "grid.pad_z_lo", "grid.pad_z_hi",
                     "grid.face_layers.get", "face_layers[("):
            assert attr not in src, (
                f"{mod.__name__} now reads {attr}; class 6's premise (one "
                "scalar profile driven at every face regardless of the "
                "per-face pads) must be re-derived, not kept out of habit")
    # and the profile is still built ONCE, from the scalar depth: this is
    # the other half of the premise (the depth ignores face_layers too).
    d1_src = Path(d1.__file__).read_text()
    assert "n = grid.cpml_layers" in d1_src, (
        "the scalar-depth premise moved; re-derive class 6")
    assert "_cpml_profile(n, grid.dt" in d1_src, (
        "the one-scalar-profile premise moved; re-derive class 6")


def test_the_nu_grid_answers_face_pads_so_the_check_can_read_it():
    """Round 4: class 6 reads ``grid.face_pads``, and ``distributed_v2``
    reaches the check with a NON-UNIFORM grid.

    ``run_distributed`` builds the NU grid at ``is_nu`` and calls the slab
    checks BEFORE the ``is_nu and use_cpml`` NotImplementedError below them
    (that raise is a documented backstop, not a guard that runs first). The
    NU grid dataclass carried the six per-face pads but not the name
    ``Grid`` uses for them, so reading ``grid.face_pads`` would have turned
    a named refusal into an ``AttributeError`` on that path -- strictly
    worse than what it did before the widening, since the old check read
    ``grid.pad_x_lo``, which the NU grid does have. A ``face_pads``
    property now sits beside the existing ``axis_pads`` one.

    Pinned both ways: the property's order matches ``Grid``'s, and the NU +
    CPML + ``devices=`` path still ends at its own NotImplementedError.
    """
    from rfx.nonuniform import make_nonuniform_grid

    nu = make_nonuniform_grid(domain_xy=(12e-3, 6e-3),
                              dz_profile=np.full(7, 1e-3, dtype=np.float64),
                              dx=1e-3)
    assert nu.face_pads == (nu.pad_x_lo, nu.pad_x_hi,
                            nu.pad_y_lo, nu.pad_y_hi,
                            nu.pad_z_lo, nu.pad_z_hi)
    # the same order Grid uses, checked against a real Grid rather than
    # assumed from the attribute names
    uni = _sym_cpml()._build_grid()
    assert uni.face_pads == (uni.pad_x_lo, uni.pad_x_hi,
                             uni.pad_y_lo, uni.pad_y_hi,
                             uni.pad_z_lo, uni.pad_z_hi)

    # and end to end: NU + CPML + devices= still gets the Phase-C refusal,
    # not an AttributeError from inside the gate
    sim = _sym_cpml()
    sim._dz_profile = np.full(9, 1e-3, dtype=np.float64)
    with pytest.raises(NotImplementedError, match="Phase B supports"):
        _run_distributed(sim, n_steps=4)


def test_a_pec_only_model_never_reaches_the_phantom_window_check():
    """``boundary='pec'`` builds no CPML window at all, so class 6 must not
    fire on it -- the check is guarded on ``use_cpml``, and the default PEC
    fixture still runs distributed (the parity control below)."""
    res = _run_distributed(_build())
    assert np.max(np.abs(np.asarray(res.time_series))) > 0

# ---------------------------------------------------------------------------
# 8. One ghost-width formula (docstring item 8; not an admission class)
# ---------------------------------------------------------------------------

def test_ghost_width_is_the_builders_value_for_k_1_to_4():
    """RED: the preflight's floor(K/2)+1 gave 1, 2, 2, 3 for K=1..4 while the
    builder shards with g=K -- short by one cell from K=3 up."""
    import math

    from rfx.runners.distributed_nu import nu_ghost_width
    old = [math.floor(k / 2) + 1 for k in (1, 2, 3, 4)]
    new = [nu_ghost_width(k) for k in (1, 2, 3, 4)]
    assert new == [1, 2, 3, 4], new
    assert old == [1, 2, 2, 3], old
    # the disagreement this unification closes, stated as a number
    assert [n - o for n, o in zip(new, old)] == [0, 0, 1, 1]


def test_the_builder_and_the_preflight_agree_at_every_k():
    """One source of truth: the builder's own ``ghost_width`` (K=1, the only
    interval it accepts today) and the preflight's value both come from
    :func:`nu_ghost_width`, and the preflight no longer carries a second
    formula."""
    from pathlib import Path

    import rfx.api._execute as _execute
    from rfx.nonuniform import make_nonuniform_grid
    from rfx.runners.distributed_nu import (
        build_sharded_nu_grid,
        nu_ghost_width,
    )

    grid = make_nonuniform_grid(
        domain_xy=(12e-3, 6e-3),
        dz_profile=np.full(7, 1e-3, dtype=np.float64),
        dx=1e-3,
    )
    sharded = build_sharded_nu_grid(grid, 2, exchange_interval=1)
    assert sharded.ghost_width == nu_ghost_width(1)

    src = Path(_execute.__file__).read_text()
    assert "ghost_width = math.floor" not in src, (
        "the NU-forward preflight grew a second ghost-width formula again")
    assert "nu_ghost_width(" in src, (
        "the NU-forward preflight must take its ghost width from the builder")


def test_the_nu_forward_preflight_admits_with_the_builders_ghost_width():
    """The unification, end to end on the lane that carries the check.

    ``nx=4`` over 2 ranks is ``nx_per_rank=2``. MEASURED on origin/main
    (d56f68eb) with the local ``floor(K/2)+1``:

        K=3  ghost_width=2, ``2 > 2`` is False -> check 3 PASSED, and the
             call fell through to build_sharded_nu_grid's own
             "exchange_interval > 1 is reserved for Phase 2E"
        K=4  ghost_width=3 -> raised, but naming 3 where the builder needs 4

    With one source of truth K=3 is refused by check 3 itself, naming the
    width the builder would actually shard with.
    """
    sim = Simulation(freq_max=15e9, domain=(3e-3, 4e-3, 4e-3), dx=1e-3,
                     boundary="pec")
    sim._dx_profile = np.full(3, 1e-3)
    sim._dy_profile = np.full(4, 1e-3)
    sim._dz_profile = np.full(4, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_source(position=(1.5e-3, 2e-3, 2e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(1.5e-3, 2e-3, 2e-3), component="ez")
        with pytest.raises(ValueError, match="ghost_width=3 exceeds"):
            sim.forward(n_steps=4, distributed=True, devices=_devices(),
                        exchange_interval=3)


# ---------------------------------------------------------------------------
# Negative controls -- the default configuration still runs, and still
# matches the single-device lane at the tolerances the shipped parity tests
# already use (tests/unit/runners/test_distributed.py:149 and :291).
# ---------------------------------------------------------------------------

def test_default_pec_configuration_still_matches_single_device():
    """Fixture and tolerance copied from
    ``test_distributed.py::test_distributed_matches_single_pec`` (rel < 1e-4).
    """
    def build():
        sim = Simulation(freq_max=3e9, domain=(0.05, 0.02, 0.02),
                         boundary="pec")
        sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")
        return sim

    single = build().run(n_steps=100)
    multi = build().run(n_steps=100, devices=_devices())
    ts_s = np.asarray(single.time_series)
    ts_m = np.asarray(multi.time_series)
    assert ts_s.shape == ts_m.shape
    peak = np.max(np.abs(ts_s)) + 1e-30
    rel = np.max(np.abs(ts_s - ts_m)) / peak
    assert rel < 1e-4, f"distributed vs single relative error {rel:.2e}"


def test_default_cpml_configuration_still_matches_single_device():
    """Fixture and tolerance copied from
    ``test_distributed.py::test_distributed_cpml_matches_single`` (rel < 1e-3).
    """
    def build():
        sim = Simulation(freq_max=3e9, domain=(0.13, 0.04, 0.04),
                         boundary="cpml")
        sim.add_source(position=(0.065, 0.02, 0.02), component="ez",
                       waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9))
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
        return sim

    single = build().run(n_steps=200)
    multi = build().run(n_steps=200, devices=_devices())
    ts_s = np.asarray(single.time_series)
    ts_m = np.asarray(multi.time_series)
    assert ts_s.shape == ts_m.shape
    peak = np.max(np.abs(ts_s)) + 1e-30
    rel = np.max(np.abs(ts_s - ts_m)) / peak
    assert rel < 1e-3, f"CPML distributed error {rel:.2e}"


def test_a_plain_single_cell_excited_port_still_runs_distributed():
    """The admitted port shape: ``extent=None``, ``excite=True``."""
    res = _run_distributed(_build(port=dict(impedance=50.0)))
    ts = np.asarray(res.time_series)
    assert np.max(np.abs(ts)) > 0, "the admitted port injects nothing"


def test_the_default_configuration_declares_none_of_the_five():
    """The admission gate is a no-op on a model that declares none of them."""
    from rfx.runners.distributed_v2 import (
        refuse_unsupported_distributed_features,
    )
    sim = _build()
    assert refuse_unsupported_distributed_features(sim, lane="test") is None


def test_the_tfsf_and_waveguide_fallbacks_are_deliberately_unchanged():
    """S3 of the direction note keeps these two as FALLBACKS, not refusals:
    they run the whole model on one device, which is the right answer rather
    than a silently wrong one. Pinned here so the admission gate does not
    quietly swallow them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=5e9, domain=(0.13, 0.04, 0.04),
                         boundary="cpml")
        sim.add_tfsf_source(f0=2.5e9, bandwidth=0.5)
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
        res = sim.run(n_steps=50, devices=_devices(), skip_preflight=True)
    assert np.max(np.abs(np.asarray(res.time_series))) > 0
    assert any("Falling back to single-device" in str(w.message)
               for w in caught), [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# The fallbacks carry the monitors too, and the two entry points must agree.
#
# REGRESSION GUARD. The gate is called from TWO places: the public dispatch
# in ``rfx/api/_execute.py`` and ``run_distributed()`` itself. The runner's
# copy sits AFTER the TFSF / waveguide single-device fallbacks on purpose --
# those run the whole model on one device, and that lane honours every one of
# the four features. The API copy therefore has to be skipped for exactly the
# models the runner will hand back to ``sim.run()``, or a working call gets
# refused at the door.
#
# MEASURED on pristine main (d56f68eb), 2 virtual CPU devices, 0.13x0.04x0.04
# m CPML box, ``add_tfsf_source(f0=2.5e9, bandwidth=0.5)``, one Ez probe at
# the centre, 30 steps: ``run(devices=...)`` fell back (1 warning) and
# returned ``flux_monitors == ['flux_x_0']`` / an ``ntff_data``, probe peak
# 1.401931e-12 -- identical to the native run and to ``run_distributed()``
# called directly. An API-level gate placed before the fallbacks turned that
# into ``NotImplementedError`` while ``run_distributed()`` on the SAME model
# still ran, i.e. the two public entry points disagreed. The three fallback
# tests in ``tests/unit/runners/test_distributed.py`` carry point probes only
# and so could not see it; these do.
# ---------------------------------------------------------------------------

def _tfsf_sim(*, flux=False, ntff=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=5e9, domain=(0.13, 0.04, 0.04),
                         boundary="cpml")
        sim.add_tfsf_source(f0=2.5e9, bandwidth=0.5)
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
        if flux:
            sim.add_flux_monitor(axis="x", coordinate=0.09, n_freqs=3)
        if ntff:
            sim.add_ntff_box((0.03, 0.01, 0.01), (0.10, 0.03, 0.03),
                             n_freqs=3)
    return sim


def _waveguide_sim(*, flux=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(0.13, 0.04, 0.04),
                         boundary="cpml")
        sim.add_waveguide_port(
            x_position=0.01, y_range=(0.005, 0.035),
            z_range=(0.005, 0.035), mode=(1, 0), mode_type="TE",
            direction="+x", f0=5e9, bandwidth=0.5,
        )
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")
        if flux:
            sim.add_flux_monitor(axis="x", coordinate=0.09, n_freqs=3)
    return sim


@pytest.mark.parametrize("kind,kwargs", [
    ("tfsf", dict(flux=True)),
    ("tfsf", dict(ntff=True)),
    ("waveguide", dict(flux=True)),
])
def test_a_fallback_model_with_a_monitor_still_falls_back_through_run(
        kind, kwargs):
    """The standard RCS / transmission setup: TFSF (or a waveguide port)
    PLUS a flux monitor or an NTFF box. ``run_distributed()`` hands it to
    the single-device lane, which populates the monitor, so the public
    ``run(devices=...)`` must do the same and not refuse it."""
    sim = (_tfsf_sim(**kwargs) if kind == "tfsf"
           else _waveguide_sim(**kwargs))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(n_steps=30, devices=_devices(), skip_preflight=True)
    assert any("Falling back to single-device" in str(w.message)
               for w in caught), [str(w.message) for w in caught]
    if kwargs.get("flux"):
        assert res.flux_monitors is not None, (
            "the fallback dropped the flux monitor")
        assert list(res.flux_monitors) == ["flux_x_0"], res.flux_monitors
    if kwargs.get("ntff"):
        assert res.ntff_data is not None, "the fallback dropped the NTFF box"


@pytest.mark.parametrize("kind,kwargs", [
    ("tfsf", dict(flux=True)),
    ("tfsf", dict(ntff=True)),
    ("waveguide", dict(flux=True)),
])
def test_both_entry_points_agree_on_a_fallback_model_with_a_monitor(
        kind, kwargs):
    """``sim.run(devices=...)`` and ``run_distributed()`` must reach the same
    verdict on the same model. Before the API gate was guarded on "the runner
    will actually shard", the first refused and the second ran."""
    from rfx.runners.distributed_v2 import run_distributed

    def build():
        return (_tfsf_sim(**kwargs) if kind == "tfsf"
                else _waveguide_sim(**kwargs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via_api = build().run(n_steps=30, devices=_devices(),
                              skip_preflight=True)
        via_runner = run_distributed(build(), n_steps=30,
                                     devices=_devices())
    np.testing.assert_array_equal(np.asarray(via_api.time_series),
                                  np.asarray(via_runner.time_series))
    assert ((via_api.flux_monitors is None)
            == (via_runner.flux_monitors is None))
    assert ((via_api.ntff_data is None) == (via_runner.ntff_data is None))


def test_a_sharding_model_with_a_monitor_is_still_refused_at_the_door():
    """The guard is on the FALLBACK, not on the monitors: a model the runner
    really will shard keeps its refusal at the public dispatch."""
    with pytest.raises(NotImplementedError, match="add_flux_monitor"):
        _run_api(_build(flux=True))


# ---------------------------------------------------------------------------
# The class-5 message must not blame an absorber the caller never declared.
# ---------------------------------------------------------------------------

PEC_X_SPEC = BoundarySpec(
    x=Boundary(lo="pec", hi="pec"),
    y=Boundary(lo="cpml", hi="cpml"),
    z=Boundary(lo="cpml", hi="cpml"),
)


def test_pec_x_faces_are_still_refused_but_the_message_says_why():
    """``x='pec'`` with y/z CPML declares NO x absorber, yet the runner
    applies both x-face CPML windows anyway -- it never reads
    ``grid.face_pads`` (the S5 gap the direction note records). So the
    refusal is right (on pristine main this died in XLA: 6x8x8 mm gave "mul
    got incompatible shapes for broadcasting: (8, 1, 1), (5, 25, 25)" and
    8x8x8 mm gave "(8, 1, 1), (6, 25, 25)"), but it must not tell the caller
    to shrink an absorber they never asked for without saying that the lane
    adds it."""
    sim = _build(boundary=PEC_X_SPEC, cpml_layers=8,
                 domain=(8e-3, 8e-3, 8e-3), source=(4e-3, 4e-3, 4e-3),
                 probes=(4e-3,))
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(sim, n_steps=8)
    msg = str(excinfo.value)
    assert "x CPML absorber" in msg, msg
    assert "declares no CPML on x-lo or x-hi" in msg, msg
    assert "grid.pad_x_lo=0" in msg and "grid.pad_x_hi=0" in msg, msg
    assert "does not read grid.face_pads" in msg, msg


def test_the_absorber_remedy_never_recommends_a_one_device_run():
    """``fits`` used to search ``range(1, n_devices)``, which contains 1 for
    every real grid -- so the remedy could read "n_devices=1 is the largest
    count that fits", i.e. "omit devices=" said twice. It now searches from
    2 up and falls through to naming the depth that would fit."""
    from rfx.runners.distributed_v2 import check_x_absorber_fits_ranks

    # n_devices=2: no smaller MULTI-device count exists. cpml_layers=10 and
    # not 8: after round 3 an 8-layer window at nx_per=8/pad_x=1 overflows
    # by exactly one cell, which is admitted.
    with pytest.raises(ValueError) as excinfo:
        check_x_absorber_fits_ranks(nx=15, n_devices=2, nx_per=8, pad_x=1,
                                    ghost=1, cpml_layers=10)
    msg = str(excinfo.value)
    assert "n_devices=1" not in msg, msg
    assert "reduce cpml_layers (<= 8 fits at n_devices=2)" in msg, msg

    # n_devices=4 with a depth that fits at 2: the recommendation is real.
    with pytest.raises(ValueError) as excinfo:
        check_x_absorber_fits_ranks(nx=40, n_devices=4, nx_per=10, pad_x=0,
                                    ghost=1, cpml_layers=15)
    msg = str(excinfo.value)
    assert "n_devices=2 is the largest count above 1 that fits" in msg, msg


# ---------------------------------------------------------------------------
# The EXPORTED v1 pmap runner is gated too (round-2 review, BLOCKING 3)
# ---------------------------------------------------------------------------
# ``rfx/runners/__init__.py`` re-exports ``run_distributed`` from
# ``rfx.runners.distributed`` -- the pmap runner -- not from
# ``distributed_v2``.  The first round of this change gated
# ``distributed_v2.run_distributed`` and the ``run(devices=...)`` dispatch
# and left ``rfx.runners.run_distributed`` running all five classes.  The
# 23 mm domain below is deliberate: nx=24 is evenly divisible by 2, so the
# runner's own "not evenly divisible" ValueError cannot bounce the call and
# hide the gap (it is what made the 24 mm fixtures look safe).

def _run_v1(sim, n_steps=N_STEPS, n_devices=2):
    from rfx.runners.distributed import run_distributed as run_v1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_v1(sim, n_steps=n_steps, devices=_devices()[:n_devices])


V1_DOMAIN = (23e-3, 12e-3, 12e-3)   # nx = 24, divisible by 2 devices


def test_the_exported_runner_is_the_pmap_one_and_it_is_gated():
    """The export is the fact that makes the gap reachable, so pin it."""
    import rfx.runners as R
    assert R.run_distributed.__module__ == "rfx.runners.distributed", (
        "if this export moves to distributed_v2 the four tests below still "
        "pass but stop testing the pmap runner -- re-point them"
    )


@pytest.mark.parametrize("name,make,match", [
    ("periodic",
     lambda: _build(periodic="y", domain=V1_DOMAIN),
     "periodic / Bloch"),
    ("extent port",
     lambda: _build(port=dict(impedance=50.0, extent=3e-3),
                    domain=V1_DOMAIN),
     r"extent="),
    ("passive port",
     lambda: _build(port=dict(impedance=50.0, excite=False,
                              waveform=GaussianPulse(f0=7.5e9,
                                                     bandwidth=0.8)),
                    domain=V1_DOMAIN),
     "excite=False"),
    ("flux monitor",
     lambda: _build(flux=True, domain=V1_DOMAIN),
     "add_flux_monitor"),
    ("ntff box",
     lambda: _build(ntff=True, domain=V1_DOMAIN),
     "add_ntff_box"),
])
def test_the_v1_pmap_runner_refuses_the_four_feature_classes(name, make,
                                                             match):
    """RED on the first-round tree, at nx=24 with 0 warnings every time:
    extent port -> pmap probe energy 0.000000e+00 vs native 9.951533e-04;
    excite=False -> pmap 1.982867e-03 vs native 0.0; flux ->
    result.flux_monitors is None; periodic 'y' -> max|dEz| 1.963800e-04 on
    a 1.090239e-03 peak (18% of peak)."""
    with pytest.raises(NotImplementedError, match=match):
        _run_v1(make())


def test_the_v1_refusal_names_the_v1_lane():
    """The message must name the lane the caller actually used, or a v1
    caller is told to change something in a file they never called."""
    with pytest.raises(NotImplementedError) as excinfo:
        _run_v1(_build(flux=True, domain=V1_DOMAIN))
    assert "distributed (v1) pmap runner" in str(excinfo.value)


def test_the_v1_pmap_runner_refuses_an_x_absorber_that_spans_ranks():
    """RED on the first-round tree, and round 3 splits it in two.

    (a) ``x_lo='cpml'``/``x_hi='pec'`` with y/z CPML, ``cpml_layers=8``,
    5x8x8 mm at dx=1 mm (nx=14, nx_per=7, pad_x=0 -- the v1 runner requires
    ``nx % n_devices == 0``, so an 8-layer window overflows the x-hi face by
    exactly ONE cell) ran at max|dEz| 2.207114e+00 on a 4.416774e+00 peak --
    49.97% of peak, 0 warnings (re-reproduced on origin/main 7b511591). One
    cell of overflow is a measured no-op, so class 5 now admits this and
    class 6 refuses it, which is the class the 49.97% actually belongs to.

    (b) ``x=('pec','pec')`` with y/z CPML, ``cpml_layers=8``, 7x8x8 mm
    (nx=8, 2 devices, nx_per=4) overflows by four and on origin/main died
    inside XLA with ``mul got incompatible shapes for broadcasting:
    (8, 1, 1), (5, 25, 25)``. That is the band class 5 keeps, and it is
    asserted through the v1 lane here.
    """
    one_cell = _build(boundary=ASYM_SPEC, cpml_layers=8,
                      domain=(5e-3, 8e-3, 8e-3), source=(2e-3, 4e-3, 4e-3),
                      probes=(1e-3, 2e-3, 4e-3))
    with pytest.raises(ValueError) as excinfo:
        _run_v1(one_cell, n_steps=8)
    msg = str(excinfo.value)
    assert "no CPML absorber" in msg, msg
    assert "distributed (v1) pmap runner" in msg, msg

    deep = _build(boundary=PEC_X_SPEC, cpml_layers=8,
                  domain=(7e-3, 8e-3, 8e-3), source=(3e-3, 4e-3, 4e-3),
                  probes=(1e-3, 3e-3, 5e-3))
    with pytest.raises(ValueError) as excinfo:
        _run_v1(deep, n_steps=8)
    msg = str(excinfo.value)
    assert "x CPML absorber" in msg, msg
    assert "distributed (v1) pmap runner" in msg, msg


def test_the_v1_pmap_runner_refuses_a_phantom_window_at_one_device():
    """Class 6 must fire on the v1 runner at n_devices=1 as well.

    MEASURED on pristine d56f68eb through this runner with ONE device
    (ASYM spec, cpml_layers=8, 24x8x8 mm, ``_build`` default source and
    probes x=12/22 mm, 60 steps): max|dEz| 1.610351e-04 on a 3.165971e-03
    peak = 5.086e-02 of peak, the x=22 mm probe 99.46% wrong on its own
    peak, 0 warnings -- bit-for-bit the same wrongness as at 2 devices, so
    the single-device fast path is not a way around the defect. The
    symmetric control through the same call is 1.395e-06 of peak.

    Note this is NOT reachable from ``sim.run(devices=[one_device])``:
    ``rfx/api/_execute.py`` dispatches distributed only for
    ``len(devices) > 1``. It is reachable from
    ``rfx.runners.run_distributed`` and as ``distributed_v2``'s
    ``n_devices == 1`` delegate.
    """
    with pytest.raises(ValueError, match="no CPML absorber"):
        _run_v1(_asym(8, 24), n_steps=8, n_devices=1)


def test_the_v1_pmap_runner_still_runs_a_symmetric_absorber_at_one_device():
    """Negative control for the two v1 slab checks: parity, unweakened.

    Pristine d56f68eb, this fixture through the v1 runner at one device:
    1.395464e-06 of peak (2 devices: 1.836136e-06). Asserted at the
    shipped CPML tolerance of 1e-3.
    """
    single = _sym_cpml().run(n_steps=60, skip_preflight=True)
    multi = _run_v1(_sym_cpml(), n_steps=60, n_devices=1)
    ts_s = np.asarray(single.time_series)
    ts_m = np.asarray(multi.time_series)
    assert ts_s.shape == ts_m.shape
    peak = np.max(np.abs(ts_s)) + 1e-30
    rel = np.max(np.abs(ts_s - ts_m)) / peak
    assert rel < 1e-3, f"v1 symmetric CPML error {rel:.2e}"


# Ordering note: the v1 gate went in AFTER ``_refuse_f0``, so the
# pre-existing thin-conductor refusal keeps firing first. That is pinned
# where it already lives --
# ``tests/unit/materials/test_sheet_impedance.py::test_g9_distributed_runners_refuse``
# calls the v1 runner on a sheet-bearing sim and asserts the f0 message; it
# would go red if the gate had been placed above ``_refuse_f0``.


# ---------------------------------------------------------------------------
# The refusals also fire on the DEFAULT run(devices=...) path
# ---------------------------------------------------------------------------
# Every other ``_run_api`` call in this file passes ``skip_preflight=True``,
# which pins nothing about the default path. MEASURED on pristine d56f68eb
# through ``sim.run(n_steps=..., devices=[d0, d1])`` with NO
# ``skip_preflight``: all six ran. periodic 'y' peak 8.938612e-04 with 0
# warnings; extent port peak exactly 0.0 with 0 warnings; excite=False peak
# 1.910045e-02 with 0 warnings; flux and ntff returned
# ``flux_monitors=None`` / ``ntff_data=None`` (the NTFF case with one
# advisory about PEC + far-field, which says nothing about the drop); and
# ``_asym(8, 6)`` ran with one advisory about a probe near the absorber
# (round 3: that fixture is now the class-6 case -- its one-cell overflow is
# a no-op -- and the class-5 row below uses ``_asym_deep()``, which on
# origin/main died inside XLA rather than running).
# So "no refusal and no warning" held through the default path too, and now
# it is a pinned fact rather than a measured one.

@pytest.mark.parametrize("name,make,exc,match", [
    ("periodic", lambda: _build(periodic="y"),
     NotImplementedError, "periodic / Bloch"),
    ("extent port", lambda: _build(port=dict(impedance=50.0, extent=3e-3)),
     NotImplementedError, r"extent="),
    ("passive port",
     lambda: _build(port=dict(impedance=50.0, excite=False,
                              waveform=GaussianPulse(f0=7.5e9,
                                                     bandwidth=0.8))),
     NotImplementedError, "excite=False"),
    ("flux monitor", lambda: _build(flux=True),
     NotImplementedError, "add_flux_monitor"),
    ("ntff box", lambda: _build(ntff=True),
     NotImplementedError, "add_ntff_box"),
    ("x absorber", lambda: _asym_deep(), ValueError, "x CPML absorber"),
    ("phantom window", lambda: _asym(8, 24), ValueError,
     "no CPML absorber"),
])
def test_every_class_is_refused_on_the_default_preflight_path(name, make,
                                                              exc, match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(exc, match=match):
            make().run(n_steps=8, devices=_devices())


# ---------------------------------------------------------------------------
# The ghost-width helper refuses a fractional interval
# ---------------------------------------------------------------------------

def test_a_fractional_exchange_interval_is_refused_not_truncated():
    """``nu_ghost_width`` did ``int(exchange_interval)`` BEFORE the ``k < 1``
    check, so K=2.5 silently became a ghost width of 2 -- a halo half a cell
    short of the interval it is sized for. Consistent across both call
    sites, so it produced no disagreement to fail on."""
    from rfx.runners.distributed_nu import nu_ghost_width
    with pytest.raises(ValueError, match="integer number of steps"):
        nu_ghost_width(2.5)
    with pytest.raises(ValueError, match="must be >= 1"):
        nu_ghost_width(0)
    # an integral float is still the same integer, and still accepted
    assert nu_ghost_width(3.0) == 3
    assert nu_ghost_width(3) == 3
