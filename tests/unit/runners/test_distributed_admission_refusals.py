"""The distributed lane refuses five silent-wrong classes instead of running them.

Five features reached ``run(devices=...)`` with **no refusal and no warning**
and came back wrong. The map is
``rfx-research-notes/accel-import-20260913/DIRECTION-distributed-preflight.md``
S3-S4 and its survey ``decomposition-survey.md`` SC1; this file is the
first layer of that note's distributed preflight -- the position-independent
admission check plus the one position-dependent slab check.

Every number below was MEASURED on ``origin/main`` (d56f68eb) before the
refusals existed, on 2 virtual CPU devices (the root ``conftest.py`` default),
with the harness these tests still use -- ``_build()`` for classes 1-4 and
``_asym()`` for class 5, both of which now carry the source position and the
probe row explicitly, because none of the digits below survive a change to
either. None of the five raised, and none of them warned.

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
   ``sim._periodic_axes`` occurs ZERO times in
   ``rfx/runners/distributed_v2.py`` (grep, 2026-09-14).

2. **Extended lumped port** (``impedance=50``, ``extent=3e-3``)::

       distributed probe energy sum(Ez^2) = 0.000000e+00  (max|Ez| = 0.0)
       native     probe energy sum(Ez^2) = 9.953718e-04  (max|Ez| = 1.412484e-02)
       warnings raised                    = 0

   Cause: the lane forks ``impedance > 0 and extent is None`` then
   ``elif impedance == 0`` (the source/termination fork in
   ``distributed_v2.py``, and its twin at ``distributed.py:1414-1422``).
   A wire port satisfies neither, so it gets no source and no resistive
   termination.

3. **Passive port** (``excite=False``), with an explicit waveform::

       distributed energy = 1.983971e-03   (max|Ez| = 1.910045e-02)
       native     energy = 0.000000e+00   (max|Ez| = 0.0)
       same port with excite=True, distributed = 1.983971e-03  (bit-identical)
       warnings raised                    = 0

   With the documented ``waveform=None`` default the lane instead died inside
   ``make_port_source`` with ``TypeError: Expected a callable value, got
   None`` -- a crash that names no feature. Cause: ``excite`` occurs ZERO
   times in either distributed runner; ``rfx/runners/uniform.py:421,440``
   honours it.

4. **Flux monitor / NTFF box**::

       distributed result.flux_monitors = None   (native: 1 monitor, 'flux_x_0')
       distributed result.ntff_data     = None   (native: NTFFData)
       distributed result.ntff_box      = None
       warnings raised                  = 0

   Same class as the #579 DFT-plane refusal: no accumulator, no cross-rank
   reduce, so a registered monitor is dropped and the field comes back None.

5. **x-absorber spanning ranks.** ``_asym(8, 6)``: ``x_lo='cpml'``/
   ``x_hi='pec'`` with y/z CPML, 6x8x8 mm at dx=1 mm, ``cpml_layers=8``,
   2 devices, 60 steps, field source at (3, 4, 4) mm, Ez probe row at
   x = 1, 3, 5, 6 mm -> ``nx=15``, ``pad_x=1``, ``nx_per=8``, so the x-hi
   window wants cells ``[0, 8)`` of a slab whose owned REAL cells are
   ``[1, 8)`` (its 9th is a #623 alignment cell)::

       native peak abs(Ez)             = 4.416633e+00  (probe x=3 mm)
       max|Ez_distributed - Ez_native| = 2.207244e+00  (49.98% of peak)
       probe at x=5 mm, one cell inside
         the x-hi PEC face             = 1.646758e-01 wrong on its own
                                         1.646768e-01 peak = 99.9994%
       warnings raised                 = 0
       the SAME fixture at nx_per=17 (``_asym(8, 24)``, 24 mm domain)
         = 1.005828e-06 on a 4.422700e+00 peak = 2.274240e-07 of peak
           (parity; this is the negative control below)

   The probe ON the x-hi face (x=6 mm) reads exactly 0.0 on both lanes, so
   the 99.9994% probe is the one a cell inside it. The earlier draft of this
   note quoted the control as ``9.7e-05`` of peak; that digit belongs to no
   fixture that could be reproduced and is replaced by the measured
   2.274240e-07 above. (For the record, the shipped ``_build`` DEFAULT
   source at (6, 6, 6) mm -- which on a 6 mm domain sits on the x-hi PEC
   plane and is discarded -- gives a different silent-wrong on the same
   model: native peak 5.554087e-03 vs a distributed peak of 2.499657e-08,
   rel 1.000003, and its 24 mm control sits at 3.05e-03 of peak, ABOVE the
   1e-3 CPML parity tolerance. That last number sizes the S5 gap for lane B
   and is recorded in the design note's S5.)

   One cell of overflow keeps every array shape valid, so nothing raises.
   Two or more cells overflow past the slab end and XLA raises a
   broadcasting error that names no feature: with the same boundary spec but
   ``cpml_layers=20`` and a 7x12x12 mm domain at dx=1 mm (nx=28, pad_x=0,
   nx_per=14, ny=nz=53) main dies with ``mul got incompatible shapes for
   broadcasting: (20, 1, 1), (15, 53, 53)``. The 8-layer/8 mm fixture the
   earlier draft attached that quote to cannot produce it -- it gives
   ``(8, 1, 1), (6, 25, 25)``.

6. **Ghost-width formula.** ``rfx/api/_execute.py`` computed
   ``floor(K/2)+1`` while ``rfx/runners/distributed_nu.build_sharded_nu_grid``
   uses ``ghost = K``::

       K:                1   2   3   4
       floor(K/2)+1:     1   2   2   3
       builder (g=K):    1   2   3   4

   The preflight was SHORT by one cell from K=3 up, i.e. it would clear a
   configuration the builder cannot shard. There is now one source of truth,
   :func:`rfx.runners.distributed_nu.nu_ghost_width`.

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


def test_x_absorber_spanning_ranks_is_refused():
    """RED: 2.207244 on a 4.416633 peak (49.98%), and the probe one cell
    inside the x-hi face wrong by 99.9994% of its own peak, 0 warnings --
    with the ``_asym`` fixture below, whose source and probe row are part of
    the measurement (see module docstring)."""
    with pytest.raises(ValueError, match="x CPML absorber"):
        _run_distributed(_asym(8, 6), n_steps=8)


def test_x_absorber_refusal_reports_the_slab_arithmetic():
    with pytest.raises(ValueError) as excinfo:
        _run_distributed(_asym(8, 6), n_steps=8)
    msg = str(excinfo.value)
    for token in ("cpml_layers=8", "nx=15", "n_devices=2", "nx_per=8",
                  "pad_x=1"):
        assert token in msg, f"{token!r} missing from: {msg}"
    # At n_devices=2 there IS no smaller multi-device count, so the message
    # must not recommend "n_devices=1" (that is "omit devices=" said twice);
    # it names the depth that would fit and the single-device way out.
    assert "n_devices=1" not in msg, msg
    assert "omit devices=" in msg, msg
    assert "reduce cpml_layers" in msg, msg


def test_x_absorber_refusal_also_fires_through_the_public_dispatch():
    with pytest.raises(ValueError, match="x CPML absorber"):
        _run_api(_asym(8, 6), n_steps=8)


def test_the_x_absorber_condition_is_the_window_arithmetic():
    """The exact boundary, unit-tested without running a simulation.

    The x-hi window is ``[nx_per + ghost - pad_x - n, nx_per + ghost -
    pad_x)`` on a slab whose owned cells are ``[ghost, ghost + nx_per)``, so
    ``n == nx_per - pad_x`` is the last admissible depth and ``n + 1`` is
    the first refused one.
    """
    from rfx.runners.distributed_v2 import check_x_absorber_fits_ranks
    for pad_x in (0, 1, 3):
        nx_per = 12
        kw = dict(nx=nx_per * 2, n_devices=2, nx_per=nx_per, pad_x=pad_x,
                  ghost=1)
        # last admissible depth: silent, returns None
        assert check_x_absorber_fits_ranks(
            cpml_layers=nx_per - pad_x, **kw) is None
        # one cell deeper: refused
        with pytest.raises(ValueError, match="x CPML absorber"):
            check_x_absorber_fits_ranks(cpml_layers=nx_per - pad_x + 1, **kw)
        # and the x-lo face has its own (looser) bound
        with pytest.raises(ValueError, match="x-lo"):
            check_x_absorber_fits_ranks(cpml_layers=nx_per + 1, **kw)


def test_a_fitting_x_absorber_is_not_refused():
    """Negative control for class 5: the SAME fixture with a wider domain
    (nx=33, nx_per=17, so the 8-layer absorber fits) ran on main at
    1.005828e-06 on a 4.422700e+00 peak = 2.274240e-07 of peak, and must
    keep running."""
    res = _run_distributed(_asym(8, 24), n_steps=8)
    assert res.time_series is not None


# ---------------------------------------------------------------------------
# 6. One ghost-width formula
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

    # n_devices=2: no smaller MULTI-device count exists.
    with pytest.raises(ValueError) as excinfo:
        check_x_absorber_fits_ranks(nx=15, n_devices=2, nx_per=8, pad_x=1,
                                    ghost=1, cpml_layers=8)
    msg = str(excinfo.value)
    assert "n_devices=1" not in msg, msg
    assert "reduce cpml_layers (<= 7 fits at n_devices=2)" in msg, msg

    # n_devices=4 with a depth that fits at 2: the recommendation is real.
    with pytest.raises(ValueError) as excinfo:
        check_x_absorber_fits_ranks(nx=40, n_devices=4, nx_per=10, pad_x=0,
                                    ghost=1, cpml_layers=15)
    msg = str(excinfo.value)
    assert "n_devices=2 is the largest count above 1 that fits" in msg, msg
