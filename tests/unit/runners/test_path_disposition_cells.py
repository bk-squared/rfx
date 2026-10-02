"""The executable cells of the path-disposition table, run.

Each cell of ``tests/contracts/path_disposition.py`` on a row with a model
here is checked on a tiny model: a 12 mm PEC box with 1 mm cells, an Ez soft
source at x = 4 mm and an Ez probe at x = 8 mm, 40 steps. The graded lanes
get a one-size ``dx_profile`` (the same mesh, so the lane is the only
difference, as in #1282); the ADI lanes (``run()`` and ``forward()``) set
``solver='adi'``; the subgridded lane gets a 20 mm tall box whose refinement
covers z = 0-14 mm, which production validation accepts.

* ``refuses``: the model is built first (or, for a ``declared`` cell, its
  declaration raises); then the path raises ``NotImplementedError`` or
  ``ValueError`` naming the cell's ``raises`` fragment, before any kernel
  scan starts.
* ``carries``: the declared input changes the probe record by more than
  ``EFFECT_FLOOR`` of its peak, compared with the same model without it. On
  the lanes that share the reference lane's mesh and time step
  (``PARITY_LANES``) the record must also agree with it to ``PARITY_TOL``:
  ``sim.run()`` of the same model, or ``run()`` of the ADI model for
  ``fwd_adi``. An observer carries when its result field is filled. On the
  ADI lanes an absorber is checked in the conductivity handed to the kernel,
  because declaring one also pads the grid.
* ``falls back``: the path warns and returns the named lane's result.
* a cell with ``wrong`` is a strict expected failure against its issue
  (``raises=AssertionError``, so a crash is not taken for it). Its check
  passes only when the path refuses the input or carries it (effect, parity
  for a ``carries`` cell, and no departure where
  ``tests/contracts/test_realized_boundary.py`` measures the boundary), so
  fixing the issue either way turns it into an unexpected pass that forces
  the table to be updated.

``EFFECT_FLOOR`` and ``PARITY_TOL`` are relative to the probe peak. The float32
noise between two lanes doing the same arithmetic in a different order on
this model is below 3e-6 (``test_thresholds_clear_float32_noise`` measures
it); the parity tolerance is the multi-device PEC criterion of
``test_distributed.TestDistributedRunner``, and the effect floor is ten times
that.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from typing import Callable, NamedTuple
from unittest.mock import patch

import jax
import numpy as np
import pytest

import rfx.adi
from rfx import (Box, DebyePole, GaussianPulse, PolylineWire, Simulation,
                 Sphere, drude_pole, lorentz_pole)
from rfx.runners import _admission as A
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests.contracts import path_disposition as T
from tests.contracts.test_realized_boundary import measured as _boundary_departures
from tests.unit.nonuniform.test_refinement_refused_on_graded_mesh import _box as _refined_box_1282

N_STEPS = 40
EFFECT_FLOOR = 1e-3
PARITY_TOL = 1e-4
PARITY_LANES = ("run_nonuniform", "run_distributed", "fwd_uniform",
                "fwd_nonuniform", "fwd_distributed_nu", "fwd_adi")
GRADED = ("run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu")
ADI = ("run_adi", "fwd_adi")
WAVEFORM = GaussianPulse(f0=5e9, bandwidth=0.8)
# Modules whose lax.scan is a time loop; a refusal must come before any of them.
_KERNEL_MODULES = ("rfx/simulation.py", "rfx/nonuniform.py", "rfx/runners/",
                   "rfx/adi.py", "rfx/subgridding/", "rfx/vmap_sweep.py",
                   "rfx/progress.py")


def mm(*v):
    return tuple(x * 1e-3 for x in v)


def _devices():
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "requires the root conftest's two CPU devices"
    return devices[:2]


def _graded(n):
    """n cells of 1 mm with the middle four split in two: the same length."""
    cells = [1e-3] * n
    mid = n // 2 - 2
    return np.array(cells[:mid] + [0.5e-3] * 8 + cells[mid + 4:])


def _simulation(lane, domain, *, ref=False, **ctor):
    """A Simulation on ``domain`` (mm) for ``lane``; ``ref`` builds it for
    ``sim.run()`` on one device, the parity reference."""
    ctor = dict(freq_max=10e9, domain=mm(*domain), dx=1e-3, boundary="pec") | ctor
    if not isinstance(ctor["boundary"], str) or ctor["boundary"] != "pec":
        ctor.setdefault("cpml_layers", 4)
    if not ref and lane in GRADED and "dx_profile" not in ctor:
        ctor["dx_profile"] = np.full(domain[0], 1e-3)
    if lane in ADI:
        ctor["solver"] = "adi"
    return Simulation(**ctor)


def _base(lane, *, ref=False, source="field", refine=True, **ctor):
    """The 12 mm box; on the subgridded lane a 20 mm tall one, refined."""
    subgrid = lane == "run_subgridded" and not ref
    sim = _simulation(lane, (12, 12, 20 if subgrid else 12), ref=ref, **ctor)
    if source:
        sim.add_source(mm(4, 6, 6), "ez", waveform=WAVEFORM, amplitude_kind=source)
    sim.add_probe(mm(8, 6, 6), "ez")
    if subgrid and refine:
        sim.add_refinement(z_range=(0.0, 14e-3), ratio=2)
    return sim


def _block(sim, lo=(5, 3, 3), hi=(7, 9, 9), **material):
    sim.add_material("block", **material)
    sim.add(Box(mm(*lo), mm(*hi)), material="block")
    return sim


def _probe(result):
    return np.asarray(result.time_series, dtype=np.float64)


class Feature(NamedTuple):
    """How to declare one input on one lane.

    ``build(lane, on, ref)`` returns the model with (``on``) or without the
    input. ``off`` names a model without the input that other features share,
    so it runs once per lane. ``variant(lane)`` names a model that differs
    between lanes beyond the lane's own mesh and solver; the parity reference
    is built per variant. ``boundary`` maps a lane to the
    ``test_realized_boundary.py`` (case, entry) that measures its walls, and
    ``adi_layers`` is the absorber thickness to find in the conductivity the
    ADI kernel receives.
    """
    build: Callable
    read: Callable = _probe
    steps: Callable = lambda lane: N_STEPS
    run_kwargs: Callable = lambda lane: {}
    variant: Callable = lambda lane: ""
    parity: bool = True
    observer: str = ""   # result field an observer fills
    off: str = ""
    boundary: dict = {}
    adi_layers: int = 0


def _added(add, **base):
    """The base model, with ``add(sim, lane)`` applied when on."""
    def build(lane, on, ref=False):
        sim = _base(lane, ref=ref, **base)
        if on:
            add(sim, lane)
        return sim
    return build


def _plus(add):
    """A feature added to the shared base model."""
    return Feature(_added(add), off="base")


def _observer(add, field, **base):
    return Feature(_added(add, **base), observer=field)


def _pole(material):
    """A dispersive block against its static part. On run_distributed, the
    #1302 check: a 4-layer CPML box, 250 steps (the two-device record leaves
    run()'s between 200 and 250 steps)."""
    static = {k: v for k, v in material.items() if k in ("eps_r", "sigma", "mu_r")}

    def build(lane, on, ref=False):
        cpml = lane == "run_distributed"
        sim = _base(lane, ref=ref, **({"boundary": "cpml"} if cpml else {}))
        return _block(sim, **(material if on else static))
    return Feature(build, steps=lambda lane: 250 if lane == "run_distributed" else N_STEPS,
                   variant=lambda lane: "cpml" if lane == "run_distributed" else "",
                   off=f"block eps_r={static['eps_r']:g}")


def _kerr(lane, on, ref=False):
    return _block(_base(lane, ref=ref), eps_r=1.0, chi3=1e3 if on else 0.0)


def _amplitude_kind(lane, on, ref=False):
    return _base(lane, ref=ref, source="current" if on else "field")


def _interface_eps(lane, on, ref=False):
    """An εr 4 block offset by half a cell, so its faces cut E edges."""
    sim = _base(lane, ref=ref, interface_eps="dual_average" if on else "sampled")
    return _block(sim, lo=(5.5, 3, 3), hi=(7.5, 9, 9), eps_r=4.0)


def _board(lane, on, ref=False):
    """A microstrip: 1 mm εr 3.66 substrate, 2 mm PEC trace, the port the only
    drive. ADI refuses any trace, so its board has none: that is the #1308 case."""
    subgrid = lane == "run_subgridded" and not ref
    length = 24
    sim = _simulation(lane, (length, 12, 10 if subgrid else 6), ref=ref)
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0, 0, 0), mm(length, 12, 1)), material="substrate")
    if lane not in ADI:
        sim.add(Box(mm(1, 5, 1), mm(length - 1, 7, 2)), material="pec")
    if on:
        sim.add_msl_port(position=mm(2, 6, 0), width=2e-3, height=1e-3, direction="+x",
                         impedance=50.0, waveform=GaussianPulse(f0=7e9, bandwidth=0.8))
    sim.add_probe(mm(12, 6, 2.5), "ez")
    if subgrid:
        sim.add_refinement(z_range=(0.0, 7e-3), ratio=2)
    return sim


def _guide(lane, on, ref=False):
    """A 12 × 6 mm PEC guide absorbing on x; the TE10 port the only drive.
    ADI takes one absorber on all six faces, so its guide is a CPML box: the
    port must reach ADI's own refusal, not the per-face one."""
    spec = ("cpml" if lane in ADI else
            BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pec", hi="pec")))
    sim = _simulation(lane, (30, 12, 6), ref=ref, freq_max=20e9, boundary=spec)
    if on:
        sim.add_waveguide_port(6e-3, direction="+x", mode=(1, 0), mode_type="TE",
                               freqs=np.linspace(13e9, 18e9, 3), f0=15e9, bandwidth=0.5,
                               probe_offset=4, ref_offset=2)
    sim.add_probe(mm(20, 6, 3), "ez")
    if lane == "run_subgridded" and not ref:
        sim.add_refinement(z_range=(0.0, 3e-3), ratio=2)
    return sim


def _plane_wave(lane, on, ref=False):
    sim = _simulation(lane, (24, 6, 6), ref=ref, boundary="cpml")
    if on:
        sim.add_tfsf_source(f0=5e9, bandwidth=0.5, margin=3)
    sim.add_probe(mm(14, 3, 3), "ez")
    if lane == "run_subgridded" and not ref:
        sim.add_refinement(z_range=(0.0, 3e-3), ratio=2)
    return sim


def _floquet_cell(scan_theta):
    """A 6 mm periodic cell absorbing on z, the Floquet port the only drive.
    On the ADI lanes a CPML box, for the reason _guide gives."""
    def build(lane, on, ref=False):
        spec = "cpml" if lane in ADI else BoundarySpec(x="periodic", y="periodic", z="cpml")
        sim = _simulation(lane, (6, 6, 20), ref=ref, boundary=spec)
        if on or scan_theta:
            sim.add_floquet_port(4e-3, axis="z", f0=5e9, scan_theta=scan_theta if on else 0.0)
        sim.add_probe(mm(3, 3, 14), "ez")
        if lane == "run_subgridded" and not ref:
            sim.add_refinement(z_range=(0.0, 3e-3), ratio=2)
        return sim
    return build


def _sphere(*, ports=False):
    """A PEC sphere between source and probe, walls declared conformal on x."""
    def build(lane, on, ref=False):
        spec = BoundarySpec(x=Boundary(lo="pec", hi="pec", conformal=on),
                            y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pec", hi="pec"))
        sim = _base(lane, ref=ref, boundary=spec, source=None if ports else "field")
        if ports:
            for x in (4, 8):
                sim.add_port(mm(x, 6, 6), "ez", impedance=50.0, waveform=WAVEFORM)
        sim.add(Sphere(center=mm(6, 6, 6), radius=1.2e-3), material="pec")
        return sim
    return build


def _s_params(result):
    return np.abs(np.asarray(result.s_params))


def _refinement(lane, on, ref=False):
    """The refusal lanes reuse #1282's box; run_subgridded carries the slab."""
    if lane == "run_subgridded":
        return _base(lane, ref=ref, refine=on)
    return _refined_box_1282(on, graded=lane in GRADED,
                             solver="adi" if lane in ADI else "yee")


def _boundary(on_ctor, *, cpml_off=False, **options):
    """A boundary declaration against the PEC box, or against the CPML box."""
    off_ctor = {"boundary": "cpml"} if cpml_off else {}

    def build(lane, on, ref=False):
        return _base(lane, ref=ref, **(on_ctor if on else off_ctor))
    return Feature(build, off="base cpml" if cpml_off else "base", **options)


def _profile(axis):
    def build(lane, on, ref=False):
        sim_domain = (12, 12, 20 if lane == "run_subgridded" else 12)
        ctor = {f"d{axis}_profile": _graded(sim_domain["xyz".index(axis)])} if on else {}
        return _base(lane, ref=ref, **ctor)
    return build


def _mode(lane, on, ref=False):
    """A 2d_tmz model against the same box in 3d: one z cell between PEC
    walls, three on the ADI lanes (a one-cell 3d box dies there)."""
    thick = 3 if lane in ADI else 1
    sim = _simulation(lane, (12, 12, thick), ref=ref, mode="2d_tmz" if on else "3d")
    sim.add_source(mm(4, 6, thick // 2), "ez", waveform=WAVEFORM, amplitude_kind="field")
    sim.add_probe(mm(8, 6, thick // 2), "ez")
    if lane == "run_subgridded" and not ref:
        sim.add_refinement(z_range=(0.0, 1e-3), ratio=2)
    return sim


def _lid(lane, on, ref=False, *, lid="cpml", validation="production", z_top=10.0, kappa=1.0):
    """A closed 12 x 12 x 16 mm PEC box with an absorbing lid on z_hi (on) or
    a PEC one (off), the base source and probe at z = 6 mm. On the subgridded
    lane the refined slab covers z = 0 to ``z_top`` mm and touches the PEC
    floor: at 10 mm, the guarded envelope production subgrid validation
    accepts."""
    spec = BoundarySpec(x=Boundary(lo="pec", hi="pec"), y=Boundary(lo="pec", hi="pec"),
                        z=Boundary(lo="pec", hi=lid if on else "pec"))
    sim = _simulation(lane, (12, 12, 16), ref=ref, boundary=spec, cpml_kappa_max=kappa)
    sim.add_source(mm(4, 6, 6), "ez", waveform=WAVEFORM, amplitude_kind="field")
    sim.add_probe(mm(8, 6, 6), "ez")
    if lane == "run_subgridded" and not ref:
        sim.add_refinement(z_range=(0.0, z_top * 1e-3), ratio=2, validation=validation)
    return sim


# The lid is 10 mm above the source: enough steps for the wave to come back.
LID_STEPS = 100


_PMC_X = BoundarySpec(x=Boundary(lo="pmc", hi="pmc"), y=Boundary(lo="pec", hi="pec"),
                      z=Boundary(lo="pec", hi="pec"))
_PERIODIC_Y = BoundarySpec(x=Boundary(lo="pec", hi="pec"), y="periodic", z=Boundary(lo="pec", hi="pec"))

FEATURES: dict[tuple[str, str], Feature] = {
    ("_precision", ""): Feature(lambda lane, on, ref=False: _base(
        lane, ref=ref, precision="mixed" if on else "float32"), off="base", parity=False),
    ("_mode", ""): Feature(_mode),
    ("_materials", "eps"): _plus(lambda s, _: _block(s, eps_r=4.0)),
    ("_materials", "sigma"): _plus(lambda s, _: _block(s, sigma=0.5)),
    ("_materials", "mu"): _plus(lambda s, _: _block(s, mu_r=4.0)),
    ("_materials", "debye"): _pole(dict(eps_r=2.0, debye_poles=[DebyePole(delta_eps=2.0, tau=1e-11)])),
    ("_materials", "lorentz"): _pole(dict(eps_r=2.0, lorentz_poles=[
        lorentz_pole(2.0, 2 * np.pi * 8e9, 2 * np.pi * 1e9)])),
    ("_materials", "drude"): _pole(dict(eps_r=1.0, lorentz_poles=[drude_pole(2 * np.pi * 10e9, 1e10)])),
    ("_materials", "kerr"): Feature(_kerr, off="block eps_r=1"),
    ("_geometry", "pec_volume"): _plus(lambda s, _: s.add(Box(mm(5, 3, 3), mm(7, 9, 9)), material="pec")),
    ("_geometry", "pec_sheet"): _plus(lambda s, _: s.add(Box(mm(6, 3, 3), mm(6, 9, 9)), material="pec")),
    ("_geometry", "pec_wire"): _plus(lambda s, _: s.add(
        PolylineWire((mm(6, 6, 3), mm(6, 6, 9)), radius=0.0), material="pec")),
    ("_thin_conductors", "lossy_sheet"): _plus(lambda s, _: s.add_thin_conductor(
        Box(mm(6, 3, 3), mm(6, 9, 9)), sigma_bulk=1e3, thickness=1e-4)),
    ("_thin_conductors", "pec_sheet"): _plus(lambda s, _: s.add_thin_conductor(
        Box(mm(6, 3, 3), mm(6, 9, 9)))),
    ("_thin_conductors", "surface_impedance"): _plus(lambda s, _: s.add_thin_conductor(
        Box(mm(6, 3, 3), mm(6, 9, 9)), sigma_bulk=1e3, thickness=1e-4, surface_impedance_f0=5e9)),
    ("_pinned_sheets", "pec_sheet"): _plus(lambda s, _: s.add_pinned_sheet(
        plane_index=6, i_range=(3, 9), j_range=(3, 9), normal_axis=0)),
    ("_ports", "source"): _plus(lambda s, _: s.add_source(
        mm(6, 8, 6), "ez", waveform=WAVEFORM, amplitude_kind="field")),
    ("_ports", "amplitude_kind"): Feature(_amplitude_kind, off="base"),
    ("_ports", "lumped_port"): _plus(lambda s, _: s.add_port(
        mm(6, 8, 6), "ez", impedance=50.0, waveform=WAVEFORM)),
    ("_ports", "passive_port"): _plus(lambda s, _: s.add_port(
        mm(6, 8, 6), "ez", impedance=50.0, excite=False)),
    ("_ports", "wire_port"): _plus(lambda s, _: s.add_port(
        mm(6, 8, 5), "ez", impedance=50.0, extent=2e-3, waveform=WAVEFORM)),
    ("_msl_ports", "msl_port"): Feature(_board),
    ("_waveguide_ports", "waveguide_port"): Feature(_guide),
    ("_coaxial_ports", "coax_port"): _plus(lambda s, _: s.add_coaxial_port(
        mm(6, 6, 0), face="bottom", pin_length=3e-3)),
    ("_floquet_ports", "floquet_port"): Feature(_floquet_cell(0.0)),
    ("_floquet_ports", "scan_angle"): Feature(_floquet_cell(30.0)),
    ("_lumped_rlc", "R"): _plus(lambda s, _: s.add_lumped_rlc(mm(6, 6, 6), "ez", R=10.0)),
    ("_lumped_rlc", "series_RL"): _plus(lambda s, _: s.add_lumped_rlc(mm(6, 6, 6), "ez", R=10.0, L=1e-9)),
    ("_tfsf", "plane_wave"): Feature(_plane_wave),
    ("_refinement", "slab"): Feature(_refinement, parity=False),
    ("_boundary", "cpml"): _boundary({"boundary": "cpml"}, adi_layers=4,
                                     boundary={lane: ("cpml", "adi") for lane in ADI}),
    ("_boundary", "upml"): _boundary({"boundary": "upml"}),
    ("_pec_faces", "pec_face"): _boundary({"boundary": "cpml", "pec_faces": {"z_lo"}}, cpml_off=True),
    ("_boundary_spec", "pmc_face"): _boundary({"boundary": _PMC_X}, boundary={
        "run_uniform": ("pmc-pec", "run"), "run_nonuniform": ("pmc-pec", "nonuniform"),
        "fwd_uniform": ("pmc-pec", "forward"), "run_distributed": ("pmc-pec", "distributed"),
        "run_adi": ("pmc-pec", "adi"), "fwd_adi": ("pmc-pec", "adi")}),
    ("_boundary_spec", "conformal"): Feature(_sphere()),
    ("_boundary_spec", "conformal_s_matrix"): Feature(
        _sphere(ports=True), read=_s_params,
        run_kwargs=lambda lane: (dict(compute_s_params=True, s_param_freqs=np.array([4e9, 5e9, 6e9]))
                                 if lane == "run_uniform" else {})),
    ("_boundary_spec", "absorbing_lid"): Feature(_lid, steps=lambda lane: LID_STEPS),
    ("_periodic_axes", "periodic"): _boundary({"boundary": _PERIODIC_Y}, boundary={
        "run_uniform": ("periodic-xy", "run"), "fwd_uniform": ("periodic-xy", "forward")}),
    ("_cpml_layers", "layers"): _boundary({"boundary": "cpml", "cpml_layers": 8}, cpml_off=True,
                                          adi_layers=8),
    ("_cpml_kappa_max", "kappa"): _boundary({"boundary": "cpml", "cpml_kappa_max": 5.0}, cpml_off=True),
    ("_interface_eps", "dual_average"): Feature(_interface_eps, parity=False),
    ("_dx_profile", "graded"): Feature(_profile("x"), off="base"),
    ("_dy_profile", "graded"): Feature(_profile("y"), off="base"),
    ("_dz_profile", "graded"): Feature(_profile("z"), off="base"),
    ("_probes", "probe"): Feature(_added(lambda s, _: None), observer="time_series", off="base"),
    ("_dft_planes", "dft_plane"): _observer(
        lambda s, _: s.add_dft_plane_probe(axis="x", coordinate=6e-3, n_freqs=3), "dft_planes"),
    ("_flux_monitors", "flux"): _observer(
        lambda s, _: s.add_flux_monitor(axis="x", coordinate=6e-3, n_freqs=3), "flux_monitors"),
    ("_ntff", "ntff_box"): _observer(
        lambda s, _: s.add_ntff_box(mm(2, 2, 2), mm(10, 10, 8), n_freqs=3), "ntff_data"),
    ("_current_moments", "block_moments"): _observer(
        lambda s, _: s.add_current_moment_monitor(mm(2, 2, 2), mm(10, 10, 10), block_size=4e-3,
                                                  freqs=np.array([5e9])),
        "current_moment_data", boundary="cpml"),
}


# ------------------------------------------------------------------ running

def _run(sim, lane, feature, *, entry=None):
    """Run ``sim`` through ``entry`` (default: ``lane``'s own) for as many
    steps as ``lane``'s cell asks."""
    entry = entry or lane
    kwargs = dict(n_steps=feature.steps(lane), skip_preflight=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry.startswith("run_"):
            if entry == "run_distributed":
                kwargs["devices"] = _devices()
            # These cells compare field records; S-request cells opt in below.
            kwargs.update(compute_s_params=False)
            kwargs.update(feature.run_kwargs(entry))
            return sim.run(**kwargs)
        kwargs["checkpoint"] = False
        if entry == "fwd_distributed_nu":
            kwargs.update(distributed=True, devices=_devices())
        return sim.forward(**kwargs)


def _build(feature, lane, on, ref=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feature.build(lane, on, ref)


_RESULTS: dict = {}
_TESTS_SINCE_CLEAR = [0]


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Every cell compiles its own programs. A draft of this file that kept
    them all aborted inside the XLA compiler after about 320 cells; dropping
    them every 100 tests also halves the resident memory (2.3 GB against
    4.3 GB, measured on CPU). The records in _RESULTS are numpy arrays."""
    yield
    _TESTS_SINCE_CLEAR[0] += 1
    if _TESTS_SINCE_CLEAR[0] >= 100:
        jax.clear_caches()
        _TESTS_SINCE_CLEAR[0] = 0


def _record(key, feature, lane, on, ref=False):
    """The probe record of one model on one lane, computed once per module."""
    if key not in _RESULTS:
        sim = _build(feature, lane, on, ref)
        _RESULTS[key] = feature.read(_run(sim, lane, feature,
                                          entry="run_uniform" if ref else lane))
    return _RESULTS[key]


def _with(name, feature, lane):
    return _record((name, lane, True), feature, lane, True)


def _without(name, feature, lane):
    return _record((feature.off or name, feature.variant(lane), lane, False), feature, lane, False)


def _reference(name, feature, lane):
    """The record the lane must agree with: run() of the ADI model for
    fwd_adi, else ``sim.run()`` of the same model on one device, which unless
    the lane's model is a variant is the run_uniform cell's own record."""
    if lane == "fwd_adi":
        return _with(name, feature, "run_adi")
    variant = feature.variant(lane)
    if not variant:
        return _with(name, feature, "run_uniform")
    return _record((name, "ref", variant), feature, lane, True, ref=True)


def relative(a, b):
    """max|a - b| over the larger peak; 0 when both records are empty."""
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    return 0.0 if peak == 0.0 else float(np.max(np.abs(a - b)) / peak)


@contextlib.contextmanager
def _watch_kernel_scans():
    """Record every lax.scan a time-stepping module starts."""
    real = jax.lax.scan
    started = []

    def scan(*args, **kwargs):
        caller = sys._getframe(1).f_code.co_filename.replace(os.sep, "/")
        if any(module in caller for module in _KERNEL_MODULES):
            started.append(caller)
        return real(*args, **kwargs)

    with patch.object(jax.lax, "scan", scan):
        yield started


def _refusal(name, feature, lane, sim):
    """The exception the path raises before any kernel scan, or None if it ran
    (its record is kept for the carried check)."""
    with _watch_kernel_scans() as started:
        try:
            result = _run(sim, lane, feature)
        except (NotImplementedError, ValueError) as exc:
            assert not started, f"{name} on {lane} raised after a kernel scan started: {started}"
            return exc
    if not feature.observer:
        _RESULTS.setdefault((name, lane, True), feature.read(result))
    return None


def _assert_refused(name, feature, lane, c):
    if c.declared:
        with pytest.raises((NotImplementedError, ValueError)) as exc:
            _build(feature, lane, True)
    else:
        exc = _refusal(name, feature, lane, _build(feature, lane, True))
        assert exc is not None, f"{name} ran on {lane}; the table says it is refused ({c.note})"
    message = str(exc.value if hasattr(exc, "value") else exc)
    assert c.raises in message, (
        f"{name} on {lane} was refused, but not by the refusal the table names "
        f"({c.raises!r}): {message[:300]}")


def _adi_absorber(name, feature, lane):
    """Problems with the absorber in the conductivity handed to the ADI
    kernel: nonzero in exactly the outer ``adi_layers - 1`` cells of every
    face (the innermost layer of the grading is zero) of the vacuum box."""
    handed = []
    real = rfx.adi.run_adi_3d

    def spy(*args, **kwargs):
        handed.append(np.asarray(args[7]))
        return real(*args, **kwargs)

    with patch.object(rfx.adi, "run_adi_3d", spy):
        _run(_build(feature, lane, True), lane, feature)
    sigma = handed[0]
    depth = np.min(np.stack(np.meshgrid(
        *[np.minimum(np.arange(n), n - 1 - np.arange(n)) for n in sigma.shape],
        indexing="ij")), axis=0)
    expected = depth <= feature.adi_layers - 2
    if np.array_equal(sigma > 0, expected):
        return []
    return [f"the ADI kernel received σ > 0 on {int(np.sum(sigma > 0))} cells, where a "
            f"{feature.adi_layers}-layer absorber covers {int(np.sum(expected))}"]


def _observed(name, feature, lane):
    if feature.observer == "time_series":   # the shared base model's record
        return bool(np.any(_without(name, feature, lane) != 0))
    result = _run(_build(feature, lane, True), lane, feature)
    value = getattr(result, feature.observer, None)
    return value is not None and (len(value) > 0 if isinstance(value, dict) else True)


def _carried(name, feature, lane, *, parity, first_only=False):
    """What keeps the lane from carrying the input; empty when it does.
    ``first_only`` (a known-wrong cell) stops at the first problem, so a cell
    that fails parity does not also run the model without the input."""
    if feature.observer:
        return [] if _observed(name, feature, lane) else [f"{feature.observer} came back empty"]
    problems = []
    if lane in ADI and feature.adi_layers:
        problems += _adi_absorber(name, feature, lane)
    elif parity and feature.parity and lane in PARITY_LANES:
        agreement = relative(_with(name, feature, lane), _reference(name, feature, lane))
        if agreement > PARITY_TOL:
            problems.append(f"it differs from its reference run by {agreement:.3e} of the peak "
                            f"(tolerance {PARITY_TOL:g})")
    if problems and first_only:
        return problems
    if not (lane in ADI and feature.adi_layers):
        effect = relative(_with(name, feature, lane), _without(name, feature, lane))
        if effect <= EFFECT_FLOOR:
            problems.append(f"declaring it moved the record by {effect:.3e} of its peak "
                            f"(floor {EFFECT_FLOOR:g})")
    if problems and first_only:
        return problems
    if lane in feature.boundary:
        departures = _boundary_departures(*feature.boundary[lane])
        if name == "_periodic_axes/periodic":
            # The periodic row judges the periodic faces. The forward lane's
            # absorber backing on z remains independently held by the B1
            # face contract (B3); it is not a periodic-period departure.
            departures = [d for d in departures if d["code"] in ("d", "e")]
        if departures:
            problems.append(f"test_realized_boundary.py {feature.boundary[lane]} departs: "
                            + "; ".join(f"{d['face']} {d['code']}" for d in departures))
    return problems


def _assert_falls_back(name, feature, lane, c):
    sim = _build(feature, lane, True)
    with pytest.warns(UserWarning, match="Falling back to single-device"):
        fell = feature.read(sim.run(n_steps=feature.steps(lane), devices=_devices(),
                                    skip_preflight=True))
    target = _record((name, c.to, True), feature, c.to, True)
    # The same program on one device: equal to a few float32 rounding steps.
    ulp = 4 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(fell, target, rtol=ulp, atol=ulp * np.max(np.abs(target)))


# -------------------------------------------------------------------- cells

_RELAXED = ("_refinement", "relaxed_validation")


def _xfail(c):
    return [pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{c.wrong}: {c.note}")]


def _executable():
    for attr, features in T.TABLE.items():
        if not T.executable(attr):
            continue
        for feature in features:
            if (attr, feature) == _RELAXED:
                continue   # test_relaxed_subgrid_validation_refuses_or_carries
            for lane in T.LANES:
                c = T.cell(attr, feature, lane)
                if c.kind in (T.NOT_REACHABLE, T.IGNORABLE):
                    continue
                yield pytest.param(attr, feature, lane, id=f"{attr}-{feature}-{lane}",
                                   marks=_xfail(c) if c.wrong else [])


@pytest.mark.parametrize("attr,feature,lane", list(_executable()))
def test_cell(attr, feature, lane):
    c = T.cell(attr, feature, lane)
    name = f"{attr}/{feature}"
    spec = FEATURES[attr, feature]
    if c.wrong:
        # Fixed either way, the cell stops failing: refused, or carried.
        if _refusal(name, spec, lane, _build(spec, lane, True)) is None:
            carries = c.kind == T.CARRIES
            # A cell the table calls carried-with-a-known-departure always
            # measures the effect too: its departure is the expected failure,
            # a lane that drops the input altogether is not (#1338 re-check).
            problems = _carried(name, spec, lane, parity=carries, first_only=not carries)
            dropped = [p for p in problems if p.startswith("declaring it moved the record")]
            if carries and dropped:
                pytest.fail(f"{name} on {lane}: the table says carried with the departure "
                            f"{c.wrong}, but the lane drops the input: " + "; ".join(dropped))
            assert not problems, f"{name} on {lane}: " + "; ".join(problems)
    elif c.kind == T.REFUSES:
        _assert_refused(name, spec, lane, c)
    elif c.kind == T.CARRIES:
        problems = _carried(name, spec, lane, parity=True)
        assert not problems, f"{name} on {lane}: " + "; ".join(problems)
    elif c.kind == T.FALLS_BACK:
        _assert_falls_back(name, spec, lane, c)
    else:
        pytest.fail(f"no check for {c.kind!r}")


RELAXED_INPUTS = {
    "debye": lambda s: _block(s, eps_r=2.0, debye_poles=[DebyePole(delta_eps=2.0, tau=1e-11)]),
    "lorentz": lambda s: _block(s, eps_r=2.0, lorentz_poles=[
        lorentz_pole(2.0, 2 * np.pi * 8e9, 2 * np.pi * 1e9)]),
    "drude": lambda s: _block(s, eps_r=1.0, lorentz_poles=[drude_pole(2 * np.pi * 10e9, 1e10)]),
    "kerr": lambda s: _block(s, eps_r=1.0, chi3=1e3),
    "rlc": lambda s: s.add_lumped_rlc(mm(6, 6, 6), "ez", R=10.0),
}
RELAXED_STATIC = {"debye": 2.0, "lorentz": 2.0, "drude": 1.0, "kerr": 1.0, "rlc": None}


def _relaxed(mode, add):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _base("run_subgridded", refine=False)
        sim.add_refinement(z_range=(0.0, 14e-3), ratio=2, validation=mode)
        add(sim)
        return sim


def _relaxed_params():
    c = T.cell(*_RELAXED, "run_subgridded")
    # 'off' runs the same runner as 'research' without the validation report;
    # it is checked on #1286's own case only.
    cases = [("research", name) for name in RELAXED_INPUTS] + [("off", "debye")]
    return [pytest.param(mode, name, id=f"{mode}-{name}", marks=_xfail(c) if c.wrong else [])
            for mode, name in cases]


@pytest.mark.parametrize("mode,name", _relaxed_params())
def test_relaxed_subgrid_validation_refuses_or_carries(mode, name):
    """The subgridded cell of _refinement/relaxed_validation: what production
    refuses, research/off must refuse too or carry (effect against the same
    model without the input's pole, χ³ or element; validation does not change
    that model's fields, so it is run once, unvalidated)."""
    spec = Feature(lambda lane, on, ref=False: None)
    sim = _relaxed(mode, RELAXED_INPUTS[name])
    c = T.cell(*_RELAXED, "run_subgridded")
    with _watch_kernel_scans() as started:
        try:
            declared = _probe(_run(sim, "run_subgridded", spec))
        except (NotImplementedError, ValueError) as exc:
            assert not started, started
            if c.kind == T.REFUSES and not c.wrong:
                assert c.raises in str(exc), (
                    f"validation={mode!r} with {name} was refused, but not by the refusal the "
                    f"table names ({c.raises!r}): {str(exc)[:300]}")
            return
    eps_r = RELAXED_STATIC[name]
    key = ("relaxed", eps_r)
    if key not in _RESULTS:
        static = _relaxed("off", (lambda s: None) if eps_r is None else (lambda s: _block(s, eps_r=eps_r)))
        _RESULTS[key] = _probe(_run(static, "run_subgridded", spec))
    static = _RESULTS[key]
    effect = relative(declared, static)
    assert effect > EFFECT_FLOOR, (
        f"validation={mode!r}: the {name} input moved the record by {effect:.3e} of its peak")


@pytest.mark.parametrize("mode", ["research", "off"])
@pytest.mark.parametrize("absorber,words", [("cpml", "a CPML absorber"), ("upml", "a UPML absorber")])
def test_relaxed_subgrid_validation_refuses_an_absorbing_box(mode, absorber, words):
    """validation='research' and 'off' ran a subgridded model in an absorbing
    box, unvalidated. What the lane carries there has not been measured, so
    lane admission refuses the absorber in every validation mode, before any
    step, as production validation does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _base("run_subgridded", refine=False, boundary=absorber)
        sim.add_refinement(z_range=(0.0, 14e-3), ratio=2, validation=mode)
    with _watch_kernel_scans() as started, pytest.raises(NotImplementedError) as exc:
        _run(sim, "run_subgridded", Feature(lambda lane, on, ref=False: None))
    assert not started, started
    assert f"{words} is not carried by the subgridded run() lane" in str(exc.value)


@pytest.mark.parametrize("validation", ["production", "research", "off"])
def test_the_guarded_lid_runs_in_every_validation_mode(validation):
    """A CPML lid on a closed PEC box, the refined slab on the PEC floor:
    production validation's guarded envelope. It runs in every validation
    mode, and in each the lid moves the record against a PEC lid (production
    runs the slab's boundary-terminated interface, research and off do not,
    so the modes do not give the same record). A UPML lid, which this lane
    runs as CPML, is refused in every mode."""
    spec = Feature(lambda lane, on, ref=False: None, steps=lambda lane: LID_STEPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lid, pec = (_probe(_run(_lid("run_subgridded", on, validation=validation), "run_subgridded", spec))
                    for on in (True, False))
    effect = relative(lid, pec)
    assert effect > EFFECT_FLOOR, f"validation={validation!r}: the lid moved the record by {effect:.3e}"
    # kappa_max on the lid is read by this lane (measured), and gated with it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stretched = _probe(_run(_lid("run_subgridded", True, validation=validation, kappa=5.0),
                                "run_subgridded", spec))
    assert np.all(np.isfinite(stretched)) and stretched.shape == lid.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        upml = _lid("run_subgridded", True, lid="upml", validation=validation)
    with _watch_kernel_scans() as started, pytest.raises(NotImplementedError) as exc:
        _run(upml, "run_subgridded", spec)
    assert not started, started
    assert "a UPML absorber is not carried by the subgridded run() lane" in str(exc.value)


@pytest.mark.parametrize("validation", ["production", "research", "off"])
def test_a_slab_reaching_the_lid_is_refused_in_every_validation_mode(validation):
    """The guarded box, its refined slab carried up to 0.01 mm under the CPML
    lid: the fine slab would overlap the absorber, which production
    validation refuses (subgrid_overlaps_absorber). research and off do not
    widen that envelope: lane admission refuses the absorber. The slab to
    10 mm, the guarded case, runs."""
    spec = Feature(lambda lane, on, ref=False: None, steps=lambda lane: LID_STEPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reaching = _lid("run_subgridded", True, validation=validation, z_top=15.99)
        guarded = _probe(_run(_lid("run_subgridded", True, validation=validation), "run_subgridded", spec))
    assert np.all(np.isfinite(guarded)) and np.max(np.abs(guarded)) > 0
    with _watch_kernel_scans() as started, pytest.raises((NotImplementedError, ValueError)) as exc:
        _run(reaching, "run_subgridded", spec)
    assert not started, started
    expected = ("[subgrid_overlaps_absorber]" if validation == "production"
                else "a CPML absorber is not carried by the subgridded run() lane")
    assert expected in str(exc.value), str(exc.value)[:400]


def _band_wire(lane, x_mm):
    """A graded x mesh of 1 mm cells with a 0.25 mm band over x = 3-4 mm, and
    a PEC PolylineWire of radius 0.3 mm along z at ``x_mm``: in the refused
    filament band in 1 mm cells (0 < 0.3 < 0.5), a volume in the fine
    band (0.3 >= 0.125)."""
    profile = np.array([1e-3] * 3 + [0.25e-3] * 4 + [1e-3] * 8)
    sim = _simulation(lane, (12, 12, 12), dx_profile=profile)
    sim.add(PolylineWire((mm(x_mm, 6.5, 3), mm(x_mm, 6.5, 9)), radius=0.3e-3), material="pec")
    sim.add_source(mm(8, 6, 6), "ez", waveform=WAVEFORM, amplitude_kind="field")
    sim.add_probe(mm(6, 6, 6), "ez")
    return sim


@pytest.mark.parametrize("x_mm,kind", [(9.5, "pec_wire"), (3.5, "pec_volume")],
                         ids=["filament_in_1mm_cells", "volume_in_the_band"])
def test_a_wire_is_judged_by_the_cells_at_its_own_vertices(x_mm, kind):
    """The assembler decides filament or volume from the cells at the wire's
    own vertices; the detector asks the same rule. The volume in the band is
    carried by both graded multi-device lanes, and gives what the same call
    gives with admission switched off, bit for bit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _band_wire("run_nonuniform", x_mm)
        grid = sim._build_nonuniform_grid()
        sheets, wires = [], []
        if kind == "pec_wire":
            with pytest.raises(ValueError, match="resolve the wire as a volume"):
                sim._assemble_materials_nu(grid, pec_sheets=sheets, pec_wires=wires)
            return
        sim._assemble_materials_nu(grid, pec_sheets=sheets, pec_wires=wires)
    assert len(wires) == (1 if kind == "pec_wire" else 0)
    assert A.DETECTORS["_geometry", kind](sim)
    other = "pec_volume" if kind == "pec_wire" else "pec_wire"
    assert not A.DETECTORS["_geometry", other](sim)
    if kind == "pec_wire":
        return
    for entry in ("run_distributed", "fwd_distributed_nu"):
        feature = Feature(lambda lane, on, ref=False: None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _probe(_run(_band_wire(entry, x_mm), entry, feature))
            with patch.object(A, "admit", lambda *args, **kw: None):
                unadmitted = _probe(_run(_band_wire(entry, x_mm), entry, feature))
        np.testing.assert_array_equal(got, unadmitted)


@pytest.mark.parametrize("attr,feature,lane,named", [
    ("_materials", "mu", "run_adi", "uniform run()"),
    ("_cpml_kappa_max", "kappa", "run_nonuniform", "multi-device run(devices=...)"),
])
def test_the_refusal_names_lanes_that_carry_the_rest_of_the_model(attr, feature, lane, named):
    """The message names, as alternatives, the lanes that carry every input
    of the model apart from the rows that choose the lane (solver, mesh
    profiles, refinement). ADI with a magnetic block names the Yee lanes; a
    graded CPML box with kappa names the multi-device run(), whose own Phase B
    refusal of a graded CPML box is not admission's."""
    spec = FEATURES[attr, feature]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _build(spec, lane, True)
        with pytest.raises(NotImplementedError) as exc:
            _run(sim, lane, spec)
    message = str(exc.value)
    assert T.cell(attr, feature, lane).raises + "." in message
    carriers = message.split("apart from the ones that choose the lane (solver, mesh profiles, refinement): ")[1]
    assert named in carriers.split(".\n")[0], message


_CONFORMAL_S = ("_boundary_spec", "conformal_s_matrix")


@pytest.mark.parametrize("call,refused", [
    ({}, True),
    ({"compute_s_params": True}, True),
    ({"compute_s_params": False}, False),
    ({"conformal_pec": False}, False),
], ids=["default", "s_params", "no_s_params", "staircase_override"])
def test_conformal_s_matrix_is_refused_only_when_run_computes_it(call, refused):
    """Conformal walls and lumped ports on the uniform run(). Its lumped-port
    S-matrix comes from a path with no conformal update (#1299), so a call
    that computes it against conformal fields (the default with ports, or
    compute_s_params=True) is refused. compute_s_params=False computes none,
    and conformal_pec=False asks for staircase fields as well as a staircase
    S-matrix: both run, and give what the same call gives with admission
    switched off, bit for bit."""
    kwargs = dict(n_steps=N_STEPS, skip_preflight=True, **call)
    sim = _build(FEATURES[_CONFORMAL_S], "run_uniform", True)
    if refused:
        with pytest.raises(NotImplementedError) as exc, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.run(**kwargs)
        assert T.cell(*_CONFORMAL_S, "run_uniform").raises in str(exc.value)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sim.run(**kwargs)
        with patch.object(A, "admit", lambda *args, **kw: None):
            unadmitted = _build(FEATURES[_CONFORMAL_S], "run_uniform", True).run(**kwargs)
    np.testing.assert_array_equal(_probe(got), _probe(unadmitted))
    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        np.testing.assert_array_equal(np.asarray(getattr(got.state, name)),
                                      np.asarray(getattr(unadmitted.state, name)))
    assert (got.s_params is None) == (unadmitted.s_params is None)
    if got.s_params is not None:
        np.testing.assert_array_equal(np.asarray(got.s_params), np.asarray(unadmitted.s_params))


# ------------------------------------------------------ the table's own terms

def test_every_executable_row_has_a_builder():
    rows = {(attr, feature) for attr, features in T.TABLE.items() if T.executable(attr)
            for feature in features} - {_RELAXED}
    assert rows == set(FEATURES), sorted(rows ^ set(FEATURES))


def test_thresholds_clear_float32_noise():
    """Two lanes doing the same arithmetic in another order differ by float32
    rounding only. On the dielectric-block model that is below 3e-6 of the
    peak; the parity tolerance and the effect floor must stand well clear."""
    spec = FEATURES["_materials", "eps"]
    noise = max(relative(_with("_materials/eps", spec, lane),
                         _reference("_materials/eps", spec, lane))
                for lane in ("run_nonuniform", "fwd_uniform", "fwd_nonuniform", "fwd_adi"))
    assert noise < 3e-6, noise
    assert PARITY_TOL >= 30 * noise and EFFECT_FLOOR >= 10 * PARITY_TOL


# ------------------------------------------------ what admission sees (J2b)
# Lane admission (rfx/runners/_admission.py) refuses what a lane does not
# carry only if a detector sees the input. A detector that never fires would
# let every drop through again, and one that always fires would refuse every
# model, so both directions are checked on the models the cells build.

def _declared_models():
    """Every cell above whose model can be built: a ``declared`` refusal
    raises in the constructor, so it has no model to look at."""
    for param in _executable():
        attr, feature, lane = param.values
        if not T.cell(attr, feature, lane).declared:
            yield pytest.param(attr, feature, lane, id=param.id)


@pytest.mark.parametrize("attr,feature,lane", list(_declared_models()))
def test_the_rows_detector_fires_on_the_cells_model(attr, feature, lane):
    sim = _build(FEATURES[attr, feature], lane, True)
    assert A.DETECTORS[attr, feature](sim), (
        f"{attr}/{feature} is declared on the {lane} cell's model, but its detector does not see it")


def _calculator_declarations():
    for attr, feature in FEATURES:
        for calculator in T.CALCULATORS:
            disposition = T.cell(attr, feature, calculator)
            if disposition.kind == T.NOT_REACHABLE:
                continue
            yield pytest.param(attr, feature, calculator,
                               id=f"{attr}-{feature}-{calculator}")


@pytest.mark.parametrize("attr,feature,calculator", list(_calculator_declarations()))
def test_calculator_declaration_admission_cell(attr, feature, calculator, monkeypatch):
    """Exercise the cell on a real declaration, without executing a kernel.

    Public entry wiring and the migrated guard rows are exercised separately
    in test_calculator_admission.py. Numerical calculator outputs remain in
    the calculator-specific tests; this check isolates the named input even
    when the common declaration contains another unsupported source or port.
    """
    sim = _build(FEATURES[attr, feature], "fwd_uniform", True)
    row = (attr, feature)
    assert A.DETECTORS[row](sim), row

    def forbidden(*args, **kwargs):
        pytest.fail("admission started lax.scan")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    disposition = T.cell(attr, feature, calculator)
    refused = A.refused(sim, calculator)
    if disposition.kind == T.REFUSES:
        assert row in refused, (calculator, row, refused)
        with pytest.raises(NotImplementedError) as caught:
            A.admit(sim, calculator)
        assert A.ROW_WORDS[row] in str(caught.value)
    else:
        assert disposition.kind in (T.CARRIES, T.IGNORABLE)
        assert row not in refused, (calculator, row, refused)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#1293: vmap's auto-mesh fallback holds the step count while dt changes")
def test_vmap_auto_mesh_cell_covers_the_requested_time_for_each_value():
    """The _dx calculator cell remains a carry with a measured duration defect."""
    from rfx.vmap_sweep import vmap_material_sweep

    def model(eps):
        sim = Simulation(freq_max=10e9, domain=mm(12, 12, 12), boundary="pec")
        sim.add_material("diel", eps_r=eps)
        sim.add(Box(mm(0, 0, 0), mm(12, 12, 4)), material="diel")
        sim.add_port(mm(4, 6, 6), "ez", impedance=50., waveform=WAVEFORM)
        sim.add_probe(mm(8, 6, 6), "ez")
        return sim

    values = (2., 3.)
    swept = vmap_material_sweep(model(values[0]), "diel.eps_r", values, num_periods=2.)
    grids = [model(value)._build_grid() for value in values]
    covered = np.array([swept.time_series.shape[1] * float(grid.dt) for grid in grids])
    requested = 2. / 10e9
    assert np.all(covered >= requested), (covered, requested)


@pytest.mark.parametrize("mode,name", [(m, n) for m, n in (p.values for p in _relaxed_params())])
def test_the_relaxed_validation_detector_fires_on_its_models(mode, name):
    assert A.DETECTORS[_RELAXED](_relaxed(mode, RELAXED_INPUTS[name]))


# The rows the base model declares, written out per lane: every model has
# freq_max and a domain, the base passes dx and adds an Ez source and an Ez
# probe, and each lane's base carries the lane's own selector (a dx_profile on
# the graded lanes, solver='adi' on the ADI lanes, a refinement on the
# subgridded lane). Leader's ruling on J2b, 2026-09-27: exactly these fire.
_EVERY_BASE = {("_freq_max", ""), ("_domain", ""), ("_dx", ""), ("_ports", "source"),
               ("_probes", "probe")}
BASE_ROWS = {
    "run_uniform": _EVERY_BASE,
    "run_nonuniform": _EVERY_BASE | {("_dx_profile", "graded")},
    "run_subgridded": _EVERY_BASE | {("_refinement", "slab")},
    "run_adi": _EVERY_BASE | {("_solver", "")},
    "run_distributed": _EVERY_BASE,
    "fwd_uniform": _EVERY_BASE,
    "fwd_nonuniform": _EVERY_BASE | {("_dx_profile", "graded")},
    "fwd_distributed_nu": _EVERY_BASE | {("_dx_profile", "graded")},
    "fwd_adi": _EVERY_BASE | {("_solver", "")},
}


@pytest.mark.parametrize("lane", T.LANES)
def test_on_the_base_model_only_its_declared_rows_fire(lane):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fired = set(A.active(_base(lane)))
    assert fired == BASE_ROWS[lane], (
        f"on the {lane} base model, extra: {sorted(fired - BASE_ROWS[lane])}, "
        f"missing: {sorted(BASE_ROWS[lane] - fired)}")


def _bare(lane="run_uniform", **ctor):
    """The base box with nothing added."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _simulation(lane, (12, 12, 12), **ctor)


_WITHOUT = {
    ("_ports", "source"): lambda: _base("run_uniform", source=None),
    ("_probes", "probe"): lambda: _bare(),
    ("_dx", ""): lambda: _bare(dx=None),
    # the lane selectors, on the uniform base
    ("_dx_profile", "graded"): lambda: _base("run_uniform"),
    ("_solver", ""): lambda: _base("run_uniform"),
    ("_refinement", "slab"): lambda: _base("run_uniform"),
}


@pytest.mark.parametrize("row", list(_WITHOUT), ids=lambda row: "/".join(row).rstrip("/"))
def test_a_base_row_does_not_fire_without_its_input(row):
    model = _WITHOUT[row]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model()
    assert not A.DETECTORS[row](sim), f"{row} fires on a model that does not declare it"
