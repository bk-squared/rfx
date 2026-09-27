"""The physics and observer cells of the path-disposition table, run.

Each cell of ``tests/contracts/path_disposition.py`` on a physics or observer
row is checked on a tiny model: a 12 mm PEC box with 1 mm cells, an Ez soft
source at x = 4 mm and an Ez probe at x = 8 mm, 40 steps. The graded lanes
get a one-size ``dx_profile`` (the same mesh, so the lane is the only
difference, as in #1282); the subgridded lane gets a 20 mm tall box whose
refinement covers z = 0-14 mm, which production validation accepts.

* ``refuses``: the path raises ``NotImplementedError`` or ``ValueError``
  before any kernel scan starts.
* ``carries``: the declared input changes the probe record by more than
  ``EFFECT_FLOOR`` of its peak, compared with the same model without it.
  On the lanes that share ``run_uniform``'s mesh and time step
  (``PARITY_LANES``) the record must also agree with ``sim.run()`` of the same
  model to ``PARITY_TOL``. An observer carries when its result field is
  filled.
* ``falls back``: the path warns and returns the named lane's result.
* a cell with ``wrong`` is a strict expected failure against its issue. Its
  check passes only when the path refuses the input or carries it (effect,
  and parity for a ``carries`` cell), so fixing the issue either way turns it
  into an unexpected pass that forces the table to be updated.

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

from rfx import (Box, DebyePole, GaussianPulse, PolylineWire, Simulation,
                 Sphere, drude_pole, lorentz_pole)
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests.contracts import path_disposition as T
from tests.unit.nonuniform.test_refinement_refused_on_graded_mesh import _box as _refined_box_1282

N_STEPS = 40
EFFECT_FLOOR = 1e-3
PARITY_TOL = 1e-4
PARITY_LANES = ("run_nonuniform", "run_distributed", "fwd_uniform",
                "fwd_nonuniform", "fwd_distributed_nu")
GRADED = ("run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu")
WAVEFORM = GaussianPulse(f0=5e9, bandwidth=0.8)
# Modules whose lax.scan is a time loop; a refusal must come before any of them.
_KERNEL_MODULES = ("rfx/simulation.py", "rfx/nonuniform.py", "rfx/runners/",
                   "rfx/adi.py", "rfx/subgridding/", "rfx/vmap_sweep.py")


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
    if not ref and lane == "run_adi":
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


def _block(sim, **material):
    sim.add_material("block", **material)
    sim.add(Box(mm(5, 3, 3), mm(7, 9, 9)), material="block")
    return sim


def _probe(result):
    return np.asarray(result.time_series, dtype=np.float64)


class Feature(NamedTuple):
    """How to declare one input on one lane.

    ``build(lane, on, ref)`` returns the model with (``on``) or without the
    input. ``off`` names a model without the input that other features share,
    so it runs once per lane. ``variant(lane)`` names a model that differs
    between lanes beyond the lane's own mesh and solver; the parity reference
    is built per variant.
    """
    build: Callable
    read: Callable = _probe
    steps: Callable = lambda lane: N_STEPS
    run_kwargs: Callable = lambda lane: {}
    variant: Callable = lambda lane: ""
    parity: bool = True
    observer: str = ""   # result field an observer fills
    off: str = ""


def _added(add, **base):
    """The base model, with ``add(sim, lane)`` applied when on."""
    def build(lane, on, ref=False):
        sim = _base(lane, ref=ref, **base)
        if on:
            add(sim, lane)
        return sim
    return build


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


def _board(lane, on, ref=False):
    """A microstrip: 1 mm εr 3.66 substrate, 2 mm PEC trace, the port the only drive."""
    subgrid = lane == "run_subgridded" and not ref
    length = 24
    sim = _simulation(lane, (length, 12, 10 if subgrid else 6), ref=ref)
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0, 0, 0), mm(length, 12, 1)), material="substrate")
    sim.add(Box(mm(1, 5, 1), mm(length - 1, 7, 2)), material="pec")
    if on:
        sim.add_msl_port(position=mm(2, 6, 0), width=2e-3, height=1e-3, direction="+x",
                         impedance=50.0, waveform=GaussianPulse(f0=7e9, bandwidth=0.8))
    sim.add_probe(mm(12, 6, 2.5), "ez")
    if subgrid:
        sim.add_refinement(z_range=(0.0, 7e-3), ratio=2)
    return sim


def _guide(lane, on, ref=False):
    """A 12 × 6 mm PEC guide absorbing on x; the TE10 port the only drive."""
    spec = BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pec", hi="pec"))
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


def _floquet_cell(lane, on, ref=False):
    spec = BoundarySpec(x="periodic", y="periodic", z="cpml")
    sim = _simulation(lane, (6, 6, 20), ref=ref, boundary=spec)
    if on:
        sim.add_floquet_port(4e-3, axis="z", f0=5e9)
    sim.add_probe(mm(3, 3, 14), "ez")
    if lane == "run_subgridded" and not ref:
        sim.add_refinement(z_range=(0.0, 3e-3), ratio=2)
    return sim


def _sphere(conformal, *, ports=False):
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
                             solver="adi" if lane == "run_adi" else "yee")


def _boundary(on_ctor, *, cpml_off=False):
    """A boundary declaration against the PEC box, or against the CPML box."""
    off_ctor = {"boundary": "cpml"} if cpml_off else {}

    def build(lane, on, ref=False):
        return _base(lane, ref=ref, **(on_ctor if on else off_ctor))
    return Feature(build, off="base cpml" if cpml_off else "base")


def _profile(axis):
    def build(lane, on, ref=False):
        sim_domain = (12, 12, 20 if lane == "run_subgridded" else 12)
        ctor = {f"d{axis}_profile": _graded(sim_domain["xyz".index(axis)])} if on else {}
        return _base(lane, ref=ref, **ctor)
    return build


def _plus(add):
    """A feature added to the shared base model."""
    return Feature(_added(add), off="base")


def _observer(add, field):
    return Feature(_added(add), observer=field)


FEATURES: dict[tuple[str, str], Feature] = {
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
        PolylineWire((mm(6, 6, 3), mm(6, 6, 9)), radius=0.2e-3), material="pec")),
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
    ("_floquet_ports", "floquet_port"): Feature(_floquet_cell),
    ("_lumped_rlc", "R"): _plus(lambda s, _: s.add_lumped_rlc(mm(6, 6, 6), "ez", R=10.0)),
    ("_lumped_rlc", "series_RL"): _plus(lambda s, _: s.add_lumped_rlc(mm(6, 6, 6), "ez", R=10.0, L=1e-9)),
    ("_tfsf", "plane_wave"): Feature(_plane_wave),
    ("_refinement", "slab"): Feature(_refinement, parity=False),
    ("_boundary", "cpml"): _boundary({"boundary": "cpml"}),
    ("_boundary", "upml"): _boundary({"boundary": "upml"}),
    ("_pec_faces", "pec_face"): _boundary({"boundary": "cpml", "pec_faces": {"z_lo"}}, cpml_off=True),
    ("_boundary_spec", "pmc_face"): _boundary({"boundary": BoundarySpec(
        x=Boundary(lo="pec", hi="pec"), y=Boundary(lo="pec", hi="pec"), z=Boundary(lo="pmc", hi="pmc"))}),
    ("_boundary_spec", "conformal"): Feature(_sphere(True)),
    ("_boundary_spec", "conformal_s_matrix"): Feature(
        _sphere(True, ports=True), read=_s_params,
        run_kwargs=lambda lane: (dict(compute_s_params=True, s_param_freqs=np.array([4e9, 5e9, 6e9]))
                                 if lane == "run_uniform" else {})),
    ("_periodic_axes", "periodic"): _boundary({"boundary": BoundarySpec(
        x=Boundary(lo="pec", hi="pec"), y="periodic", z=Boundary(lo="pec", hi="pec"))}),
    ("_cpml_layers", "layers"): _boundary({"boundary": "cpml", "cpml_layers": 8}, cpml_off=True),
    ("_cpml_kappa_max", "kappa"): _boundary({"boundary": "cpml", "cpml_kappa_max": 5.0}, cpml_off=True),
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
            return sim.run(**kwargs, **feature.run_kwargs(entry))
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
    """``sim.run()`` of the same model on one device. Unless the lane's model
    is a variant, that is the run_uniform cell's own record."""
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


def _raises_before_stepping(name, feature, lane):
    """The exception the path raises before any kernel scan, or None if it ran."""
    with _watch_kernel_scans() as started:
        try:
            _run(_build(feature, lane, True), lane, feature)
        except (NotImplementedError, ValueError) as exc:
            assert not started, f"{name} on {lane} raised after a kernel scan started: {started}"
            return exc
    return None


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
    if parity and feature.parity and lane in PARITY_LANES:
        agreement = relative(_with(name, feature, lane), _reference(name, feature, lane))
        if agreement > PARITY_TOL:
            problems.append(f"it differs from sim.run() by {agreement:.3e} of the peak "
                            f"(tolerance {PARITY_TOL:g})")
            if first_only:
                return problems
    effect = relative(_with(name, feature, lane), _without(name, feature, lane))
    if effect <= EFFECT_FLOOR:
        problems.append(f"declaring it moved the record by {effect:.3e} of its peak "
                        f"(floor {EFFECT_FLOOR:g})")
    return problems


# -------------------------------------------------------------------- cells

_RELAXED = ("_refinement", "relaxed_validation")

def _executable():
    for attr, features in T.TABLE.items():
        if T.ROW_CLASS[attr] not in (T.PHYSICS, T.OBSERVER):
            continue
        for feature in features:
            if (attr, feature) == _RELAXED:
                continue   # test_relaxed_subgrid_validation_refuses_or_carries
            for lane in T.LANES:
                c = T.cell(attr, feature, lane)
                if c.kind in (T.NOT_REACHABLE, T.IGNORABLE):
                    continue
                marks = ([pytest.mark.xfail(strict=True, reason=f"{c.wrong}: {c.note}")]
                         if c.wrong else [])
                yield pytest.param(attr, feature, lane, id=f"{attr}-{feature}-{lane}", marks=marks)


@pytest.mark.parametrize("attr,feature,lane", list(_executable()))
def test_cell(attr, feature, lane):
    c = T.cell(attr, feature, lane)
    name = f"{attr}/{feature}"
    spec = FEATURES[attr, feature]
    if c.wrong:
        # Fixed either way, the cell stops failing: refused, or carried.
        if _raises_before_stepping(name, spec, lane) is None:
            problems = _carried(name, spec, lane, parity=c.kind == T.CARRIES, first_only=True)
            assert not problems, f"{name} on {lane}: " + "; ".join(problems)
    elif c.kind == T.REFUSES:
        assert _raises_before_stepping(name, spec, lane) is not None, (
            f"{name} ran on {lane}; the table says it is refused ({c.note})")
    elif c.kind == T.CARRIES:
        problems = _carried(name, spec, lane, parity=True)
        assert not problems, f"{name} on {lane}: " + "; ".join(problems)
    elif c.kind == T.FALLS_BACK:
        sim = _build(spec, lane, True)
        with pytest.warns(UserWarning, match="Falling back to single-device"):
            fell = spec.read(sim.run(n_steps=spec.steps(lane), devices=_devices(),
                                     skip_preflight=True))
        target = _record((name, c.to, True), spec, c.to, True)
        np.testing.assert_array_equal(fell, target)
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
    marks = [pytest.mark.xfail(strict=True, reason=f"{c.wrong}: {c.note}")] if c.wrong else []
    # 'off' runs the same runner as 'research' without the validation report;
    # it is checked on #1286's own case only.
    cases = [("research", name) for name in RELAXED_INPUTS] + [("off", "debye")]
    return [pytest.param(mode, name, id=f"{mode}-{name}", marks=marks) for mode, name in cases]


@pytest.mark.parametrize("mode,name", _relaxed_params())
def test_relaxed_subgrid_validation_refuses_or_carries(mode, name):
    """The subgridded cell of _refinement/relaxed_validation: what production
    refuses, research/off must refuse too or carry (effect against the same
    model without the input's pole, χ³ or element; validation does not change
    that model's fields, so it is run once, unvalidated)."""
    spec = Feature(lambda lane, on, ref=False: None)
    sim = _relaxed(mode, RELAXED_INPUTS[name])
    with _watch_kernel_scans() as started:
        try:
            declared = _probe(_run(sim, "run_subgridded", spec))
        except (NotImplementedError, ValueError):
            assert not started, started
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


# ------------------------------------------------------ the table's own terms

def test_every_executable_row_has_a_builder():
    rows = {(attr, feature) for attr, features in T.TABLE.items()
            if T.ROW_CLASS[attr] in (T.PHYSICS, T.OBSERVER)
            for feature in features} - {_RELAXED}
    assert rows == set(FEATURES), sorted(rows ^ set(FEATURES))


def test_thresholds_clear_float32_noise():
    """Two lanes doing the same arithmetic in another order differ by float32
    rounding only. On the dielectric-block model that is below 3e-6 of the
    peak; the parity tolerance and the effect floor must stand well clear."""
    spec = FEATURES["_materials", "eps"]
    noise = max(relative(_with("_materials/eps", spec, lane),
                         _reference("_materials/eps", spec, lane))
                for lane in ("run_nonuniform", "fwd_uniform", "fwd_nonuniform"))
    assert noise < 3e-6, noise
    assert PARITY_TOL >= 30 * noise and EFFECT_FLOOR >= 10 * PARITY_TOL
