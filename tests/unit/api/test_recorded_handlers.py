"""Recorded diagnostics preserve failures and propagate unexpected exceptions."""

import builtins
from dataclasses import replace

import numpy as np


import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation


def _ntff_warnings(freqs):
    sim = Simulation(freq_max=10e9, domain=(0.04, 0.04, 0.04),
                     dx=0.002, boundary="pec")
    sim._ntff = ((0.01, 0.01, 0.01), (0.03, 0.03, 0.03), freqs)
    sim._ports = [SimpleNamespace(position=(0.015, 0.02, 0.02))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._validate_ntff_inverse_design()
    return [str(w.message) for w in caught if "NTFF face x_lo" in str(w.message)]


def test_ntff_none_frequencies_use_declared_maximum():
    messages = _ntff_warnings(None)
    assert len(messages) == 1
    assert "f_max=10.00GHz" in messages[0]


def test_ntff_traced_frequencies_use_declared_maximum():
    messages = []

    def check(freqs):
        messages.extend(_ntff_warnings(freqs))
        return freqs

    jax.make_jaxpr(check)(jnp.array([5e9]))
    assert len(messages) == 1
    assert "f_max=10.00GHz" in messages[0]


def test_ntff_empty_frequencies_use_declared_maximum():
    messages = _ntff_warnings(jnp.array([]))
    assert len(messages) == 1
    assert "f_max=10.00GHz" in messages[0]


def test_ntff_concrete_frequencies_use_their_maximum():
    messages = _ntff_warnings(jnp.array([3e9, 5e9]))
    assert len(messages) == 1
    assert "f_max=5.00GHz" in messages[0]


def test_ntff_malformed_concrete_frequencies_raise():
    # Conversion and size inspection succeed; reading the maximum as a real
    # frequency fails. A fallback around float(jnp.max(...)) must not hide it.
    freqs = jnp.asarray([5e9 + 1j])
    assert freqs.size == 1
    with pytest.raises(TypeError):
        _ntff_warnings(freqs)


MESSAGE = "injected recorded-handler failure"


def _raising(error):
    def fail(*args, **kwargs):
        raise error(MESSAGE)
    return fail


class _BrokenArray:
    shape = (2, 2, 2)
    dtype = np.dtype("float32")
    size = 8

    def __init__(self, error):
        self.error = error

    def __array__(self, *args, **kwargs):
        raise self.error(MESSAGE)


def _small_sim():
    return Simulation(freq_max=10e9, domain=(0.008,) * 3, dx=0.001, boundary="pec")


# Each case enters the production handler through its actual caller. The return
# value is the field exposed to a reader, not an assertion about source syntax.
G5_CASES = (
    "host_poles", "linear_accumulators", "invariance",
    "import_rfx", "import_jax", "jax_devices", "jax_config", "core_deps",
    "optional_deps", "fidelity", "coax", "cutoff", "mask_error", "source", "tfsf",
    "point_probe", "plane_probe", "flux", "ntff",
    "cells_per_wavelength", "error_witnesses", "judge_record",
)


def _g5_call(case, monkeypatch, error):
    fail = _raising(error)
    if case == "host_poles":
        from rfx.ringdown._jax import host_poles
        # Execute the callback synchronously so JAX does not wrap the exception
        # in XlaRuntimeError at the device boundary.
        monkeypatch.setattr(jax, "pure_callback", lambda fn, shapes, *a, **kw: fn(*a))
        return lambda: host_poles(fail, jnp.ones(4), 1, 0.1, [1.0])[2]
    if case == "linear_accumulators":
        from rfx.api._ad_carry import linear_accumulators
        sim = SimpleNamespace(_precision="float32", _ntff=(None, None, None),
                              _dft_planes=(), _flux_monitors=(), _uses_nonuniform_mesh=False)
        monkeypatch.setattr(jax, "eval_shape", fail)
        return lambda: linear_accumulators(sim, None, 1)[1]
    if case == "invariance":
        from rfx.boundaries.tfsf import invariant
        return lambda: invariant(_BrokenArray(error), 0, (2, 2, 2))[1]
    if case in {"import_rfx", "import_jax", "jax_devices", "jax_config",
                "core_deps", "optional_deps"}:
        from rfx import diagnostics as diag
        report = diag._Report()
        if case in {"import_rfx", "import_jax"}:
            original = builtins.__import__
            target = "rfx" if case == "import_rfx" else "jax"
            def importing(name, *args, **kwargs):
                if name == target:
                    fail()
                return original(name, *args, **kwargs)
            monkeypatch.setattr(builtins, "__import__", importing)
        elif case == "jax_devices":
            monkeypatch.setattr(jax, "devices", fail)
        elif case == "jax_config":
            monkeypatch.setattr(jax.config, "read", fail)
        elif case in {"core_deps", "optional_deps"}:
            monkeypatch.setattr(diag.importlib, "import_module", fail)
        check = {"import_rfx": diag._check_rfx_import, "import_jax": diag._check_jax,
                 "jax_devices": diag._check_jax, "jax_config": diag._check_jax,
                 "core_deps": diag._check_core_deps, "optional_deps": diag._check_optional_extras}[case]
        def diagnostic():
            check(report)
            return report.lines
        return diagnostic
    if case in {"source", "tfsf", "point_probe", "plane_probe", "flux", "ntff", "cells_per_wavelength"}:
        from rfx import realized_placement as placement
        grid = SimpleNamespace(shape=(3, 3, 3), position_to_index=fail)
        ctx = SimpleNamespace(grid=grid, lane="uniform")
        entry = SimpleNamespace(impedance=0.0, position=(0.,) * 3, component="ez",
                                axis="z", coordinate=0., name="probe", polarization="ez")
        sim = SimpleNamespace(_ports=[entry], _tfsf=None, _probes=[], _dft_planes=[],
                              _flux_monitors=[], _ntff=None, _freq_max=10e9)
        monkeypatch.setattr(placement, "field_index", fail)
        if case == "source":
            return lambda: placement._sources(sim, ctx)[0].error
        if case == "tfsf":
            sim._tfsf, sim.tfsf_box_indices = entry, fail
            return lambda: placement._tfsf_row(sim, ctx).error
        if case in {"point_probe", "plane_probe"}:
            if case == "point_probe":
                sim._probes = [entry]
            else:
                sim._dft_planes = [entry]
            return lambda: placement._probes(sim, ctx)[0].error
        if case == "flux":
            return lambda: placement._flux_row(sim, ctx, entry, fail).error
        if case == "ntff":
            from rfx import farfield
            sim._ntff = (None, None, None)
            monkeypatch.setattr(farfield, "make_ntff_box", fail)
            return lambda: placement._ntff_row(sim, ctx, fail).error
        from rfx.model import materials
        monkeypatch.setattr(materials, "volume_eps_mu", fail)
        domain = [SimpleNamespace(padding=(0, 0), n_cells=1)] * 3
        mats = SimpleNamespace(eps_r=np.ones((1, 1, 1)), mu_r=1.)
        return lambda: placement.cells_per_wavelength(sim, [np.ones(1)] * 3,
                                                       domain, mats, None).error
    if case == "coax":
        from rfx.model.conductors import RealizedConductors, coax_kernel_conductors
        root = RealizedConductors("uniform", None, None, None, None, (), (), (), (), {})
        sim = SimpleNamespace(_campaign_ctx=fail)
        return lambda: coax_kernel_conductors(sim, None, None, None, (), (), root=root).stamp_check_findings
    if case == "cutoff":
        from rfx.preflight import waveguide
        monkeypatch.setattr(waveguide, "_preflight_aperture_cells", lambda *a: {"x": 2, "y": 2})
        monkeypatch.setattr(waveguide, "resolve_waveguide_port_freqs", fail)
        entry = SimpleNamespace(direction="+z", mode_profile="discrete", n_modes=2)
        grid = SimpleNamespace(cells=lambda axis: [0.001])
        spans = {a: {"rasterized": 0.002} for a in "xy"}
        return lambda: waveguide._waveguide_port_cutoff_reading(None, entry, grid, spans)["build_error"]
    if case == "mask_error":
        from rfx import Box, realized_geometry as geometry
        sim = _small_sim()
        sim.add(Box((0.001,) * 3, (0.0011, 0.0011, 0.002)), material="pec")
        ctx = sim._campaign_ctx()
        assembled = geometry._assembly(sim, ctx)
        ctx.entry_realizations()
        ctx.interior_pec_entries()
        monkeypatch.setattr(geometry, "_assembly", lambda *a: assembled)
        monkeypatch.setattr(type(ctx), "rasterize", fail)
        return lambda: geometry._build_record(sim, ctx).entities[0].mask_error
    if case == "fidelity":
        from rfx import Box
        from rfx.fidelity import fidelity_report
        sim = _small_sim()
        sim.add(Box((0.002,) * 3, (0.004,) * 3), material="pec")
        record = sim.realized_geometry()
        # The record preserves a failed rasterization; the report reconstructs
        # that error for the unbounded conductor's finding.
        row = replace(record.entities[0], mask_error=f"{error.__name__}: {MESSAGE}")
        table = replace(record.realization_table, rows=tuple(
            replace(table_row, declared_bounds_m=None) if table_row.label == "geometry[0]" else table_row
            for table_row in record.realization_table.rows))
        monkeypatch.setattr(sim, "realized_geometry", lambda: replace(
            record, entities=(row,), realization_table=table))
        sim._geometry[0] = SimpleNamespace(material_name="pec", shape=SimpleNamespace())
        return lambda: fidelity_report(sim, print_report=False)[1]["findings"]
    if case in {"error_witnesses", "judge_record"}:
        from rfx import ringdown
        monkeypatch.setattr(ringdown, "early_start_witness", fail)
        spec = ringdown.RingdownSpec()
        if case == "error_witnesses":
            owner = SimpleNamespace(spec=spec, source_ratio_long=0., n_steps=20)
            w2 = SimpleNamespace(value=0., spectra=None)
            return lambda: ringdown.RingdownRun._error_witnesses(
                owner, None, .1, [1.], 1., lambda x: x, None, w2, {})[0].note
        from rfx.measurement import ringdown_tail
        monkeypatch.setattr(ringdown_tail, "plain_dft", lambda *a: np.zeros(1))
        monkeypatch.setattr(ringdown_tail, "tail_dft", lambda *a: np.zeros(1))
        monkeypatch.setattr(ringdown_tail, "completion_tail", lambda *a, **kw: np.zeros(1))
        monkeypatch.setattr(ringdown, "identify", lambda *a, **kw: None)
        monkeypatch.setattr(ringdown, "_tail_influence", lambda *a: (None, None))
        monkeypatch.setattr(ringdown, "_floor_pole", lambda *a: (0., None, (), 0))
        monkeypatch.setattr(ringdown, "_passivity_value", lambda *a: 0.)
        monkeypatch.setattr(ringdown, "judge_growing_poles", lambda *a, **kw:
                            SimpleNamespace(n_kept_growing=0, ok=True, status="ok", reason=""))
        return lambda: ringdown._judge_record(None, .1, [1.], 20, 10, spec,
                                               ref_hz=1., observable=lambda x: x)["note"]
    raise AssertionError(case)


@pytest.mark.parametrize("case", G5_CASES)
def test_g5_failure_is_recorded(case, monkeypatch):
    result = _g5_call(case, monkeypatch, RuntimeError)()
    if case == "host_poles":
        from rfx.ringdown._jax import STATUS_FAILED
        assert result == STATUS_FAILED
    elif case == "optional_deps":
        assert len(result) == 4
        assert all("not installed (" in line for line in result)
    elif case == "invariance":
        assert "RuntimeError: invariance unavailable" in result
    else:
        assert MESSAGE in str(result)


@pytest.mark.parametrize("case", ["laplace", "compile_switch", "readable"])
def test_narrowed_handlers_propagate_runtime_error(case, monkeypatch):
    fail = _raising(RuntimeError)
    if case == "readable":
        from rfx.measurement.msl_planes import readable_values
        call = lambda: readable_values(_BrokenArray(RuntimeError))
    else:
        original = builtins.__import__
        target = "scipy.sparse" if case == "laplace" else "jax._src"
        if case == "laplace":
            from rfx.sources._laplace import _solve_laplace_2d
            call = lambda: _solve_laplace_2d(None, None, None, 1., 1.)
        else:
            from rfx.core.jax_utils import compile_time_switch
            call = compile_time_switch.__wrapped__
        def importing(name, *args, **kwargs):
            if name == target:
                fail()
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError, match=MESSAGE):
        call()


def test_staging_probe_missing_api_is_conservative(monkeypatch):
    from rfx.stepping.rank import _staged_by_enclosing_program
    monkeypatch.delattr(jax.lax, "add")
    assert _staged_by_enclosing_program() is True


def test_staging_probe_recognizes_an_enclosing_trace():
    from rfx.stepping.rank import _staged_by_enclosing_program
    assert _staged_by_enclosing_program() is False

    def check():
        assert _staged_by_enclosing_program() is True
        return jnp.zeros(())

    jax.make_jaxpr(check)()


def test_traced_invariance_and_readable_values_do_not_coerce():
    from rfx.boundaries.tfsf import invariant
    from rfx.measurement.msl_planes import readable_values
    def check(value):
        assert invariant(value, 0, (2, 2, 2))[0] is None
        assert readable_values(value) is None
        return value
    jax.make_jaxpr(check)(jnp.ones((2, 2, 2)))


def _recorded_probe(case, monkeypatch, error):
    fail = _raising(error)
    if case in {"memory_outer", "memory_inner"}:
        from tests.unit.autodiff.test_estimate_ad_memory import _uniform_sim
        sim = _uniform_sim()
        monkeypatch.setattr(jax, "local_devices", fail if case == "memory_outer" else lambda: [
            SimpleNamespace(platform="mps", memory_stats=fail)])
        return lambda: sim.estimate_ad_memory(4).warning
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["memory_outer", "memory_inner"])
def test_recorded_probe_preserves_fallback(case, monkeypatch):
    result = _recorded_probe(case, monkeypatch, RuntimeError)()
    if case in {"to_dict", "asdict"}:
        assert result["type"] == "SimpleNamespace"
        assert "fail" in result["repr"]
        result = result["error"]
    assert f"RuntimeError: {MESSAGE}" in result
