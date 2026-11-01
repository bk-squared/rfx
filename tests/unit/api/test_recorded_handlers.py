"""Recorded diagnostics preserve failures and propagate unexpected exceptions."""

import builtins
from dataclasses import replace
from pathlib import Path
import runpy
import sys
from unittest.mock import MagicMock

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
    "host_poles", "linear_accumulators", "peak_error", "bounding_box",
    "mesh_error", "audit_error", "issues", "invariance", "dashboard",
    "import_rfx", "import_jax", "jax_devices", "jax_config", "core_deps",
    "optional_deps", "smoke", "fidelity", "coax", "cutoff", "mask_error",
    "source", "tfsf", "point_probe", "plane_probe", "flux", "ntff",
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
    if case in {"peak_error", "bounding_box", "mesh_error", "audit_error", "issues"}:
        from rfx import artifacts
        if case == "peak_error":
            return lambda: artifacts._array_summary(_BrokenArray(error), include_peak=True)["peak_error"]
        if case == "bounding_box":
            return lambda: artifacts._bbox(SimpleNamespace(bounding_box=fail))
        if case == "issues":
            return lambda: artifacts._preflight_summary(SimpleNamespace(preflight=fail),
                                                        n_steps=None, available_memory_gb=None)["issues"]
        grid = SimpleNamespace(shape=(2, 2, 2), dx=0.001, dt=1e-12)
        sim = SimpleNamespace(_build_grid=fail if case == "mesh_error" else lambda: grid)
        if case == "audit_error":
            sim.mesh_intelligence_report = fail
        def mesh():
            result = artifacts._mesh_summary(sim, n_steps=None, available_memory_gb=None)
            return result["error"] if case == "mesh_error" else result["audit"]["error"]
        return mesh
    if case == "invariance":
        from rfx.boundaries.tfsf import invariant
        return lambda: invariant(_BrokenArray(error), 0, (2, 2, 2))[1]
    if case in {"import_rfx", "import_jax", "jax_devices", "jax_config",
                "core_deps", "optional_deps", "smoke"}:
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
        else:
            monkeypatch.setattr(Simulation, "run", fail)
        check = {"import_rfx": diag._check_rfx_import, "import_jax": diag._check_jax,
                 "jax_devices": diag._check_jax, "jax_config": diag._check_jax,
                 "core_deps": diag._check_core_deps, "optional_deps": diag._check_optional_extras,
                 "smoke": diag._check_smoke}[case]
        def diagnostic():
            check(report)
            return report.lines
        return diagnostic
    if case == "dashboard":
        class Stopped(BaseException):
            pass
        st = MagicMock()
        st.number_input.side_effect = lambda *a, **kw: kw["value"]
        st.selectbox.side_effect = lambda label, options, **kw: options[0]
        st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
        st.session_state = {}
        st.button.return_value = True
        st.stop.side_effect = Stopped
        components = MagicMock()
        components.geometry_panel.return_value = []
        components.source_panel.return_value = ([], [])
        monkeypatch.setitem(sys.modules, "streamlit", st)
        monkeypatch.setitem(sys.modules, "rfx.dashboard.components", components)
        monkeypatch.setattr(Simulation, "_build_realized_grid", fail)
        def dashboard():
            with pytest.raises(Stopped):
                runpy.run_path(str(Path(__file__).parents[3] / "rfx/dashboard/app.py"))
            return str(st.error.call_args) + str(st.code.call_args)
        return dashboard
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
        monkeypatch.setattr(sim, "realized_geometry", lambda: replace(record, entities=(row,)))
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


@pytest.mark.parametrize("case", G5_CASES)
def test_g5_timeout_propagates(case, monkeypatch):
    call = _g5_call(case, monkeypatch, TimeoutError)
    with pytest.raises(TimeoutError, match=MESSAGE):
        call()


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


@pytest.mark.parametrize("error", [TypeError, ValueError, AttributeError, NotImplementedError, RuntimeError])
def test_scalar_conversion_known_failure_preserves_value(error):
    from rfx.artifacts import _scalar
    value = SimpleNamespace(item=_raising(error))
    assert _scalar(value) is value


def test_scalar_type_dispatch_propagates_timeout():
    from rfx.artifacts import _scalar
    with pytest.raises(TimeoutError, match=MESSAGE):
        _scalar(SimpleNamespace(item=_raising(TimeoutError)))


def test_git_undecodable_output_preserves_default(monkeypatch):
    from rfx import artifacts

    def undecodable(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(artifacts.subprocess, "run", undecodable)
    assert artifacts._git(["status", "--porcelain"], cwd=Path.cwd()) is None


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


def test_artifact_traced_values_use_metadata_without_host_conversion():
    from rfx import artifacts
    def check(value):
        assert artifacts._safe_float(value) is None
        assert artifacts._jsonable(value, array_values=True)["shape"] == [2]
        assert "traced value" in artifacts._array_summary(value, include_peak=True)["peak_error"]
        assert artifacts._profile_summary(value)["count"] == 2
        assert artifacts._freq_summary(value)["size"] == 2
        return value
    jax.make_jaxpr(check)(jnp.ones(2))


PROBE_CASES = (
    "version", "backend", "devices", "device_metadata", "analysis_attribute",
    "analysis_call", "analysis_field", "scope", "memory_outer", "memory_inner",
    "safe_float", "array_values", "to_dict", "asdict", "sequence", "profile",
    "grid_shape", "frequencies", "git", "distribution_version", "compile_probe",
)
NARROWED_PROBES = ("safe_float", "array_values", "sequence", "profile",
                   "grid_shape", "frequencies", "git", "distribution_version")
RECORDED_PROBES = tuple(case for case in PROBE_CASES if case not in NARROWED_PROBES)


def _recorded_probe(case, monkeypatch, error):
    fail = _raising(error)
    class BrokenAttribute:
        def __getattr__(self, name):
            fail()
    if case in {"version", "backend", "devices", "device_metadata"}:
        from rfx.api import _compiled_memory as memory
        fake_jax = SimpleNamespace(__version__="0.10.2", default_backend=lambda: "cpu",
                                   local_devices=lambda: [BrokenAttribute()])
        if case == "version":
            fake_jax = BrokenAttribute()
        elif case == "backend":
            fake_jax.default_backend = fail
        elif case == "devices":
            fake_jax.local_devices = fail
        monkeypatch.setattr(memory, "jax", fake_jax)
        return lambda: memory._environment_summary()["reason"]
    if case in {"analysis_attribute", "analysis_call", "analysis_field", "scope"}:
        from rfx.api import _compiled_memory as memory
        if case == "scope":
            return lambda: memory._compiled_introspection_scope(BrokenAttribute())[1]
        compiled = BrokenAttribute() if case == "analysis_attribute" else SimpleNamespace(
            memory_analysis=fail if case == "analysis_call" else lambda: BrokenAttribute())
        return lambda: memory._normalize_compiled_memory_analysis(compiled).reason
    if case in {"memory_outer", "memory_inner"}:
        from tests.unit.autodiff.test_estimate_ad_memory import _uniform_sim
        sim = _uniform_sim()
        monkeypatch.setattr(jax, "local_devices", fail if case == "memory_outer" else lambda: [
            SimpleNamespace(platform="mps", memory_stats=fail)])
        return lambda: sim.estimate_ad_memory(4).warning
    if case == "distribution_version":
        from rfx import diagnostics
        monkeypatch.setattr(diagnostics.metadata, "version", fail)
        return lambda: diagnostics._version_of("rfx")
    if case == "compile_probe":
        from rfx.core.jax_utils import compile_time_switch
        monkeypatch.setattr(jax, "make_jaxpr", fail)
        def probe():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert compile_time_switch.__wrapped__() is None
            return str(caught[0].message)
        return probe
    from rfx import artifacts
    if case == "safe_float":
        class BrokenFloat:
            __float__ = fail
        return lambda: artifacts._safe_float(BrokenFloat())
    if case == "array_values":
        return lambda: artifacts._jsonable(_BrokenArray(error), array_values=True)
    if case in {"to_dict", "asdict"}:
        value = SimpleNamespace(**{"to_dict" if case == "to_dict" else "_asdict": fail})
        return lambda: artifacts._jsonable(value)
    if case == "sequence":
        class BrokenSequence:
            __len__ = fail
        return lambda: artifacts._sequence_summary(BrokenSequence())
    if case == "profile":
        return lambda: artifacts._profile_summary(_BrokenArray(error))
    if case == "grid_shape":
        class BrokenDimension:
            __int__ = fail
        return lambda: artifacts._grid_shape(SimpleNamespace(shape=(BrokenDimension(),)))
    if case == "frequencies":
        return lambda: artifacts._freq_summary(_BrokenArray(error))
    if case == "git":
        monkeypatch.setattr(artifacts.subprocess, "run", fail)
        return lambda: artifacts._git(["rev-parse", "HEAD"], cwd=Path.cwd())
    raise AssertionError(case)


@pytest.mark.parametrize("case", PROBE_CASES)
def test_recorded_probe_timeout_propagates(case, monkeypatch):
    call = _recorded_probe(case, monkeypatch, TimeoutError)
    with pytest.raises(TimeoutError, match=MESSAGE):
        call()


@pytest.mark.parametrize("case", RECORDED_PROBES)
def test_recorded_probe_preserves_fallback(case, monkeypatch):
    result = _recorded_probe(case, monkeypatch, RuntimeError)()
    if case in {"to_dict", "asdict"}:
        assert result["type"] == "SimpleNamespace"
        assert "fail" in result["repr"]
        result = result["error"]
    assert f"RuntimeError: {MESSAGE}" in result


@pytest.mark.parametrize("case", NARROWED_PROBES)
def test_default_probe_runtime_error_propagates(case, monkeypatch):
    with pytest.raises(RuntimeError, match=MESSAGE):
        _recorded_probe(case, monkeypatch, RuntimeError)()


@pytest.mark.parametrize("case, error", [
    (case, error)
    for case, errors in (
        ("safe_float", (TypeError, ValueError, OverflowError)),
        ("grid_shape", (TypeError, ValueError, OverflowError)),
        ("array_values", (TypeError, ValueError)),
        ("profile", (TypeError, ValueError)),
        ("frequencies", (TypeError, ValueError)),
        ("sequence", (TypeError,)),
        ("git", (OSError,)),
        ("distribution_version", (ImportError, OSError)),
    ) for error in errors
])
def test_default_probe_expected_failure_preserves_fallback(case, error, monkeypatch):
    result = _recorded_probe(case, monkeypatch, error)()
    if case in {"safe_float", "grid_shape", "git"}:
        assert result is None
    elif case == "distribution_version":
        assert result == "?"
    elif case == "sequence":
        assert result == {"present": True, "count": None, "type": "BrokenSequence"}
    else:
        assert result["shape"] == [2, 2, 2]
        assert result["dtype"] == "float32"
        assert result["size"] == 8
        assert "min" not in result


def test_git_subprocess_timeout_preserves_default(monkeypatch):
    import subprocess
    from rfx import artifacts
    def expired(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 2)
    monkeypatch.setattr(artifacts.subprocess, "run", expired)
    assert artifacts._git(["status"], cwd=Path.cwd()) is None


def test_animation_writer_runtime_error_propagates(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from rfx.animation import save_field_animation
    monkeypatch.setattr(animation, "FFMpegWriter", _raising(RuntimeError))
    try:
        with pytest.raises(RuntimeError, match=MESSAGE):
            save_field_animation({"ez": np.zeros((2, 2, 2))}, tmp_path / "failed.mp4")
    finally:
        plt.close("all")


@pytest.mark.parametrize("probe, target", [("_mpl3d", "mpl_toolkits.mplot3d.art3d"),
                                          ("_pyvista", "pyvista")])
def test_visualization_import_probe_runtime_error_propagates(monkeypatch, probe, target):
    from rfx import visualize3d
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == target:
            raise RuntimeError(MESSAGE)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError, match=MESSAGE):
        getattr(visualize3d, probe)()
