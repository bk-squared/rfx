"""Device-memory peak of the differentiated FDTD run predicted by compiled AD."""
from collections import namedtuple
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.api._ad_memory import _AdMemoryMixin
from rfx.api import _ad_memory_compiled as api
from rfx.api._ad_memory_types import AD_MemoryEstimate, ADMemoryUnavailable
from rfx.api._spec import ForwardResult


NEW_FIELDS = (
    "compiled_bytes", "resident_bytes", "allocator_rounding_bytes",
    "allocator_rounding_max_bytes", "compiled_total_bytes",
)


class Device:
    platform = "gpu"

    def __init__(self):
        self.compiled = False
        self.reads = []

    def memory_stats(self):
        value = 2_853_632 if self.compiled else 1_205_460
        self.reads.append(value)
        return {"bytes_in_use": value}


class Scene:
    """Small differentiable records; compilation and device stats are stubbed."""
    _uses_nonuniform_mesh = True
    _refinement = None
    _precision = "float32"

    def __init__(self):
        self.extra = {}
        self.calls = []
        self.analytic = AD_MemoryEstimate(1., 2., 3., 0., None, None, ad_segmented_gb=.5)

    def _build_realized_grid(self):
        return SimpleNamespace(shape=(2, 3, 4))

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        eps, sigma = kwargs["eps_override"], kwargs["sigma_override"]
        return ForwardResult(
            time_series=eps + sigma,
            waveguide_ports={"port": (2 * eps, jnp.asarray(1, dtype=jnp.int32))},
            # N3 carries its frequencies as a NumPy array here: a constant, not pulled back.
            waveguide_sparams={"S": (eps + 1j * sigma).sum(),
                               "freqs": np.linspace(4.5e9, 8e9, 3, dtype=np.float32)},
            grid={"coordinates": eps}, **self.extra)

    def estimate_ad_memory(self, n_steps, **kwargs):
        if kwargs.get("compiled", False):
            return _AdMemoryMixin.estimate_ad_memory(self, n_steps, **kwargs)
        self.analytic_options = dict(n_steps=n_steps, **kwargs)
        return self.analytic


@pytest.fixture
def harness(monkeypatch):
    device, sim = Device(), Scene()
    monkeypatch.setattr(api, "_default_device", lambda: device)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    analysis = SimpleNamespace(temp_size_in_bytes=517_738_488,
                               argument_size_in_bytes=90_536,
                               output_size_in_bytes=90_536, alias_size_in_bytes=0)
    state = SimpleNamespace(device=device, sim=sim, analysis=analysis, traces=[], compiles=0)
    real_jit = jax.jit

    def jit(function):
        def trace(*args):
            traced = real_jit(function).trace(*args)
            state.traces.append((traced, args))

            def compile_only():
                state.compiles += 1
                device.compiled = True
                return SimpleNamespace(memory_analysis=lambda: state.analysis)

            return SimpleNamespace(lower=lambda: SimpleNamespace(compile=compile_only))
        return SimpleNamespace(trace=trace)

    monkeypatch.setattr(api.jax, "jit", jit)
    return state


@pytest.mark.parametrize("case,message", [
    ("uniform", "uniform path"), ("distributed", "distributed path"),
    ("subgridded", "subgridded path"), ("cpu", "backend cpu"),
    ("METAL", "backend METAL"), ("tpu", "backend tpu"),
    ("missing_stats", "memory_stats bytes_in_use"),
    ("no_stats_method", "memory_stats bytes_in_use"),
    ("missing_bytes", "memory_stats bytes_in_use"),
    ("stats_exception", "memory_stats bytes_in_use"),
    ("unset_preallocate", "PREALLOCATE=false"),
    ("true_preallocate", "PREALLOCATE=false"),
    ("zero_preallocate", "PREALLOCATE=false"),
    ("capital_preallocate", "PREALLOCATE=false"),
    ("occupancy", "whole-grid permittivity"), ("design_box", "whole-grid permittivity"),
    ("float64", "float64"), ("segments", "checkpoint plan"),
    ("none", "checkpoint plan"), ("one_level_tuple", "checkpoint plan"),
    ("three_levels", "checkpoint plan"), ("boolean", "checkpoint plan"),
    ("warmup", "n_warmup"), ("unsegmented_int", "does not segment"),
    ("unsegmented_tuple", "does not segment"),
])
def test_1_refusals(harness, monkeypatch, case, message):
    """Unmeasured paths cannot claim the device-memory peak of a differentiated FDTD run."""
    sim, device = harness.sim, harness.device
    options = dict(compiled=True, checkpoint_every=2)
    if case == "uniform":
        sim._uses_nonuniform_mesh = False
    elif case == "distributed":
        options["distributed"] = True
    elif case == "subgridded":
        sim._refinement = {"z_range": (.001, .002), "ratio": 2}
    elif case in ("cpu", "METAL", "tpu"):
        device.platform = case
    elif case == "missing_stats":
        device.memory_stats = lambda: None
    elif case == "no_stats_method":
        monkeypatch.setattr(api, "_default_device", lambda: SimpleNamespace(platform="gpu"))
    elif case == "missing_bytes":
        device.memory_stats = lambda: {}
    elif case == "stats_exception":
        def unavailable():
            raise RuntimeError("no allocator stats")
        device.memory_stats = unavailable
    elif case == "unset_preallocate":
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE")
    elif case.endswith("_preallocate"):
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", {
            "true_preallocate": "true", "zero_preallocate": "0", "capital_preallocate": "False",
        }[case])
    elif case == "occupancy":
        options["design"] = "occupancy"
    elif case == "design_box":
        options["design_box"] = ((0.,) * 3, (.001,) * 3)
    elif case == "float64":
        sim._precision = "float64"
    elif case == "segments":
        options["checkpoint_segments"] = 2
    elif case == "warmup":
        options["n_warmup"] = 1
    else:
        options["checkpoint_every"] = {
            "none": None, "one_level_tuple": (2,), "three_levels": (4, 2, 1), "boolean": True,
            "unsegmented_int": 4, "unsegmented_tuple": (8, 5),
        }[case]
    with pytest.raises(ADMemoryUnavailable, match=message):
        sim.estimate_ad_memory(4, **options)
    assert harness.compiles == 0


def test_1_path_refusals_precede_cpu_backend():
    """Path refusals for the differentiated FDTD device peak remain reachable on CPU."""
    if jax.default_backend() != "cpu":
        pytest.skip("pins the real CPU backend refusal")
    sim = Scene()
    sim._uses_nonuniform_mesh = False
    with pytest.raises(ADMemoryUnavailable, match="uniform path"):
        sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2)
    sim._uses_nonuniform_mesh = True
    with pytest.raises(ADMemoryUnavailable, match="distributed path"):
        sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2, distributed=True)
    sim._refinement = {}
    with pytest.raises(ADMemoryUnavailable, match="subgridded path"):
        sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2)
    sim._refinement = None
    with pytest.raises(ADMemoryUnavailable, match="backend cpu"):
        sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2)


@pytest.mark.parametrize("field", [name for name in ForwardResult._fields if name not in (
    "time_series", "waveguide_ports", "waveguide_sparams", "grid")])
def test_1_unmeasured_output_refusal(harness, field):
    """Every other inexact output invalidates the differentiated FDTD device-peak estimate."""
    harness.sim.extra[field] = {"nested": [jnp.ones(1, dtype=jnp.complex64)]}
    with pytest.raises(ADMemoryUnavailable, match=f"output {field} not measured"):
        harness.sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2)
    assert harness.compiles == 0


def test_1_future_output_refusal():
    """A new record cannot silently change the predicted differentiated FDTD device peak."""
    result = namedtuple("FutureResult", "time_series waveguide_ports waveguide_sparams grid future")
    with pytest.raises(ADMemoryUnavailable, match="output future not measured"):
        api._measured_outputs(result(None, None, None, None, jnp.ones(1)))


@pytest.mark.parametrize("analysis,message", [
    (None, "memory_analysis.*returned None"),
    (SimpleNamespace(temp_size_in_bytes=1), "missing required byte field"),
    (SimpleNamespace(temp_size_in_bytes=-1, argument_size_in_bytes=0,
                     output_size_in_bytes=0, alias_size_in_bytes=0), "non-negative"),
    (SimpleNamespace(temp_size_in_bytes=100, argument_size_in_bytes=20,
                     output_size_in_bytes=20, alias_size_in_bytes=4), "alias_size_in_bytes"),
])
def test_1_analysis_refusal(harness, analysis, message):
    """Incomplete or aliased analysis cannot predict a differentiated FDTD device peak."""
    harness.analysis = analysis
    with pytest.raises(ADMemoryUnavailable, match=message):
        harness.sim.estimate_ad_memory(4, compiled=True, checkpoint_every=2)


@pytest.mark.parametrize("temp,rounding,maximum", [
    # N3: ceil256(t)=517738496; 2 MiB * 2**8 = 536870912; difference=19132416.
    (517_738_488, 19_132_416, 134_217_728),
    # N1: ceil256(t)=1742162688; R=2147483648; difference=405320960 >=128 MiB.
    (1_742_162_640, 0, 134_217_728),
    # 62.1 MB: ceil256=62100224; R=67108864; difference=5008640.
    (62_100_000, 5_008_640, 62_100_224),
    (2_097_152, 0, 2_097_152), (1_048_575, 1_048_576, 1_048_576),
    (1, 2_096_896, 256), (0, 2_097_152, 0),
    # Exactly the 128 MiB split threshold, then one 256-byte step below it.
    (402_653_184, 0, 134_217_728), (402_653_440, 134_217_472, 134_217_728),
])
def test_2_rounding(temp, rounding, maximum):
    """Hand-built BFC excess predicts part of the differentiated FDTD device-memory peak."""
    assert api._allocator_rounding(temp) == (rounding, maximum)


@pytest.mark.parametrize("cotangents,plan", [(1, 2), (3, (3, 2))])
def test_3_composition(harness, cotangents, plan):
    """E_c + post-compile R_res + rho predicts the differentiated FDTD device-memory peak."""
    sim = harness.sim
    before = sim.estimate_ad_memory(4, compiled=False)
    result = sim.estimate_ad_memory(4, compiled=True, checkpoint_every=plan,
                                    cotangents=cotangents, checkpoint=False, n_warmup=0)
    # A6000 N3, by hand: 517738488 + 90536 + 90536 = 517919560.
    assert result.compiled_bytes == 517_919_560
    assert result.resident_bytes == 2_853_632
    assert result.allocator_rounding_bytes == 19_132_416
    assert result.allocator_rounding_max_bytes == 134_217_728
    assert result.compiled_total_bytes == 539_905_608
    assert harness.device.reads == [1_205_460, 2_853_632]
    assert harness.compiles == 1
    assert result[:-5] == before[:-5]
    assert tuple(getattr(before, name) for name in NEW_FIELDS) == (None,) * 5
    assert {k: v for k, v in result.to_dict().items() if k not in NEW_FIELDS} == before.to_dict()
    assert json.loads(result.to_json()) == result.to_dict()
    assert tuple(result._asdict())[-5:] == NEW_FIELDS
    assert "compiled_total_bytes=539905608" in repr(result)
    traced, args = harness.traces[0]
    assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(args))
    assert [x.shape for x in args[:2]] == [(2, 3, 4)] * 2
    assert len(args[2]) == 3  # Time series, port float array, complex S; no integer, NumPy or grid leaf.
    assert [x.shape for x in args[2]] == (
        [(2, 3, 4), (2, 3, 4), ()] if cotangents == 1
        else [(3, 2, 3, 4), (3, 2, 3, 4), (3,)])
    assert [x.shape for x in traced.out_info] == [
        ((2, 3, 4) if cotangents == 1 else (3, 2, 3, 4))] * 2
    assert sim.analytic_options["cotangents"] == cotangents
    for call in sim.calls:
        assert call["skip_preflight"] is True
        assert call["checkpoint"] is False
        assert call["checkpoint_every"] == plan
        assert call["n_warmup"] == 0


def _small_graded_scene():
    sim = Simulation(freq_max=10e9, domain=(.012,) * 3, dx=.002,
                     boundary="pec", cpml_layers=0, dz_profile=np.full(6, .002))
    sim.add_source((.004,) * 3, "ez", amplitude_kind="current")
    sim.add_probe((.008,) * 3, "ez")
    return sim


def test_3_real_forward_composition(harness):
    """The real graded FDTD pullback's predicted device peak retains its analytic fields."""
    sim = _small_graded_scene()
    analytic = sim.estimate_ad_memory(4, checkpoint_every=2, checkpoint=False)
    result = sim.estimate_ad_memory(4, checkpoint_every=2, checkpoint=False, compiled=True)
    assert result[:-5] == analytic[:-5]
    assert result.compiled_total_bytes == 539_905_608
    assert result.resident_bytes == 2_853_632
    assert harness.compiles == 1


def test_compiled_false_unchanged(monkeypatch):
    """Opting out preserves every analytic field for the differentiated FDTD device peak."""
    sim = _small_graded_scene()
    def forbidden(*args, **kwargs):
        pytest.fail("compiled helper entered on analytic path")
    monkeypatch.setattr("rfx.api._ad_memory.compiled_estimate", forbidden)
    default = sim.estimate_ad_memory(4, checkpoint_every=2)
    explicit = sim.estimate_ad_memory(4, checkpoint_every=2, compiled=False)
    assert default._asdict() == explicit._asdict()
    assert default == explicit
    assert default.to_json() == explicit.to_json()
    assert all(getattr(default, name) is None for name in NEW_FIELDS)
    assert not set(NEW_FIELDS).intersection(default.to_dict())


@pytest.mark.parametrize("as_platform", [False, True])
def test_default_device_honors_context(as_platform):
    """Resident accounting follows the device placing the differentiated FDTD peak."""
    device = jax.local_devices()[0]
    with jax.default_device(device.platform if as_platform else device):
        assert api._default_device() == jax.local_devices()[0]


@pytest.mark.gpu
def test_n3_gpu_fresh_process():
    """Predict the device-memory peak of the measured N3 differentiated FDTD run."""
    if jax.default_backend() != "gpu":
        pytest.skip("N3 requires GPU memory_stats; never run the FDTD measurement on CPU")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    env.pop("PYTHONPATH", None)
    # Import only the row builder below in a fresh interpreter; never run pytest recursively.
    code = (
        "import runpy; "
        f"runpy.run_path({str(Path(__file__).resolve())!r})['_n3_gpu_row']()"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[3],
                            text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def _n3_gpu_row():
    """The archive's WR-90 build(300) and eps-only N3 objective, without measurement records."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("N3 requires GPU")
    dx = 299_792_458.0 / 8.0e9 / 20.0
    sim = Simulation(freq_max=8.0e9, domain=(.12, 22 * dx, 11 * dx), dx=dx,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
                     dz_profile=np.full(11, dx))
    sim.add_waveguide_port(.03, direction="+x", mode=(1, 0), mode_type="TE",
                          freqs=jnp.asarray(np.linspace(4.5e9, 8.0e9, 300)),
                          f0=6.25e9, bandwidth=.56, ref_offset=3, probe_offset=15, name="left")
    original = api._normalize_compiled_memory_analysis
    aliases = []

    def capture_analysis(compiled):
        analysis = original(compiled)
        aliases.append(analysis.fields["alias_size_in_bytes"])
        return analysis

    api._normalize_compiled_memory_analysis = capture_analysis
    estimate = sim.estimate_ad_memory(50_000, checkpoint=False,
                                      checkpoint_every=(1000, 40), compiled=True)
    assert aliases == [0]
    eps = jnp.ones(sim._build_realized_grid().shape, jnp.float32)

    def scalar(eps):
        result = sim.forward(n_steps=50_000, checkpoint=False, checkpoint_every=(1000, 40),
                             eps_override=eps, skip_preflight=True)
        return sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(result.waveguide_sparams)
                   if hasattr(x, "dtype") and jnp.iscomplexobj(x))

    measured = jax.jit(jax.grad(scalar)).lower(eps).compile()
    assert original(measured).fields["alias_size_in_bytes"] == 0
    jax.block_until_ready(measured(eps))
    peak = api._default_device().memory_stats()["peak_bytes_in_use"]
    assert 0.99 <= estimate.compiled_total_bytes / peak <= 1.05, (estimate, peak)
