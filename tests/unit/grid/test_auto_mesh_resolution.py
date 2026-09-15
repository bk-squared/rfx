"""The documented declarations are judged on the mesh execution selects.

The three documentation variants exercise real preflight and material assembly
through both public execution entry points, stopping immediately after assembly
to avoid six large scans. The original plate additionally executes a real one-
step run and forward. Small models exercise tracing and invalidation separately.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.api._mesh import AutoMeshWarning
from rfx.nonuniform import NonUniformGrid


def _documented_model(variant="plate"):
    # materials-geometry.mdx's first Simulation and 1 mm PEC plate block.
    sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.02), boundary="cpml")
    sim.add(Box((0.005, 0.005, 0.005), (0.045, 0.045, 0.0066)), material="fr4")
    sim.add(Box((0.005, 0.005, 0.004), (0.045, 0.045, 0.005)), material="pec")
    if variant == "chained":
        (sim.add_material("sub", eps_r=3.55, sigma=0.0)
         .add_material("metal", eps_r=1.0, sigma=5.8e7)
         .add(Box((0, 0, 0), (0.05, 0.03, 0.001)), material="sub")
         .add(Box((0.005, 0.005, 0.001), (0.045, 0.025, 0.0015)), material="metal"))
    elif variant == "via":
        sim.add(Cylinder((0.025, 0.015, 0.000), radius=0.0005,
                         height=0.002, axis="z"), material="copper")
    return sim


def _pec_errors(report):
    return [issue for issue in report.errors if issue.code in {
        "pec_box_subcell", "pec_zero_cells", "pec_realization_refused",
    }]


class _Assembled(Exception):
    """Stop after the real refusal boundary, before allocating a scan."""


@pytest.mark.parametrize("variant", ["plate", "chained", "via"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_documentation_legality_matches_execution_assembly(monkeypatch, variant, entry):
    import rfx.runners.nonuniform as runner

    sim = _documented_model(variant)
    report = sim.preflight()
    assert not _pec_errors(report), [issue.to_dict() for issue in report.errors]
    expected = sim._build_realized_grid()
    assert isinstance(expected, NonUniformGrid)
    original = runner.assemble_materials_nu
    observed = []

    def assemble_then_stop(active_sim, grid, *args, **kwargs):
        assembled = original(active_sim, grid, *args, **kwargs)
        assert assembled[3] is not None and bool(jnp.any(assembled[3]))
        observed.append(grid)
        raise _Assembled

    with monkeypatch.context() as patch:
        patch.setattr(runner, "assemble_materials_nu", assemble_then_stop)
        with pytest.raises(_Assembled):
            getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    np.testing.assert_array_equal(observed[0].dz, expected.dz)
    assert observed[0].dx == expected.dx


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_documented_plate_completes_real_one_step_solve(entry):
    # A fresh declaration also proves callers need not preflight beforehand.
    sim = _documented_model()
    result = getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert isinstance(result.grid, NonUniformGrid)
    assert np.isfinite(np.asarray(result.time_series)).all()
    assert not _pec_errors(sim.preflight())
    assert result.grid.dx == pytest.approx(0.0005)


def _small_static_model():
    sim = Simulation(freq_max=10e9, domain=(0.008, 0.008, 0.008), boundary="pec")
    sim.add_material("dielectric", eps_r=4.0)
    sim.add(Box((0.002, 0.002, 0.002), (0.006, 0.006, 0.006)), material="dielectric")
    return sim


def test_preflight_resolution_is_idempotent_and_preserves_declaration(monkeypatch):
    sim = _small_static_model()
    names = ("_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile")
    declared = {name: sim.__dict__[name] for name in names}
    original = sim._auto_configure_mesh
    calls = []

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(sim, "_auto_configure_mesh", counted)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = sim.preflight()
        grid = sim._build_realized_grid()
        second = sim.preflight()
        result = sim.run(n_steps=1, skip_preflight=True)
    assert calls == [1]
    assert sum(str(w.message).startswith("Auto mesh:") for w in caught) == 1
    assert not any(str(issue).startswith("Auto mesh:") for issue in first + second)
    assert all(sim.__dict__[name] is value for name, value in declared.items())
    assert grid.shape == result.grid.shape
    assert grid.dt == result.grid.dt
    fresh_result = _small_static_model().run(n_steps=1, skip_preflight=True)
    assert fresh_result.grid.shape == result.grid.shape
    assert fresh_result.grid.dt == result.grid.dt


def test_geometry_added_after_preview_recomputes_the_mesh():
    sim = _small_static_model()
    old = sim._resolve_mesh()
    assert old["_dz_profile"] is None
    sim.add(Box((0.002, 0.002, 0.006), (0.006, 0.006, 0.0065)), material="dielectric")
    with pytest.warns(AutoMeshWarning, match="Auto mesh:"):
        new = sim._resolve_mesh()
    assert new is not old
    assert new["_dz_profile"] is not None
    assert sim.__dict__["_dx"] is None
    assert sim.__dict__["_dz_profile"] is None
    fresh = _small_static_model()
    fresh.add(Box((0.002, 0.002, 0.006), (0.006, 0.006, 0.0065)), material="dielectric")
    np.testing.assert_array_equal(new["_dz_profile"], fresh._dz_profile)
    assert new["_dx"] == fresh._dx


def test_declared_mesh_remains_available_before_and_after_resolution(monkeypatch):
    sim = _documented_model()
    declared = sim._declared_mesh
    assert declared["_dx"] is None
    assert declared["_dz_profile"] is None
    assert declared["_domain"] == (.05, .05, .02)

    def unexpected_resolution():
        pytest.fail("reading the declaration must not plan a mesh")

    with monkeypatch.context() as patch:
        patch.setattr(sim, "_auto_configure_mesh", unexpected_resolution)
        assert sim._declared_mesh == declared
    resolved = sim._resolve_mesh()
    assert resolved["_dz_profile"] is not None
    assert resolved["_dx"] is not None
    assert sim._declared_mesh == declared
    # The returned mapping is a snapshot, not writable Simulation state.
    declared["_dx"] = .001
    assert sim._declared_mesh["_dx"] is None


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_first_forward_under_jit_grad_resolves_static_mesh_on_host(
    monkeypatch, nonuniform, skip_preflight,
):
    # Only the independent reference is resolved before tracing. The actual
    # forward's first mesh read occurs inside the jit(grad(...)) invocation.
    def model():
        sim = _small_static_model()
        if nonuniform:
            sim.add(Box((.002, .002, .006), (.006, .006, .0065)), material="dielectric")
        return sim

    shape = model()._build_realized_grid().shape
    sim = model()
    # Microamp source keeps reverse-mode intermediates within float32 range.
    sim.add_source((0.004, 0.004, 0.004), "ez", waveform=lambda t: 1e-6 * jnp.ones_like(t),
                   amplitude_kind="current")
    sim.add_probe((0.004, 0.004, 0.004), "ez")
    original = sim._auto_configure_mesh
    calls = []

    def counted():
        config = original()
        assert not isinstance(config.dx, jax.core.Tracer)
        assert not any(isinstance(value, jax.core.Tracer)
                       for value in jax.tree.leaves(config.dz_profile))
        calls.append(1)
        return config

    monkeypatch.setattr(sim, "_auto_configure_mesh", counted)

    def loss(eps):
        result = sim.forward(n_steps=6, eps_override=jnp.full(shape, eps),
                             skip_preflight=skip_preflight)
        return jnp.sum(result.time_series ** 2)

    derivative = jax.jit(jax.grad(loss))
    first = float(derivative(jnp.float32(4.0)))
    second = float(derivative(jnp.float32(4.2)))
    assert np.isfinite(first) and first != 0.0
    assert np.isfinite(second) and second != 0.0
    assert calls == [1]
    assert sim.__dict__["_dx"] is None
    assert not isinstance(sim._dx, jax.core.Tracer)


def test_traced_geometry_requires_explicit_mesh_inputs():
    def selected_dx(width, *, dx=None):
        sim = Simulation(freq_max=10e9, domain=(0.008, 0.008, 0.008),
                         dx=dx, boundary="pec")
        sim.add(Box((0.001, 0.001, 0.001), (width, 0.006, 0.006)), material="fr4")
        return sim._dx

    with pytest.raises(ValueError, match="Auto mesh requires static geometry.*dx="):
        jax.jit(selected_dx)(jnp.float32(0.006))
    fixed = jax.jit(lambda width: selected_dx(width, dx=0.001))
    assert float(fixed(jnp.float32(0.006))) == pytest.approx(0.001)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_true_subcell_pec_still_errors_on_explicit_realized_grid(entry):
    sim = Simulation(freq_max=10e9, domain=(0.008, 0.008, 0.008),
                     dx=0.001, boundary="pec")
    sim.add(Box((0.002, 0.002, 0.002), (0.006, 0.006, 0.0025)), material="pec")
    findings = [issue for issue in sim.preflight().errors if issue.code == "pec_box_subcell"]
    assert findings and all(issue.severity == "error" for issue in findings)
    assert "run()/forward() will raise" in str(findings[0])
    with pytest.raises(ValueError, match="thinner than one cell"):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)


def test_jit_keeps_static_nonuniform_subcell_refusal():
    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     dz_profile=np.full(8, .001), boundary="pec")
    sim.add(Box((.002, .002, .002), (.006, .006, .0025)), material="pec")

    def loss(eps):
        return sim.forward(n_steps=1, eps_override=eps,
                           skip_preflight=True).time_series.sum()

    with pytest.raises(ValueError, match="thinner than one cell"):
        jax.jit(loss)(jnp.ones((9, 9, 9)))
