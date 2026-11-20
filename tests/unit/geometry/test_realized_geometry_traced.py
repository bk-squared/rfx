"""A material trace refuses public host geometry and preserves differentiable solves."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.errors import RfxRefusal
from rfx.realized_geometry import RealizedGeometry


def _scene(p):
    d = 1e-3
    sim = Simulation(freq_max=10e9, domain=(8*d, 8*d, 8*d), dx=d,
                     boundary="pec")
    sim.add_material("dielectric", eps_r=1 + p)
    sim.add(Box((2*d, 2*d, 2*d), (6*d, 6*d, 6*d)), material="dielectric")
    sim.add_source((3*d, 4*d, 4*d), "ez", waveform=GaussianPulse(f0=5e9),
                   amplitude_kind="field")
    sim.add_probe((5*d, 4*d, 4*d), "ez")
    return sim


def test_public_geometry_refuses_and_names_material_trace(capsys):
    started = time.perf_counter()

    def objective(p):
        sim = _scene(p)
        with pytest.raises(RfxRefusal, match=r"material\['dielectric'\]\.eps_r") as caught:
            sim.realized_geometry()
        message = str(caught.value)
        assert message == (
            "realized_geometry() is unavailable: traced inputs: material['dielectric'].eps_r. "
            "Call realized_geometry() outside the transform on concrete values; "
            "the geometry does not depend on a material value.")
        with capsys.disabled():
            print(f"public refusal: {message}")
        return p

    assert jax.jacfwd(objective)(0.5) == 1
    with capsys.disabled():
        print(f"public refusal wall time: {time.perf_counter() - started:.6f}s")


def test_material_trace_preserves_forward_and_run(capsys):
    for entry in ("forward", "run"):
        started = time.perf_counter()
        observations = []

        def objective(p):
            sim = _scene(p)
            try:
                sim.realized_geometry()
            except RfxRefusal:
                pass
            result = getattr(sim, entry)(n_steps=16, skip_preflight=True)
            record = result.realized_geometry
            assert record is None or isinstance(record, RealizedGeometry)
            observations.append("None" if record is None else "record")
            return jnp.sum(jnp.square(result.time_series[:, 0]))

        derivative = jax.jacfwd(objective)(0.5)
        assert np.isfinite(derivative)
        assert len(observations) == 1
        with capsys.disabled():
            print(f"{entry}: trace admitted; Result.realized_geometry={observations[0]}; "
              f"objective derivative={float(derivative)}; "
              f"wall time={time.perf_counter() - started:.6f}s")


def test_concrete_entity_masks_equal_concrete_build(capsys):
    started = time.perf_counter()
    sim = _scene(0.5)
    record = sim.realized_geometry()
    build = _scene(0.5)
    masks = []
    build._assemble_materials(build._build_grid(), geometry_masks=masks)
    observed = dict(masks)
    assert len(record.entities) == len(build._geometry) == len(observed) == 1
    for entity, entry in zip(record.entities, build._geometry, strict=True):
        np.testing.assert_array_equal(entity.mask, observed[id(entry)])
    with capsys.disabled():
        print(f"concrete masks wall time: {time.perf_counter() - started:.6f}s")
