"""An MSL port on a traced mesh is refused by name at the line-port admission.

The admission reads realized node coordinates on the host, so a traced
``dx_profile``, ``dy_profile`` or ``dz_profile`` meets a raw JAX error there
or later. The refusal comes first, on run() and forward(), under jit and grad,
with preflight on and off, and before any mesh build.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.model import NominalProfile
from rfx.preflight.line_stub import line_stub_admission

MM = 1e-3
DZ = np.concatenate([np.full(6, 0.5), 0.5 * 1.25 ** np.arange(1, 6)]) * MM
DY = np.full(12, 1.0 * MM)
W_Z = np.zeros(DZ.size)
W_Z[4], W_Z[6] = 0.1 * MM, -0.1 * MM
W_Y = np.zeros(DY.size)
W_Y[1], W_Y[2] = 0.1 * MM, -0.1 * MM
AXIS_Z = "the cell widths on axis z are traced"
AXIS_Y = "on axis y are traced"
KEEP = "Keep the cell-width profiles concrete in a model with an MSL or coaxial port."


def _scene(v=None, strip="box", axis="z"):
    """The graded MSL board: eps_r 3 substrate (z 1-3 mm) on a PEC ground."""
    extra = {}
    dz = DZ
    if axis == "z" and v is not None:
        dz = DZ + v * W_Z
        extra["nominal_profiles"] = {"z": NominalProfile(cells=DZ, traced_cells=(4, 7))}
    if axis == "y" and v is not None:
        extra["dy_profile"] = DY + v * W_Y
        extra["nominal_profiles"] = {"y": NominalProfile(cells=DY, traced_cells=(1, 3))}
    sim = Simulation(freq_max=16e9, domain=(24 * MM, 12 * MM, float(DZ.sum())),
                     dx=1.0 * MM, boundary="pec", dz_profile=dz,
                     dt=1e-12, dt_min_cell=0.5e-3, **extra)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((2 * MM, 2 * MM, 1 * MM), (14 * MM, 10 * MM, 3 * MM)), material="sub")
    sim.add(Box((2 * MM, 2 * MM, 0), (14 * MM, 10 * MM, 1 * MM)), material="pec")
    if strip == "box":
        sim.add(Box((5 * MM, 5 * MM, 3 * MM), (22 * MM, 7 * MM, float(np.cumsum(DZ)[7]))),
                material="pec")
    else:
        sim.add_pinned_sheet(plane_index=6, i_range=(5, 22), j_range=(5, 7), name="strip")
    sim.add_msl_port(position=(5 * MM, 6 * MM, 1 * MM), width=2 * MM,
                     height=2 * MM, direction="+x", impedance=50.0,
                     waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    sim.add_probe((15 * MM, 6 * MM, 2 * MM), "ez")
    return sim


def _solve(sim, entry, skip_preflight):
    if entry == "run":
        result = sim.run(n_steps=40, skip_preflight=skip_preflight)
    else:
        result = sim.forward(n_steps=40, skip_preflight=skip_preflight, checkpoint=False)
    return jnp.sum(result.time_series ** 2)


def _assert_refused(excinfo, axis_literal):
    assert type(excinfo.value) is TracedStructureRefusal
    assert axis_literal in str(excinfo.value)
    assert KEEP in str(excinfo.value)


@pytest.mark.parametrize("strip", ["sheet", "box"])
@pytest.mark.parametrize("skip_preflight", [False, True], ids=["preflight", "no_preflight"])
@pytest.mark.parametrize("transform", [jax.jit, jax.grad], ids=["jit", "grad"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_traced_z_msl_port_is_refused_by_name(entry, transform, skip_preflight, strip):
    def objective(v):
        return _solve(_scene(v, strip=strip), entry, skip_preflight)

    with pytest.raises(TracedStructureRefusal) as excinfo:
        transform(objective)(jnp.float32(0.0))
    _assert_refused(excinfo, AXIS_Z)


def test_traced_y_msl_port_is_refused_by_name():
    def objective(v):
        return _solve(_scene(v, strip="box", axis="y"), "forward", False)

    with pytest.raises(TracedStructureRefusal) as excinfo:
        jax.jit(objective)(jnp.float32(0.0))
    _assert_refused(excinfo, AXIS_Y)


def test_traced_z_msl_s_matrix_is_refused_by_name():
    def objective(v):
        _scene(v, strip="box").compute_msl_s_matrix(n_steps=40, n_freqs=3)
        return v

    with pytest.raises(TracedStructureRefusal) as excinfo:
        jax.jit(objective)(jnp.float32(0.0))
    _assert_refused(excinfo, AXIS_Z)


def test_traced_mesh_refusal_precedes_mesh_build(monkeypatch):
    def objective(v):
        sim = _scene(v, strip="sheet")

        def unexpected(*args, **kwargs):
            pytest.fail("the refusal must come before any mesh build")
        monkeypatch.setattr(sim, "_build_realized_grid", unexpected)
        return _solve(sim, "forward", True)

    with pytest.raises(TracedStructureRefusal) as excinfo:
        jax.jit(objective)(jnp.float32(0.0))
    _assert_refused(excinfo, AXIS_Z)


@pytest.mark.parametrize("strip", ["sheet", "box"])
def test_concrete_mesh_msl_port_passes_admission(strip):
    line_stub_admission(_scene(strip=strip))
