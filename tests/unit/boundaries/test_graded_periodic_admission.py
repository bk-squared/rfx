"""Unsupported periodic combinations refuse on both graded public entries."""
import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from tests.unit.boundaries.graded_periodic_cases import simulation


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_dual_average_periodic_graded_mesh_refuses(entry):
    sim = simulation(True, 20e9, (.012, .012, .012), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(x="periodic", y="pec", z="pec"),
                     interface_eps="dual_average")
    with pytest.raises(NotImplementedError, match=(
            "interface_eps='dual_average'.*periodic axes 'x'.*"
            "use the default interface_eps on a periodic axis")):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("skip_preflight", [False, True])
@pytest.mark.parametrize("axis", list("xyz"))
def test_nonconstant_periodic_profile_refuses(entry, skip_preflight, axis):
    cells = np.full(12, .001)
    cells[3:5] = (.0009, .0011)
    sim = Simulation(20e9, (.012, .012, .012), dx=.001, cpml_layers=0,
                     boundary=BoundarySpec(**{a: "periodic" if a == axis else "pec" for a in "xyz"}),
                     **{f"d{axis}_profile": cells})
    with pytest.raises(NotImplementedError, match=f"axis '{axis}'.*use a constant spacing along a periodic axis"):
        getattr(sim, entry)(n_steps=1, skip_preflight=skip_preflight)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("skip_preflight", [False, True])
@pytest.mark.parametrize("domain", [(.020, .003, .005), (.020, .028, .021)],
                         ids=["broadside", "induced-wrap-3d"])
def test_plane_wave_with_periodic_graded_mesh_refuses(entry, skip_preflight, domain):
    # The graded lane with one x cell size: y and z are periodic (constant by
    # rule) and x carries the wave, so a graded x would meet the graded
    # propagation-axis refusal (#2078) before the periodic one pinned here.
    sim = Simulation(20e9, domain, dx=.001, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="periodic", z="periodic"),
                     dx_profile=np.full(round(domain[0] / .001), .001))
    sim.add_tfsf_source(f0=10e9, margin=3)
    with pytest.raises(NotImplementedError, match="plane wave.*periodic.*'yz'.*use the uniform mesh"):
        getattr(sim, entry)(n_steps=1, skip_preflight=skip_preflight)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_bloch_graded_keeps_existing_refusal(entry, skip_preflight):
    sim = simulation(True, 20e9, (.020, .028, .021), dx=.001, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="periodic", z="periodic"))
    sim.add_tfsf_source(f0=10e9, margin=3, angle_deg=20)
    text = (r"TFSF oblique incidence \(angle_deg != 0\) is not yet supported"
            if skip_preflight else "TFSF oblique incidence is not yet supported on a nonuniform mesh")
    with pytest.raises(ValueError, match=text):
        getattr(sim, entry)(n_steps=1, skip_preflight=skip_preflight)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_multidevice_periodic_keeps_existing_refusal(entry, graded, skip_preflight):
    from tests.contracts.run_forward_census import build_b_periodic
    sim = build_b_periodic("graded" if graded else "uniform")
    kwargs = dict(devices=jax.devices("cpu")[:2], n_steps=1, skip_preflight=skip_preflight)
    assert len(kwargs["devices"]) == 2
    if entry == "forward":
        kwargs["distributed"] = True
    with pytest.raises(NotImplementedError, match="periodic / Bloch boundaries are not supported"):
        getattr(sim, entry)(**kwargs)
