"""API guard tests for the canonical Phase-1 z-slab lane."""

import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def test_subgrid_touching_cpml_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml", dx=2e-3)
    with pytest.raises(ValueError, match="boundary='pec' only|CPML/UPML coexistence|all-PEC BoundarySpec"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3)


def test_subgrid_accepts_all_pec_boundaryspec():
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=BoundarySpec.uniform("pec"),
        dx=2e-3,
    )
    sim.add_refinement(z_range=(0.01, 0.03), ratio=3)

    assert sim._refinement["ratio"] == 3


@pytest.mark.parametrize(
    "boundary",
    [
        BoundarySpec(x="pec", y="pec", z=Boundary(lo="pec", hi="cpml")),
        BoundarySpec(x="pec", y="pec", z=Boundary(lo="pmc", hi="pec")),
        BoundarySpec(x="periodic", y="pec", z="pec"),
        BoundarySpec(
            x="pec",
            y="pec",
            z=Boundary(lo="cpml", hi="pec", lo_thickness=4),
        ),
    ],
)
def test_subgrid_rejects_non_pec_boundaryspec(boundary):
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary=boundary,
        dx=2e-3,
    )
    with pytest.raises(ValueError, match="all-PEC BoundarySpec"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3)


def test_subgrid_rejects_late_periodic_axes_on_run():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("x")

    with pytest.raises(ValueError, match="all-PEC BoundarySpec"):
        sim.run(n_steps=10)


def test_partial_xy_refinement_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    with pytest.raises(ValueError, match="does not support xy_margin"):
        sim.add_refinement(z_range=(0.01, 0.03), ratio=3, xy_margin=1e-3)


def test_source_outside_zslab_fails():
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.020), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.032), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.032), component="ez")
    with pytest.raises(ValueError, match="outside .*z-slab fine grid|Widen z_range"):
        sim.run(n_steps=10)


@pytest.mark.parametrize(
    ("attach_feature", "message"),
    [
        (
            lambda sim: sim.add_ntff_box((0.008, 0.008, 0.008), (0.032, 0.032, 0.032)),
            "does not support NTFF",
        ),
        (
            lambda sim: sim.add_dft_plane_probe(axis="z", coordinate=0.018, component="ez"),
            "does not support DFT plane probes",
        ),
        (
            lambda sim: sim._waveguide_ports.append(object()),
            "does not support waveguide ports",
        ),
        (
            lambda sim: setattr(sim, "_tfsf", object()),
            "does not support TFSF sources",
        ),
        (
            lambda sim: sim.add_lumped_rlc((0.02, 0.02, 0.02), component="ez", R=50.0),
            "does not support lumped RLC",
        ),
        (
            lambda sim: sim.add_coaxial_port((0.02, 0.02, 0.04), face="top"),
            "does not support coaxial ports",
        ),
        (
            lambda sim: sim.add_floquet_port(position=0.01, axis="z"),
            "does not support Floquet ports",
        ),
    ],
)
def test_unsupported_phase1_features_fail_fast(attach_feature, message):
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    attach_feature(sim)
    with pytest.raises(ValueError, match=message):
        sim.run(n_steps=10)


@pytest.mark.parametrize(
    ("attach_port", "message"),
    [
        (
            lambda sim: sim.add_port(
                position=(0.02, 0.02, 0.02),
                component="ez",
                impedance=50.0,
            ),
            "soft point sources only|impedance point ports",
        ),
        (
            lambda sim: sim.add_port(
                position=(0.02, 0.02, 0.018),
                component="ez",
                impedance=50.0,
                extent=0.004,
            ),
            "soft point sources only|wire/extent ports",
        ),
    ],
)
def test_subgrid_rejects_impedance_ports(attach_port, message):
    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    attach_port(sim)
    with pytest.raises(ValueError, match=message):
        sim.run(n_steps=10)
