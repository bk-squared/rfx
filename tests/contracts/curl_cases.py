"""Small asymmetric models used by the runtime E-curl contract and B3a records."""
import numpy as np

from rfx import Box, DebyePole, GaussianPulse, Simulation, Sphere, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec


def model(feature="plain", mesh="uniform", mode="3d", wall="pmc"):
    domain = (12.3e-3, 9.7e-3, 7.4e-3)
    if feature == "waveguide":
        domain = (30.3e-3, 12.7e-3, 6.4e-3)
    if feature == "msl":
        domain = (24.3e-3, 12.7e-3, 6.4e-3)
    if feature in ("tfsf", "coax"):
        domain = (24.3e-3, 12.7e-3, 12.4e-3)
    if mode != "3d":
        domain = (*domain[:2], 1e-3)
    boundary = wall
    if wall == "pmc":
        boundary = BoundarySpec(x=Boundary(lo="pmc", hi="pec"), y="pec", z="pec")
    if wall == "periodic":
        boundary = "pec"
    if feature in ("tfsf", "coax", "msl"):
        boundary = "cpml"
    if feature == "msl":
        boundary = BoundarySpec(x="cpml", y="pec", z="pec")
    if feature == "waveguide":
        boundary = BoundarySpec(x="cpml", y="pec", z="pec")
    if feature == "conformal":
        boundary = BoundarySpec(x=Boundary(lo="pec", hi="pec", conformal=True),
                                y="pec", z="pec")
    kwargs = {}
    if mesh == "graded":
        # Exact declared length, genuinely graded, distinct transverse lengths.
        n = round(domain[0] / 1e-3)
        length = 12e-3 if wall == "periodic" else domain[0]
        interior = np.linspace(.87, 1.13, n - 2)
        interior *= (length - 2e-3) / interior.sum()
        kwargs["dx_profile"] = np.r_[1e-3, interior, 1e-3]
    sim = Simulation(freq_max=20e9, domain=domain, dx=1e-3, mode=mode,
                     boundary=boundary, cpml_layers=3, **kwargs)
    if wall == "periodic":
        # Explicit periodic dx requires a commensurate length (#1221 B2).
        sim = Simulation(freq_max=20e9, domain=(12e-3, domain[1], domain[2]),
                         dx=1e-3, mode=mode, boundary=BoundarySpec(x="periodic", y="pec", z="pec"), cpml_layers=0, **kwargs)
    pulse = GaussianPulse(f0=12e9, bandwidth=1.2)
    component = "ex" if mode == "2d_tez" else "ez"
    z = 0.0 if mode != "3d" else 2.1e-3
    if feature in ("debye", "lorentz", "mixed", "subpixel"):
        material = dict(eps_r=2.4)
        if feature in ("debye", "mixed"):
            material["debye_poles"] = [DebyePole(delta_eps=1.1, tau=1e-11)]
        if feature in ("lorentz", "mixed"):
            material["lorentz_poles"] = [lorentz_pole(delta_eps=.6, omega_0=8e10, delta=2e9)]
        sim.add_material("block", **material)
        sim.add(Box((4.3e-3, 2.2e-3, 1.4e-3), (8.6e-3, 6.9e-3, 5.3e-3)), material="block")
    if feature == "conformal":
        sim.add(Sphere(center=(6.2e-3, 4.1e-3, 3.3e-3), radius=1.3e-3), material="pec")
    if feature in ("lumped", "wire"):
        kw = {"extent": 2e-3} if feature == "wire" else {}
        sim.add_port((2.2e-3, 3.1e-3, z), component, impedance=50.0,
                     waveform=pulse, **kw)
    elif feature == "msl":
        sim.add_material("substrate", eps_r=3.66)
        sim.add(Box((0, 0, 0), (24.3e-3, 12.7e-3, 1e-3)), material="substrate")
        sim.add(Box((1e-3, 5e-3, 1e-3), (23e-3, 7e-3, 2e-3)), material="pec")
        sim.add_msl_port(position=(2e-3, 6e-3, 0), width=2e-3, height=1e-3,
                         direction="+x", impedance=50.0, waveform=pulse)
    elif feature == "waveguide":
        sim.add_waveguide_port(6e-3, direction="+x", mode=(1, 0), mode_type="TE",
                               freqs=np.array([14e9, 16e9]), f0=15e9,
                               bandwidth=.5, probe_offset=4, ref_offset=2)
    elif feature == "coax":
        sim.add_coaxial_port((8e-3, 6e-3, 4e-3), pin_radius=1e-3,
                             outer_radius=3e-3, waveform=pulse)
    elif feature == "tfsf":
        sim.add_tfsf_source(f0=12e9, bandwidth=1.2)
    else:
        sim.add_source((2.2e-3, 3.1e-3, z), component, waveform=pulse, amplitude_kind="field")
    for pos in [(4.2e-3, 3.1e-3, z), (7.1e-3, 5.2e-3, z)]:
        sim.add_probe(pos, component)
    return sim
