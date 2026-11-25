"""Ports and S-parameters -- match the port to the structure.

A port is both an excitation and a field measurement.  Its field shape must
match the transmission structure, or the reported reflection includes the
bad launch as well as the device response.

This tutorial runs two very small single-cell-port S11 calculations, then
builds representative microstrip, rectangular-waveguide, and coaxial models
and checks each setup without running the larger calculations.

One honest limitation applies to strongly reflecting devices.  In a
single-run reflection measurement, voltage and current can both pass through
a standing-wave null at the port at some frequencies.  The extractor marks
those bins unreliable and warns.  Do not read ``|S11| > 1`` there as physics.

Run as::

    python examples/tutorials/ports_and_sparams_101.py

The lumped R+C sits one cell from the feed because a component inside the feed cell is in parallel with the source, which changes the drive level but not S11.
"""

from __future__ import annotations

import numpy as np

from rfx import (
    Box,
    GaussianPulse,
    Simulation,
)



PORT_POSITION = (9.3e-3, 9.3e-3, 9.3e-3)
RLC_POSITION = (9.3e-3 + 0.020 / 15, 9.3e-3, 9.3e-3)
S11_FREQS = np.asarray([4.5e9, 5.0e9, 5.5e9], dtype=np.float32)
S11_STEPS = 600


def build_generic_port_demo(*, add_component: bool) -> Simulation:
    """Build the tiny lumped-feed model used for the live S11 runs.

    When the series R+C component is enabled on a cell that carries no port conductance, resistance below roughly 180 ohms can produce non-finite fields.
    """
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.020 / 15,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_port(
        PORT_POSITION,
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=5.0e9, bandwidth=0.9),
    )
    if add_component:
        sim.add_lumped_rlc(
            RLC_POSITION,
            component="ez",
            R=500.0,
            C=0.20e-12,
            topology="series",
        )
    return sim


def run_generic_s11(*, add_component: bool) -> np.ndarray:
    """Preflight and run one inexpensive generic-port reflection case."""
    sim = build_generic_port_demo(add_component=add_component)

    report = sim.preflight()
    if any(issue.code != "thin_absorber" for issue in report):
        raise RuntimeError("Generic-port setup has unexpected advisories")

    result = sim.run(
        n_steps=S11_STEPS,
        compute_s_params=True,
        s_param_freqs=S11_FREQS,
        skip_preflight=True,
    )
    if result.s_params is None:
        raise RuntimeError("Generic port did not produce S-parameters")
    return np.abs(np.asarray(result.s_params)[0, 0, :])


def build_microstrip_ports() -> Simulation:
    """Build a short, lossy microstrip line with the correct modal ports.

    The metal and dielectric end before the absorbing cells. The 1 mm
    substrate's x and y faces are realized 87.5 µm (x) and 74.4 µm (y) inside
    the drawn faces, where sheet-edge registration takes precedence; the side
    clearance exceeds twice the substrate height.

    General preflight reports ``line_stub_behind_port`` for both short feeds
    and ``msl_port_geometry`` for the automatic probe-offset interval solve,
    which is skipped on the graded propagation axis; the MSL S-parameter
    preflight reports all checks passed. The three-layer absorber, the
    zero-thickness conductors and the substrate realization also give
    advisories. The tutorial only builds this setup; it does not quote a
    settled microstrip S-matrix.
    """
    from rfx.mesh_edges import edge_aware_profiles

    domain = (0.044, 0.034, 0.004)
    dx = 0.25e-3
    margin = 12e-3
    ground = Box((0.75e-3, 0.75e-3, 1.25e-3), (19.25e-3, 9.25e-3, 1.25e-3))
    substrate = Box((0.75e-3, 0.75e-3, 1.25e-3), (19.25e-3, 9.25e-3, 2.25e-3))
    trace = Box((0.75e-3, 4.50e-3, 2.25e-3), (19.25e-3, 5.50e-3, 2.25e-3))
    def shifted(box):
        return Box(tuple(v + margin if a < 2 else v for a, v in enumerate(box.corner_lo)),
                   tuple(v + margin if a < 2 else v for a, v in enumerate(box.corner_hi)))

    ground, substrate, trace = map(shifted, (ground, substrate, trace))
    edge_dx = dx
    pad = 8 * edge_dx

    def interior_box(box):
        return Box(tuple(v - pad if a < 2 else v for a, v in enumerate(box.corner_lo)),
                   tuple(v - pad if a < 2 else v for a, v in enumerate(box.corner_hi)))

    profiles = edge_aware_profiles(
        (domain[0] - 2 * pad, domain[1] - 2 * pad, domain[2]), 0.25e-3, boundary_cell=edge_dx,
        sheets=[interior_box(ground), interior_box(trace)],
        solids=[interior_box(substrate)],
        faces={"x": [4e-3 + margin - pad, 16e-3 + margin - pad], "y": [5e-3 + margin - pad]}, axes="xy",
    )
    profiles = {key: np.concatenate((np.full(8, edge_dx), cells, np.full(8, edge_dx)))
                for key, cells in profiles.items()}
    sim = Simulation(
        freq_max=6.0e9,
        domain=domain,
        dx=dx,
        dz_profile=np.full(16, 0.25e-3),
        **profiles,
        boundary="cpml",
        cpml_layers=3,
    )
    sim.add_material("substrate", eps_r=3.2, sigma=0.01)

    sim.add_thin_conductor(ground)
    sim.add(
        substrate,
        material="substrate",
    )
    sim.add_thin_conductor(trace)

    common = {
        "width": 1.0e-3,
        "height": 1.0e-3,
        "eps_r_sub": 3.2,
    }
    sim.add_msl_port(
        (4.0e-3 + margin, 5.0e-3 + margin, 1.25e-3),
        direction="+x",
        name="left",
        **common,
    )
    sim.add_msl_port(
        (16.0e-3 + margin, 5.0e-3 + margin, 1.25e-3),
        direction="-x",
        name="right",
        **common,
    )

    record = sim.realized_geometry()
    walls = record.wall_planes(2)
    z_nodes = record.nodes[2]
    k_gnd = int(np.argmin(np.abs(z_nodes - 1.25e-3)))
    k_trace = int(np.argmin(np.abs(z_nodes - 2.25e-3)))
    if walls != sorted({k_gnd, k_trace}):
        raise RuntimeError(
            f"realized z wall planes {walls} != declared "
            f"{sorted({k_gnd, k_trace})} — the ground and trace sheets did "
            "not land on the substrate faces"
        )
    gap = float(z_nodes[k_trace] - z_nodes[k_gnd])
    if abs(gap - 1.0e-3) > 1e-12:
        raise RuntimeError(
            f"realized strip-to-ground gap {gap * 1e3:.4f} mm != the 1.000 mm "
            "height the MSL ports were declared with"
        )
    print(
        f"Microstrip realized conductor planes along z: {walls} "
        f"(ground {float(z_nodes[k_gnd]) * 1e3:.2f} mm, trace "
        f"{float(z_nodes[k_trace]) * 1e3:.2f} mm, strip-to-ground "
        f"{gap * 1e3:.2f} mm = the declared port height)"
    )
    return sim


def build_waveguide_ports() -> Simulation:
    """Build a two-port hollow rectangular guide above TE10 cutoff."""
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.050, 0.020, 0.010),
        dx=1.0e-3,
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
        cpml_layers=4,
    )
    common = {
        "y_range": (0.0, 0.020),
        "z_range": (0.0, 0.010),
        "mode": (1, 0),
        "mode_type": "TE",
        "freqs": np.asarray([8.0e9], dtype=np.float32),
        "f0": 8.0e9,
    }
    sim.add_waveguide_port(0.010, direction="+x", name="left", **common)
    sim.add_waveguide_port(0.040, direction="-x", name="right", **common)
    return sim


def build_coaxial_port() -> Simulation:
    """Build a coax declaration for the line calculators.

    Experimental uniform single-device run()/forward() also build the feed line
    and read S11 at position, using the built impedance renormalized to impedance.
    They require sufficient line length behind the reference plane, one coax port,
    top or bottom feed, float32 second-order 3-D Yee, CPML on every face and no
    other S-port family, periodic axes, TF/SF, conformal PEC, subpixel smoothing
    or material overrides. A declared NTFF box must contain the whole port.
    """
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.008, 0.008, 0.080),
        dx=0.4e-3,
        boundary="cpml",
    )
    sim.add_coaxial_port(
        (0.004, 0.004, 0.030),
        face="top",
        pin_length=5.0e-3,
        pin_radius=0.635e-3,
        outer_radius=2.055e-3,
        impedance=50.0,
    )
    return sim


def main() -> None:
    unloaded_s11 = run_generic_s11(add_component=False)
    loaded_s11 = run_generic_s11(add_component=True)

    print("Generic lumped-port S11:")
    for freq, unloaded, loaded in zip(S11_FREQS, unloaded_s11, loaded_s11):
        print(
            f"  {freq / 1e9:.1f} GHz: "
            f"without component={unloaded:.4f}, series-RC load={loaded:.4f}"
        )
    max_change = float(np.max(np.abs(loaded_s11 - unloaded_s11)))
    print(f"RLC changed max |S11| by: {max_change:.4f}")


    microstrip = build_microstrip_ports()
    microstrip_report = microstrip.preflight()
    microstrip_route = microstrip.preflight_sparameters(calculator="msl")
    print(
        "Microstrip port setup ready: "
        f"{microstrip_report.ok and microstrip_route.ok}"
    )

    waveguide = build_waveguide_ports()
    waveguide_report = waveguide.preflight()
    waveguide_route = waveguide.preflight_sparameters(calculator="waveguide")
    print(
        "Waveguide port setup ready: "
        f"{waveguide_report.ok and waveguide_route.ok}"
    )
    print(
        "Waveguide setup advisories: "
        + ", ".join(sorted({issue.code for issue in waveguide_route}))
    )

    coaxial = build_coaxial_port()
    coaxial_report = coaxial.preflight()
    coaxial_route = coaxial.preflight_sparameters(calculator="coaxial")
    expected_build_only_advisory = bool(coaxial_report.by_code("no_sources"))
    print(f"Coax build-only advisory observed: {expected_build_only_advisory}")
    print(f"Coaxial port setup ready: {not len(coaxial_route)}")


if __name__ == "__main__":
    main()
