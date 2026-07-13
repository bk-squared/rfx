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
"""

from __future__ import annotations

import numpy as np

from rfx import Box, GaussianPulse, Simulation


# Port decision tree:
#
# microstrip line                 -> sim.add_msl_port(...)
# hollow rectangular guide       -> sim.add_waveguide_port(...)
# coax                            -> sim.add_coaxial_port(...)
# generic lumped feed or load    -> sim.add_port(...)
# R/L/C component in the circuit -> sim.add_lumped_rlc(...)

PORT_POSITION = (9.3e-3, 9.3e-3, 9.3e-3)
S11_FREQS = np.asarray([4.5e9, 5.0e9, 5.5e9], dtype=np.float32)
S11_STEPS = 600


def build_generic_port_demo(*, add_component: bool) -> Simulation:
    """Build the tiny lumped-feed model used for the live S11 runs."""
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
        # add_lumped_rlc() represents a circuit component; it is not another
        # port.  Co-locating it with this feed makes its effect on S11 easy to
        # see.  Two non-zero values select the full series RLC update.
        sim.add_lumped_rlc(
            PORT_POSITION,
            component="ez",
            R=50.0,
            C=0.05e-12,
            topology="series",
        )
    return sim


def run_generic_s11(*, add_component: bool) -> np.ndarray:
    """Preflight and run one inexpensive generic-port reflection case."""
    sim = build_generic_port_demo(add_component=add_component)

    # Expect "All checks passed" for both generic-port models.  The explicit
    # call makes the complete report visible once, so run() skips its repeat.
    report = sim.preflight()
    if report:
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
    """Build a short, lossy microstrip line with the correct modal ports."""
    sim = Simulation(
        freq_max=6.0e9,
        domain=(0.020, 0.010, 0.004),
        dx=0.25e-3,
        boundary="cpml",
        cpml_layers=3,
    )
    sim.add_material("substrate", eps_r=3.2, sigma=0.01)

    # The metal and dielectric end before the absorbing cells.  The 1 mm
    # substrate has four cells through its height, and the side clearance is
    # greater than twice that height, so preflight should pass cleanly.
    sim.add(
        Box((0.75e-3, 0.75e-3, 0.75e-3), (19.25e-3, 9.25e-3, 1.25e-3)),
        material="pec",
    )
    sim.add(
        Box((0.75e-3, 0.75e-3, 1.25e-3), (19.25e-3, 9.25e-3, 2.25e-3)),
        material="substrate",
    )
    sim.add(
        Box((0.75e-3, 4.50e-3, 2.25e-3), (19.25e-3, 5.50e-3, 2.75e-3)),
        material="pec",
    )

    common = {
        "width": 1.0e-3,
        "height": 1.0e-3,
        "eps_r_sub": 3.2,
    }
    sim.add_msl_port(
        (4.0e-3, 5.0e-3, 1.25e-3),
        direction="+x",
        name="left",
        **common,
    )
    sim.add_msl_port(
        (16.0e-3, 5.0e-3, 1.25e-3),
        direction="-x",
        name="right",
        **common,
    )

    # A one-cell wire port on a microstrip badly undersamples the mode field
    # between strip and ground.  Use add_msl_port() for microstrip.
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
    """Build one SMA-sized coaxial probe entering through the top face."""
    sim = Simulation(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.4e-3,
        boundary="pec",
    )
    # The 1.42 mm annulus between the pin and outer conductor spans more than
    # 3.5 cells at this dx.  A coaxial TEM field needs radial resolution.
    sim.add_coaxial_port(
        (0.010, 0.010, 0.015),
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

    # Registered RLC elements affect run() and the uniform, single-device
    # forward() path.  To optimize R/L/C values with jax.grad, pass tracers as
    # forward(..., rlc_values_override={0: {"R": R, "C": C}}).  Registered
    # constants are still in the model but do not become variables themselves.

    microstrip = build_microstrip_ports()
    # Expect the general and MSL-family checks to pass.  Only construction and
    # preflight are needed here; a settled microstrip S-matrix costs much more
    # than the small generic-port demonstrations above.
    microstrip_report = microstrip.preflight()
    microstrip_route = microstrip.preflight_sparameters(calculator="msl")
    print(
        "Microstrip port setup ready: "
        f"{not microstrip_report and not microstrip_route}"
    )

    waveguide = build_waveguide_ports()
    # The 20 mm broad wall gives TE10 a 7.49 GHz cutoff, so the 8 GHz source
    # propagates.  Both preflight calls should pass.
    waveguide_report = waveguide.preflight()
    waveguide_route = waveguide.preflight_sparameters(calculator="waveguide")
    print(
        "Waveguide port setup ready: "
        f"{not waveguide_report and not waveguide_route}"
    )

    coaxial = build_coaxial_port()
    # Simulation.run() does not accept add_coaxial_port().  Because this model
    # deliberately ends after construction, general preflight should report
    # that there is no generic run source.  The coaxial-family check should
    # still pass and confirms the port was routed to its dedicated calculator.
    coaxial_report = coaxial.preflight()
    coaxial_route = coaxial.preflight_sparameters(calculator="coaxial")
    expected_build_only_advisory = bool(coaxial_report.by_code("no_sources"))
    print(f"Coax build-only advisory observed: {expected_build_only_advisory}")
    print(f"Coaxial port setup ready: {not coaxial_route}")


if __name__ == "__main__":
    main()
