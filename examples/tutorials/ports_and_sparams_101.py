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

from rfx import (
    Box,
    GaussianPulse,
    Simulation,
    realized_pec_edge_masks,
    realized_wall_planes,
)


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
    if len(report):   # PreflightReport refuses bool() (#980)
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
    #
    # FOIL IS A SHEET, A PLATE IS A VOLUME.  A microstrip ground plane and its
    # trace are etched copper foil — tens of microns on a 1 mm board — so they
    # are declared with ``add_thin_conductor`` on ZERO-THICKNESS Boxes.  A
    # sheet is a footprint on ONE node plane: it owns no cell, writes no
    # permittivity, zeroes the two in-plane E components on its plane, and
    # leaves the E through it live.  That is the whole model of a foil, and it
    # is the answer for every etched conductor in this repository.
    #
    # ``sim.add(Box(...), material="pec")`` is the OTHER declaration — a
    # VOLUME.  A volume shorts every E edge incident to one of its cells, so a
    # drawn slab realizes walls on BOTH of its faces with the interior shorted.
    # That is right for a plate, an iris, a post or a machined wall, and wrong
    # for foil.  Both of these conductors used to be drawn as 0.5 mm PEC slabs
    # (two cells each), which put 1 mm of solid metal into a 2 mm stack-up.
    # Lattice ownership contract, #931 §1.1-§1.3.
    sim.add_thin_conductor(
        Box((0.75e-3, 0.75e-3, 1.25e-3), (19.25e-3, 9.25e-3, 1.25e-3))
    )
    sim.add(
        Box((0.75e-3, 0.75e-3, 1.25e-3), (19.25e-3, 9.25e-3, 2.25e-3)),
        material="substrate",
    )
    # A sheet Box footprint is sampled CLOSED on its two in-plane axes, so the
    # drawn 1.0 mm trace width is realized as 1.0 mm (nodes 4.50 .. 5.50 mm at
    # dx = 0.25 mm).  The half-open node sampling this file used to get gave
    # 0.75 mm of realized trace against a 1.0 mm declaration.
    sim.add_thin_conductor(
        Box((0.75e-3, 4.50e-3, 2.25e-3), (19.25e-3, 5.50e-3, 2.25e-3))
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
    #
    # Build-time check (no solve): the two declared foil planes must be the
    # two realized tangential-wall planes along z, and the gap between them
    # must be the 1.0 mm the ports were told.  Read from the contract's own
    # realization (``realized_pec_edge_masks`` then ``realized_wall_planes``,
    # #931 §1.7) over the arrays the assembly hands the stepper, so the check
    # and the solve cannot drift apart.  ``_assemble_materials`` is private
    # only because the realized edge set has no public accessor yet; the two
    # functions it feeds are public (``from rfx import ...``).
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    grid = sim._build_grid()
    sheets: list = []
    _mat, _deb, _lor, pec_cells, _s, _w, _c = sim._assemble_materials(
        grid, pec_sheets=sheets)
    walls = realized_wall_planes(
        realized_pec_edge_masks(pec_cells, sheets=sheets,
                                periodic=sim._periodic_flags()), 2)
    # Node indices are offset by the CPML pad, so read the declared planes off
    # the built grid's own z node line rather than from z / dx.
    z_nodes = np.asarray(coords_from_uniform_grid(grid).z, dtype=float)
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
    # Expect the MSL-family checks to pass.  Only construction and preflight
    # are needed here; a settled microstrip S-matrix costs much more than the
    # small generic-port demonstrations above.
    #
    # The general report is NOT empty on this model and readiness is read off
    # report.ok (no error-severity finding), the same way the waveguide leg
    # below does it.  What it draws is one advisory, twice: preflight
    # assembles the material arrays without a PEC-sheet collector, so the
    # sheet-declared ground and trace are absent from the cell mask it reads
    # and rfx says so instead of quietly dropping them.  That is a gap in
    # preflight's own plumbing (#931 §6), not a finding about this geometry —
    # the realized planes were checked at build time in
    # build_microstrip_ports() and are exactly the two declared foils.  When
    # preflight reads the realized edge set the advisory goes away and the
    # report is empty again.
    microstrip_report = microstrip.preflight()
    microstrip_route = microstrip.preflight_sparameters(calculator="msl")
    print(
        "Microstrip port setup ready: "
        f"{microstrip_report.ok and microstrip_route.ok}"
    )

    waveguide = build_waveguide_ports()
    # The 20 mm broad wall gives TE10 a 7.49 GHz cutoff, so the 8 GHz source
    # propagates.  Neither preflight call reports an error.
    #
    # The waveguide S-parameter preflight also runs three SETUP audits that
    # speak in input units, and two of them have something to say about this
    # small teaching model, so readiness is read off report.ok (no
    # error-severity finding) rather than off an empty report:
    #
    #   * record_shorter_than_far_boundary_round_trip -- at the default
    #     num_periods=20 the record is 2.4 far-boundary round trips long, and
    #     the message names the num_periods that reaches 3.
    #   * port_index_mirror_known_e_plane_offset -- informational: the '-'
    #     port's E correction sits one cell inward of its mirror image, a known
    #     constant of the source, not an asymmetry in this geometry.
    #
    # The third audit, layout_measured_from_band_low_edge, is silent here and
    # that is the interesting part: it reports the layout in guide wavelengths
    # only when the band's lowest bin is within 6 % of the port's own discrete
    # cutoff, and 8.0 / 7.4871 = 1.0685 sits just outside.  Move the source
    # down toward 7.8 GHz and the note appears.
    #
    # Two codes this model does NOT draw, for completeness: a band whose lowest
    # frequency sits at or below the port's own cutoff reports
    # record_far_boundary_band_below_cutoff and no ratio, and a grid or mode
    # solve that fails reports waveguide_setup_audit_skipped instead of raising.
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
    # Simulation.run() does not accept add_coaxial_port().  Because this model
    # deliberately ends after construction, general preflight should report
    # that there is no generic run source.  The coaxial-family check should
    # still pass and confirms the port was routed to its dedicated calculator.
    coaxial_report = coaxial.preflight()
    coaxial_route = coaxial.preflight_sparameters(calculator="coaxial")
    expected_build_only_advisory = bool(coaxial_report.by_code("no_sources"))
    print(f"Coax build-only advisory observed: {expected_build_only_advisory}")
    print(f"Coaxial port setup ready: {not len(coaxial_route)}")


if __name__ == "__main__":
    main()
