"""Materials and dispersion — choose what the fields travel through.

This tutorial makes one material decision at a time:

* Use a library name when a standard material already fits the job.
* Register a custom dielectric when you have measured permittivity or loss.
* Add Debye or Lorentz poles when permittivity changes with frequency.

The important loss rule is simple. A perfectly lossless dielectric in an open
domain can keep ringing forever in the model, so a resonance calculation can
report an artificially infinite Q. Real laminates dissipate energy. When a
datasheet gives loss tangent, convert it to conductivity at the frequency of
interest before interpreting Q.

Run as::

    python examples/tutorials/materials_and_dispersion.py

Five very small simulations run at the end: a library material, lossless and
lossy custom materials, and one material for each dispersion model.
"""

from __future__ import annotations

import numpy as np

from rfx import Box, DebyePole, LorentzPole, MATERIAL_LIBRARY, Simulation


DOMAIN = (0.016, 0.016, 0.016)
DX = 1.0e-3
FREQ_MAX = 8.0e9
N_STEPS = 100

EPSILON_0 = 8.854_187_812_8e-12  # vacuum permittivity, F/m
FR4_EPS_R = 4.4
FR4_TAN_DELTA = 0.02
LOSS_FREQUENCY = 5.0e9


def make_sim() -> Simulation:
    """Build the small open domain shared by every material example."""
    return Simulation(
        freq_max=FREQ_MAX,
        domain=DOMAIN,
        dx=DX,
        boundary="cpml",
        cpml_layers=3,
    )


def add_sample(sim: Simulation, material: str) -> None:
    """Place one material sample, source, and probe in a simulation."""
    sim.add(Box((0.004, 0.004, 0.004), (0.012, 0.012, 0.012)), material=material)
    sim.add_source((0.008, 0.008, 0.008), component="ez")
    sim.add_probe((0.010, 0.008, 0.008), component="ez")


def peak_ez(sim: Simulation) -> float:
    """Run a preflighted simulation and return its largest probe magnitude."""
    result = sim.run(
        n_steps=N_STEPS,
        compute_s_params=False,
        skip_preflight=True,
    )
    trace = np.asarray(result.time_series)[:, 0]
    return float(np.max(np.abs(trace)))


def has_infinite_q_advisory(findings: list[str]) -> bool:
    """Recognize the plain-language loss advisory returned by preflight."""
    return any("artificially infinite q" in str(item).lower() for item in findings)


def main() -> None:
    # MATERIAL_LIBRARY is part of the public rfx package. Its keys are the
    # names accepted by sim.add(..., material="name") without registration.
    # Printing the sorted keys is safer than copying a list that may grow.
    print(f"Public material names: {sorted(MATERIAL_LIBRARY)}")

    # Library material: pass the string directly. There is no add_material()
    # call here; "fr4" resolves from the public catalog.
    library_sim = make_sim()
    add_sample(library_sim, "fr4")

    # Custom material: eps_r sets stored electric energy. sigma sets loss in
    # siemens per metre. Setting sigma to zero deliberately makes this lossless.
    lossless_sim = make_sim()
    lossless_sim.add_material("fr4_lossless", eps_r=FR4_EPS_R, sigma=0.0)
    add_sample(lossless_sim, "fr4_lossless")

    # Expect a preflight advisory now: the open box contains a dielectric with
    # no loss, so a resonance study would give an artificially infinite Q.
    lossless_findings = lossless_sim.preflight()

    # A datasheet loss tangent becomes conductivity at the frequency where the
    # value is specified:
    # sigma = 2 * pi * f * epsilon_0 * eps_r * tan_delta
    fr4_sigma = (
        2.0
        * np.pi
        * LOSS_FREQUENCY
        * EPSILON_0
        * FR4_EPS_R
        * FR4_TAN_DELTA
    )
    lossy_sim = make_sim()
    lossy_sim.add_material("fr4_lossy", eps_r=FR4_EPS_R, sigma=fr4_sigma)
    add_sample(lossy_sim, "fr4_lossy")

    # Expect no infinite-Q advisory here: sigma represents the laminate loss.
    lossy_findings = lossy_sim.preflight()

    # Debye relaxation is useful when polarization follows the field with a
    # characteristic time tau. eps_r is the high-frequency baseline and
    # delta_eps is the pole's added low-frequency permittivity.
    debye_sim = make_sim()
    debye_sim.add_material(
        "debye_demo",
        eps_r=2.5,
        debye_poles=[DebyePole(delta_eps=1.5, tau=8.0e-12)],
    )
    add_sample(debye_sim, "debye_demo")

    # A Lorentz pole describes a damped material resonance. omega_0 is its
    # angular frequency, delta is damping, and kappa is coupling strength.
    omega_0 = 2.0 * np.pi * 6.0e9
    lorentz_sim = make_sim()
    lorentz_sim.add_material(
        "lorentz_demo",
        eps_r=2.0,
        lorentz_poles=[
            LorentzPole(
                omega_0=omega_0,
                delta=2.0e9,
                kappa=1.0 * omega_0**2,
            )
        ],
    )
    add_sample(lorentz_sim, "lorentz_demo")

    # The catalog material and both dispersive materials already include a
    # loss mechanism, so these preflight calls should not report infinite Q.
    library_sim.preflight()
    debye_sim.preflight()
    lorentz_sim.preflight()

    print(f"Lossless advisory observed: {has_infinite_q_advisory(lossless_findings)}")
    print(f"Lossy advisory observed: {has_infinite_q_advisory(lossy_findings)}")
    print(f"FR4 conductivity at 5 GHz: {fr4_sigma:.4e} S/m")

    simulations = [
        ("library fr4", library_sim),
        ("lossless custom", lossless_sim),
        ("lossy custom", lossy_sim),
        ("Debye", debye_sim),
        ("Lorentz", lorentz_sim),
    ]
    for label, sim in simulations:
        print(f"{label} peak |Ez|: {peak_ez(sim):.6e}")


if __name__ == "__main__":
    main()
