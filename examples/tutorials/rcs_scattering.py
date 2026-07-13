"""PEC-sphere RCS -- subtract the empty reference before reading a pattern.

Radar cross section turns the scattered far field into an equivalent area.
This tutorial makes one important measurement decision: remove the incident
field from the collection box before interpreting any angle.

Always pass ``subtract_incident_reference=True`` — it runs a second empty
reference simulation and subtracts the incident field the collection box
would otherwise integrate.

The example uses a small sphere with ``ka`` close to one.  No public Mie
helper is exposed by rfx, so the printed ``pi*r^2`` geometric-optics limit is
only a scale check.  That limit describes spheres much larger than a
wavelength; it is not an accurate value for this electrical size.

One honest limitation: deep pattern nulls very close to the collection box
carry larger errors.  This is a known limitation of near-field-to-far-field
integration; do not tune the setup trying to remove it.

Run as::

    python examples/tutorials/rcs_scattering.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx import Grid, Simulation, Sphere, compute_rcs
from rfx.geometry import rasterize
from rfx.materials import MaterialArrays


C0 = 299_792_458.0
F0 = 3.0e9
FREQ_MAX = 1.5 * F0
BANDWIDTH = 0.5

DOMAIN = 0.100
DX = (C0 / F0) / 40.0
CPML_LAYERS = 8
N_STEPS = 500

RADIUS = 15.9e-3
CENTER = (DOMAIN / 2.0,) * 3
SPHERE = Sphere(center=CENTER, radius=RADIUS)

THETA_OBS = np.asarray([np.pi / 2.0])
PHI_OBS = np.linspace(0.0, np.pi, 13)


def build_preflight_model() -> Simulation:
    """Mirror the scattering geometry in Simulation for public checks."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(DOMAIN, DOMAIN, DOMAIN),
        dx=DX,
        boundary="cpml",
        cpml_layers=CPML_LAYERS,
    )
    sim.add(SPHERE, material="pec")

    # This is the plane wave used for the setup check.  The production call
    # below creates the same +x, Ez-polarized TFSF illumination itself.
    sim.add_tfsf_source(
        f0=F0,
        bandwidth=BANDWIDTH,
        polarization="ez",
        direction="+x",
        margin=3,
    )
    return sim


def build_rcs_inputs() -> tuple[Grid, MaterialArrays]:
    """Rasterize the conducting sphere for the functional RCS API."""
    grid = Grid(
        freq_max=FREQ_MAX,
        domain=(DOMAIN, DOMAIN, DOMAIN),
        dx=DX,
        cpml_layers=CPML_LAYERS,
    )

    # A large conductivity represents the PEC sphere on the material grid.
    eps_r, sigma = rasterize(grid, [(SPHERE, 1.0, 1.0e7)])
    materials = MaterialArrays(
        eps_r=eps_r,
        sigma=sigma,
        mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
    )
    return grid, materials


def main() -> None:
    sim = build_preflight_model()

    # Expect "All checks passed": the sphere stays away from the absorbing
    # cells, and the TFSF boundary has vacuum on both sides.  compute_rcs() is
    # a functional API, so this matching Simulation makes its public setup
    # checks visible without reaching into private state.
    report = sim.preflight()
    if report:
        raise RuntimeError("Sphere scattering setup has unexpected advisories")
    print(f"TFSF plane-wave setup ready: {not report}")

    grid, materials = build_rcs_inputs()

    # compute_rcs() creates the TFSF source and collection box, runs the
    # target, and transforms the collected fields to the requested angles.
    # Always request the empty reference: without it, residual incident field
    # is included in the angular pattern.
    result = compute_rcs(
        grid,
        materials,
        N_STEPS,
        f0=F0,
        bandwidth=BANDWIDTH,
        theta_inc=0.0,
        phi_inc=0.0,
        polarization="ez",
        theta_obs=THETA_OBS,
        phi_obs=PHI_OBS,
        freqs=np.asarray([F0]),
        boundary="cpml",
        cpml_layers=CPML_LAYERS,
        tfsf_margin=3,
        ntff_offset=1,
        subtract_incident_reference=True,
    )

    # For +x incidence, (theta=pi/2, phi=pi) is backscatter.  It is the final
    # sample in this H-plane cut, after the empty-reference subtraction.
    backscatter = float(np.asarray(result.rcs_linear)[0, 0, -1])
    if not np.isfinite(backscatter) or backscatter <= 0.0:
        raise RuntimeError("Backscatter RCS is not finite and positive")

    wavelength = C0 / F0
    ka = 2.0 * np.pi * RADIUS / wavelength
    geometric_optics = np.pi * RADIUS**2
    ratio = backscatter / geometric_optics

    print("Incident-reference subtraction enabled: True")
    print(f"Electrical size ka: {ka:.3f}")
    print(f"Backscatter RCS: {backscatter:.6e} m^2")
    print(f"Backscatter RCS: {10.0 * np.log10(backscatter):.3f} dBsm")
    print(f"Geometric-optics limit pi*r^2: {geometric_optics:.6e} m^2")
    print(f"Backscatter / geometric-optics limit: {ratio:.3f}")
    print(
        "Accuracy note: ka is about 1, not much greater than 1, so pi*r^2 "
        "is only a scale check here."
    )
    print(
        "Deep-null limitation: values near a pattern null can carry larger "
        "near-field-to-far-field integration errors."
    )


if __name__ == "__main__":
    main()
