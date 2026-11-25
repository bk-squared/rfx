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

The setup prints the drawn diameter and the solved x/y/z spans of the
centre-sampled PEC sphere from its grid record. Quote those solved spans
with the cell size when describing the sphere used by the setup check.

Record length: the 800-step record lies past step 524, which the run's
record-end guard prints for the NTFF box. That step is the drive's end plus a
light-speed transit to the box, a geometric bound; it does not show that the
sphere's response has died away.

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
from rfx.core.yee import MaterialArrays


C0 = 299_792_458.0
F0 = 3.0e9
FREQ_MAX = 1.5 * F0
BANDWIDTH = 0.5

DOMAIN = 0.100
DX = (C0 / F0) / 40.0
CPML_LAYERS = 8
N_STEPS = 800

RADIUS = 6 * DX
CENTER = (20 * DX,) * 3
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

    sim.add_tfsf_source(
        f0=F0,
        bandwidth=BANDWIDTH,
        polarization="ez",
        direction="+x",
        margin=3,
        closed_box=True,
    )
    return sim


def build_rcs_inputs() -> tuple[Grid, MaterialArrays]:
    """Rasterize the conducting sphere for the functional RCS API.

    The conductivity fill approximates the conductor as a lossy volume rather than an exact PEC edge set.
    """
    grid = Grid(
        freq_max=FREQ_MAX,
        domain=(DOMAIN, DOMAIN, DOMAIN),
        dx=DX,
        cpml_layers=CPML_LAYERS,
    )

    eps_r, sigma = rasterize(grid, [(SPHERE, 1.0, 1.0e7)])
    materials = MaterialArrays(
        eps_r=eps_r,
        sigma=sigma,
        mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
    )
    return grid, materials


def main() -> None:
    """Run the sphere scattering tutorial and print its results.

    For the +x incident wave, polar angle 90 degrees and azimuth 180 degrees denote backscatter.
    """
    sim = build_preflight_model()

    report = sim.preflight()
    if len(report):
        raise RuntimeError("Sphere scattering setup has unexpected advisories")
    print(f"TFSF plane-wave setup ready: {not len(report)}")

    grid, materials = build_rcs_inputs()

    record = sim.realized_geometry()
    solved_mm = tuple(axis.extent_m * 1e3 for axis in record.entities[0].axes)
    print(f"Sphere drawn diameter: {2 * RADIUS * 1e3:.5f} mm; "
          f"PEC solved spans x/y/z: {solved_mm} mm; cells: {grid.dx * 1e3:.5f} mm")
    n_pec = int(record.pec_mask.sum())
    n_sigma = int((np.asarray(materials.sigma) > 0.0).sum())
    n_analytic = (4.0 / 3.0) * np.pi * RADIUS**3 / DX**3
    print(
        f"Sphere cells: PEC volume (centre-sampled) {n_pec}, sigma fill "
        f"(node-sampled) {n_sigma}, analytic {n_analytic:.1f}"
    )
    for label, n in (("PEC volume", n_pec), ("sigma fill", n_sigma)):
        if abs(n - n_analytic) > 0.05 * n_analytic:
            raise RuntimeError(
                f"{label} sphere occupies {n} cells against an analytic "
                f"{n_analytic:.1f} — more than the 5 % a rim-cell difference "
                "explains"
            )

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

    report_record_coverage("RCS run", result.settling_witness)
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


def report_record_coverage(label, witness):
    """Print the run's record-end guard and stop if a monitor is not covered."""
    rows = (witness or {}).get("coverage", ())
    print(f"  {label}: settling witness {(witness or {}).get('status')}")
    for row in rows:
        print(f"    {row['name']}: {row['verdict']} (drive past the box by step {row['end_step']}, record {row['n_record']} steps)")
    uncovered = [row["name"] for row in rows if row["verdict"] != "covered"]
    if not rows or uncovered:
        raise RuntimeError(f"{label}: the record ends before the drive has passed {uncovered or 'the monitors'}")


if __name__ == "__main__":
    main()
