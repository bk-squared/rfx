# rfx Simulation Methodology

This guide describes how to configure and assess an rfx simulation. It does not
define one mesh or error tolerance for every model. Choose those values from the
observable, frequency band, geometry, material representation, and applicable
validation result.

## 1. Define the result before building the model

Record the following before choosing a grid:

- the frequency band and requested observable;
- the source or port family and its reference impedance or normalization;
- the geometry dimensions and material model that must be resolved;
- the boundary condition and required clearance from absorbers;
- the comparison metric, reference result, and acceptance threshold.

Use a soft field source plus a probe for resonance or ring-down studies where a
port load would alter the response. Use a documented port and its matching
S-parameter calculator when the result must be impedance referenced. A field
peak from a soft-source run is not return loss.

The current port-specific restrictions are listed in
[S-parameter support](sparameter_support_matrix.md). The broader solver and mesh
restrictions are listed in [support boundaries](support_matrix.md).

## 2. Choose and inspect the mesh

Uniform Cartesian Yee grids are the general default. A graded `dz_profile` can
reduce the cell count for supported thin-z geometries, but it does not make
every source, port, boundary, or observable available on a nonuniform grid.
Check the support documents before switching mesh type.

`auto_configure(...)` derives a starting `SimConfig` from geometry and a
frequency range. `plan_mesh(...)` returns the same setup with resolution,
absorber, CFL, and memory information in a serializable `MeshPlan`. These tools
do not establish mesh independence or electromagnetic accuracy.

For every model:

1. Confirm that important dielectric regions occupy enough cells to represent
   their interfaces and thickness.
2. Represent a zero-thickness conductor with `add_thin_conductor(...)`, or use
   an edge-aligned PEC volume that occupies at least one cell.
3. Inspect the final cell edges after grading. A requested physical thickness
   can still rasterize to zero cells.
4. Respect the adjacent-cell ratio reported by preflight.
5. Repeat the requested observable on at least one finer mesh before using it
   as an RF result.

See [Nonuniform mesh](../public/guide/nonuniform-mesh.mdx) for supported graded-z
usage and [Automation](../public/api/automation.mdx) for the exact planning API.

## 3. Model materials and conductors explicitly

Material names do not imply a loss model. Set `eps_r`, `mu_r`, conductivity, or
the supported dispersive parameters required by the calculation, and document
the values used.

rfx promotes materials with `sigma >= 1e6 S/m` to the PEC mask. Such a material
does not retain finite volumetric conductivity in the update equations. Use
PEC when that is the intended approximation. A finite sheet-resistance study
requires a supported finite-conductivity representation below that threshold
and a mesh/convergence study for the loss observable.

PEC geometry is rasterized to the Yee grid. Subcell warnings identify geometry
that may disappear or change effective dimensions; they are not evidence that
the approximation is acceptable. See
[Geometry and materials](../public/api/geometry-materials.mdx).

## 4. Keep active geometry clear of absorbers

Place sources, ports, probes, conductors, dielectrics, and NTFF surfaces outside
the effective CPML or UPML region unless the documented workflow explicitly
allows otherwise. Check clearance against the final grid and absorber cell
count, not only against nominal model coordinates.

For NTFF calculations, rfx applies conservative preflight heuristics to the
projected face clearance from PEC geometry. Passing those checks does not prove
far-field separation. The transform may use a closed surface in the near field;
mesh convergence and an analytic or independent reference remain necessary.

See [Far field and RCS](../public/guide/farfield-rcs.md) for the six-face box,
clearance calculation, settling checks, and reference-subtraction procedure.

## 5. Run preflight and act on every finding

Call `sim.preflight()` before a long run and retain the report with the inputs.
Use `report.raise_for_failure()` to stop on errors, but also read every warning.
Warnings such as geometry inside an absorber, zero-cell rasterization,
port/PEC overlap, poor mesh resolution, or an unsupported configuration can
make a finite array unusable as an RF result.

Preflight checks configuration consistency. It does not validate resonance,
return loss, Q, radiation pattern, RCS, or a derivative. A short successful run
only confirms that the selected execution path produced output.

## 6. Run long enough for the requested observable

Source duration, resonator Q, losses, boundary reflections, and the requested
frequency resolution determine the required run time. Do not infer convergence
from `n_steps` alone.

On the supported uniform CPML/UPML runner, `until_decay` monitors an interior
sum-of-squared-fields proxy outside the absorber. The non-uniform
(`dx/dy/dz`-profile) runner supports the same stop on CPML/UPML boundaries,
weighting each cell by its own volume; closed-boundary non-uniform runs warn
and execute the fixed step count. The proxy does not measure total
electromagnetic energy and does not prove convergence of every DFT,
S-parameter, Harminv, or NTFF quantity. Other runners may reject or ignore that
option as documented by their emitted diagnostics.

When the post-run ring-down advisory is emitted, the tail of a recorded probe
suggests incomplete settling. Increase the run duration or use `until_decay`
where supported, then verify the actual observable separately. Closed PEC
cavities do not decay through an absorber; use a fixed run and inspect the
probe signal.

## 7. Interpret the requested observable, not a convenient proxy

- Remove source-dominated transients before applying Harminv or an FFT to a
  ring-down signal.
- Treat Harminv frequency and Q as estimates from the supplied time window;
  compare stability across windows and meshes.
- For MSL S-parameters, `reliable[p, k] == False` means the voltage/current
  split for driven port `p` is too weak at bin `k`; exclude column `S[:, p, k]`
  from RF interpretation. `True` is not a general accuracy guarantee.
- On nonuniform MSL models, use the supported normalization documented by the
  calculator. Do not assume a uniform-grid normalization option is accepted.
- For RCS, subtract an incident-only reference field before forming scattered
  power. A total-field NTFF result is not RCS.
- AD/FD agreement checks differentiation of the implemented computation. It
  does not validate the physical model that produced the objective.

See [Probes and S-parameters](../public/guide/probes-sparams.mdx) and
[Simulation](../public/api/simulation.mdx) for return fields and warning scope.

## 8. Establish evidence for the stated use

Use all applicable checks:

1. **Configuration:** preflight contains no unresolved error or relevant
   warning.
2. **Numerical convergence:** the same physical model is repeated on refined
   meshes and, where relevant, with increased absorber clearance and run time.
3. **Analytic reference:** compare with a closed-form result inside its stated
   assumptions.
4. **Independent comparison:** compare the same geometry, materials, ports,
   reference planes, frequency samples, and metric with another solver.
5. **Regression check:** retain the inputs, raw arrays, plots, metric, and
   threshold so the comparison can be rerun.

A unit test, finite output, preflight pass, internal consistency check, or UI
replay can detect software regressions without establishing RF accuracy. State
exactly which comparison ran; a missing optional external solver is a skipped
comparison, not a pass. Current executable comparisons and their thresholds are
listed in [Validation](../public/guide/validation.mdx).
