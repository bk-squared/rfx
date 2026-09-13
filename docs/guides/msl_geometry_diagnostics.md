# MSL conductor gaps and material extents

An MSL port's voltage interval runs between its ground and trace conductor
planes. A dielectric material array can extend beyond that interval,
including into a PEC volume. Those two extents describe different objects.

The geometry advisories distinguish:

| Quantity | Meaning |
|---|---|
| Conductor-plane gap | Separation of the validated ground and trace planes on the run grid |
| Normal intervals | Number of Yee intervals between those planes |
| Material-column extent | Contiguous sample slots carrying the first dielectric permittivity above the ground |
| Declared-face fraction | Where the declared substrate top falls between grid nodes, before snapping |

For example, the historical 254µm substrate with a one-cell PEC trace
volume has these current uniform-grid observations:

| Grid spacing | Conductor gap | Normal intervals | Dielectric material extent |
|---:|---:|---:|---:|
| 80µm | 240µm | 3 | 320µm in 4 sample slots |
| 60µm | 240µm | 4 | 300µm in 5 sample slots |

The 320/300µm values do not establish that the RF gap became thicker.
Some of those dielectric slots lie inside the conductor. Nor does a
material's geometric bounding width establish an electrical effective width
for an analytic microstrip model.

Preflight validates attachment against the original realized PEC and
surface-impedance sheet geometry before reporting a gap. It then uses the
same snapped port interval and canonical node coordinates as the port
geometry routines. A failed attachment remains an error with an unavailable
gap; preflight does not choose another conductor wall to repair the input.

The declared-face fraction remains independent of the snapped gap. A port
can attach to a valid snapped plane while its declaration is between nodes.
Alignment suggestions use the nodes bracketing that declaration, regardless
of whether the same dielectric continues beyond the trace.

The existing four-interval recommendation, mixed-cell fraction window and
8.3% geometry advisory threshold are screening heuristics. They do not
bound Z0 error, validate every layer of an inhomogeneous structure, or
certify S-parameter accuracy. The material observation is from the declared
material assembly. Both diagnostics describe that original assembly;
later material or fractional-occupancy overrides are not validated here.
The gap diagnostic does not change the port source, V/I observation,
reference impedance or autodiff path.

## Historical Z0 sweep

The old six-point MSL sweep and realized-anchor JSON files remain immutable
historical records. The old generator combined their measured Z0 with the
current rasterized geometry and overwrote the anchor file. That operation
cannot establish a current comparison, even if some dimensions happen to
match. Generation through those legacy commands is therefore retired.

To inspect the retained records without a field solve or file writes:

```sh
python scripts/diagnostics/msl_z0_bias_floor_sweep.py --show-archive
python scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py --show-archive
```

Both commands verify the recorded file hashes and label their output as
historical. The old inferred precision and geometry metadata remain part
of that record; inspection does not independently verify the original run.

A new quantitative comparison needs field data and geometry provenance
from the same run: conductor planes, active intervals, material distribution,
trace geometry, grid, source/probe locations, precision, frequencies and
extraction convention. It must state whether the analytic reference describes
the declared design or the realized structure, and whether that analytic
model supports the conductor thickness and dielectric stack being used.
Neither the historical “within 0.4%” result nor a quiet geometry preflight
supplies a current error bound.
