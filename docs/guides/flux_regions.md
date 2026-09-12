# Finite flux-monitor regions

`add_flux_monitor(size=..., center=...)` selects a rectangular window on a
plane. The two values follow the tangential coordinate axes in this order:

| Plane normal | Size and centre axes |
|---|---|
| x | y, z |
| y | x, z |
| z | x, y |

Each supplied size/centre must contain exactly two finite real values. Both
sizes must be positive. A centre outside the domain is allowed if the window
still resolves to at least one interior cell along each tangential axis.

The finite window snaps to cell-bounding nodes and clamps to the grid's
physical interior. Every requested endpoint outside that interior produces
a clamp warning, even when the overflow is smaller than a cell. A window
that resolves to no interior cell raises an error before field evolution.
Endpoint differences within the rounding uncertainty of the centre, size
and boundary arithmetic do not count as physical overflow; this guard uses
floating-point ULPs, not a fraction of the mesh spacing.
The CPML padding and trailing bounding-node slot are excluded. In a uniform
2D simulation the collapsed z axis represents one integration cell of
thickness dx.

Uniform grids retain their existing round-to-even index choice and declared
domain midpoint when the centre is omitted. Graded grids use nearest nodes
on the canonical float64 coordinate spine, choosing the first/lower node on
an exact tie, and the realized interior midpoint by default. The returned
metadata makes the actual choices visible. Graded area weights still come
from the per-axis cell-size arrays used by the monitor.

## Read the actual window before solving

After registering finite monitors on a simulation:

```python
report = sim.preflight()
for region in report.flux_regions:
    print(region["name"], region["tangential_axes"])
    print("Requested:", region["requested_bounds_m"])
    print("Realized:", region["realized_bounds_m"])
    print("Cell slices:", region["cell_slices"])
```

The report also records the requested/realized normal-plane coordinate and
the clamp status of each endpoint. `report.to_dict()` and `report.to_json()`
include these records under `flux_regions`. They are metadata, not findings:
an aligned window does not add a warning, change report truthiness or cause
strict preflight to fail. Actual clamps are warning findings; invalid or
empty windows are errors. Unavailable geometry is reported explicitly.

`size=None` preserves the legacy full allocated plane, which can include
padding. It has no finite-window record. `center` only affects a finite
window. Use an explicit physical `size` when the intended observable is the
flux through an interior aperture.

Window correctness does not certify RF accuracy, settling, source purity or
the normal-axis H interpolation on a graded mesh. Those require their own
validation. The finite-window resolver changes geometry bookkeeping; it does
not alter field evolution or DFT normalization. Field/material dependence
stays on the existing autodiff tape; changing the selected aperture can
change both the flux value and its gradient.
