Wire ports accept `radius=` in metres with `extent=`. PEC `PolylineWire`
filaments also carry their declared radius through shared component H materials.
All H coefficient builders use the same material entry; no μ averaging is added.
A missing radius record retains the cell permeability for all H components.
The local model requires square, uniform transverse cells, uniform axial cells,
and `0 < radius <= 0.20 * d_min`; filament radii between 0.20 and 0.50 cells
refuse. Wires of radius at least 0.50 cells retain volume realization.
Larger wire ports require a coax feed or a volume wire. Distributed, subgridded,
ADI, UPML and dispersive radius boards refuse before stepping.
The full-height port accuracy result does not cover a one-edge PEC-filament feed.
