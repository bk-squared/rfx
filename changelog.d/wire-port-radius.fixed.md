Wire ports accept `radius=` in metres with `extent=`, using shared component
H materials. All H coefficient builders use that entry without μ averaging;
a missing record retains the cell permeability for every H component.
The port model requires square, uniform transverse cells, uniform axial cells,
and `0 < radius <= 0.20 * d_min`. Larger ports require a resolved volume wire
or coax feed. Distributed, subgridded, ADI, UPML and dispersive radius boards refuse.
PEC `PolylineWire` now refuses every `0 < radius < 0.50 * d_min`: resolve the
wire as a volume, `a >= 0.5*d`. Its filament correction failed the independent
main-grid impedance witness and is removed. Volume wires and legacy radius=0
filaments retain their existing realization; the wire-port radius model remains.
