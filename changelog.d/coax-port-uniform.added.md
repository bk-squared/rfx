### Added — Use a coaxial feed and reference-plane S11 in uniform single-device run() and forward(). (#PR)

Experimental `add_coaxial_port()` builds the line, matched feed, TEM drive and voltage planes.
S11 uses the built line impedance and is renormalized to the declared `impedance`; the coax calculators remain available.
Supports one top or bottom coax port, float32, three-dimensional second-order Yee and CPML on all six faces.
The whole port must fit inside a declared NTFF box, with room for the line and lateral absorber clearance.
Other S-port families, periodic axes, TF/SF, conformal PEC, subpixel smoothing and eps/sigma overrides are refused.
Graded, distributed and subgridded lanes are refused. Port dimensions and impedance are static; PEC occupancy gradients are available.
An antenna-level result against an independent solver is not yet shown.
The port's source end is a capped, shielded line end, so nothing leaves the line there and far-field and flux monitors see only the structure.
`realized_geometry()` reports the port's realized line impedance and pin cells before a run, or `run()`'s refusal in the port row.
