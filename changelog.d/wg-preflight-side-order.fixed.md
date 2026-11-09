### Fixed — waveguide preflight side order and cutoff (#PR)

Waveguide preflight now uses the solver's side order and cutoff. An evanescent
launch with the larger side on the v axis is now warned, and a propagating mode
is no longer falsely warned because its transverse sides were sorted by size.
Multimode checks name the lowest-cutoff mode the port drives.
