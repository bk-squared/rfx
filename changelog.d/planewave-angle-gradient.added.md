### Added — forward() takes tfsf_angle_override= for the incidence-angle gradient of the uniform Bloch line (#PR)

Differentiate reconstructed probe time series with respect to incidence angle
in degrees, at fixed carrier frequency, on the uniform single-device Bloch line.
The override supports grad, jvp, jit and vmap angle batches; the declaration
keeps the source kind and auxiliary-grid sizes fixed. Declare the largest
absolute angle used to check the cutoff advisory for the full angle range.
Unsupported kinds and traced declaration scalars now give named refusals.
