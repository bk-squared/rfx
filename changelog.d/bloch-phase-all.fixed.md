### Fixed — Apply the Bloch phase to smoothed and dispersive electric fields (#PR)

Oblique Bloch plane waves now support diagonal subpixel smoothing, dielectric
Kottke smoothing, and Debye, Lorentz, Drude and mixed pole materials on the
uniform grid. Angle overrides use the same phase, and pole states preserve
the complex field polarization. Off-diagonal node coupling and the unsupported
PEC smoothing combinations remain refused. Combining poles with a directly
supplied permittivity tensor now raises an error instead of ignoring the tensor.
