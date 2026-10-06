### Fixed — source coefficients use final material stamps (#PENDING)
Single-device uniform/graded port and current-source drives now read Debye/Lorentz coefficients after all loads are stamped.
In first-step witnesses, the Debye port's 1.014× excess and a periodic soft source's 2.5× excess are removed.
Soft-source declaration order no longer omits a later port load (1.861× excess in the witness).
Unsmoothed UPML interior sources use the kernel's cell-owned coefficient (1.322× excess removed at an interface).
Ports and sources inside UPML pads now raise instead of injecting with a mismatched coefficient (1.147× in the witness).
The currently unreachable MSL eigenmode J+M builder uses separate Ey/Ez coefficients; default Laplace MSL results are unchanged.
