### Fixed — source coefficients follow the realized E update (#PENDING)
Port and current-source drives now read Debye/Lorentz coefficients after all loads are stamped.
In the first-step witnesses, the Debye port's 1.014× excess and a periodic soft source's 2.5× excess are removed.
Soft-source declaration order no longer omits a later port load (1.861× excess in the witness).
The MSL J+M builder uses each driven component's coefficient; its plain-interface Ey witness previously read 1.333×.
