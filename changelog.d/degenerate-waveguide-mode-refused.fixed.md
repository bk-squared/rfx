### Fixed — Refuse ambiguous discrete waveguide modes (#PR)

A square-guide TE10 port could launch a 45-degree mix of TE10 and TE01
that depends on the machine; a 2:1 guide's second mode likewise mixed TE01
and TE20. Discrete profiles are now refused when the requested mode shares
its discrete cutoff with another mode within roundoff. The error names the
modes and aperture cells. Use `mode_profile="analytic"`, or an aperture
whose cell counts give the modes distinct cutoffs.
