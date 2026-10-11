### Fixed — Refuse NTFF requests without an absorbing exterior

NTFF admission now refuses PEC, PMC, periodic and zero-depth absorber faces,
and plane-wave sources without a closed scattered-field box, before stepping.
The shared rule covers run, forward, runner admission and the RCS calculator.
