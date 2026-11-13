### Fixed — Preserve requested modes on degenerate waveguide apertures (#PR)

Square and 2:1 apertures now launch the requested discrete mode independently of the eigensolver basis.
Before, square TE10 could be a machine-dependent 0.707/0.707 TE10/TE01 mix; 2:1 TE01 mixed with TE20.
Measured max |Ey| / max |Ez|: D1 TE10 9.22e-15; D2b TE20 6.41e-14.
Default float32 public TE10 (D9): 9.22e-15 uniform and 7.33e-15 graded-x.
