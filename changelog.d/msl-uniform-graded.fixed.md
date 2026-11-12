### Fixed — A microstrip port declared with `mode="uniform"` on a graded grid was fed and terminated with the static-Laplace profile; it now gets the uniform feed, as on a uniform grid (#PR)

On identical cells the uniform and graded runners now give bitwise-equal port records and S for the same
declaration. On the contract line the graded |S11| changes by up to 0.034 (2 GHz), 0.015 (6 GHz) and 0.024 (10 GHz).
`mode="laplace"`, the default, and the graded refusal of `mode="eigenmode"` are unchanged.
