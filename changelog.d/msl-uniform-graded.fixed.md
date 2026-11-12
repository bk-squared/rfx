### Fixed — A microstrip port declared with `mode="uniform"` on a graded grid was fed and terminated with the static-Laplace profile; it now gets the uniform feed, as on a uniform grid (#PR)

On identical cells the uniform and graded runners now give bitwise-equal port records and S for the same
declaration. On the contract line the graded S11 moves from 0.1827 to 0.1975 at 2 GHz, from 0.3736 to 0.3600 at
6 GHz and from 0.2827 to 0.3067 at 10 GHz; S21 moves by 0.021, 0.0095 and 0.0097 at the same frequencies.
`mode="laplace"`, the default, and the graded refusal of `mode="eigenmode"` are unchanged.
