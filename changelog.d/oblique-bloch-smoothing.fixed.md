### Fixed — An oblique Bloch plane wave with subpixel smoothing, conformal PEC, a Kottke PEC occupancy or a pole material returned wrong fields without warning; it is now refused before any step (#PR)

An oblique Bloch plane wave now refuses subpixel smoothing (`True` or `'full_tensor'`),
`'kottke_pec'` smoothing, conformal PEC, a PEC occupancy override with
`pec_occupancy_model='kottke'`, and Debye/Lorentz pole materials by name before stepping.
These electric-field updates omit the Bloch phase. The refusal replaces |T| 0.832
against Airy 0.749 at 30° on a 24 mm εr 4 slab (+0.91 dB), and the Debye scan-carry error.
Use normal incidence, or `subpixel_smoothing=False`, conformal PEC off,
no `pec_occupancy_override`, or a material without poles, as applicable.
