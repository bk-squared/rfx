### Fixed — Subpixel smoothing solved a later shape as an earlier material when its eps_r repeated one declared before (#PR)

- With `subpixel_smoothing=True`, `"kottke_pec"` or `"full_tensor"`, a later shape whose eps_r equals an earlier shape's eps_r is now painted in declared order, so a later entry wins as it does with smoothing off.
- Before, an εr 4 Box nested in an εr 10.2 Box read 10.2 inside when another εr 4 shape was declared first (about 155 % in that body's permittivity). Adding that small block moved a cavity resonance by −1.71 %; it now moves it by −0.0145 %.
- Same-eps shapes with no other material declared between them, or none within two cells, still form one body with no interface between them.
- Uniform and graded meshes; scenes with no repeated eps_r are unchanged bit for bit.
