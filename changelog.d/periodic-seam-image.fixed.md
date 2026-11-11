### Fixed — Image selection for periodic seam geometry (#PR)

Conformal PEC near the HIGH seam now uses the nearest image: the recorded
sphere's free edge is 0.70 instead of 0.97 (reflected-power error up to 2.4e-3
at λ/21). Dielectric smoothing and full-tensor operands beside periodic seams
also change where a far image was read: review cases differed by up to 0.628
in node operands, 0.148 in Kottke permittivity, and 0.166 beside a Box.
