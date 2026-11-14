### Changed — named warning for closed curved PEC smoothing (#PR)

- A closed curved PEC Sphere or Cylinder under `kottke_pec` now warns by body kind and material name: on a PEC sphere at ka 6 with cells of λ/20, extinction missed Mie by 20–51 % (staircase: 13–14 %), with a late field of 3–5 % of the peak.
- The warning recommends `subpixel_smoothing=False` (the staircase) or judging three refinements, and points to the closed curved metal body known limitation.
- Other PEC volumes keep the existing warning.
