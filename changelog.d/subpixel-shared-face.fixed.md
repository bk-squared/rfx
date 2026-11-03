### Fixed — shared dielectric faces use both bulk permittivities (#PR)

- Subpixel smoothing uses the bulk media on both sides of touching dielectric
  volumes on uniform and graded meshes, including `"full_tensor"` and supported
  `"kottke_pec"` paths, independently of the two volumes' declaration order.
- In the 20 × 13 × 7 mm PEC cavity with touching εr 3.38 and 10.2 layers,
  the 0.5 mm mesh's TM resonance error changes from +1.5171% to −0.0571%
  against the closed form; the identification indicator passes in both runs.
