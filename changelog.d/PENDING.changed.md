### Changed — Boundary feature rewrites now require compatible declarations or reported defaults (#PENDING)

Full-aperture waveguides whose transverse absorbers realized zero depth now default
omitted transverse boundaries to PEC and warn once; explicit absorbers there are
refused. Non-uniform guides retain their transverse absorbers. TF/SF plane waves
refuse non-invariant transverse absorber or compatible-wall replacements, even
with preflight skipped. Invariant cases keep the reported legacy periodic wrap;
traced overrides report when invariance cannot be judged. Declare periodic faces
for an array, or use `closed_box=True` for a finite scatterer. UPML remains refused.
