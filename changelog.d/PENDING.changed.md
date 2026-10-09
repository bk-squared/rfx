### Changed — Boundary feature rewrites now require compatible declarations or reported defaults (#PENDING)

Full-aperture waveguides whose transverse absorbers realized zero depth now default
omitted transverse boundaries to PEC and warn once; explicit absorbers there are
refused. Graded full-aperture guides now refuse an omitted boundary: declare the
side walls explicitly; explicitly declared absorbers retain the open cross-section. Uniform TF/SF plane waves
refuse non-invariant transverse absorber or compatible-wall replacements, even
with preflight skipped. Invariant cases keep the reported legacy periodic wrap;
traced overrides report when invariance cannot be judged. Declare periodic faces
for an array, or use `closed_box=True` for a finite scatterer. UPML remains refused.
Graded plane waves keep their absorbers; oblique tilt axes are reported, not judged.
