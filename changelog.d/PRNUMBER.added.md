### Added — NTFF reciprocity adjoint (#PRNUMBER)

- `forward(gradient="adjoint")` accepts NTFF far-field and H DFT-plane objectives on the uniform Yee design-permittivity path.
- NTFF adjoint sources use the accumulator's spatial transpose for both face-centre and lower-corner sampling.
- The gradient is for the settled spectrum. `forward()` has no flux-monitor output. Design conductivity overrides remain refused.
- `scripts/benchmarks/adjoint_vs_autodiff.py --ntff` selects an NTFF directivity objective.
