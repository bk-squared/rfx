# Graded-wall checks for #729

Current main e32e386d fails all four new checks before the source correction.
The candidate passes all four (+x/-x/+y/-y; unequal x/y/z grading).
The source-independent bounds are the realized PEC ground and trace planes.
Voltage reads the solver normal-edge lengths; Joule power reads its E-update
control volumes. These are modal-source/load identities, not claims that the
static Laplace profile is an exact graded-grid eigenmode or that RF input
impedance equals the termination parameter. The intentionally abrupt grading
exercises metric discrimination only; it is not a validated propagation mesh.

Candidate: `pytest -q tests/unit/ports/test_msl_physical_substrate_span.py -k graded`.
Baseline: import this same test file with rfx pinned to e32e386d, then call
`test_graded_source_voltage_and_load_use_actual_walls` for its four parameters.
Every baseline failure occurs at the independent wall-span assertion.
