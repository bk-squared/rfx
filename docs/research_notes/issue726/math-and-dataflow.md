# What the paired observation can establish

These are exact single-mode algebra and synthetic extraction interventions,
not new RF results or reasons to relax the existing run gates.

## Fitted diagnostics versus measured inputs

`compute_msl_s_matrix` constructs A and B from the first probe's measured
V/I and its analytic Hammerstad-Jensen reference impedance, then solves
S = B A^-1. Fitted beta and Z0 are returned diagnostics. No subsequent
operation translates S from those probe planes to the physical feed planes.

The new `test_fitted_impedance_and_beta_cannot_change_production_s` uses the
public extractor with synthetic DFT planes and a nonsingular drive matrix.
Changing only the fit's returned Z0 and beta changes those diagnostics while
leaving raw V/I and S exactly equal. Keeping that same fit but altering the
measured H at one drive/port changes I and S while the low-signal mask stays
true. This validates a data dependency. It does not quantify an RF error or
establish which physical mechanism would cause that field change.

## Observation position and impedance reference

For a single lossless propagating mode with real positive modal impedance
Zc and fixed real positive reference Zr, let

    V(x) = a exp(-j beta x) + b exp(+j beta x)
    I(x) = [a exp(-j beta x) - b exp(+j beta x)] / Zc.

Writing Gamma_c(x) = (b/a) exp(2j beta x) and m = (Zc-Zr)/(Zc+Zr), the
reflection formed from those fields is

    Gamma_r(x) = [V(x)-Zr I(x)]/[V(x)+Zr I(x)]
               = [m+Gamma_c(x)]/[1+m Gamma_c(x)].

Thus a constant modal reflection magnitude need not imply a constant
reference-normalized magnitude. The reproducible example in
`phasor_counterexamples.py` uses Zc=60 ohm, Zr=50 ohm and |Gamma_c|=0.6.
Two positions give |Gamma_r|=0.655172 and 0.538462, a 1.704 dB difference,
although no higher mode or evanescent field exists in this constructed model.
This is a counterexample to identifying every magnitude change as modal
contamination; it is not an estimate for cv06b.

The limitation matters for #726: if |Gamma_c|=1, the same real-impedance
transformation preserves |Gamma_r|=1 at every position. Reference mismatch
alone does NOT explain a magnitude error on an ideal perfect reflector.
The finite open microstrip experiment has not established that ideal limit.

## Low signal versus an ill-conditioned wave solve

The second construction is a reciprocal lossless two-port with

    S = [[r, j sqrt(1-r^2)], [j sqrt(1-r^2), r]],

with r=1 at the middle bin and r=0.3 in the neighboring bins. Independent
matched drives give A=I, hence B=S. The production solve returns the planted
S exactly and cond(A)=1 throughout. Nevertheless, the middle bin's passive
V/I are zero, and the current per-record relative low-signal rule marks both
ports false. The raw record therefore refutes the blanket claim that a
small passive V/I pair itself proves corruption of the entire S slice.

It does NOT prove noisy finite-time FDTD results are safe at a transmission
zero. An absolute uncertainty estimate, finite-window error and mode/port
validity remain separate questions. No mask formula, numerical threshold,
or run acceptance rule is changed here. `phasor_counterexamples.json` is
an algebraic receipt, not a physical validation score.
