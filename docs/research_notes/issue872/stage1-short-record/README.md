# Short-record CI failure, retained before policy correction

The original stage1 NU cavity gate fails on the #872 candidate; this is an
unresolved compatibility finding, not a passing validation artifact.

The exact estimator input is `tests/fixtures/harminv_decimation/stage1/input.npz`.
Its receipt records the captured source and byte hash. Both old and new
estimators consume those same bytes. The main source differs from the tested
candidate only in unrelated #967 code at the baseline point.

The 1,846-sample input covers 7.61 cycles. The current #872 automatic plan crops
it to 186 samples after factor 9 and then 18 after factor 5; the final pencil
has only 5 columns. The previous stage has 10 singular values above the same
relative 0.001 cutoff, while the final pencil saturates all 5 columns. Thus a
ten-sample minimum is insufficient for this physical multi-mode input.

The closed-form p=0 Yee frequency, including time dispersion, is
5.298418220 GHz, versus continuum 5.299632000 GHz. The native estimate is
5.298428569 GHz and stopping after factor 9 gives 5.298430753 GHz; both satisfy
the unchanged 0.03 percent continuum gate. Full automatic decimation gives
5.295250456 GHz (0.0827 percent error) and fails. The old automatic result's
0.0144 percent apparent continuum error is partly cancellation with estimator
bias; it is not a tighter physical discretization result.

`diagnosis.json` and compressed logs retain the failing result and raw controls.
The two diagnostic scripts document capture and one-stage analysis; their
`/tmp` paths are session-local locations, not a reusable public CLI.
No solver threshold or production decimation policy was changed for this
capture. The general rank-capacity fallback proposal awaits PI decision.
