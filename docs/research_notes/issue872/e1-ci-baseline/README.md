# Existing main CI failure: balance the same scattering equations

The #872 CI merge included main b7ad4e93 and its newly merged #960 replay.
That replay failed without calling Harminv. The full CI log is retained here.
A detached b7ad4e93 checkout with no #872 changes reproduces both failed
assertions and the exact CI value when `OPENBLAS_CORETYPE=Haswell` is selected.
The default local BLAS kernel passes on the same source. This is an existing
platform-dependent baseline error, not a new Harminv regression.

The scalar NU recurrence contributes rows proportional to 1/d^2, while the
Bloch-wave boundary rows have coefficients of order one. For the failed arm,
row magnitudes span 1 to about 8 million. Dividing every row AND its RHS by
that row's maximum coefficient leaves the linear problem unchanged but
balances its elimination. The code changes only this numerical operation.

## Reproduction

At main b7ad4e93, run:

```sh
OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell JAX_PLATFORMS=cpu python -m pytest -q \
  'tests/unit/nonuniform/test_e1_band_law_replay.py::test_model_side_replays_frozen_predictions[N60_r1.2]' \
  'tests/unit/nonuniform/test_e1_band_law_replay.py::test_model_side_of_results_replays[N60_r1.2]'
```

This fails twice with the scalar recorded in `receipt.json`. Apply only the
row-balancing change in `validation/research/multiband_nu/chain_model.py` and
run both `test_e1_band_law_replay.py` and `test_band_builder_chain_model.py`:
all 56 tests pass with Haswell selected and with the default local kernel.
The full logs are retained. These existing always-on tests caught the defect;
no window is widened, no frozen prediction is re-pinned, and no FDTD is rerun.
