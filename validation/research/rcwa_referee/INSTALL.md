# grcwa install recipe (RCWA referee, #491 step 1)

Validated 2026-08-09 on the primary rfx checkout
(`/root/workspace/byungkwan-workspace/research/rfx`, Python 3.11.15, numpy 2.4.6).

## Install

The rfx venv is uv-managed and has **no pip** (`.venv/bin/pip` does not exist,
`python -m pip` fails with "No module named pip"). Use uv against the venv's
interpreter:

```bash
cd /root/workspace/byungkwan-workspace/research/rfx
uv pip install --python .venv/bin/python grcwa
```

Installed: `grcwa==0.1.2` + its only extra dependency `autograd==1.9.1`
(numpy was already present). Pure Python — no compiler, no system packages,
installs in under a second. This is why grcwa was preferred over S4
(C++ build with BLAS/Lapack wiring) as the first-choice referee.

## Verify (reproduce-gate)

Run the released package's own shipped test suite verbatim (the sdist carries
the tests; the wheel does not):

```bash
cd <scratch>
curl -sL -o grcwa-0.1.2.tar.gz \
  https://files.pythonhosted.org/packages/a9/4d/d79d7dfcf73bb402890bc573ad482036d524f39d1868383da7c4a47c2a4b/grcwa-0.1.2.tar.gz
tar xzf grcwa-0.1.2.tar.gz && cd grcwa-0.1.2
/root/workspace/byungkwan-workspace/research/rfx/.venv/bin/python -m pytest \
  tests/test_rcwa.py tests/test_kbloch.py -v
# expected: 10 passed  (reference log: logs/20260809_grcwa012_shipped_tests.log)
```

Note: run from the sdist root, pytest imports the sdist's own `grcwa/` source
tree (identical 0.1.2 code, verified by the traceback paths). The committed
gate script `rcwa_referee_step1.py` additionally reruns the canonical numbers
against the **installed** venv package and prints its import path.

Then run the gate script:

```bash
cd /root/workspace/byungkwan-workspace/research/rfx
.venv/bin/python validation/research/rcwa_referee/rcwa_referee_step1.py
# expected: "RESULT: ALL GATES PASS", exit 0
```

## Gotchas

- grcwa 0.1.2 (PyPI) predates the GitHub HEAD: no `fmm_method="pol"` argument
  (`grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)` is the full
  signature). The HEAD repo's `test_rcwa_pol` tests do not apply to 0.1.2.
- Conventions: c = 1, `exp(-i omega t)`, frequency in units of c/period-unit.
  `RT_Solve(normalize=1)` is required when the output medium is not vacuum or
  incidence is oblique.
- Patterned layers: the Fourier grid `Ny` must cover the reciprocal orders the
  circular truncation selects — `Ny=2` IndexErrors in `Epsilon_fft` even for a
  y-uniform (1D lamellar) grating with a small y-period; `Ny=16` is safe for
  the configurations used here.
- No artificial loss is added to `freq` internally, so lossless energy
  conservation reaches ~1e-13 and uniform-slab R/T match closed form at
  machine precision.
