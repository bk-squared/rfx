# Fresh #752 evidence

`covered-run/` contains the six completed, sequential cases from VESSL
369367260699. `weak-pulse-controls/` retains completed h3/h4 and partial h5
from run369367260698; that run was stopped after the original differentiated
pulse failed the unchanged high-band relative-signal screen. Incomplete
records are not additional completed mesh points.

Both runs are terminal and deleted only after their full provider logs and
raw artifacts were backed up and verified. `covered-manifest.json` and
`weak-pulse-manifest.json` record file hashes. Text logs are gzip-compressed;
their decompressed bytes retain the original provider/runner output.

Each completed case includes the pre-solve geometry/reference plan, actual
per-drive material and conductor edge arrays, source waveforms, witness and
DFT registrations, raw DFT and time records, raw port V/I dumps, returned
results and diagnostic flags. `driver.py`, protocol, environment and frozen
input hashes preserve how they were produced. The numerical RFX source is
commit `cbdc0976bd67a7c5fc318f3bc1967cd947399495`; `git archive` at that commit
has the recorded source.tar hash. The complete source archive also remains
in the original workspace run directories; it is not duplicated here.

The final repository's change to `msl_eigenmode.py` only corrects the
reference helper's docstring. `reference-docstring-parity.json` proves its
executable AST is identical to the source used by the field runs.

Re-evaluate stored data without advancing fields, using a NEW output path:

```sh
python scripts/diagnostics/analyze_msl_z0_matched_geometry_sweep.py \
  --input-root docs/research_notes/issue752/fresh/covered-run/artifacts \
  --out /tmp/rfx752-new-analysis
```

The analysis command's successful exit means the report was written. Read
its scientific verdict and case eligibility; an analysis error, missing
case, or unusable record cannot establish a passing six-point comparison.
The production Z0 values are never replaced by the independent fit.

The final run's independent replay reports all six cases as
`usable_under_declared_screens`. The primary port-0 band means are 46.36530,
47.95702, 46.39500, 47.41100, 46.35699 and 44.92623 ohm in the order
`h3,h4,h5,h6,dx80,dx60`. Relative to the repository reference they are
−3.947%, −2.903%, −2.145%, −1.781%, −3.964% and −2.720%, respectively.
The corresponding full HJ1980 comparison is −3.488% through −1.311%.
Both comparisons refute the predeclared all-six 0.4% hypothesis.

The analyzer includes explicit cross-file and provenance rejection checks.
Its tests cover altered production/raw Z0, nonfinite passive records, drive
duplication, material/PEC/source drift, current-plane/convention changes,
probe/DFT changes, and weighted-current/voltage-record inconsistencies.
Those are evidence-integrity checks; they do not impose a new RF physics
threshold or repair the production values.

The final replay is in `final-analysis-v3/`; its summary records all six
cases as usable and the all-six hypothesis as `refuted` for both references.
The replay completed without a field solve and its summary hash matches the
shipping analyzer source.

`protocol-weak-pulse.md` is the original frozen protocol. `protocol.md` is
its explicit excitation-coverage revision, frozen before the replacement
six cases. Neither changes the 0.4% comparison or physics rejection screens.
This is not an absolute accuracy qualification, a general mesh-convergence
claim, or a resolution of the separate #726 MSL power issue.
