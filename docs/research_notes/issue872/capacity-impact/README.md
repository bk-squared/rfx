# Model-capacity safeguard: impact on retained inputs

`report.json` compares the prior crop/joint-fit implementation (retrieve with
`git show 5ee9539b:rfx/harminv.py`) and the approved capacity safeguard.
The report contains both source hashes and every input hash, plan, duration,
mode list and wall time. `replay.py --before <old-source> --out <fresh-report>`
reproduces the check from the repository root without FDTD or external jobs.

Of 53 inputs, 52 retain exactly the same automatic plan and mode output in
this common NumPy/SciPy environment. These cover all 30 original cv02/cv24
records, the candidate input from each Meep version, and the ten probe traces
from each fed/unfed GPU run. This is an estimator-change check on retained
physical data, not a fresh external solver or CPU/GPU bit-identity claim.

Only the 1,846-sample stage1 record changes: factor 9 then 5 / 18 samples is
replaced by factor 9 / 186 samples. Its continuum error falls from 0.0826764%
to 0.0226667%, passing the original 0.03% gate. The p=0 Yee time/space dispersion
predicts 0.0229031%; the old estimator's smaller 0.0144% continuum error partly
cancelled that physical discretization shift. The original failed result and
native controls remain separately retained in `../stage1-short-record/`.

The live stage1 gate also passes. The synthetic regression deliberately keeps
all twelve poles below the previous plan's Nyquist frequency, so it isolates
dimension loss from aliasing; substituting the previous estimator makes it
fail. No continuum, lattice, stationarity, matching or Q gate was loosened.
