# Issue #726: measurement scope and extraction diagnostics

The warning and experiment repairs are ready for review. The paired RF
record is complete, but its existing low-signal screen returns `not_read`;
this is not a calibrated accuracy PASS and does not close #726. The user
has not yet decided the clearance-result policy or the proposed spatial
alignment correction. No numerical extraction or gate threshold is changed.

## What changed

The two-wave fit includes forward and backward waves. Preflight no longer
predicts a universal 5–10 dB S error from a layout threshold, or calls the
fixed epsilon/f_max proxy a conservative whole-band bound. It describes the
registered-geometry estimate and the source/ladder/reflector space needed
when the offset interval is empty.

The fitted-Z0/beta guards now state only that those fitted outputs do not
enter production S; they do not certify their shared measured V/I. The result
docs distinguish the nominal first E-probe reference from the feed plane and
the fitted Z0 from the analytic normalization impedance. They also disclose
the spatial E/H staggering identified below.

The old comparison moved the physical source and load along with its probes.
Its short fixture further assumed exact unit reflection for a finite open
microstrip, without excluding radiation or other channels. On current grid
geometry its near probe lies ON the realized short front at 19.896666667 mm:
all three substrate Ez edges are PEC, while the old unsnapped check reports
76 um clearance. Both old first probes also violate the existing source
standoff (10 versus 15 cells). The historical builder is retained for
reproduction but its accuracy experiment refuses to execute.

`audit_original_short.py`, `original-short-build.json`, the lossless console
log and manifest preserve the build-only evidence. The baseline diagnostic
source is loaded through git; the receipt was generated with the checkout's
unchanged baseline numerical library before the warning edits.

## Paired current-cv06b record

VESSL 369367260508 used source `130b9071e5e6aaa0efda0dd1d6a36fb7e8d519bd`.
Both physical ports, waveforms, loads, materials, mesh and DUT stayed fixed.
Only p1's observation offset changed; p2's ladder stayed identical. Both
realized ladders were certified before the first field step. Five probes
with two-cell spacing were used in both arms; this short fixed fit span is
part of the experiment, not a claim about all auto-placed port ladders.

| Recorded quantity | Control | Near |
|---|---:|---:|
| First probe to realized stub | 9.398 mm | 0.5715 mm |
| Last probe to realized stub | 8.890 mm | 0.0635 mm |
| Worst settling over the two drives | -120.006 dB | -118.409 dB |
| Raw S11 at the control notch bin, 3.77125 GHz | +0.026895 dB | +0.018065 dB |
| Raw S21 at that same bin | -47.730755 dB | -47.720779 dB |
| cond(A) at that bin | 1.006287 | 1.005992 |
| `reliable` at that bin | false, false | false, false |
| p1 beta-railed bins in 3–5 GHz | 0 / 51 | 51 / 51 |
| Maximum raw column power, 3–5 GHz | 1.006511 | 1.007819 |

The raw S11 difference at the common bin is **-0.008830 dB**. Across the
predeclared 3–5 GHz band its raw magnitude difference ranges from -0.017537
to +0.006562 dB. This is observed sensitivity at different nominal probe
planes, not a calibrated physical error. No passivity projection was used;
the small column-power excess remains visible. The data do not identify
whether fit conditioning, field content outside the model, or another
mechanism produced the beta rails.

All four field solves completed. The producer then exited 2 because its
existing notch low-signal screen rejected the automatic comparison. Thus
VESSL's terminal `failed` state describes the retained analysis verdict,
not a failed FDTD execution. No gate was relaxed to obtain a PASS.

`gpu-369367260508/` contains the raw V/I, S, diagnostics, input certificate,
environment and launch records, complete execution log, and paired audit.
The full provider log is in `../vessl_logs/369367260508_failed.log.gz`.
Every backup was checked before the terminal run was deleted. The audit
reproduces identically from the retained copy. A NumPy BA^-1 replay from raw
V/I differs from production by at most 2.3e-7 / 2.2e-7, using the pinned
producer's reconstructed HJ reference (47.8948 ohm), not its 50-ohm load or
its fitted Z0. This checks algebra and provenance, not physical accuracy.

The earlier staging failure 369367260507 occurred before any field step:
the archive omitted a relative comparator import. Its complete logs and
receipt are retained. The replacement archive was unpacked and built in
isolation with field execution forbidden before GPU submission.

## Interpretation and remaining decisions

`math-and-dataflow.md` and `phasor_counterexamples.py` show why fit-output
independence does not certify V/I, why partial reflection can change
magnitude with observation position under a different reference impedance,
and why a true transmission zero can trip the relative low-signal mask even
with A=I. Real positive reference mismatch preserves unit magnitude for an
ideal perfect reflector; that limitation is explicit.

`spatial-alignment.md` and `yee_spatial_stagger.py` document another source
finding: uniform MSL combines node-plane voltage with the same-index H at
half a cell away, after correcting time staggering only. The analytic
matched-wave falsifier has O(dx) false reflection; bracketing H interpolation
would reduce it to O(dx^2). That correction is only proposed. Staggering
alone does not establish the cause of the measured power excess.

The existing low-signal mask's blanket whole-matrix-error wording is also
unresolved. Its formula and the experiment's screen remain unchanged.
The clearance diagnostic policy and the H-interpolation choice await the
user; #953's separate G1 question is unchanged.

Validation at `501b73e9`: 98 focused checks, two affected checks after final
edits, scoped Ruff, independent source/algebra reviews, and all 21 required
CI checks passed. The final PR checks must also pass for the later evidence and
nominal-reference wording. These checks do not turn the RF record into
an accuracy certification.
