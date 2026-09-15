# Short: the historical interval measured a disconnected network

The disposition is causal attribution, not a new reflection-accuracy pin. The
historical `(2.25, 3]` interval summed a reflected wave and a second measurement
of the same cavity as though they were independent outgoing ports. The repaired
fixture removes that contribution. Its unchanged advisory-witness assertion
must remain red; no gate, test, or production operator is changed here.

## Named VESSL result, 2026-09-09

Run **369367259618**, `rfx-931-fixture-repair-short`, tested pinned source
**b36fc46cdf21d1c57f221e6a057654bcad60bae2** from its node-local clone. Its
submitter-recorded `run_id.txt` identifies the run; the provenance does not
depend on an environment variable inside the pod. Both coarse and fine
captures completed with return code 0. Pytest returned **15 passed, 1 failed,
0 errors, 0 skipped**; the sole failure is the predicted unchanged
`test_soft_advisory_real_coarse_pec_short_witness`. This is an ordinary failing
assertion, not an xfail. The overall job returned 1 because of that failure.

| Capture | Maximum column power, original float32 capture | Original interval | Left / right settling, dB |
|---|---:|---|---|
| Coarse | 1.044479250907898 | Fails `(2.25, 3]`, as predicted | −25.4948906971 / −49.5955696612 |
| Fine | 1.0494229793548584 | Passes `<= 2.25` | −47.4758522315 / −103.1566072296 |

Both drives have nonzero incident and reflected waves, and both transmission
terms and every opposite-side V/I record are exactly zero. Offline
adjudication completed successfully. The causal question is closed: the former
interval encoded an invalid outgoing-power sum that the repaired geometry and
port placement no longer produce. Coarse absolute reflection accuracy remains
unqualified, and the physical advisory test remains red.

Live files are in
`output/931-fixture-repair-vessl/issue931-fixture-repair-short-20260909T015010Z-b36fc46cdf21-101/`;
the audited report is
`output/931-fixture-repair-live-20260909/short-adjudication.json`.
The report contains submitter provenance, source SHA, file hashes, original
capture maxima and independently recomputed complex128 powers. Their roughly
1e-7 arithmetic differences are recorded separately, not silently substituted
for the original readings.

**Exact identity applies to the committed historical controlled pair**, not to
the fresh VESSL run versus another host. The fresh maximum complex S difference
from the archived repaired capture is **3.7558317e-6** coarse and
**7.1714693e-6** fine. The largest V/I trace difference divided by its archived
record peak is **3.3856793e-7** coarse and **5.1581407e-7** fine; 12/16 coarse
records and 8/16 fine records remain exactly identical. These small cross-host
differences do not change the topology, interval disposition, or settling
classification. No exact cross-host reproducibility claim is made.

The fresh [inspection figure](short-live-369367259618.png) contains all coarse
and fine S bins, left/right V/I time-record envelopes, the historical
double-counted power decomposition, and every-bin cross-host S differences.
It was rendered and inspected with `view_image`: the coarse 4 GHz deficit,
late left-drive lobe, and absence of cross-plate transmission remain visible.
The envelope panels use block maxima normalized to each individual record;
their plotted dB values are **not** the API's end/peak-energy settling metric.
Reproduce the compact figure with:

```bash
python scripts/diagnostics/plot_931_short_adjudication.py \
  --report output/931-fixture-repair-live-20260909/short-adjudication.json \
  --output docs/design_notes/931_migration/fixture-repair-evidence/short-live-369367259618.png
```

## Reproduce with named live evidence

Run after the short VESSL job finishes, retaining the submitter's `run_id.txt`
above the capture directories:

```bash
python scripts/diagnostics/adjudicate_931_short_repair.py \
  --coarse "$SHORT_OUT/coarse/advisory_current.json" \
  --fine "$SHORT_OUT/fine/advisory_current_fine.json" \
  --output "$SHORT_OUT/short-adjudication.json"
```

If the captures have been copied away from that directory, supply `--run-id`
with the actual submitter-recorded numeric ID. The adjudicator rejects a
placeholder, checks that both captures identify the same source SHA, and hashes
every input and its own source. Its report is the named live result; this note
does not imply that an unexecuted or local capture is VESSL qualification.

The JSON retains all six bins of every complex S entry, each reflection and
reported transmission contribution, both column powers, driven a/b spectra,
trace comparisons, settling, and all production warnings. Original interval
passes are reported separately from whether the causal adjudication succeeds.
Successful adjudication therefore does **not** mean the original pytest passed.

## Mechanism and independent evidence

The #931 contract samples volume occupancy at cell centres and makes every
incident edge PEC; an aligned slab owns both faces. See the normative
[`20260906_plan_realign_lattice_ownership.md`, lines 42–51 and 55–74](../../20260906_plan_realign_lattice_ownership.md).
The corrected builder retains the coarse realized plate at 84–86 mm, with
sources at 10 and 110 mm and physical reference/probe offsets of 6 and 20 mm
([`tests/_pec_short_advisory_fixture.py`, lines 16–42](../../../../tests/_pec_short_advisory_fixture.py)).
The compiled-plane contract checks source/reference/probe coordinates
10/16/30 mm and 110/104/90 mm, each in its own connected vacuum region
([`test_pec_short_advisory_geometry.py`, lines 11–28](../../../../tests/contracts/test_pec_short_advisory_geometry.py)).

The historical right source was at 90 mm, but its reference and probe were at
84 and 70 mm. The extractor reads the reference V/I records, then divides each
receiving outgoing wave by the driven incident wave
([`waveguide_port.py`, lines 1834–1837 and 2177–2189](../../../../rfx/sources/waveguide_port.py)).
With the current 84–86 mm plate, that old reference sits on the wall: zero
electric voltage and nonzero magnetic current produce equal modal a/b
magnitudes and a fictitious `|S21|` near 0.5. With the historical node sampling,
the plate moves to 86–88 mm and that reference samples the same left cavity as
the left port: the fictitious `|S21|` becomes approximately one.

The committed controlled intervention held compiled ports, material inputs,
grid, timestep, and duration fixed. Node sampling with the current volume
operator restored maximum column power **2.527903795**, nearly matching legacy
sigma damping's **2.527910233**. At 6 GHz the node-volume result combines
`|S11| = 1.133861328` and spurious `|S21| = 1.114568210`. Their squared sum
crosses the historical lower bound. This is same-cavity double counting,
compounded by a weak incident denominator at the band edge. The historical
right-drive exact-volume records are all zero: its probes cannot see its source.
See [the original controlled evidence, lines 19–93](../slow-consumer-evidence/advisory-result-note.md).

An additional exact comparison isolates the measurement repair from the
reflecting field itself. Committed historical `advisory_current.json` and
repaired `short-coarse.json` have **bit-identical S11 in all six bins**, and their
four left-drive/left-port V/I time records are also **bit-identical**. The old
spurious transmission contributes these powers:

| GHz | Left reflection power | Old falsely counted transmission power |
|---:|---:|---:|
| 4.0 | 0.30398699 | 0.26694266 |
| 4.4 | 1.04448018 | 0.25543059 |
| 4.8 | 1.01201042 | 0.24819883 |
| 5.2 | 0.99926635 | 0.24741284 |
| 5.6 | 0.99131092 | 0.25094141 |
| 6.0 | 0.89148843 | 0.23449574 |

The repair removes the second column in this table without changing the first.
It also makes the right drive observable. Both transmission terms and all
opposite-side traces are zero, as required for the closed full-cross-section
plate. The adjudicator strictly asserts the committed controlled identity and
reports fresh-run equality and all numerical differences independently; it does
not assume distinct CPU environments are bitwise identical.

## Predictions, falsifiers, and limits

The named live prediction is a repeated coarse miss, a fine control below 2.25,
nonzero incident/reflected waves at both driven ports, and exactly zero
cross-plate transmission and time records. A live interval hit, nonzero
cross-plate transmission, missing incident/reflected waves, changed fixture
coordinates/settings, or failure of the committed exact-identity comparison
falsifies this adjudication and causes the script to fail. Full fresh-vs-archive
differences remain visible even when these causal checks pass.

The prior repaired coarse maximum **1.0444802046** is a finite-window result.
Its left settling is **−25.4949 dB**, failing the unchanged **−40 dB** criterion;
the right settles to **−49.5956 dB**. Both retained absorbers are thinner than
the production diagnostic's documented floor. The fine control's maximum
**1.0494127274** passes its unchanged upper bound, and its settling passes, but
its band, duration and physical absorber thickness also differ. It is not a
fixed-configuration mesh-convergence experiment. These caveats are emitted in
the live adjudication and remain limits on absolute reflection accuracy.

The previously inspected historical [coarse figure](short-coarse-inspection.png) and
[fine figure](short-fine-inspection.png) show every bin, the nonzero driven
spectra, and voltage envelopes. The coarse 4 GHz deficit and late lobe are
visible; neither should disappear behind the near-unity maximum. These figures
remain historical inspection artifacts. The named live figure above is
generated directly from the VESSL capture paths in the adjudication report.

The causal finding closes the question of what the old interval meant. It does
not establish a replacement physical advisory witness or an absolute short
accuracy baseline. The live test retains its original interval assertion
([`test_sparam_passivity_guard.py`, lines 234–271](../../../../tests/unit/sparams/test_sparam_passivity_guard.py));
the independent synthetic tests retain advisory threshold coverage
(lines 170–230 of that same file). No production change is indicated by this
fixture attribution.
