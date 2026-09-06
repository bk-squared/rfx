# Evidence, calibration, expectation — the ownership decision (issue #928)

Date: 2026-09-06. Status: the methodology section is the durable part; the
findings and decisions below are dated and specific to this arc.

**Why this file and not `docs/agent-memory/development_methodology.md`.** That
file is the home of this repository's method, and this section belongs in it.
It is also gitignored in this public repo, so it exists only in the primary
checkout: a rule written there has no history, no diff review and no backup,
and a reader of the public repo cannot see it at all. The text is therefore
written here, in a tracked file, and the agent-memory copy — when it is next
edited from the primary checkout — should point at this note rather than
restate it.

---

## 1. What went wrong, stated plainly

cv04 measures a band-mean reflection and transmission envelope on a slab rig.
It never wrote that envelope down. cv22 and cv23 derive their windows from it,
so when they needed the numbers they cited the nearest committed copies: the
STUDIO UI fixture `tests/fixtures/golden_workflows/multilayer_fresnel.json` and
a source comment in `validation/crossval/04_multilayer_fresnel.py`. A gate test
then pinned the consumer equal to the UI fixture and grepped the source comment
for the string `max|R+T-1| = 0.0487`.

Two things were wrong, and only the second is interesting:

* a display fixture had become a physics source — the shallow defect;
* the slab rig itself, and cv04's envelope, were **declared in a module named
  after a consumer** (`comparators/cv22_dispersive_gates.py`), which ten
  modules imported, the producer among them. The direction was backwards, so
  every fix that stayed inside that module made the coupling tidier without
  making it right.

The diagnosis in the issue body — that the results directories were gitignored
— was wrong; they are tracked (`git check-ignore` exits 1, cv22 has 27
committed artifacts and cv23 has 39). The producer simply never emitted its own
envelope.

## 2. The method: evidence, calibration, expectation

Three things that were one thing, and what each is allowed to be:

1. **Evidence** is what a producer measured, in the producer's own artifact,
   recomputable from raw data, append-only by revision, hashed. It is never a
   gate and never a window.
2. **Calibration** is a consumer's decision to adopt one revision of that
   evidence. It lives in the consumer, names the revision, pins that revision's
   hash and the realized rig's hash, and records the gate policy in force when
   it was made. Changing it is an edit to a declaration, reviewed under
   "no silent gate loosening" — never a consequence of a run.
3. **Expectation** is the window, derived from the two by the shared policy and
   written down nowhere else.

The rules that follow from that split, in the form they were argued into:

* **No self-certification.** Re-running a producer must not widen the gates
  that judge it. Mechanically: a new revision appended to the artifact moves
  nothing until a consumer's adoption record names it. The artifact and an
  adoption record that cites it may not change in the same diff, with one
  `bootstrap: true` exemption per artifact for the revision that creates it.
* **A guarantee is a re-derivation from outside.** A test that reads the
  producer's artifact, recomputes the window with the policy arithmetic written
  out on its own side, and compares it against what the evaluator actually
  applies — the per-bin windows after the model-error term, the bin mask, the
  pass/fail boundary — not only the headline scalars. A reference syntax like
  `cite()` is a convenience, not a guarantee: replacing a grep for `1.5` with a
  grep for `cite(` changes nothing. Ownership is bound in the manifest instead:
  an artifact a comparator derives a gate from is registered under the case
  whose id equals its `producer`.
* **A hash is only worth what it covers.** Hash the configuration the run
  actually realized, as values (cells, timestep, record length, mask,
  preflight), not the configuration a script declares it wants.
* **Cancelling errors show up only against a varied rig.** A number agreeing
  with a reference at one rung is not converged. The witness is a sibling
  artifact with one rig variable changed (methodology §3.2's run-length or mesh
  2× check) plus an independent invariant — closure, passivity, energy. Where
  no such sibling exists, the value is `carried-unwitnessed` and says so.
* **Only a copy of the SAME measurement gets replaced by a reference.** Same
  realized-rig hash and same estimator. An independent oracle re-implementation
  and a frozen regression value are second witnesses, not copies; they carry
  their purpose and their source revision. Rig-varied siblings — a dx ladder, a
  longer-record diagnostic — are not copies of each other by construction.
* **A recorded window is an output, not a second source.** Artifacts record the
  window a run was judged against, and public pages cite those recordings by
  value. That is what makes a claim checkable. Exactly one INPUT source; any
  number of recorded outputs.
* **Per-arm gate tables are closed sets** — required / diagnostic / not
  applicable with a reason — and an unclassified arm fails.
* **Migration is a global rule plus an explicit classification**, not two cases
  fixed by hand. Fixing the two loudest cases leaves the rest in a worse mixed
  state than before.
* **Refactoring and measurement changes do not travel together.** A code motion
  merges only under a bit-identity gate; a changed number is a separate,
  explicitly adopted revision in its own change.

Two additions the reviews argued for, recorded here because they generalize
beyond this arc:

* **Completeness.** A required gate whose evidence is missing, skipped, None or
  non-finite cannot aggregate to PASS. Diagnostic invocations may legitimately
  omit a witness; claims-bearing ones may not, and the aggregate must name what
  was missing rather than skipping it silently.
* **A claims-bearing case whose gate tests skip in CI when an artifact is
  absent has the wrong role.** A case that can only be checked where the
  artifact happens to exist is a diagnostic reporter until its evidence is
  committed or its lane retains the run's output. (Stated here; the mechanical
  check goes with the manifest evidence rule.)

## 3. What this arc changed

* `validation/crossval/_04_fresnel_results/envelope.json` — the producer's own
  envelope, revision r1, with the four values, where each was previously kept,
  the measurement identity, the realized rig as values with per-key provenance,
  and the two hashes. Unknowns (the run's timestep, its commit) are null with
  the reason, not reconstructed from today's defaults.
* `comparators/slab_family.py` — a leaf module holding the slab rig, the gated
  band, the incident-pulse and ring-down helpers, the cell bookkeeping and the
  auxiliary-echo geometry, `staged_commit`, the envelope loader and the verdict
  aggregate. cv22 and `slab_rig` re-export every name, so their importers are
  unchanged; `04_multilayer_fresnel.py` reads the leaf and reaches no consumer,
  directly or through `lattice_witness` / `slab_rig` — a review found both of
  those longer edges after the direct one was removed, and
  `tests/crossval/test_producer_import_graph.py` now holds the property
  statically (an AST scan that also sees an import inside a function) and
  dynamically (a fresh interpreter per producer-side import).
* cv22 and cv23 each declare `CV04_ADOPTION`, DERIVE every window from it —
  cv23 included, which used to re-export cv22's three and therefore moved
  whenever cv22 re-adopted, with its own record untouched — and hold none of
  the envelope values in any spelling.
* Contracts, four of them and not three: the outside re-derivation with its
  three evidence scenarios (unadopted evidence inert, altered evidence refused
  from two directions, an explicit re-adoption moving only the adopter), and
  the same-commit guard, which is a fourth and is where the review found the
  hole — its exemption keyed on "the adopted revision is marked bootstrap",
  and r1 stays marked bootstrap forever, so every later rewrite was waived. It
  keys on the artifact's absence at the diff base now, and a written revision
  is immutable regardless. Alongside them: the policy pin, the staleness
  report, the ownership and fan-out contracts, the completeness flip (with the
  case's declared gate-name set, so an ABSENT gate is incomplete rather than
  invisible) and the code-motion identity gate.
* `tests/_git_tracked.py` — one shared answer to "is this a committed
  artifact?", used by both evidence gates. `Path.exists()` was not it.

What was deliberately NOT done, with the reason:

* The r3 record-length recipe stayed in cv22. It is that case's declaration,
  adopted by cv23; moving it into the shared module would relocate a consumer's
  recipe and re-create a wide re-export shim that cuts no coupling — the shape
  the 2026-06-23 review declined (see §4).
* cv04 was not re-run. Its corrected-injection rung (a per-bin closure reported
  as 0.0043 against r1's 0.0487) is a measurement change: revision r2, adopted
  explicitly, in its own change.
* No `validation/crossval/rigs/` directory, no `cite()` helper, no free-text
  witness fields. A field nothing checks is paperwork.

## 4. Two prior decisions this repository made and could not cite

Both were only in session memory. They are recorded here with their original
scope and what would reopen them — a STOP is not a permanent prohibition, it is
a claim about evidence.

### 2026-06-23 — refactoring review: 2 of 5 candidates shipped

Scope: a five-candidate refactoring review of the simulation/API surface. Two
shipped (PRs #218, #219). The rest were declined as churn, the load-bearing one
being a proposed registry shape for `rfx/api/_preflight.py`: it would have
replaced a readable sequence of `_validate_*` functions with a table plus a
dispatcher, moving code without removing a dependency or enabling a caller.
Also declined in the same review: splitting the coax module.

Scope limit: this says nothing about restructuring in general. It is a verdict
on re-export shims and registry tables that cut no coupling.

Supersession: a concrete caller that cannot be written without the new shape,
or a measured defect the current shape caused. "It would be tidier" is not new
evidence. This note's own leaf-module move was checked against that bar: the
producer could not stop importing a consumer without it.

### 2026-07-23 — the "sufficient milestone" decision

Scope: rfx reached research-sufficient capability — validated core physics,
differentiability, and an external cross-validation set. The decision was to
STOP proactive completeness work: no more features, coverage or polish
undertaken because a gap exists rather than because a user or a claim needs it.

Scope limit: it does not forbid correctness work, and it does not forbid
finishing something a live claim depends on. It forbids self-assigned breadth.

Supersession: a request from the PI, or a defect that a claim already made
depends on. Under that bar the present arc qualifies — public pages carry
numbers whose provenance was a UI fixture — and the work stayed inside that
justification rather than expanding to the whole manifest.

## 5. Two measurements this arc took, worth keeping

**The examples axis is already clean.** `tests/_example_fidelity_lib.py`
registers crossval comparators as path strings in a classification table and
loads them by path; it does not import them as a package. So the restructure
here is crossval-internal: no example, tutorial or public-docs consumer moves
with it, and the directory-restructure proposal floated in the issue (proposal
D) is not needed for this problem.

**Putting `docs/public/guide/benchmarks.mdx` under the numeric-provenance gate
cost nothing.** All 93 artifact references in it resolve against the committed
artifacts at this commit — zero failures on first run. The page that carries
the word "validated" was outside the gate for no reason other than that nobody
had opted it in, which is why the opted-in set is now enumerated with a checked
exclusion for every file that stays out.

## 6. Pointers

* Envelope artifact: `validation/crossval/_04_fresnel_results/envelope.json`
* Loader and rig: `validation/crossval/comparators/slab_family.py`
* Policy: `tests/_gate_policy.py::gate_from_envelope` (multiplier and quantum;
  a consumer's adoption record pins the values it adopted under)
* Contracts: `tests/contracts/test_calibration_envelope_ownership.py`,
  `tests/contracts/test_evidence_numeric_provenance.py`,
  `tests/crossval/test_slab_family_code_motion_identity.py`,
  `tests/crossval/test_producer_import_graph.py`
* Identity baseline: `tests/fixtures/slab_family_windows_baseline.json`
