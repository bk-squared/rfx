# Which lattice the witness is judged against — reference selection by ARRIVAL (#1015 remainder)

Date: 2026-09-15 · Lane: `feat/1015-cv26-reference-selection` · Standard under
revision: `docs/design_notes/20260903_lattice_witness_standard.md` §3 (the
window), §4 (GL1 per bin, GL2 band mean), §13 (GL1's validity domain, #1028).

## 0. Read this first — what kind of document this is

**This is NOT a falsifiable pre-declaration. It is a regression pin, and saying
otherwise would be dishonest.** Every condition G-A…G-G below was evaluated on
the committed records BEFORE this file was written, and every one of them
passed. The 0.90 bar in G-C was chosen after 0.823 had been measured. Nothing
in G-A…G-G can come out differently unless the in-comparator implementation
diverges from the scratch computation that produced the numbers — which is
precisely, and only, what these conditions are for.

The earlier `20260914_lattice_witness_gl1_predeclaration.md` was a genuine
pre-declaration: it fixed a rule for choosing among (A), (B) and (C) before the
table that decides it existed. This file is the other thing — a diagnosis that
was already complete, written down with the conditions that must keep holding
after it is implemented. Both are useful. They are not the same instrument and
this note does not dress one as the other.

What IS genuinely open is in §5 (the R2 question) and §6 (the questions for the
PI). Those are not measurements and this lane does not settle them.

## 1. The finding, in one sentence

`record["t_safe_cpml_steps"]` and the #892 auxiliary-echo arrival split cv26's
twelve witness entries in two, and that split — computed from the rig's
GEOMETRY, never from a residual — predicts which of two lattice references the
measurement actually matches, on 10 of 10 entries with a defined window: the
five entries whose absorber echoes are outside the record by ARRIVAL match the
absorber-FREE lattice, and the five that admit an echo by AMPLITUDE match the
lattice with the realized absorbers. cv26 judges all ten against the realized
one.

## 2. The predicate, written as code, and it reads no residual

```python
arrival_safe = (record["t_safe_cpml_steps"] >= run["n_steps"]
                and aux_echo_arrival_report(spec, cells, dx=..., dt=...,
                                            n_steps=...)["arrival_steps"] > run["n_steps"])
```

Both inputs come from the rig's layout. `t_safe_cpml_steps` is derived in
`oblique_fresnel.derive_record` from the CPML standoff and the fastest gated
group velocity (cv04's 0.95 rule); `arrival_steps` is
`oblique_fresnel.aux_echo_arrival_report`'s flight-minus-lead, pure geometry by
its own docstring. Neither reads `R_rfx`, any residual, any window or any
breach count. This is the SAME test the standard's §3 (Z1) and §13.3 already
use to declare the family's absorber terms zero by construction; it is not a
new envelope.

**Honest caveat on the "10/10".** The predicate's two clauses are perfectly
collinear on this data — `t_safe_cpml_steps ≥ n_steps` and `arrival_steps >
n_steps` agree in sign on all twelve entries — so no measurement here
distinguishes the AND from either clause alone. The 10/10 is ten entries, not
twenty independent bits.

## 3. The mechanism hypothesis, the observable, the intervention

- **Mechanism.** cv26's comparator hands `evaluate_e2` a witness reference that
  contains absorber content the record, by arrival, cannot contain. The GL1
  breaches on those entries are manufactured by the comparator's choice of
  reference, not by rfx.
- **Observable.** `mean |R_rfx − R_ref|` and `mean |T_rfx − T_ref|` over the
  gated band, for the realized-absorber reference and for the absorber-free
  one, against the geometry predicate of §2. It is a comparison of two
  references; it never reads a window and never counts a breach.
- **Intervention family.** Reference selection. No window moves, no budget term
  is added, `witness_domain`'s predicate is untouched, and the amplitude-capped
  branch is not touched at all.

## 4. The conditions — G-A … G-G

All seven must hold. They are implementation-fidelity pins (see §0).

**G-A — the predicate is geometry.** `arrival_safe` must be True on exactly
`{te_00, tm_00, te_30, te_00__settle60, tm_00__settle60}` and False on
`{te_45, te_60, tm_45, tm_60, graze_vac, graze_pec, graze_te}`, computed from
`t_safe_cpml_steps` and `aux_echo_arrival_report(...)["arrival_steps"]` only.

**G-B — the mechanism's own witness (R5).** On the ten entries with a defined
window, the sign of `mean|X_rfx − X_realized| − mean|X_rfx − X_absorber_free|`
must agree with `arrival_safe` on 10 of 10, for X = R **and** for X = T.

**G-C — the number the attempt buys.** On every arrival-safe entry, judged
against the absorber-free reference: `GL1_R_bins_beyond == 0` and
`GL1_T_bins_beyond == 0` over ALL gated bins, and `max(|Δ|/W) ≤ 0.90` on both R
and T. cv26's `verdict.gl1_breaches_in_domain` goes 135 → 73 and the all-gated-
bin total 1291 → 989.

**G-D — no reported failure may be erased (#1028's A2 rule, re-applied).**
`_GL2_EXPECTED` unchanged for all seven primary arms; in particular `tm_60`
keeps `GL2_R = False` at 244 % of its window. Any flip is an outright reject.

**G-E — cv26's own gates do not move.** `gates`, `e2_ok`, `verdict.exit_code`
identical on all arms; `mean_dR_gated`, `mean_dT_gated`, `mean_window_R`,
`mean_window_T` unchanged; the four amplitude-capped arms' and the three
grazing arms' witness numbers unchanged.

> **G-E's baseline is a FRESH PRE-CHANGE REGENERATION, not the committed
> artifact, and its threshold is the measured run-to-run bound.** Regenerating
> `lattice_witness_replay.json` on unmodified `541f703f` and diffing it against
> the committed copy already gives **138 float mismatches, worst 1.785e-09
> relative** (`arms/te_45/domain_R/max_unmodelled_over_window_gated`
> 1.9106385190859112 → 1.9106385224958224), on exactly the arms G-E names, with
> **zero** count or boolean differences. Two consecutive runs on one revision
> ARE byte-identical after dropping `replay_commit` / `date_utc`. So "unchanged
> to 1e-12 against the committed file" is unsatisfiable on main today and would
> have been a false gate. G-E is evaluated as: fresh run on the unmodified
> worktree vs fresh run on the changed one, same session, same interpreter;
> counts and booleans must be EXACTLY equal and floats equal to ≤ 1e-9
> relative, the measured pre-existing drift bound. The earlier plan's claim that
> the replay reproduces the committed artifact "to ~1e-12" is **withdrawn** —
> it does not.

**G-F — the family does not move.** `lattice_witness.build_from_results` output
for **cv22 and cv23**, serialized with sorted keys, identical between
`origin/main`'s `lattice_witness.py` and the changed one, evaluated in one
process.

> **cv04 is NOT a leg of G-F, and the earlier plan was wrong to list it.**
> `rungs_from_results("validation/crossval/_04_fresnel_results")` returns `[]`
> — that directory holds `envelope.json`, `fringe_gate_geometry.json`,
> `lattice_witness.json` and `RECOMPUTE.md`, and no `rfx.json` for the glob at
> `lattice_witness.py:747-748` to find — so `build_from_results("cv04", ...)`
> yields a 147-byte zero-rung document and the comparison is two empty
> documents comparing equal. That is a guaranteed pass that proves nothing. The
> standard's own §13.7 falsifier used cv22 + cv23 only (3 and 9 rungs); this
> one does the same, and adds a direct check that no existing callable in
> `lattice_witness.py` changed at all.

**G-G — the attempt does not overclaim.** `GL1_gated` stays `false` and
`verdict.gl1_gated` stays `false`. The 73 remaining in-domain breaches and
`tm_60`'s GL2_R failure are REPORTED, not closed. Issue items 2 and 3 do not
close.

## 5. R2 — the ledger, stated under both readings, including the one that goes against this lane

The standard's own §13.7 reads, verbatim:

> R2: attempt 1 on "what does GL1's window omit".

That sentence counts option (C) — the validity domain shipped in #1028 — as
attempt 1 on a mechanism whose name is broad enough to contain this lane. **It
must be confronted, not omitted.** The ledger under each reading:

**Broad reading — mechanism = "what does GL1's window omit".** Then the spent
attempts are: the exact second-order term `(δ_scat + δ_round)²` (FALSIFIED,
58 → 54 and 110 → 109; close note §10.2/§10.4, standard §13.7); option (A),
carrying `U` in the budget (measured and REJECTED on A2/A4/A5, standard §13.3);
option (C), the validity domain (SHIPPED, #1028, and what §13.7 calls attempt
1). Under this reading reference selection is **attempt 2 or 3** on one
mechanism, the RF/EM N≥1 intensifier forbids it, and this lane should not be
merged without an explicit PI override.

**Narrow reading — mechanism = "the comparator's REFERENCE contains absorber
content the record cannot contain".** Then attempts spent: **0**, and this is
attempt 1. The three above all act on the BUDGET or on the DOMAIN; this one
acts on neither. It is decided by a predicate that reads no residual, it moves
no window's definition, and on the five entries it touches it makes the
omitted term zero rather than bounding it or excluding it.

**Which reading this lane acts on, and why.** The narrow one — but the argument
that carries it is not "it is a new mechanism", it is **"it is not a mechanism
attempt at all"**. §3 (Z1) and §13.3 already declare the rule: an absorber echo
outside the record by arrival is zero by construction and the reference has no
absorber in it. Five of cv26's ten entries satisfy that arrival test and were
nonetheless judged against a reference carrying the realized absorbers. That is
a case not applying a rule the standard already binds it to — a conformance
defect in the comparator, the same class as the 13 entries in
`docs/agent-memory/rfx-known-issues.md`'s comparator-bug ledger — not a new
hypothesis about the window.

**This is a judgement, not a fact, and the PI owns it.** §6 puts it as an open
question. The PR is opened, not merged, for that reason.

## 6. Open questions for the PI — none of these is settled here

1. Does §13.7's "attempt 1 on 'what does GL1's window omit'" fence reference
   selection? §5 argues no; the text's scope is a PI call.
2. #1028's override was argued on "(C) moves nothing". This lane moves cv26's
   reported GL1 / domain numbers on five of twelve entries (toward zero), while
   moving no window definition, no family number and no reported failure. Does
   that clear the same bar?
3. The self-cancelling `U`. At all 20 of the in-domain breaches that exceed
   `W + U`, the two components `cpml3d_term` and `aux_echo_term` each
   individually exceed the net `U` by 1–10×; using `U_cpml + U_aux` in the same
   predicate would take the residue 135 → 96, or → 3 in-domain breaches beyond
   the matching bound. That changes the predicate #1028 shipped and is a
   SECOND attempt with its own pre-declaration. **Explicitly NOT bundled here.**
4. Items 1–3 of the issue do not all close and this lane does not pretend they
   do. Proposed disposition: item 1 closes PARTIALLY (135 → 73); item 2
   (`GL1_gated`) does NOT close; item 3 (`tm_60` GL2_R) stays a pinned failure
   with its trace attached to the already-named absorber candidate, untested.

## 7. The STOP rule

If any of G-A…G-G fails on the implementation, the change is reverted, nothing
is merged, and #1015 gets the audit table of §2 alone.

## 8. What this lane does NOT touch

`lattice_witness.evaluate`, `budget_terms`, `windows_from_terms`,
`witness_domain`'s predicate, `ceiling_windows`, `ringdown_rate`; cv26's E2 /
Fresnel gates (`gates`, `e2_ok` are assembled at `oblique_fresnel.py:1416`,
BEFORE the `if cells is not None` lattice block at `:1420`, and never read the
lattice — the gate-invariance of G-E is a property of the code, not a hope);
`evaluate_grazing_pec` / `evaluate_grazing_slab`; `lattice_margin`;
`mean_W_lat_R_gated` / `mean_W_lat_T_gated`, which stay on the REALIZED lattice
so the live console line at `26_oblique_slab_fresnel.py:487` does not move.
