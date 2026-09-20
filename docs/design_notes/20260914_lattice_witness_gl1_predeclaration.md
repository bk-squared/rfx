# GL1's per-bin window and the unmodelled absorber term — pre-declaration (#1015)

Date: 2026-09-14 · Lane: `fix/1015-lattice-witness-gl1` · Standard under revision:
`docs/design_notes/20260903_lattice_witness_standard.md` §3 (the window) and §4
(GL1 per bin, GL2 band mean).

**This file is written and committed BEFORE any number in it is computed.** It
exists so the option chosen for #1015 can be checked against a rule that was
fixed in advance, rather than against a rule read off the answer. Nothing below
is a result.

## 0. What #1015 found, and what is still open

cv26 is the first oblique consumer of the standard. With the corrected source
tau (#1011), GL1 breaches on five of its seven primary arms and `tm_60` fails
GL2_R at 244 % of its window. The breach count separates the ends of the range
by the ratio "the absorber term the standard's budget does not model, divided by
the window" (0.07× / 0.04× → 0 breaches; 24.4× → 357), is loose in the middle,
and `graze_te` (172×, 0 breaches) is a counter-example to reading it as a law.
The earlier reflection-null reading is withdrawn.

#1015 lists three options and investigates none:

- **(A)** carry the absorber / auxiliary-echo term in the budget where the case
  measures one;
- **(B)** demote GL1 to reported, keep GL2 as the gate;
- **(C)** declare GL1's validity domain and say where it does not apply.

## 1. The quantity, named before it is measured

**cv26's absorber term** is a property of the REFERENCE, not of rfx. In
`validation/crossval/comparators/oblique_fresnel.py::evaluate_e2`,

    absorber_term_R_gated_max = max_{gated bins} | R_lattice(rig absorbers)
                                                 − R_lattice(ideal_absorber=True, aux="plane") |

— the size of the 2-D-at-fixed-`k_y` lattice's own dependence on the two
absorbers the rig carries (the 3-D CPML and the auxiliary grid's own, #888).
No rfx number enters it. It is written to
`validation/crossval/_26_oblique_results/lattice_witness_replay.json` as
`arms.<arm>.absorber_term_R_gated_max`, with `absorber_term_over_window_R` the
ratio against `mean_W_witness_R_gated`.

**The slab family's equivalent.** cv04 / cv22 / cv23 records carry no such key.
The measurement below must therefore say what CAN be computed for them and what
cannot:

- their reference (`dispersive_eps.yee_lattice_slab_rt_model`) is the INFINITE
  lattice's steady state and contains no absorber of any kind, so the same
  difference is identically zero by construction, not by measurement;
- their two zero-by-construction preconditions are asserted per rung and can be
  read from the committed artifact: `gates.precond_cpml_gate`
  (`t_safe_cpml_steps ≥ n_steps`) and `gates.precond_aux_echo_record`
  (`aux_echo.ok`, the #888 arrival invariant).

The closest computable analogue for the family is the #888 arrival margin
`aux_echo.echo_arrival_steps / n_steps`. It is an ARRIVAL statement, not an
amplitude one; the measurement will report it as such and will not present it as
the same quantity cv26 measures.

## 2. Decision rules — fixed here, applied to the table later

### Pick (A) — carry the term in the budget — iff ALL of:

- **A1.** The term is definable from a key every consumer either carries in its
  record or can assert identically zero for, with the scaling by which it enters
  `W_witness` stated explicitly (which power of `R_lat`, which side of the
  first-order expansion).
- **A2.** Adding it flips NO currently-reported failure to a pass. Concretely:
  `tm_60`'s `GL2_R = false` in `lattice_witness_replay.json` must stay false.
  *If it flips, (A) is rejected outright* — a budget change that erases a
  reported failure absorbs a finding instead of reporting it.
- **A3.** Every slab-family window is unchanged: `lattice_witness.json` for
  cv04, cv22 and cv23 rebuilds byte-identical.
- **A4.** With the term carried, GL1 holds on every cv26 arm that has a defined
  window.
- **A5.** §3's "what makes it not tuned" survives. In particular claim 2 — the
  a-priori `ceiling_windows` bounds a passing run's window from above — must
  still hold, or the note must declare an a-priori BAR for the new term the way
  `SETTLING_BAR` and `PURITY_BAR` are declared for the existing ones. A term
  whose only available size is the measured one is a fitted term, and the
  standard's §3 claim 1 ("none of them is the residual being gated") forbids it.

### Pick (B) — demote GL1 to reported — iff:

- **B1.** Over EVERY committed rung × falsifier kind of cv04, cv22 and cv23
  (`lattice_witness.json::rungs.<rung>.falsifiers.<kind>.gates`), there is no
  case where GL1 fails while GL2 passes. That is: GL1 is never the only gate
  that fires, so demoting it costs no discriminating power. AND
- **B2.** (A) is rejected.

### Pick (C) — declare GL1's validity domain — iff:

- **C1.** B1 is FALSE — at least one committed falsifier fires GL1 while GL2
  passes, so GL1 carries discriminating power GL2 does not have and demoting it
  loses something measurable. AND
- **C2.** The domain predicate leaves every slab-family rung INSIDE the domain,
  so no numeric gate moves anywhere and `lattice_witness.json` rebuilds
  byte-identical for cv04 / cv22 / cv23; and puts cv26's breaching arms outside
  it. AND
- **C3.** The predicate is stated as a PRECONDITION on the window's validity —
  "the omitted term must sit inside the window that omits it" — and NOT as a
  prediction of breach count. Under that reading `graze_te` (172×, zero
  breaches) is outside the domain and is not a counter-example; a predicate
  phrased as a prediction would be refuted by it.

**The threshold, declared now and not fitted later.** The predicate is

    unmodelled_term / mean W_witness  ≤  1

— one, because a term LARGER than the window that omits it cannot be argued to
be inside a first-order budget. The measurement will report where every arm
falls against this threshold; the threshold will NOT be moved to improve the
agreement with the breach counts. If the data wants a different number, that is
reported as the data disagreeing with the pre-declaration.

### Tiebreak

If both (B) and (C) are satisfiable, take **(C)**: it keeps a gate that has
demonstrated discriminating power, and it moves no committed number.

If none of (A), (B), (C) is satisfiable under its own rule, nothing is
implemented and #1015 gets the table plus the statement that all three options
fail their pre-declared test.

## 3. What the measurement must produce before any option is chosen

One table over EVERY consumer arm of the standard — cv04's rung, cv22's rungs,
cv23's rungs, cv26's twelve entries — carrying, per arm:

1. the current per-bin window (`mean_W_witness_R_gated`, and T);
2. the per-bin residual `|rfx − lattice|` (`mean_dR_lattice_gated`, and T);
3. the GL1 breach count on the gated bins;
4. the GL2 verdict;
5. the arm's absorber term as cv26 defines it, or — for the family — the
   statement that the quantity is zero by construction plus the #888 arrival
   margin that IS computable.

and for (A) additionally: each consumer's window before and after, per arm. A
window that widens is a gate loosening and goes in the table and the PR body; it
is not hidden.

## 4. R2 and R3

R2: this is attempt 1 on the "what does GL1's window omit" mechanism. The one
previous attempt on the adjacent mechanism — carrying the exact second-order
term `(δ_scat + δ_round)²` — is recorded FALSIFIED in the close note §10.4
(58 → 54, 110 → 109) and is not retried here.

R3 falsifier, declared before the work: the slab family's committed
`lattice_witness.json` (cv04, cv22, cv23) must rebuild byte-identical from the
committed records after the change. If any of the three moves, the change
touched the family and the claim "no numeric gate moves" is false.
