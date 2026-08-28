# Issue #683 decision experiment: wire-port V/I sampling order (pre vs post source injection)

Status: PROTOCOL, written and committed BEFORE any measurement run.
Author lane (agent), 2026-08-29. Branch `agent/issue-683-vi-sampling-order`.

## 1. The question

The uniform lane accumulates the wire-port V/I DFT **before** soft-source
injection (`rfx/simulation.py`, issue #72 contract); the non-uniform lane
accumulates **after** (`rfx/nonuniform.py:1394-1436`, status quo ante). Both
injections are additive/soft, so source semantics do not separate the lanes
(the hard-vs-soft argument is WITHDRAWN, per the #683 comment history and the
block comment at the NU accumulation site). Which sampling instant yields the
physically meaningful port V is OPEN. This protocol decides it by
measurement against a ground truth that does NOT run through the contested
S-parameter normalization.

## 2. Physical ground truth (independent of the contested normalization)

At a driven wire port whose gap is connected, through PEC conductors, to a
known external resistance `R_L`, the *terminal* phasors must satisfy the
quasi-static circuit law

    ρ(f) := +V_mid(f) / I_mid(f) = (R_L + jωL_loop) / n_live        (QS)

where `V_mid = -E_c·d` and `I_mid` (Ampère loop, right-handed about the
component axis — the repo convention under which a PASSIVE port cell reads
`-V/I = +Z`) are the sampled per-cell quantities at the port's mid cell,
`n_live` is the number of live wire cells (the per-cell V is 1/n_live of
the whole-column V by symmetry of identical cells), and `L_loop` is the
small loop inductance (est. < 10 nH, i.e. |ωL| < ~3 Ω at 0.1 GHz). Hence
at the lowest bins:

    Re ρ(f_low) · n_live = R_L,   independent of the driven port's own Z0.

Sign note (amended pre-measurement, before the harness existed; the first
draft wrote `-V/I` here by analogy with the passive-port convention, which
is WRONG for this loop): both columns are sampled with the same +z
orientation, while the loop current runs UP one column and DOWN the other,
so `I_drv = -I_load`; KVL around the PEC loop gives per-cell
`V_drv = V_load`, and the passive load column obeys the repo-verified
oracle `V_load = -(R_L/n_live)·I_load`. Substituting:
`V_drv = +(R_L/n_live)·I_drv`, i.e. the decision quantity is
`ρ = +V/I` with POSITIVE slope `1/n_live` in `R_L`.

Derivation: KCL at the port face makes the Ampère-loop `I` the series loop
current; KVL around the PEC loop (PEC edges contribute no EMF, quasi-static
flux term = jωL_loop·I) makes the driven-column voltage equal
`I·(R_L + jωL_loop)`. Every internal detail of the port cell (its folded
σ = n_live·d/(Z0·dp1·dp2), its gap capacitance, the impressed current) drops
out of the ratio — *provided V and I are a consistent terminal pair*. A V
sampled at an instant that is not any field time level (i.e. missing the
same-step injection increment at the sampled cell) breaks the pair by a term
proportional to the drive, which is what the experiment detects.

Deliberately NOT used as oracle: `S11`, the `(1-n_live)/(1+n_live)` closed
form (documented self-consistency lock on the known-wrong #313/#318 per-cell
normalization), or any quantity referencing the whole-port Z0.

## 3. Fixture (absolute physical coordinates, preflight ON)

Uniform cubic mesh `dx = 1 mm`; domain `(16, 12, 12) mm`; boundary `pec`
(closed box; lowest cavity resonance ≈ 15.6 GHz, two decades above the
measurement bins, so the band is quasi-static). If the NU lane cannot run
`boundary="pec"` (engineering failure at setup time, before any decision
quantity is read), BOTH lanes fall back to `boundary="cpml", cpml_layers=6`
with the same interior geometry, and the fallback is recorded. No other
fixture change is permitted after the first decision run.

- Driven port: `add_port(position=(5e-3, 6e-3, 5e-3), component="ez",
  impedance=50, extent=1e-3, excite=True,
  waveform=GaussianPulse(f0=2e9, bandwidth=0.9), direction="+x")`
  → live cells (5,6,5) and (5,6,6), `n_live = 2`, mid cell (5,6,6).
- Load port (the known load): `add_port(position=(11e-3, 6e-3, 5e-3),
  component="ez", impedance=R_L, extent=1e-3, excite=False,
  direction="+x")` → live cells (11,6,5), (11,6,6); per-cell
  σ = 2·d/(R_L·A) so the column is a resistor of total value R_L.
- PEC loop closing the circuit (2-cell-thick bars so interior edges exist):
  - bottom bar: `Box((4e-3, 5e-3, 3e-3), (12e-3, 7e-3, 5e-3))`, material pec
  - top bar:    `Box((4e-3, 5e-3, 7e-3), (12e-3, 7e-3, 9e-3))`, material pec
  Bars attach to the column end nodes via the PEC Ez stubs (i,6,4) and
  (i,6,7) at i = 5 and 11; bars do not touch the domain walls and do not
  overlap any port extent cell (ports live at k = 5,6).
- Frequencies: `[0.05, 0.1, 0.2, 0.5] GHz`; decision bins f1 = 0.05 GHz,
  f2 = 0.1 GHz. `n_steps = 4096` (t_end ≈ 7.8 ns ≫ pulse end ≈ 1.4 ns; all
  discharge time constants < 0.1 ns through the loop resistances).
- `R_L` sweep: `{12.5, 25, 50, 100, 200, 400} Ω`. (R_L = 0 is excluded —
  `add_port` requires positive impedance; a PEC-short anchor fixture, load
  column replaced by `Box((11e-3, 5e-3, 5e-3), (12e-3, 7e-3, 7e-3))` pec and
  no load port, is run as a *diagnostic only*, expected Re r ≈ 0.)

## 4. The two arms

The toggle is NOT a production edit. On a uniform-valued `dz_profile` the
two lanes discretise bit-identical geometry, and PR #684 established
(separability proof recorded in the research ledger) that with #672/#673
merged the only live difference at an excited wire port is the sampling
slot. Therefore:

- **Arm PRE** (accumulate before injection): uniform lane,
  `sim.forward(n_steps=4096, port_s11_freqs=FREQS)` → raw
  `(v_dft, i_dft)` from `fr.wire_port_sparams` (contract slot:
  `rfx/simulation.py` scan, before the soft-source loop).
- **Arm POST** (accumulate after injection): NU lane, same `Simulation`
  kwargs plus `dz_profile=np.full(12, 1e-3)`,
  `sim.run(n_steps=4096, compute_s_params=True, s_param_freqs=FREQS)`
  (preflight ON). Raw accumulators surfaced by a purely ADDITIVE result key
  `wire_sparams_raw` added to `rfx/nonuniform.py::run_nonuniform`'s result
  dict (no ordering change, no behavioral change to existing keys), captured
  by a harness-local wrapper around
  `rfx.runners.nonuniform.run_nonuniform`.

Arm-lane confound is not assumed away; it is *gated* (G2 below): the two
lanes must be shown, on this very fixture, to differ exactly by the
same-step injection increment, or the experiment reports itself unable to
decide.

## 5. Gates (fixture validity — checked BEFORE reading any decision quantity)

- **G0 — preflight**: both lanes' runs execute with preflight enabled and
  raise no error. Warnings are recorded verbatim.
- **G1 — the load is in the driven circuit** (the failure mode of the
  earlier independent repro). Uses only ordering-independent observables
  (I reads H only; the load port is passive so its V is unaffected by the
  injection slot on either lane):
  - G1a: `|I_drv(f1)|` is strictly decreasing across the R_L sweep and
    `|I_drv(f1; R_L=12.5)| / |I_drv(f1; R_L=400)| ≥ 2` in EACH arm
    (circuit prediction ≈ (50+400)/(50+12.5) ≈ 7.2).
  - G1b: `max/min over R_L of |V_load(f1)| ≥ 1.5` in EACH arm (circuit
    prediction ≈ 4.4).
  If G1 fails in either arm, the fixture is INVALID and the session stops
  and reports so. No tuning, no second fixture.
- **G2 — lane difference is exactly the injection increment** (separability
  de-confound). Define, per lane, the mid-cell injected table `W(t_n)` (the
  ΔE added per step, captured from the actual source specs handed to the
  scan) and `Ŵ(f) = Σ_n W(t_n)·e^{-jω n dt}·dt`, and the normalized
  transfer `G(f) = V̂(f)/Ŵ(f)`. From `V_pre(t_n) = V_post(t_n) + d·W(t_n)`
  (V = -E·d, injection E += W):

        G_PRE(f) − G_POST(f) = d_par = 1e-3 m.

  Gate: `|G_PRE(f) − G_POST(f) − 1e-3| ≤ 0.1e-3` at f1 and f2 on the
  R_L = 50 fixture, and dt identical across lanes. If G2 fails, the lanes
  are not a clean pre/post toggle and the session reports INCONCLUSIVE
  (arm confound unresolved) — no verdict, no production change.

## 6. Pre-declared decision falsifiers

For each arm, least-squares fit `Re ρ(f1)` against `R_L` over the six sweep
points, `ρ = +v_dft/i_dft` at the driven mid cell (§2 sign note): slope
`a`, intercept `b`.

- An arm **PASSES the circuit law** iff `n_live·a ∈ [0.90, 1.10]` AND
  `n_live·|b| ≤ 10 Ω`.
- **F1 (decision)**: exactly one arm passes ⇒ that arm's sampling instant
  is the physically correct wire-port V/I sampling order, and the other is
  refuted. **F2 (falsifier)**: both arms pass, or both fail ⇒ the
  experiment is INCONCLUSIVE; report it as such, stop; no lane change, no
  re-tuning, one attempt only.
- Robustness clause (declared now): the same fit at f2 must give the same
  verdict; if f1 and f2 disagree on which arm passes, report INCONCLUSIVE.

Prediction (recorded for honesty, not load-bearing): the terminal-consistent
arm reads `n·a ≈ 1, b ≈ 0`; the contaminated arm reads
`r ≈ r_true + Cb·Ŵ-scaled·(Z0+R_L)`-type behavior — slope far above 1 with a
large positive intercept. If instead the PRE arm passes, then the uniform
lane's #72 contract is the correct one and the NU lane is the outlier; the
protocol accepts either outcome.

## 7. Corroborating (non-deciding) check — the discrete Ampère identity

The previously UNEXAMINED claim: post-injection E satisfies the port-cell
Ampère/update identity. Exact discrete form used (derived from the scan
update `E^{n+1} = Ca·E^n + Cb·(curlH)^{n+1/2} + W^n`, all three sampled
quantities stamped `t = n·dt` by the accumulators, zero initial field,
decayed tail):

    (1 − Ca·e^{−jω dt})·Ê(f) − Cb·Î(f)/A_dual − Ŵ(f) ≈ 0,  Ê = −V̂/d

(the shift factor is `e^{−jω dt}`: the accumulators stamp `E^{n+1}` with
phase `e^{−jω n dt}`, so `Σ E^n e^{−jωn dt} = e^{−jω dt}·Σ E^{n+1}
e^{−jω n dt}` for a zero-initial, fully decayed signal; corrected
pre-measurement together with the §2 sign note)

with Ca, Cb built from the mid-cell ε and σ (port σ included). Report the
relative residual `|LHS| / |Cb·Î/A_dual|` for both arms at f1, f2. Expected:
small (≲ few %) for the arm whose Ê is the true field level, O(1) for the
other. This corroborates but does NOT decide (decision is §6 only).

## 8. Outputs and stop conditions

The harness `validation/research/issue683_sampling_order_decision.py` runs:
G0 → G1 → G2 → sweep fits → F1/F2 verdict → §7 residuals → PEC-short
anchor, prints a machine-readable summary, and exits nonzero on gate
failure. Failed runs, fired falsifiers, and "inconclusive at CPU scale" are
valid, reportable outcomes. Estimated cost: 14 runs of a 16×12×12-cell,
4096-step FDTD — minutes on CPU; no GPU/VESSL needed. If any single run
exceeds the CPU budget, a VESSL YAML is emitted instead of running locally.

Production sampling order is NOT changed in this session regardless of
verdict; the xfail witness `test_excited_port_lane_ordering_disagreement_
is_open_683` stays as-is. The verdict and artifacts feed the follow-up
change under #683 review.
