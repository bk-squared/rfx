# Stage-0 verdict — record length fixed by measurement, and the benchmark premise gets its first datum (2026-08-27)

Jobs 369367256474-478. Classical two open stubs (λ/4 at 5.25 and 5.775 GHz, 8 mm apart)
inside the bounded 12 × 9 mm box, imperative hard-PEC solve, rfx ring-down witness.

| periods | n_steps | record | DFT res | settling | lower notch | upper notch | peak between | wall |
|---|---|---|---|---|---|---|---|---|
| 20 | 9 178 | 2.22 ns | 0.450 GHz | −43.6 dB | −60.2 @ 5.050 | −37.7 @ 5.500 | 1.9 dB merged | 29 s |
| 30 | 13 766 | 3.33 ns | 0.300 GHz | −66.6 dB | −63.9 @ 4.975 | −42.6 @ 5.500 | 2.6 dB merged | 34 s |
| **45** | 20 650 | 5.00 ns | 0.200 GHz | −102.8 dB | −66.5 @ 4.975 | −43.1 @ 5.500 | **3.4 dB resolved** | 48 s |
| 90 | 41 299 | 10.00 ns | 0.100 GHz | −125.9 dB | −71.2 @ 4.975 | −43.1 @ 5.500 | 3.4 dB resolved | 83 s |
| 140 | 64 243 | 15.56 ns | 0.064 GHz | −126.5 dB | −71.2 @ 4.975 | −43.1 @ 5.500 | 3.4 dB resolved | 129 s |

## Windows (decided, no longer a guess)

- **Descent / sweep window: 45 periods.** First window where the two notches separate
  (3.4 dB peak between them) and where the answer stops moving — 45, 90 and 140 agree on
  both notch centres and on the separation. Settling −102.8 dB, far inside the −40 dB
  requirement. 48 s per forward solve on one 4090.
- **Verification window: 90 periods.** Notch depth is still deepening between 45 and 90
  (−66.5 → −71.2 dB) and then stops (140 gives −71.2 too), so 90 is where depth is
  converged as well as position. 83 s.
- The Phase-1 window (10 periods) is disqualified for this spec: the CPU check failed
  the settling witness at −18.8 dB and merged the notches entirely.

Cost is much better than the plan feared (the plan estimated ~200 s/iteration at
90 periods from Phase-1 scaling; measured forward is 83 s, and the sweep runs at 48 s).

## First datum on the benchmark premise — and two corrections to how it was first read

The textbook design lands its notches at **4.975 and 5.500 GHz against targets of 5.25
and 5.775 GHz: −5.2 % and −4.8 %.** When first reported (2026-08-27, chat) I attributed
that pull to inter-stub coupling and called it consistent with the classical literature.
**That was wrong, and our own data refutes it.**

**Correction 1 — the pull is a calibration offset, not coupling.** The Phase-1 stub
sweep contains six *single*-stub solves with no second stub anywhere. Analytic λ/4
frequency vs measured notch:

| L (mm) | 5.80 | 6.10 | 6.40 | 6.70 | 7.00 | 7.37 |
|---|---|---|---|---|---|---|
| offset | −5.6 % | −4.9 % | −6.0 % | −4.6 % | −5.1 % | −5.1 % |

Mean ≈ −5.2 %, i.e. **the same offset the two-stub design shows**. It is the open-end
and T-junction fringing extension plus the stub's own ε_eff differing from the line's —
a constant an engineer removes by calibrating `f(L, W)` once. Three further arguments
agree: a shunt stub's transmission zero is pinned at its own resonance for any
separation (S21 = 0 ⟺ Y_stub → ∞); an ideal-line ABCD sweep over 2–24 mm separation
moves the zeros by +0.03 %; and Rahman's 10–16 % is *mutually coupled CSRRs*, a
different mechanism from shunt stubs. Also worth noting: 8 mm ≈ λg/4 at 5.5 GHz, so
Stage-0 happened to pick the canonical inverter spacing, not a pathological one.

**Correction 2 — the window criterion was stated wrongly.** The plan sized the record by
DFT resolution 1/T. For a *fully settled* transient the DFT is exact at any evaluation
frequency; 1/T bounds only the separation of two continuing sinusoids, i.e. what may be
*claimed*, not the accuracy. The measurement proves it: scored on the frozen metric, the
Stage-0 design reads **M = 23.74 / 23.75 / 23.75 / 23.75 / 23.75** at 20/30/45/90/140
periods — identical to 0.01 dB. **Settling is the accuracy gate** (the 10-period run was
invalid because it settled only to −18.8 dB, not because of resolution), and the
resolution figure survives only as an interpretation limit: no feature narrower than
100 MHz may be claimed, no notch centre quoted better than ±50 MHz.

**Correction 3 — what actually makes this spec hard.** Scoring the Stage-0 design on the
pre-registered metric is more informative than its notch frequencies: it already gets
**S_L = 0 and S_U = 0 — both WLAN bands are rejected by more than 20 dB.** Its entire
score comes from **S_G = 15.0** (the 5.45–5.625 GHz gap between the bands must transmit
at ≤ 10 dB and is instead fully blocked) and **S_P = 8.75** (passband loss). The naive
design is not two notches; it is one wide merged notch.

So the benchmark's difficulty is **not** hitting two close frequencies — that is easy.
It is producing skirts sharp enough to leave a transmitting gap between bands 525 MHz
apart while keeping the passband, inside a box that admits about two resonator planes
where the Butterworth shape factor says the spec needs four to five per band. That is a
better-identified and more defensible premise than the one the plan opened with, and it
is what arm D must be given every chance to defeat.

## Next

Arm-D protocol is being designed under adversarial review before any sweep runs
(the Phase-1 retraction came from an un-calibrated baseline, so the baseline design is
the thing to get right). The gate it must produce: a precise inequality under which we
declare classical *can* meet the spec in this box — in which case this benchmark is
discarded and we choose another problem rather than running the gradient arms.
