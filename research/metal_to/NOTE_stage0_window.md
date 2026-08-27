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

## First datum on the benchmark premise

The textbook design — each stub a quarter wave at its own band centre — lands its
notches at **4.975 and 5.500 GHz against targets of 5.25 and 5.775 GHz: −5.2 % and
−4.8 %, both pulled DOWN**, with the two notches barely separated (3.4 dB). The pull
direction and magnitude are consistent with the coupling error the classical literature
reports for closely spaced notch resonators (Rahman et al. measured 10–16 % on this same
WLAN pair and named strong inter-resonator coupling as the cause; the prescribed cure is
3λg/4 = 24 mm of separation, which this 12 mm box forbids).

**This is not yet evidence that classical fails.** A competent engineer would
pre-distort the stub lengths to compensate the pull, and would also have stub width
(hence impedance and bandwidth) as a free parameter. Arm D exists to give the classical
design exactly those chances. What Stage 0 establishes is only that the *uncalibrated*
textbook design misses by about a notch bandwidth, which is why the calibrated sweep is
mandatory rather than optional.

## Next

Arm-D protocol is being designed under adversarial review before any sweep runs
(the Phase-1 retraction came from an un-calibrated baseline, so the baseline design is
the thing to get right). The gate it must produce: a precise inequality under which we
declare classical *can* meet the spec in this box — in which case this benchmark is
discarded and we choose another problem rather than running the gradient arms.
