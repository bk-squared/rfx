# Lane 2b — closing the recorded limits of E6 and AD-Q without touching CPML

**Status:** pre-declaration, committed BEFORE any arm runs. **Branch**
`feat/nu-lane2b-inplane-ad` on `feat/nu-full-functionality` (c8981752), which
stacks on the AD-Q branch. **No `rfx/` change.** Overlap check (2026-09-15):
nothing in flight touches `validation/research/multiband_nu/` or the NU AD
instruments; the in-flight CPML work (#1012, #1047) is irrelevant here — every
fixture below is PEC-closed (`cpml_layers = 0`).

## What is being closed, and where each limit is recorded

| recorded limit | where | arm |
|---|---|---|
| E6 measured the x axis only | Lane 2 note, "y not measured" | **y** |
| E6's band POSITION control is FD-supported only: a 43x lead/tail cancellation makes the loss quadratic-dominated along x_c, so the order test is structurally unavailable (curvature guard) | Lane 2 note, second pass | **pos** |
| AD-Q's resonant DFT-power thickness gradients are NOT verified: the fixed-frequency observable is rippled ~1 % in thickness; FD fires at a 3 % step | AD-Q note, "L2 thickness" | **zsmooth** |

## Instrument

`validation/research/multiband_nu/e7_lane2b.py` (committed with this note).
It imports E6's `MeshMap`, strip-mask construction, waveform and Hann+comb
spectral observable, E4's z-stack runner, and the AD-Q judges
(`adq_designvar.fit_order / fd_budget / selfcheck / coverage / losses_along`)
verbatim. A forward-only smoke (one loss evaluation per fixture, no gradient,
no ladder) is run before committing to catch index/shape errors; it is not a
measurement of any quantity below.

- **y**: E6's fixture rotated onto y — `dy_profile` = the same
  0.5 / 0.1 / 0.5 mm band at cap 1.3 with 0.5 mm pinned ends, x uniform
  (12 cells), PEC strip on the z-node plane `K_S = 12` between the band-edge
  y nodes (20, 40), attached by node index; ez source at (6, 8, 6) in the
  lead segment, ez probe at (6, 43, 14) beyond the band. Controls `w`
  (width; column (−1/2, 1, −1/2) on lead/band/tail) and `y_c` (position;
  (1, 0, −1)); ladder displacement `h × 2 mm` for both. Losses: L1 = 600-step
  probe energy; L2s = 2400-step, physical-time Hann (`T_w = 0.8 N2 dt(p0)`)
  and 0.5 GHz comb (k = −10..10 around 20 GHz) — E6's exactly.
- **pos**: E6's x fixture, source unchanged at (8, 6, 6), but the probe moved
  to the SOURCE side of the strip, (17, 6, 14) — three cells before the
  band's left edge node 20. Moving the band then changes the source→strip
  round trip monotonically instead of shortening one leg while lengthening
  the other. Controls `w`, `x_c`; both losses.
- **zsmooth**: the AD-Q stack map (`adq.stack_cells`, `P0`, the four
  thickness controls at fixed 44 mm) with the L2 drive at the analytic
  nominal resonance `f_nom = 10.5617 GHz`, 8000 steps, checkpoint 100 — and
  the observable replaced by the Hann+comb form: `T_w = 0.8 N2 dt(p0)`,
  comb k = −10..10 at 0.25 GHz spacing around `f_nom` (±2.5 GHz, wider than
  the ~1 % thickness ripple's period so the ripple integrates out — that is
  the point).

## Frozen rules and windows

- **Judge self-test** (`adq.selfcheck`) runs before every arm; an arm refuses
  to run if it fails (true slope → R1 ≈ 2.00 HELD; 10 %-wrong → R1 ≈ 0.91 FIRED).
- **Induced-direction check**: every control's Jacobian column must be
  exactly constant on the dt-setting tied set (asserted; the map guarantees
  it — E6 m3).
- **Order gate, lane-2b form**: fitted `R0` slope in [0.9, 1.1] AND fitted
  `R1` slope ≥ 1.8. Lower side only: a wrong gradient drives R1 toward 1 and
  never above 2 (AD-Q note, reviewer note); AD-Q's upper edge 2.2 is not
  reused. The eligibility rule (R1 > 32 × noise floor, curvature guard,
  longest monotone run ≥ 4 points) is AD-Q's unchanged. No eligible run →
  INCONCLUSIVE.
- **FD gate**: `|g·v − D(h)| ≤ 3 B(h)` at every Richardson-valid step
  (`rho ∈ [2, 8]`), INCONCLUSIVE where `3B/|g| > 15 %`; AD-Q's `fd_budget`
  unchanged.
- **Noise floor**: measured per control on a fresh 64-point grid
  `h ∈ [1e-5, 1e-4]` (quadratic-fit RMS, cubic check ≥ 0.9), used as
  `sigma` for eligibility and FD roundoff — the AD-Q second-attempt method,
  adopted from the start for every arm here (the L2-class losses were
  measured at 14-44 ulp there).
- **Revert-proof**: for `w` (y arm) and `h_thin` (zsmooth), the same judges
  re-run with `stop_gradient(dt)` must FIRE (order or FD); the forward
  value must be bit-identical with and without it; the dt share is
  recorded. (E6: dt share 0.59 for `w`; AD-Q: 0.185 for `h_thin`.)
- **pos eligibility (declared before its gradient is computed)**: the
  cancellation ratio `|g_lead + g_tail| / (|g_lead| + |g_tail|)` of the
  x_c column, computed from the cell-wise gradient at the nominal design,
  must be ≥ 0.5 for the arm's `x_c` order verdict to count; below it the
  arm records `INCONCLUSIVE (precheck)` for `x_c` — the fixture, not the
  gradient, is then at fault (E6's value was ≈ 1/43 ≈ 0.023).
- **Expectations, not gates** (from E6 / AD-Q): y `w` R1 in the 1.9-2.0
  range with FD bars ≈ 1e-3 / 1e-4 (E6 x: 1.945 / 1.962; 2.0e-3 / 7.2e-4);
  zsmooth `h_thin` order HELD if the comb integrates the ripple out, FD bar
  well under the 15 % INCONCLUSIVE line (AD-Q's transient arm gave 2.8e-4).
- **Coverage** reported per loss: all controls, and verified controls only.

One attempt per arm; a FIRED window is a result; nothing above changes
after this commit. Results are appended below, per arm, as each finishes
and is committed.

## Results

(appended per arm)

### Arm y — E6 rotated onto the y axis (run once on c2025d34, 68 s; `e7_y.json`)

Self-test passed; induced directions constant on the 20-cell tied set.
(`git_dirty: true` in the provenance is the instrument's own untracked
output and `.started` sentinel under `validation/` — the same class Lane 1
recorded; no tracked file differed.)

| loss | control | R0 | R1 | pts | order | FD | narrowest 3B/\|g\| | floor |
|---|---|---|---|---|---|---|---|---|
| L1 (600 steps) | **w** | 1.038 | **1.943** | 5 | **HELD** | **HELD** | 4.4e-3 | 2.9 ulp |
| L1 | y_c | — | — | 1 | INCONCLUSIVE | HELD | 1.6e-2 | 2.1 ulp |
| L2s (Hann + comb) | **w** | 0.950 | **1.962** | 6 | **HELD** | **HELD** | 7.0e-4 | 14.0 ulp |
| L2s | y_c | — | — | 3 | INCONCLUSIVE | HELD | 3.4e-4 | 1.9 ulp |

Revert-proof (`stop_gradient(dt)`, forward bit-identical): `w` FIRES on both
losses — L1 R1 1.023, L2s R1 0.979, FD FIRED; dt share 0.588 (L1) / 0.767
(L2s). Coverage: L1 0.802 (verified 0.802), L2s 0.680 (verified 0.671).

Reading: the y axis reproduces E6's x results to the third digit
(x: R1 1.945 / 1.962, dt share 0.588 / 0.767, coverage 0.802 / 0.680) —
the in-plane width gradient is now verified on both in-plane axes, dt path
included. `y_c` is INCONCLUSIVE on order for the same reason as E6's `x_c`
(source and probe on opposite sides of the strip; the fixture, not the
gradient) — which is exactly what the `pos` arm is for.


### Arm pos — position control with the probe on the source side (run once on d0fab837; `e7_pos.json`)

Eligibility pre-check (declared ≥ 0.5): cancellation ratio **1.0** on both
losses — `g_lead` and `g_tail` now carry the SAME sign (L1: 5.264 and 0.175;
L2s: 4.53e-22 and 3.89e-22), so the x_c direction is no longer a
lead-minus-tail difference. The fixture is eligible.

| loss | control | R0 | R1 | pts | order (lane rule) | FD | narrowest 3B/\|g\| | floor |
|---|---|---|---|---|---|---|---|---|
| l1 | w | 0.8997 | 2.025 | 5 | FIRED | HELD | 2.3e-03 | 2.3 ulp |
| l1 | x_c | 0.9950 | 2.233 | 4 | HELD | HELD | 6.7e-04 | 1.9 ulp |
| l2s | w | 0.9501 | 1.962 | 6 | HELD | HELD | 7.2e-04 | 14.6 ulp |
| l2s | x_c | 1.0105 | 1.989 | 5 | HELD | HELD | 7.5e-03 | 1.9 ulp |

**Band POSITION is now order-verified**: `x_c` R1 = 2.233 (L1, 4 points,
h 1.6e-2..1.25e-1) and **1.989** (L2s, 5 points), FD inside its bar on both
(3B/|g| 6.7e-4 and 7.5e-3). E6's "structurally unavailable" was the
fixture's symmetry, not the gradient — moving the probe to the source side
is all it took. For the record: AD-Q's own judge, which keeps an upper R1
edge at 2.2, would have FIRED L1 `x_c` on that upper side (R1 2.233 in a
4-point window, the same class as AD-Q's `h_air`); the lane-2b rule
dropped that edge before any arm ran (above), and the L2s window (5 points,
R1 1.989) sits inside both rules.

**Recorded FIRED, not a gradient signal**: `w` on L1 fires the lane order
rule on the R0 bound alone — R0 = 0.899689 (fit standard error 0.0253)
against the frozen lower edge 0.9, while its R1 = 2.025 (the correctness
statistic) and its FD gate (3B/|g| 2.3e-3) both hold, as they did on E6-x
(R1 1.945) and on arm y (1.943). R0 is the slope of the FIRST difference
`|L(x+hv) − L(x)|`, a sanity check that the ladder is in the linear regime;
a wrong gradient does not move it. The rule is kept as declared; the
verified-coverage figure for L1 (0.174) therefore excludes `w` by rule,
while the same control is verified on L2s (coverage 0.675).

### Arm zsmooth — instrument fix before the attempt (no ladder was measured)

The first launch (on 094cc595) raised `operands could not be broadcast
together with shapes (8,) (4,)` inside the shared judge before the first
ladder point: `arm_zsmooth` passes the full 8-parameter `P0` as `x0` with
only the four thickness controls named, and `controls_arm` built the step
direction over the names, not over `x0`. Fix: the direction is built in
`x0`'s space (`v = zeros(len(x0)); v[i] = scale`), which is a no-op for arms
y and pos (their `x0` and names have the same length — recorded ladders
unaffected). No loss value of the zsmooth fixture was produced, so this is
not a measurement attempt; the `.started` sentinel was reset and the arm
relaunched once.
