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
