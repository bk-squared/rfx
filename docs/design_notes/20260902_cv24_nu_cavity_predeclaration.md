# cv24 — non-uniform (graded-z) PEC cavity eigenfrequencies vs the exact Pozar spectrum: pre-declaration

Date: 2026-09-02. Gap lane 3 of the cross-validation campaign (PI-approved 2026-09-02). Written and committed BEFORE any arm runs. Numbers here are derived from committed data and from an exact lattice model of the solver's own difference operators, never from a run of this case. Every constant lives once, in `validation/crossval/comparators/nu_cavity_gates.py`; the note quotes it.

## 0. Why this case exists

The non-uniform (graded) mesh has no observable-level accuracy evidence. The evidence rule (`docs/guides/physics_validation_evidence_rule.md`) marks NU S-parameter paths "Shadow unless they pass a co-refined analytic/external oracle"; cv05's graded z is defective (#325); cv10's NU path uses uniform dz; PR #785 documented the multi-band graded z mesh **as a mesh** (energy, per-transition reflection, order, AD) and said so ("not any observable computed on it"); #810 proposes the E4 campaign on S-parameter cases. Before the S-parameter cases, the cleanest observable in the repo — the eigenfrequency spectrum of a closed PEC cavity against Pozar's closed form (cv14, E2, exact oracle, no meshing tolerance) — is the right first claims-bearing observable for the graded mesh: it is z-dependent (every `l >= 1` mode), it is exact, and the discretization error is fully modellable a priori, so every window can be derived instead of chosen.

What this case claims, if it passes, is deliberately narrow: **an eigenfrequency computed on a graded-z Yee mesh inside the #785 envelope is as accurate as on the uniform mesh, to within a derived allowance, and equals the exact discrete-lattice prediction to 4 ppm, for one closed lossless air-filled PEC cavity, z grading only.** It says nothing about S-parameters, ports, absorbers, materials, or in-plane grading. The E4 leg for the NU lane (the committed Palace graded-dy WR-90 fixture) is a FOLLOW-UP case (section 10).

## 1. Rig — cv14's cavity, three ways plus the cost control

`validation/crossval/14_rect_cavity_pozar.py`, verbatim where it matters: air-filled PEC cavity `(a, b, d) = (50, 30, 40) mm` (x, y, z), `boundary="pec"`, `freq_max = 10 GHz` (default `GaussianPulse` f0 = 5 GHz, bandwidth 0.8, cutoff 3: τ = 79.6 ps, pulse over by `2 t0 = 0.477 ns`), three soft point sources (`ex` at (13, 11, 17) mm, `ey` at (19, 23, 13) mm, `ez` at (31, 13, 23) mm), probe at (37, 17, 29) mm, `rfx.harminv` (matrix pencil, `pencil_parameter = 0.33`, `sv_threshold = 1e-3`, subsampled to ≤ 8000 samples) on the post-pulse record, oracle `f_mnl = (c/2) sqrt((m/a)² + (n/b)² + (l/d)²)` re-derived in the comparator.

Two deliberate differences from cv14, applied to EVERY arm including the control so the pipeline is identical across arms:

1. three scalar probes (`add_probe` for `ex`, `ey`, `ez`) at cv14's probe point instead of one `add_vector_probe` — the non-uniform lane's probe path is the scalar one;
2. the record length is derived (section 5), not cv14's `num_periods = 200`.

Every source and probe z (13, 17, 23, 29 mm) sits on a locally uniform node of every profile in section 2 (no source on a grading transition; the `source_on_graded_node` preflight advisory is expected to be absent, and the preflight report is captured verbatim per arm).

## 2. Arms — profiles, cells, dt, steps (all derived by `predict_arm`)

The graded arms' finest cell is `DZ_FINE = 0.5 mm`; the coarse cell is cv14's `1.0 mm`; every transition is the chain `0.5 → 0.7 → 0.8 → 1.0 mm` (adjacent ratios 1.4, 1.143, 1.25 — inside `_MULTIBAND_RATIO_CAP = 1.4`, `rfx/api/_preflight.py:5226`). Every profile sums to exactly 40.000 mm (the walls land on nodes; #562's bounding node gives N cells N + 1 nodes, so the realized extent is the profile sum — `rfx/nonuniform.py:146` `_append_bounding_node`). x and y are uniform at the arm's `dx` on every arm (`nx = 50`, `ny = 30` cells at 1 mm; 100, 60 at 0.5 mm).

| arm | lane | z profile (mm) | z cells | nodes (x, y, z) | cells total | dt (ps) | n_steps | cell·steps |
|---|---|---|---|---|---|---|---|---|
| (a) `uniform` — cv14's mesh, the control | uniform | 40 × 1.0 | 40 | 51 × 31 × 41 | 60 000 | 1.9066 | 16 794 | 1.01e9 |
| (b) `single_band` | nonuniform | 8 × 1.0 \| 0.8 0.7 \| 10 × 0.5 \| 0.7 0.8 \| 24 × 1.0 | 46 | 51 × 31 × 47 | 69 000 | 1.3482 | 23 751 | 1.64e9 |
| (c) `multi_band` (small–large–small–large) | nonuniform | 6 × 0.5 \| 0.7 0.8 \| 10 × 1.0 \| 0.8 0.7 \| 9 × 0.5 \| 0.7 0.8 \| 18 × 1.0 | 49 | 51 × 31 × 50 | 73 500 | 1.3482 | 23 751 | 1.75e9 |
| (d) `uniform_fine` — uniform at the graded arms' finest cell (cost control, #810) | uniform | 80 × 0.5 | 80 | 101 × 61 × 81 | 480 000 | 0.9533 | 33 587 | 1.61e10 |

Envelope check (`envelope_check`): (b) ratio 1.4, 1 fine band, 6 steps in 2 chains; (c) ratio 1.4, 2 fine bands, 9 steps in 3 chains; runways against the walls 8 / 24 cells (b) and 6 / 18 cells (c), all ≥ `R_WALL_CELLS = 4`. Arm (c) is the widest small–large–small–large profile the fixture holds with its sources on uniform nodes; it is inside #785's "≤ 3 fine bands / 4 transitions".

dt is the global min-cell CFL on every lane (`rfx/nonuniform.py:415-416`, identical to `rfx/grid.py:96` on a uniform mesh): the graded arms run at 0.707× the control's dt, the fine arm at 0.5×. `dt` is unchanged by #785 and by this case.

Cost at equal dt (#810's "what the mesh buys"): cells 60 000 / 69 000 / 73 500 / 480 000 — the multi-band mesh puts 0.5 mm cells where it wants them at 15 % of the fine arm's cell count; at equal dt the step count is equal, so the cell·step ratio (d)/(c) is 6.5. At each arm's own CFL dt the ratio is 9.2 (the table's last column). Wall time is recorded per arm and reported, not gated.

## 3. The discrete dispersion of an eigenfrequency on a graded lattice — derived before the run

### 3.1 The update modelled (file:line)

Non-uniform lane: `rfx/core/yee.py:411` `update_h_nu` (the H curl differences two E nodes across ONE cell: `inv_d_h[k] = 1/d[k]`, line 438 for the z term) and `rfx/core/yee.py:456` `update_e_nu` → `curl_h_nu` (`rfx/core/yee.py:323`, z term line 333: the E curl differences two H cell-centres whose separation is the MEAN of the two cells, `inv_d_e[k] = 2/(d[k-1] + d[k])`, `inv_d_e[0] = 1/d[0]`). The arrays come from `rfx/nonuniform.py:203` `_profile_to_inv_arrays` (lines 235, 239) on the padded profile with the #562 bounding node (`rfx/nonuniform.py:146`). Uniform lane: the same stencil with `1/dx` everywhere (bit-identical on a uniform profile per the `_profile_to_inv_arrays` docstring). Leapfrog in time with the CFL dt above.

### 3.2 Separation on a tensor grid (exact)

For the empty PEC box each E component of a discrete mode is a product of 1-D eigenvectors. Along an axis where the component is node-centred (tangential to that axis's walls, PEC-pinned) the 1-D operator is the primal Dirichlet operator

    (A E)[k] = -inv_e[k] ( inv_h[k] (E[k+1] - E[k]) - inv_h[k-1] (E[k] - E[k-1]) ),   k = 1 .. N-1,

i.e. `A = -D_e^{-1} inv_h^T inv_h` in matrix form (`D_e = diag(1/inv_e)`); along the axis where it is edge-centred (normal to those walls, free) the operator is the dual `B = -inv_h D_e^{-1} inv_h^T`. `A` and `B` are `XY` and `YX` and share their nonzero spectrum; `B` adds the zero eigenvalue (index 0). With the discrete divergence-free constraint the Yee curl-curl on a tensor grid is exactly the sum of the three 1-D operators, so every discrete eigenfrequency of the box is

    sin(ω dt / 2) = (c0 dt / 2) sqrt( μ_x(m) + μ_y(n) + μ_z(l) ),     μ_axis(0) = 0,

with `μ_axis(i)` the i-th eigenvalue of that axis's primal operator on that axis's profile (`lattice_freq`). On a uniform axis of N cells `h`: `μ(i) = (2/h)² sin²(iπ/(2N))`. This is `validation/research/multiband_nu/analytic_dispersion.py`'s TE_m0p model extended to all three indices; the gate test assembles the dense 3-D discrete curl-curl of a small graded box from the SAME `inv_e / inv_h` arrays and checks its spectrum against the sums (so the separation is verified, not assumed), and checks `μ` against the uniform closed form.

### 3.3 Second-order form (the analytic statement asked for)

For the continuum eigenfunction `φ = sin(kz)` sampled at nodes `z_k`, the non-uniform second difference has the truncation expansion

    (A φ)[k] = k² φ_k + (k³/3) Δh_k cos(k z_k) - (k⁴/12) h̃²_k sin(k z_k) + O(h³),
    Δh_k = h_k - h_{k-1},   h̃²_k = h_k² - h_k h_{k-1} + h_{k-1}²,

and `A` is self-adjoint in the inner product weighted by the dual spacing `w_k = (h_{k-1} + h_k)/2`, so first-order perturbation gives

    μ = k² - (k⁴/12) ⟨h̃²⟩_w + (k³/3) ⟨Δh · sin cos⟩_w / ⟨sin²⟩_w,        (`second_order_mu`)

where `⟨·⟩_w` is the `w_k sin²(k z_k)`-weighted node average (first term) and the plain `w_k`-weighted node sum (second). The first term is the familiar dispersion `-k²h²/12` on a uniform mesh (there the second vanishes identically); the second is the TRANSITION term, nonzero only at unequal neighbours and of the same order `O(k³ h Δh)` since `Δh = (r-1) h`. Then

    f = (1/(π dt)) asin( (c0 dt/2) sqrt(Σ μ) ),   Δf/f = Δ(K²)/(2K²) + (ω dt)²/24 + …,

so the per-mode shift has a spatial part (mode-share-weighted per axis) and a time part `+(ω dt)²/24` that depends only on dt. Agreement of `second_order_mu` with the exact eigenvalue on every arm is `≤ 3e-6` of `μ_z` (TE102, the coarsest-resolved; 0.06 in 24 617), `≤ 1e-7` on the `l = 1` modes — the neglected terms are `O(h³)` as claimed.

### 3.4 Predicted per-mode shifts, BEFORE the run (`predict_arm`; ppm = 1e-6 relative to Pozar)

`lattice` = the raw frequency the solver should report (time term included); `spatial` = the dt-free spatial eigenvalue (`spatial_freq` inverts the leapfrog relation exactly; this is the equal-dt quantity #810 asks for); `time` = the leapfrog term alone.

| mode | Pozar (GHz) | (a) uniform lattice / spatial / time | (b) single_band | (c) multi_band | (d) uniform_fine |
|---|---|---|---|---|---|
| TE101 | 4.79902 | −83.2 / −220.9 / +137.7 | −149.7 / −218.5 / +68.8 | −141.1 / −209.9 / +68.8 | −20.8 / −55.2 / +34.4 |
| TM110 | 5.82692 | −176.6 / −379.5 / +203.0 | −278.0 / −379.5 / +101.5 | −278.0 / −379.5 / +101.5 | −44.1 / −94.9 / +50.8 |
| TE011 | 6.24568 | −151.8 / −384.9 / +233.2 | −267.0 / −383.5 / +116.6 | −261.9 / −378.4 / +116.6 | −37.9 / −96.2 / +58.3 |
| TM111 | 6.92792 | −56.7 / −343.6 / +287.0 | −199.1 / −342.5 / +143.5 | −195.0 / −338.4 / +143.5 | −14.2 / −85.9 / +71.8 |
| TE201 | 7.07059 | −246.6 / −545.2 / +298.8 | −394.9 / −544.2 / +149.4 | −390.9 / −540.2 / +149.4 | −61.6 / −136.3 / +74.7 |
| TM210 | 7.80485 | −211.5 / −575.5 / +364.2 | −393.6 / −575.5 / +182.0 | −393.6 / −575.5 / +182.0 | −52.9 / −143.9 / +91.1 |
| TE102 | 8.07216 | −519.7 / −908.7 / +389.3 | −821.3 / −1015.6 / +194.5 | −182.0 / −376.7 / +194.8 | −129.9 / −227.2 / +97.4 |

Reading the table (these are the things the run must reproduce):

- The `l = 0` modes (TM110, TM210) have identical SPATIAL deviation on (a), (b), (c) — the z grading cannot touch them (`μ_z(0) = 0` on any profile) — while their RAW deviation differs by the dt term alone (−176.6 vs −278.0 ppm on TM110). A raw comparison between arms is a comparison of dt, not of the mesh; this is why the allowance gate is stated on the spatial (equal-dt) quantity (section 4).
- On the `l ≥ 1` modes the graded arms differ from the control by a few ppm in the spatial deviation because two second-order terms nearly cancel: the fine band REDUCES z dispersion (TE101 (b): `⟨h̃²⟩` term −2.654 vs −3.171 in μ_z, +26 ppm in f) and the transition term SUBTRACTS (−0.469 in μ_z, −23 ppm in f). The transition term is 6× the estimator floor (section 4.1), so the lattice gate sees it.
- (c)'s TE102 is the one striking prediction: the two fine bands and the positive transition term (+27.9 in μ_z) put it at −182 ppm, closer to Pozar than the CONTROL's −520 ppm. If the run reports (c)'s TE102 near −820 ppm instead, the transition term's sign in the model is wrong.
- (d) is 4× better than (a) on every mode (second order), the convergence witness cv14's `--converge` leg already shows.

Committed anchors reproduced by the same model, a priori (`anchor_residuals`; `tests/test_nonuniform_cavity_accuracy.py`, a = 40, b = 35 mm, TM111, dx = 1 mm, the docstring's sensitivity table): uniform z → measured 0.0011 %, model −10.50 ppm (residual 0.50 ppm, the datum's quoting resolution); 4:1 graded z (0.25 mm band smoothed at 1.3, 52 cells, d = 41.375 mm) → measured 0.0252 % (`_MEASURED_ENVELOPE_PCT`), model −254.4 ppm (residual 2.39 ppm). The model reproduces a committed graded-mesh measurement made on a different cavity, a different profile and a different estimator window to 1 % of the value.

## 4. Windows — derived, not chosen

### 4.1 The estimator floor `W_est = 4e-6` (4 ppm)

`W_est = gate_from_envelope(max residual of the two anchors, quantum = 1e6)` = round-up(1.5 × 2.39e-6 at 1e-6) = **4e-6** (`estimator_floor`; `ENVELOPE_GATE_MULTIPLIER = 1.5`, `tests/_gate_policy.py:81`, re-stated in the comparator and asserted equal by the gate test). Cross-check from the estimator itself: the matrix pencil is exact for a noiseless sum of exponentials; its bias from a component dropped at the rank floor (`sv_threshold = 1e-3`, `rfx/harminv.py:49`) is ≲ 1e-3 × (1/T)/(2π) ≈ 5 kHz = 1 ppm at 5 GHz on the 31.5 ns record of section 5; float32 accumulation over 2e4 steps is ~1e-5 in amplitude and negligible in frequency. The floor is used three ways: as the lattice gate window, as the noise term of the allowance gate, and as the stationarity witness bound.

### 4.2 The claims gate: cv14's committed tolerance, unchanged

G1: TE101 `|Δf/f| < 1 %` and at least one higher mode `< 2 %` (`CV14_TOL_TE101`, `CV14_TOL_HIGHER`), on every arm. This is the control's own committed gate and is kept so the control IS cv14; it is 10–40× looser than the predicted deviations and does not carry the NU claim.

### 4.3 The allowance gate (#810), on the equal-dt spatial deviation

G2, graded arms (b), (c), per mode: `|dev_sp(g)| ≤ |dev_sp(a)| + A(mode, profile) + W_est`.

`A` is derived from #785's per-transition reflection budget (`allowance`): the frozen chain-model amplitudes `R_model(r)` of the multi-band note's section 3.3 (abrupt column: 1.1 → 4.358e-4, 1.2 → 9.139e-4, 1.4 → 1.998e-3 (−54.0 dB; measured −53.9 dB), 1.5 → 2.605e-3, 2.0 → 6.298e-3) at the battery's resolution — `λ_g/34.6` fine cells per AXIAL wavelength (30 per free-space wavelength at 10 GHz) — scaled by the `(dz/λ)²` law (−12 dB per doubling, note item 8). The chain model's variable is the axial Bloch wavenumber, so the fixture's own cells-per-wavelength is taken along z: `λ_z = 2d/l` (80 mm for `l = 1`, 40 mm for `l = 2`), `N = λ_z / (fine cell of the step)`; log-linear interpolation in `r` between table points. Each sub-step of a chain is one reflector; amplitudes add coherently (worst case). The map from reflection to eigenfrequency: a thin lossless scatterer of amplitude reflection ρ at `z_t` in a 1-D cavity of length d shifts mode `l` by `|Δf/f| = (2ρ/(lπ)) sin²(k z_t) ≤ 2ρ/(lπ)` (first-order energy perturbation of a thin slab of thickness δ: `Δω/ω = -(ε-1)(δ/d) sin²`, with `ρ = (ε-1) k δ/2`); in 3-D only `k_z²` is perturbed, so multiply by `k_z²/K²`. `l = 0` modes carry `A = 0`.

Allowances (ppm): (b) TE101 218, TE011 129, TM111 105, TE201 101, TE102 618; (c) TE101 328, TE011 193, TM111 157, TE201 151, TE102 926; TM110, TM210: 0 on both. (Free-space-wavelength alternative, reported not used: 269 / 404 on the `l = 1` modes.)

**Stated before the run: the allowance is a loose bound for eigenfrequencies.** The exact lattice model puts the actual transition term at −23 ppm (TE101, (b)), 10× under `A`; and the net graded-minus-uniform spatial difference is +2 ppm because the fine band's dispersion gain cancels it. So G2 cannot fail for an in-envelope profile unless the solver's transition metric is wrong by an O(1) factor — which is exactly the defect class the gate exists for (F1c, section 6), and which it catches by 5.8–16×. G2 is #810's gate as asked for; the gate that measures the mesh to its physics is G3.

### 4.4 The lattice gate

G3, every arm, every declared mode: `|f_meas / f_lattice(arm) − 1| ≤ W_est = 4e-6`. This is the a-priori witness of cv23 §12.2 made a gate here, because for a lossless closed cavity the lattice model is complete (no absorber, no interface, no material term) and the two committed anchors show it holds to the floor.

### 4.5 Mode count

G4: the number of clusters harminv finds in `BAND_HZ = [4.0, 8.5] GHz` (union of the three channels, lines below `AMP_FLOOR_REL = 1e-3` of their channel's strongest dropped — the estimator's own rank floor — clustered within `CLUSTER_REL = 0.5 %`, a quarter of the closest declared pair) must equal the declared count **7**, and every declared mode must own exactly one cluster. Identification is by (m, n, l) through Voronoi windows (`id_windows`; midpoints to the neighbours, the band edge below TE101, TM211 at 8.658 GHz above TE102): a line outside every window is an orphan, never assigned; two clusters in one window is ambiguous, never resolved by nearest. TE/TM-degenerate triples (TM111) are one frequency and count once.

### 4.6 Witnesses (gated)

- G5 energy: the #785 F-S1 envelope, `|E(n) − E_0| / E_0 ≤ K u sqrt(n)` with `K = 20`, `u = 2⁻²⁴` (`validation/research/multiband_nu/w1_energy_drift.py:26-27`, re-stated as `FS1_K`, `U32`; asserted equal by the gate test), on the Remis dual-cell energy of a source-free run of the arm's OWN grid and materials for the arm's `n_steps` (sampled every 500 steps, `evaluate_fs1`'s `n ≥ 1e4` rule). This is the closed-cavity analogue of the settling witness: energy must neither grow nor drift. The audit steps the production NU kernel (`_build_nu_scan`) on every arm's grid, including the two uniform arms' (uniform profile on the NU path is stencil-identical) — stated: for arms (a) and (d) it witnesses the MESH metric on the NU kernel, not the uniform lane's kernel. Predicted: drift ≤ 3e-6 on every arm (#785 measured 2.9e-6 over 1e6 steps; the envelope at 2.4e4 steps is 1.8e-4).
- G6 stationarity: each mode's frequency extracted on two overlapping sub-windows (`[t_start, t_start + 2T/3]` and `[t_start + T/3, t_start + T]`, each holding the closest pair at ≥ 3 pencil units, section 5) must agree to `W_est`. An undamped cavity's spectrum is stationary; a scatter above the floor means the extraction cannot support G3 and the run reads as a RIG defect (exit 1), never as a physics verdict — and the remedy is a longer record, never a wider floor.
- Fit residual (`HarminvMode.error`, the pole's distance from the unit circle) is REPORTED per mode; harminv Q is a window artefact on a lossless cavity and is not printed as physics (cv14's rule).

## 5. Record length — derived from the closest declared pair

Closest declared pair: TM111 / TE201, `Δf_min = 142.67 MHz` (2.04 %; harminv's dedup merges within 1 %, `rfx/harminv.py:210`, so the pair survives it). The matrix pencil resolves two exponentials when their phase difference across the pencil span exceeds 2π: `Δf · L dt ≥ 1` with `L = 0.33 N` (`rfx/harminv.py:120`), i.e. the post-pulse record must hold `T ≥ 3/Δf_min` (`PENCIL_RESOLUTION_UNITS = 3`). The stationarity sub-windows are 2/3 of the record and must each satisfy the same rule, so

    T_post = 3 / ((2/3) Δf_min) = 31.54 ns,   t_start = 2 t0 = 0.477 ns  (cv14's rule),

`n_steps = ceil(t_start/dt) + ceil(T_post/dt)` = 16 794 / 23 751 / 23 751 / 33 587 (table in section 2). The full record holds the pair at 4.5 pencil units, each sub-window at 3.0. No record is borrowed from another case; cv14's `num_periods = 200` (20 ns) would put the sub-windows at 1.9 units, below the rule.

## 6. Falsifiers — pre-declared, with predicted magnitudes and reading rules

Each is `--falsifier <name>` of the case script, runs the uniform control plus the defective graded arm, and MUST exit 1. Predictions are from the lattice model (spatial deviation vs the DECLARED cavity; "excess" = `|dev_sp(g)| − |dev_sp(a)|`).

| name | defect | predicted | must fail |
|---|---|---|---|
| F1 `ratio2_abrupt` | (b)'s bands with ABRUPT 0.5 ↔ 1.0 steps (ratio 2.0): 9 × 1.0 \| 10 × 0.5 \| 26 × 1.0, 45 cells | envelope: ratio 2.0 > 1.4. Physics: excess −4.6 / −2.8 / −2.2 / −2.1 / **+116.8** ppm (TE101 / TE011 / TM111 / TE201 / TE102) against `A` = 229 / 135 / 110 / 105 / 647 → **G2 predicted to PASS on every mode** | `envelope` |
| F1c `metric_defect` | (b) run with the pre-CORE-C2 metric swap (`swapped_inv_arrays`: H gets the mean spacing, E the local width — the defect `rfx/nonuniform.py:203`'s docstring records), injected by monkeypatching `rfx.nonuniform._profile_to_inv_arrays` for that arm only | excess +3515 / +2074 / +1685 / +1618 / +3614 ppm vs `A + W_est` = 222 / 133 / 109 / 105 / 622 (≥ 5.8×); lattice residual −3519 / −2077 / −1688 / −1621 / +5546 ppm vs 4; `l = 0` modes unchanged | `allowance`, `lattice` |
| F2 `grading_at_wall` | (b)'s bands with the chain starting AT the z = 0 wall: 0.8 0.7 \| 10 × 0.5 \| 0.7 0.8 \| 32 × 1.0 | envelope: grading within `R_WALL_CELLS = 4` of the wall. Physics: excess −65 / −39 / −31 / −30 / −195 ppm (all inside `A`) — a transition next to a PEC wall is harmless in the lattice (sin·cos → 0 there); F2 is the ENVELOPE exclusion #785 carries for absorber-adjacent grading, transplanted to walls as a declared runway | `envelope` |
| F3 `extent_plus_one_fine_cell` | (b) with ONE extra 0.5 mm fine cell: realized d = 40.5 mm, oracle keeps 40.0 | by name: TE101 −7503 ppm excess, TE011 −4423, TM111 −3593, TE201 −3450, TE102 −10 699 (each ≫ `A`, each ≫ `W_est`, each still inside its Voronoi window so it is identified as ITSELF and fails as itself); TM110, TM210: excess 0, pass. The realized profile also fails `envelope` (extent ≠ declared d) | `allowance`, `lattice` (on exactly the five `l ≥ 1` modes) |
| F4 `mode_count_drop_te102` | search band closed at 8.0 GHz (below TE102, 8.072 GHz); the oracle still declares 7 | count 6 ≠ 7; TE102 NOT FOUND | `mode_count` |

A variant of F1 with a 2 mm far band (0.5 → 1.0 → 2.0, ratio 2.0 twice) was evaluated and REJECTED as a falsifier: its TE102 excess 1960 ppm sits on its own allowance 1944 ppm (1.01×) because the reflection budget and the coarse-cell dispersion both scale as `h²` — a pass/fail there would be a coin toss (cv22's Debye τ × 1.3 rule). F1 as literally proposed in #810 ("a ratio-2.0 profile must fail the allowance gate") is predicted, a priori, NOT to fail G2 on this fixture: an abrupt ratio-2 step's transition term (+117 ppm on TE102 at most) is under the reflection-derived bound at its own ratio, as any valid bound must be. Reading rules: (i) F1 fails `envelope` and passes G2 — as predicted, recorded; (ii) F1 fails G2 — the lattice model is wrong for abrupt transitions and the entire prediction column of section 3.4 is suspect: stop and re-derive; (iii) F1c passes G2 — the allowance gate cannot see the defect class it exists for and G2 is retired as a gate (kept as a report); (iv) F3 fails on an `l = 0` mode — the mode identifier or the x/y grid is wrong, a rig defect.

Unit-level (no FDTD, `tests/test_cv24_nu_cavity_gates.py`): every falsifier's prediction above is recomputed from the profiles; the identifier is exercised on synthetic spectra (7 exact lines → 7 identified; F3's shifted spectrum → the five `l ≥ 1` modes identified as themselves with the predicted deviation, the two `l = 0` unchanged; a dropped line → count 6; an orphan line → not assigned; a duplicated line → ambiguous); the 3-D separation of section 3.2 is checked on a dense assembly.

## 7. Gates and exit contract

Per arm: G1 cv14 tolerance; G2 allowance (graded arms only); G3 lattice; G4 mode count; G5 energy; G6 stationarity; plus `envelope` (the declared envelope check of the arm's profile). Exit 0 iff every gate passes on every arm run; 1 otherwise (a falsifier arm MUST exit 1); 2 is unreachable (no external reference). `--smoke` (≤ 20 s: the uniform and single-band arms at 1/8 of the record on the real grids, no gates, exit 0 if finite; a smoke artifact is never evidence and the gate test skips it).

## 8. Artifacts and keys (prose only until the run lands)

`validation/crossval/_24_nu_cavity_results/rfx.json` (schema `rfx.cv24_nu_rect_cavity_pozar.v1`): `commit` (from `.staged_commit`), `arms.<arm>.{profile_mm, lane, dx, cells, nodes, dt, n_steps, record{...}, preflight[], modes.<name>{mnl, f_pozar_hz, f_lattice_hz, f_spatial_lattice_hz, f_meas_hz, dev_raw, dev_spatial, resid_lattice, allowance, allowance_bound, stationarity, channels, error_max}, n_clusters_in_band, orphans, energy{E0, max_drift, drift_at_end, envelope_at_end, fs1_fired}, gates{...}, ok, cost{n_cells, n_steps, cell_steps, wall_run_s, wall_energy_s, wall_harminv_s}}`, `verdict{exit_code, summary}`. Falsifiers: `rfx__falsifier_<name>.json`, same schema plus `falsifier{name, expected_failing_gates, fired, as_declared}`. The gate test replays every gate from the stored per-mode frequencies and profiles (it recomputes predictions, allowances and windows from the comparator, never from the artifact's own copies).

## 9. What the VESSL run owes, and what would refute this note

Owed: the four arms with every gate green; the five falsifiers each exit 1 for the declared reason; the gate test green on the committed set; the cost table filled. Refutations accepted before the run: any arm's lattice residual above 4 ppm on a mode whose stationarity witness is inside 4 ppm (then the model of section 3 is incomplete — a real finding about the solver, not the rig); (c)'s TE102 landing near −820 ppm instead of −182 (the transition term's sign); an energy drift above the F-S1 envelope on a graded arm (then #785's 3-D witness does not transfer to this box). A stationarity scatter above 4 ppm is a rig failure and re-derives the record, not the floor — and if that happens the section that does it is appended here before the re-run, with the scatter measured and the new record derived from it.

## 10. The Palace graded-dy WR-90 fixture — the E4 leg, and why it is a follow-up

`tests/fixtures/waveguide_nu_broad_e4/waveguide_wr90_nu_flux_broad_e4_comparison.json` (rfx NU graded-dy flux S-matrix vs Palace_r_h2, 5 pairs over empty / PEC-short / slab, grading ratio 2.0, gated by `tests/test_waveguide_nu_broad_e4_comparison_gates.py`) is the only NU evidence with an external reference. It cannot be registered as a manifest case without changing its numbers: the manifest requires a case SCRIPT with the 0/1/2 exit contract that produces or re-derives the comparison, and the fixture's producer path (the 2026-08-06 generation) is not a crossval script — registering it would mean either a new script that re-runs the rfx side (new numbers; the fixture's rfx column would then be regenerated on today's main, which the cv11 provenance note shows changes the value) or a script that only replays the frozen JSON (which the gate test already is, and which the evidence rule scores as a committed-fixture replay, not a case). Also its grading is dy (in-plane, ratio 2.0), outside #785's z-only, ratio-1.4 envelope, so it is evidence for a different envelope than this case's. What the E4 leg takes: a case script `25_nu_wr90_palace_graded_dy.py` that builds the WR-90 NU flux lane at the fixture's `setup` (dx 1 mm, graded dy, 60 periods), re-derives the 5 pairs against the committed Palace column, and gates them under the fixture's own derived tolerances (0.013 / 0.005) — with the rfx column regenerated and the old one kept as the 2026-08-06 datum. That is a one-day lane once cv24 has landed; it is not this case.

## 11. Scope statement (the row this case will carry)

Closed lossless air-filled PEC cavity, one geometry (50 × 30 × 40 mm), z grading only, fine cell 0.5 mm / coarse 1.0 mm, chain ratio ≤ 1.4, ≤ 2 fine bands, eigenfrequencies of the seven modes below 8.5 GHz, E2 against the exact Pozar spectrum with the exact-lattice prediction as the gate. No S-parameters, no ports, no absorber, no material, no in-plane grading, no other cavity. The support-matrix row for the multi-band mesh stays MESH-ONLY for S-parameters until #810's Tier-1 cases run; what this case can add to it, if green, is one sentence: "eigenfrequencies on the graded-z mesh: E2, cv24".

## 12. Addendum (2026-09-02, same day, BEFORE any VESSL arm ran) — the rig check, one estimator defect fixed, and what it showed

Written after the comparator, script and gate test existed and before `scripts/vessl_cv24_nu_cavity.yaml` was emitted. Nothing in sections 1–11 is edited; no window, allowance, record or gate moved. Local runs here are RIG CHECKS (the ≤ 20 s smoke budget: the uniform and single-band arms at dx = 1 mm take 2–5 s of FDTD each); their numbers are not evidence and are not committed as artifacts — the VESSL run is.

### 12.1 The rig check fired the stationarity witness, as section 9 said it could

Uniform arm, full record (16 794 steps), cv14's extraction verbatim (8000-sample subsampling, harminv defaults incl. `decimate="auto"`): lattice residual +1.78 / −0.35 / −1.76 / −0.34 / +4.48 / +3.89 / −9.90 ppm (TE101 … TE102) and two-window scatter 3.1 / 6.1 / 6.8 / 13.5 / 2.5 / 13.5 / 22.5 ppm — the witness (≤ 4 ppm) fires on five modes and the lattice gate on three. Per section 9 that is a rig failure, and the remedy is not the floor.

### 12.2 Mechanism, found on the same time series without re-running the FDTD

Extraction variants on the SAME record (max |lattice residual| / max scatter, ppm): cv14's rig 9.9 / 22.5; no subsampling, auto-decimation 11.0 / 37.7; 8000 samples, `decimate=False` **0.48 / 0.12**; auto-decimation with `sv_threshold = 1e-4` 0.69 / 1.9; `1e-5` admits 18 spurious lines; `pencil_parameter = 0.5` 8.8 / 16.1; a 2× record with cv14's rig 2.3 / 6.4. So the record derivation of section 5 is adequate (doubling it does not reach the floor) and the floor is harminv's **auto-decimation**: `rfx/harminv.py` multiplies its rank threshold by `sqrt(decimation factor)` after the FIR stage (factor 7 here, 1e-3 → 2.6e-3), which drops the weakly excited members of this 12-mode signal (the five modes between 8.66 and 9.74 GHz are excited at 3–37 % of TE101, and every mode is weak on two of the three probe channels), and the retained poles absorb their residual. With the raw 8000 samples the estimator is exact to 0.5 ppm — 8× under the 4 ppm derived floor.

### 12.3 What changed, declared here before the run

`HARMINV_DECIMATE = False` in the case script (`_lines`), the only change; cv14's subsampling to 8000 samples, harminv's `pencil_parameter = 0.33` and `sv_threshold = 1e-3` (the counting floor `AMP_FLOOR_REL` is derived from the latter) stay. Cost: the SVD of the 5333 × 2667 pencil is ~85 s per arm for the three windows × three channels; no GPU is needed and the VESSL lane stays on the CPU image. The manifest's `cpu_runner` exclusion states this cost.

### 12.4 Rig check after the change (local, not evidence)

Uniform arm: lattice residual −0.10 / +0.16 / −0.15 / −0.43 / −0.07 / −0.00 / −0.48 ppm; scatter ≤ 0.12 ppm; 7 / 7 clusters. Single-band arm (b): residual +0.00 / −0.01 / −0.00 / −0.00 / +0.01 / +0.08 / −0.06 ppm; scatter ≤ 0.11 ppm; 7 / 7 — including TE101 at −149.7 ppm measured vs −149.7 predicted, i.e. the −23 ppm transition term of section 3.4 is measured, not only modelled. The falsifier paths execute at 1/8 record (`--smoke --falsifier …`): the metric swap reads −3519 ppm on TE101 against the predicted −3519; the extra fine cell −7507 ppm excess against 7502; the narrowed band counts 6. Every one of section 6's predictions stands; the note's predictions column is unchanged.

### 12.5 Two records corrected in passing

- F3 (`extent_plus_one_fine_cell`): the mis-realized grid shifts every node above the fine band by 0.5 mm and puts the `ex` source (z = 17 mm) on a transition node, so the `source_on_graded_node` preflight advisory IS expected on that arm (section 1's "absent" applies to the four arms and the other falsifiers); eigenfrequencies do not depend on source position. F3 is judged against the DECLARED profile (b) — oracle, lattice prediction, allowance — with a separate `extent` gate on the realized z extent, so it fails `lattice`, `allowance` and `extent`, not `envelope` as section 6's row said (the declared profile is inside the envelope; the realization is what is wrong).
- The gate test's dense 3-D assembly of the Yee curl-curl on a graded 3 × 2 × 7-cell box reproduces the separable spectrum of section 3.2 to 1e-9 relative, multiplicities included, and the second-order formula of section 3.3 agrees with the exact eigenvalue to 6.5e-8 (`l = 1`) and 2.5e-6 (`l = 2`) of `μ_z` on every arm.
