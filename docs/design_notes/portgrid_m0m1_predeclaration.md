# SPEC-02 portgrid M0+M1 — pre-declaration (append-only)

Lane: `agent/portgrid-m0m1` · Tracker: #781 · Base: main `bdcf9ea` · Date: 2026-08-29 (KST)
Author agent session: https://claude.ai/code/session_016E2cSPq3RYrGrrJvS5TLaH

This note is committed BEFORE any measurement. Falsifier windows below are frozen at this
commit; they may not be widened after measurement (SPEC-00 §0.2-2).

## 1. Sources actually obtained (not from memory)

| Source | How obtained | Status |
|---|---|---|
| Bekmambetova/Zhang/Triverio, arXiv:1606.08761v1 (= IEEE TAP 65(2):751, 2017, 2D) | arXiv PDF, read in full | in hand |
| Bekmambetova/Zhang/Triverio, arXiv:1705.02274v2 (= IEEE TAP 66(12):7156, 2018, 3D) | arXiv PDF, read in full | in hand |
| Corrections, IEEE TAP 70(4):3132, 2022, DOI 10.1109/TAP.2022.3140321 | author preprint, waves.utoronto.ca/triverio/papers/jnl-2022-tap-fdtd-3D-dissipativity-corrections.pdf, read in full | in hand |

Equations implemented below cite the arXiv numbering of each paper.

## 2. Premise-verification results (SPEC-00 §0.2-1)

- main is at `bdcf9ea` as the routing prompt claims. venv CPU JAX 0.10.2 works.
- `validation/research/` exists; contains `subgrid/` (3D disjoint prototype, research-only,
  explicitly not long-time energy-stable). No `portgrid/` dir exists → this lane creates it.
  No file overlap with the wire-port stack (#771–#779) or the crossval lanes.
- Ledger NFS is mounted; `rfx-known-issues.md` confirms the disjoint-subgrid / SBP-SAT history.
- PR #90 body re-read (read-only): confirmed 2D lessons — (1) point-source area scaling
  ratio² when injecting the same pulse into grids of different cell area, (2) interface
  penalty signs must be outward-normal-consistent per face, (3) Yee boundary derivative
  staggering scale (2/dx vs 1/dx confusion). Carried below as sanity notes.
- **Stale/imprecise spec claim (recorded, not silently absorbed):** SPEC-02 header calls the
  2017 paper "2D TM". The paper itself (Sec. II) says the region operates "in a TE mode with
  components Ex, Ey, Hz" (out-of-plane Hz; TEz in the usual convention). Identical component
  set; only the mode label differs. This lane implements the paper's actual component set
  (Ex, Ey, Hz) and calls it TEz.
- **Second nuance vs spec:** SPEC-02 says "arbitrary odd ratio". The odd-only restriction is
  the 3D paper's (arXiv:1705.02274 Sec. V: "refinement factors are odd integers"). The 2D
  paper explicitly supports "an arbitrary integer refinement ratio r" (Sec. I and IV), and its
  own stability fixture (Sec. V-A, Fig. 4) uses r = 4. M0 operators are implemented for general
  integer r ≥ 1 with tests focused on odd r ∈ {3,5,7}; M1 runs both the paper-exact r = 4 arm
  and an odd-lane r = 5 arm.

## 3. The equations being implemented (2D, arXiv:1606.08761 numbering)

- Region dynamical system (14a)-(14b) with matrices R (17), F (18), B (19), L (20);
  storage function (22)/(25); supply rate (23)/(28).
- Dissipativity certificate = Theorem 1 conditions:
  (29a) R = Rᵀ > 0 — generalized CFL; equivalent to Δt < 2/s_k for all singular values s_k
  of S in (40); classical CFL (47) is a sufficient condition;
  (29b) F + Fᵀ ≥ 0 — reduces to σ ≥ 0 per edge (34)-(36);
  (29c) B = L·(LᵀB) with LᵀB the signed boundary-length diagonal (27).
- Interpolation rule: fine tangential E replicated from coarse, Ê_S = E_N·T (55); coarse
  hanging H = average of fine hanging H, H_N = Tᵀ·Ĥ_S / r (56). Supply rate of the rule is
  exactly zero (62)-(63) → lossless interconnect.
- P-norm adjoint pair: with P_c = ℓ·I_m (coarse tangential edge length) and
  P_f = (ℓ/r)·I_{mr}, the rules satisfy T_c2f = P_f⁻¹ · T_f2cᵀ · P_c exactly.
  Consequently the reverse-mode pullback of T_c2f equals P_c ∘ T_f2c ∘ P_f⁻¹.
- Interface update = (61) (explicit, with ε̂,σ̂ the fine-side averages (58) entering as ε̂/r,
  σ̂/r because the fine half-cell is Δ/(2r) thick); fine interface E then via (55).
  Corners of an embedded fine island need no special treatment (paper Sec. IV, p.8).

## 4. 2022 Corrections — what binds M0/M1 (checklist)

The Corrections replace only the *strong equalization* of fine hanging variables in the 3D
paper — planar (43a)-(43c)[TAP] = (39a)-(39c)[arXiv], and edge conditions (61a)-(61c), (67),
(68)[TAP] — by weaker averaged/signed-circulation conditions ((1), (3)-(5) of the Corrections),
and re-derive the corresponding supply-rate cancellations. Explicitly: "All the other
equations, theorems and numerical results in [1] are correct as stated."

- [x] 2D interpolation rule (55)/(56): NOT affected (2D has no equalization conditions among
  fine hanging variables; only their average is constrained). → no change to M0/M1 code.
- [x] 2D update equation (61) and stability proof (62)-(63): NOT affected.
- [ ] M2 (3D) checklist item pre-seeded: implement planar hanging-variable interpolation with
  the **averaged** condition (Corrections eq. (1)) and edge/corner conditions (Corrections
  eqs. (3)-(5)), not the original (39)/(57) equalities.

Because the proof corrections exist at all, every claim in this lane carries a numerical
energy audit; no proof is trusted un-audited (SPEC-02 §6).

## 5. Architecture (frozen for M0/M1)

- Code: `validation/research/portgrid/` — research-first; no `rfx/` body changes.
- `operators.py`: interface operator generators (T_c2f, T_f2c, P_c, P_f) for general integer
  r and m coarse edges; adjoint-residual and lossless-supply-rate checkers.
- `certificate.py`: 2D region matrix assembly ((17)-(20)) for arbitrary per-edge ε,σ and
  per-cell μ; certificate = (29a)/(29b)/(29c) numeric checks + Δt_max = 2/s_max from (40)
  + classical-CFL cross-check (47).
- `sim2d.py`: two-region 2D TEz prototype — pytree carry, single `lax.scan`, all-static
  shapes, masked coarse arrays (fine island hole), precomputed interface gather/scatter
  (replication/averaging), interface update per (61), global single dt = 0.99 × fine CFL.
  Per-step conserved-energy output using the paper's staggered storage (25) with region-wise
  half-cell areas (coarse half + fine half at interface edges).
- Physical absolute coordinates in SI units throughout fixtures; no preflight suppression.

## 6. Falsifiers (windows frozen NOW; derivations attached)

### F-M0-a — P-norm adjoint residual
For every (r, m) ∈ {3,5,7,4} × {1,2,5,8}, ℓ ∈ {1.0, 1e-3 m}:
`max|T_c2f − P_f⁻¹·T_f2cᵀ·P_c| / max|T_c2f| ≤ 1e-13`, and the jax.vjp pullback of the
applied T_c2f must equal P_c∘T_f2c∘P_f⁻¹ to the same window on random f64 vectors.
Derivation: the relation is an exact identity in real arithmetic (Sec. 3 above); only f64
rounding remains; 1e-13 ≈ 450× machine eps. Spec class is "> 1e-12 fires"; we tighten to
1e-13. Exceeding → FIRE.

### F-M0-b — lossless interconnect (supply-rate cancellation, eq. (63))
On 64 random f64 draws of (E_Nⁿ, E_Nⁿ⁺¹, Ĥ_S): |s| / Σ|individual terms| ≤ 1e-13.
Derivation: exact algebraic cancellation (63). Exceeding → FIRE.

### F-M0-c — certificate vs paper stability conditions
On uniform-lossless and random-material (ε ∈ [ε₀,3ε₀], σ ∈ [0,50µS/m], per-edge) regions,
sizes up to 8×6:
1. R = Rᵀ (residual ≤ 1e-13 rel) and eigmin(R) > 0 at Δt = 0.99·Δt_max_cert, eigmin(R) < 0
   at Δt = 1.01·Δt_max_cert, where Δt_max_cert = 2/s_max(S) from (40)-(41).
2. Δt_max_cert ≥ Δt_CFL_classical (47) − 1e-12 rel for uniform media (paper: (47) is a
   sufficient condition for (29a)).
3. F + Fᵀ ≥ 0 exactly for σ ≥ 0, and a deliberately negative-σ edge must make it indefinite.
4. `‖B − L·(LᵀB)‖_max / ‖B‖_max ≤ 1e-13` and LᵀB equals the signed diagonal (27) exactly
   in structure.
Any inconsistency → FIRE (certificate disagrees with paper's stability condition).

### F-M1a — ≥10⁶-step lossless energy non-growth
Fixture = paper Sec. V-A / Fig. 4 class: PEC cavity 60 mm × 40 mm; coarse Δx = 1 mm,
Δy = 2 mm; centered fine region 40 mm × 20 mm.
Arm A: r = 4 (paper-exact). Arm B: r = 5 (odd lane). Vacuum, lossless, f64.
dt = 0.99 × fine-grid CFL (i.e. 0.99·(√(ε₀μ₀))/√(r²/Δx² + r²/Δy²)).
Source: modulated-Gaussian magnetic current on one coarse Hz cell, f₀ = 3.75 GHz,
HWHM bandwidth 0.74 GHz (paper values); waveform support ends by step n_off (compact
support; identically zero after n_off).
Window: for ALL n ∈ (n_off, 10⁶]: (E_n − E_ref)/E_ref ≤ +1e-8, with
E_ref = E_{n_off+1} and E the staggered storage (25) summed over both regions; and no
NaN/Inf anywhere.
Derivation: the coupled lossless scheme conserves E exactly in exact arithmetic
(Theorem 1 with F+Fᵀ=0, PEC absorbs nothing, interconnect supply ≡ 0 by (63)); f64
round-off accumulates ≤ ~10⁶ × 2.2e-16 ≈ 2.2e-10 relative even under a pessimistic linear
model; window +1e-8 gives ~50× margin and is far below any physical instability signature
(which grows exponentially). Growth beyond window or non-finite → FIRE.
Runtime rule: pilot 10⁴ steps first; if extrapolated wall time for 10⁶ steps > 20 min per
arm on CPU → emit VESSL yaml for that arm and mark partial_gpu_pending; the window is NOT
shrunk.

### F-M1b — interface-only reflection at the paper's reported class
Paper numbers extracted (Sec. V-C + Fig. 9 bottom panel, coarse Δ = 1 mm = λ/10 at 30 GHz,
r ∈ {2,4,6}): interface-only |S11| rises from below −100 dB at low GHz to ≈ −40 dB near
30 GHz; the paper's text claim is that interface reflections are "significantly lower" than
the four-rod scatterer reflections (−35…−5 dB over the band).
Fixture here: parallel-plate waveguide (PEC at y = 0, 40 mm), vacuum, coarse Δx = Δy = 1 mm,
domain length 400 mm, PEC x-ends with time gating; island 20 mm × 20 mm centered in y,
front face at x = 200 mm; magnetic-current line source (all y) at x = 60 mm, probe at
x = 120 mm; identical-dt reference run without the island; reflected = probe_island −
probe_ref, gated to exclude far-end/second-order echoes; |S11|(f) = |R(f)/I(f)| with I the
gated direct incident.
Window (class boundaries sit ≥ 5 dB above the worst Fig.-9 curve values to absorb
figure-reading uncertainty and the PML→time-gating fixture difference):
for every r ∈ {2,3,4,5,6}: max|S11| over [2, 20] GHz ≤ −45 dB AND max|S11| over
[2, 30] GHz ≤ −35 dB. Exceeding either → FIRE.

### F-M1-grad — jax.grad vs central FD through an interface-crossing objective
Fixture: small cavity with island; parameter θ = εr of a block inside the fine island with
one face ON the south interface row (so θ flows through the interface ε̂ coefficients);
source in coarse region on one side, probe in coarse region on the other; J(θ) = Σ_n
Ex_probe². f64, ~2000 steps. Central FD at h ∈ {1e-4, 1e-5, 1e-6}·θ.
Window: min over the h-sweep of |g_AD − g_FD|/|g_FD| ≤ 1e-6.
Derivation: J is a polynomial (linear dynamics) in fields, smooth in θ; f64 central-FD
floor ≈ 1e-9…1e-11 relative at optimal h; 1e-6 is the spec's "~1e-8 class" M3a window
relaxed by the standard FD-truncation allowance (spec M3a proper re-runs at M3). Exceeding
→ FIRE.

### F-M1-vjp — P-adjoint structure of the reverse pass
1. Operator level: pullback of applied T_c2f equals P_c∘T_f2c∘P_f⁻¹ (with F-M0-a).
2. Stepper level: on a tiny fixture, extract from jax.jacrev of one full step the block
   mapping the coarse-interface-Ex cotangent to the fine boundary-row Hz cotangent; it must
   equal the transpose of the forward block (which is cb·(1/r)·segment-mean), i.e. the
   replication-structured cb·(1/r)·T pattern, residual ≤ 1e-12 rel; equivalently
   jacrev(step) ≡ jacfwd(step)ᵀ on the full tiny state to ≤ 1e-12 rel.
Failure → FIRE (backprop is not the transpose of the same dissipative interconnect).

## 7. Sanity notes carried from PR #90 (2D)

- Any source injected INTO the fine region must be scaled consistently with cell area
  (current density × cell area; a raw per-cell pulse differs by r² between grids). M1's
  sources live in the coarse region only; the note binds future fine-region sources.
- Interface sign conventions are derived once from the continuous curl updates
  (Ex: +(H_above−H_below)/Δy; Ey: +(H_west−H_east)/Δx) and tested, not tuned.
- Half-cell (Δ/2, Δ/(2r)) factors at the interface follow (50)/(52) exactly; no ad-hoc 2/dx.

## 8. Do-not-repeat compliance

Single global dt = 0.99×fine CFL (no local time stepping); no SBP operator derivation;
no Huygens/filter stabilization; unmodified Yee everywhere except the paper's interface
update (61); research-first directory, no rfx/ body entry before M3.
