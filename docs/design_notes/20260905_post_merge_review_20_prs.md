# Post-merge review of the 20 PRs this session merged (2026-09-02..04)

Workflow wf_a4c219ab-751, 38 agents: 20 reviewers, 17 adversarial verifiers, 1 synthesis. Lens: the arrival-clipping finding of `waveguide_vi_envelope_sweep_results.md` sections 2-3. Annotation E was reproduced by hand and filed as #904.

The merged record is sound. Nothing reopens. The arrival-clipping finding does not move one committed number: every chain-battery cell sits at T/tau_far = 2.12-2.14, under the brief's line of 3, but 40 -> 80 -> 120-period twins on the committed cells moved no |S11| bin by more than 3.3e-5 and no derived gate at all, because at f/f_c >= 1.28 the far-wall term is 1e-6 to 1e-5, not the 4.3e-5 the brief measured at f/f_c ~ 1.02. What the review did find is a different absorber effect in the same fixture: a first-order echo from its 0.756 lambda_g pad, fully inside the record, which answers questions the PRs left open and shows two mechanisms mislabeled in the merged text.

1. Reopen

None. No gate, pin, verdict or committed measurement in the twenty PRs is wrong or record-clipped. The three closest calls and why they stay closed: #893's empty-guide "no attributable order" is the contract-correct reading of the pre-declared branch (the number is right, the explanation is missing); #880's diagnostic script crashes on main, but the committed report reproduces bit-for-bit from the merge tree and #889 already strict-xfailed the live tie; #874's support matrix is stale, but its headline "NOT chain-closed" is still true on main (9 red, two families).

2. Annotations on the merged record

A. Chain-battery fixture, far-absorber echo (#893 x3, #880 x1). The pad is ceil(0.75 lambda_g/dx) = 43.18 mm = 0.756 lambda_g at every rung. Its return (round trip 1.1-1.3 ns, ripple period ~0.9-1 GHz, fitted reflector 117-120 mm from the reference plane, i.e. 16 mm inside the far pad) is first order in dx (0.0153 / 0.0072 / 0.0035) and adds to a second-order port term (0.0186 / 0.0052 / 0.0014). Their sum is the committed 0.041048 / 0.016405 / 0.007028 with ratios 2.50 / 2.33, which is exactly why no order could be attributed; at pad x4 the curve is monotone, ratios 3.71 / 3.85, and the |S11|-|S22| asymmetry drops 7e-3 -> 2e-4. Four texts to correct: (i) tests/oracle/test_waveguide_chain_battery_guide_cell_aperture.py:556-560 says the 8.6 GHz worst bin is "where a residual aperture mismatch would sit"; it is the first constructive-interference bin of the echo, and with the echo removed the residual rises with frequency to 11.6 GHz. (ii) PR #893 section 5.4, "convergence order not explained" and the paragraph that ruled out a rising-with-frequency term, was reasoned on the echo-contaminated curve. (iii) PR #893 section 5.7 / test_referee_at_the_claims_rung: 35 % of the 0.012620 Airy magnitude residual is echo (0.008157 at pad x4), and the interval floor 0.009029 carries it too (0.007853 at pad x4); without the echo the "below the interval" branch would have fired, not "as predicted". The Airy phase reading is pad-robust; the 0.05 gate passes with >4x margin either way. (iv) docs/design_notes/waveguide_false_lane_column_power_results.md section 4 says the ripple matches a round trip to the near absorber; it is the far one (a near-pad period would be 1.8-5.6 GHz, invisible across a 3.2 GHz band). The NON-CLOSING verdict stands. Re-adjudicating the port's order on a thicker pad needs its own pre-declaration under R2.

B. Record-length wording (#882). rfx/api/_sparams.py:884-886 says all eight drives settle below -40 dB "so none is a record-truncation artifact". The settling witness reads ring-down at the port planes and cannot see a far-wall return that arrives after T. The claim survives on a witness the PR did not run: slab|fine|false 6.982211e-3 -> 6.981880e-3 at 40 -> 120 periods (rel 5e-5, T/tau 2.12 -> 6.36). Put that twin and T/tau = 2.12 in the docstring or known-issues.

C. Unitarity gate provenance (#870 x2). The 0.005 anchor 3.151894e-3 (WR-28 dx50 er4) is a leakage number from a 0.064 lambda_g absorber, and the fixture's own absorber_discipline.note says "do not pin it as extractor behaviour" (31x better at 0.75 lambda_g; extractor floor ~1e-4). Neither the test's block comment nor the PR body mentions the absorber, and regeneration would re-derive the gate to 0.001. "Nothing read them" is also wrong: tests/locks/test_absorber_discipline_witness.py has gated the same keys at 5e-3 since #595 with the setup attribution attached, and #870's gate-policy tripwire does not list it. The repo carries two 5e-3 pins, one caveated.

D. Public docs lag (#874). support_matrix.md:36 and sparameter_support_matrix.{md,json} carry the 24-red / four-family census and rotation residual 6.60 deg from fixture.json (N+1 port). On main's port (#889, re-measured by #893) it is 9 red in two families, rotation 0.0317 deg, all three ladders interpretable. Refresh the rows; the headline stands.

E. Reproducer broken on main (#880). waveguide_false_lane_column_power_suspects.py builds the port config live; after #889 the live cutoff equals the guide cutoff, S1 = 0, and main() dies with ZeroDivisionError at :412. Its --out default is the committed suspects.json, so following the PR's "re-run and re-commit" instruction overwrites the report with mixed-provenance numbers (product explains 22 % instead of 65 %) and then crashes. Re-keying to the fixture's fc_port_hz alone does not restore the numbers (S3's q_ratio moves with the aperture). Retract the instruction; pin the reproducer to the merge tree or re-measure under #868.

F. Docstring worst case (#887). "159x the worst value measured" is a macOS/Accelerate worst. On Linux x86-64, where the gate runs, MMD_ATA gives u_refined 1.558e-13, so the ensemble margins are U1 64x, U3 9.4x, U2 9.4e3, 2.5x tighter than quoted. Default-ordering margins (177x, 25.9x, 2.59e4) hold; nothing is red.

G. Plan note (#879 x2). (i) Line 146's cv19 Palace falsifier borrows -105 MHz/cell (cavity length) for an iris-thickness step whose recorded sensitivity is +2.4 to +3.3 MHz/cell; it cannot fire the 19 MHz gate, and "2.38 mm (one a/90 cell)" is arithmetically 2.254 mm. Use a +1-cell cavity-length step (-102 MHz) for f0 and keep the thickness step for the bandwidth gate. (ii) Line 93 gates cv04's new openEMS/Palace legs with W_T = 0.017; #886, merged 3 h later, recorded that more than half of that envelope is 719-step truncation and must not be called a discretisation number, while deciding that no window moves. L1's pre-declaration must state the caveat and choose: carry 0.017 under #886's decision, or run #886 section 5.4's re-derivation first. The Meep half is moot; #835 gates it pointwise.

H. Stale pointers (#877). Nine file:LINE pointers into the merged test files kept their old line numbers (six in design notes and the WR-90 fixture JSON, plus sparameter_support_matrix.{md,json} -> test_coax_two_port_smatrix.py:699, which should be 738). Nothing computes from them; tests/contracts/test_evidence_citation_pointers.py does not cover those documents.

3. Sound

#864 cv22: rfx leg re-run from the merge tree reproduces all 446 fields to 1e-9; no port; record ends before any absorber return by design; #886 bounds the residual as the lattice term.
#865 wire-thru ladder: SVD of the stored S reproduces every e to 2e-16; T/tau >= 20.5; settling <= -126 dB; the sign flip is the pre-declared observable crossing zero.
#866 cv23: 363 citations resolve, ladder x0.25 per halving, no port; the T-window caveat was already written by #886.
#867 battery run 1: 133 passed / 18 xfailed on the merge tree; census 103/56/23/3 reproduces; PEC-short cells at T/tau 7.1; no thru/slab gate below 1e-2.
#870: 44 tests pass incl. the live re-measurement; closure witness 2.146e-5 vs 0.02; unitarity is the leakage observable (caveat C).
#871 cv24: 33 passed; closed PEC cavity, no absorber, energy drift <= 4.3e-8; pre-declaration and comparator timestamps precede the run.
#874: docs-only; every number is in fixture.json; census correct at merge (caveat D).
#875: 133 duration keys renamed 1:1; collected node-id set identical modulo directory; 548 passed.
#877: checker prints PROOF HOLDS; AST comparison 510/532 identical, the 15 differences are the note's list (caveat H).
#879: 41 citations resolve; cv11/cv18/cv19 targets at T/tau 6.6 / 3.2 / 31, each with a record-doubling witness (caveats G).
#880: script from the merge tree regenerates suspects.json with 0 differing leaves; remainder first order (caveats A-iv, E).
#881: 0/480 records misclassified by the float32 floor; #893 read -102 dB live on the same cell; PEC-short paths T/tau >= 5.8.
#882: 0.011 recomputes from the envelope; record twin changes it by 3e-5 relative (caveat B).
#883: the three rewritten controls go red on the pre-merge tree; mesh digests reproduce; the clamp returns the old depths on the battery grid.
#886: 24 passed; an independent 1-D Yee march agrees with the lattice model to 2.7e-6; record/t_safe 0.88-0.94 by design.
#887: committed 1.4655e-9 bit-exact on Linux; default-ordering margins hold (caveat F).
#889: 12-test lock gives 9 fail / 3 pass on the parent; re-measure to the last digit; record doubling moves |S11| <= 3.3e-5.
#891: docs-only pre-declaration; 178 tests; lattice search reproduces; #893 landed on its predictions (0.0317 vs 0.032 deg).
#892: 13 committed ratios reproduce; four cv04 falsifier rows reproduce; same tau as the brief, exclusion polarity.
#893 battery run 2: 234 replay tests; CPU re-measure <= 1e-6; 120-period record moves nothing above 1e-5 (caveats A).

4. Most consequential finding

The brief needs two qualifiers before it is used as a lens again. Item 4 ("CPML absorption is not the limit at any reachable setting") was measured at K >= 3 lambda_g; at the committed fixture's 0.756 lambda_g the absorber return is the dominant first-order term in the empty-guide |S11| magnitude, and it is what kept #893 section 5.4 open. Item 3 ("every convergence order taken with T/tau < 3 is a clipping artifact") is a near-cutoff statement; at f/f_c >= 1.28 the record twins put the term at 1e-6 to 1e-5. The practical consequence: the port's convergence order on the empty-guide leg has not been measured at this fixture, only the sum of port and pad; a thick-pad run (3 lambda_g gives ratios 3.7 / 3.85) would measure it, and needs a new pre-declaration.

5. Where the twenty reviews disagreed

T/tau = 2.1 < 3 as a defect: four reviews (#891, #867, #881, and the fixed-40-period record rule) filed it as ANNOTATE; the refutations, backed by 40 -> 80/120-period twins moving nothing above 3.3e-5, took all four down to NOTE. The #891 reviewer had pre-declared "<1 % => NOTE" and filed ANNOTATE after a <0.1 % result. Resolved: NOTE, one line in the fixture README.

Whether the far-wall return is inside the record: #870's status says the 8.4 GHz round trip "is not inside the record"; #881's refutation and #893's status compute the first return peaking at 2.6 ns of a 3.45 ns record. The latter is right; "clipped" in the brief means T/tau < 1.

#889's convergence-rate hedge: the per-PR status still calls the O(dx^1.3) hedge "contradicted by #893", but the refutation shows that compares the one-sided max excess (dx^2 as pre-declared) with a two-sided band RMS (dx^1.6-1.8 on the same artifact). The refutation stands; read that caveat as withdrawn.

#879 L1's T window: the reviewer said L1 "must re-derive"; the refutation notes #886 decided no window moves and the truncation content was on record 12 h before #879. The annotation should offer the choice, not mandate re-derivation.

Echo side: the #893 and #880 reviewers agree (far pad, 117-120 mm); only the merged #880 note says near.