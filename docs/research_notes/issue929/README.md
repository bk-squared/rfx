# Wire-port gaps and explicit galvanic case contracts

2026-09-11. Baseline main: `81467c81907548c49b126ffa4c0a457c2c5f2698`.
The user approved retaining general API warnings while requiring explicitly
galvanic crossval/test cases to reject a bad connection before computation.
No port-declaration API or physical acceptance tolerance is added or changed.

## General API observation

The baseline two-plate build reports 0/1/0/0 findings for gaps of 0/1/2/3
cells. Its fixed one-node look-ahead misses larger gaps.

The actual source occupies half-open E edges. A lower terminal is the first
edge's start node; an upper terminal is the last edge's end node. The check
now finds the nearest outward contact on that column in the assembled grid,
using realized wall planes and source-axis PEC edges. An axial PEC edge k
begins at node k and ends at k+1, so a filament can supply contact even when
it supplies no tangential wall. The gap is the node-index difference in
cells and the actual node-coordinate difference in metres.

This observation changes no source, geometry or solver field. A separation
is a warning and does not establish a capacitive-only coupling mechanism.
The remedy is conditional on intended contact and accounts for moving the
lower endpoint as well as the extent. The existing NU classification-
unavailable path remains explicit; it now asks users to inspect live/dead
edges and intended contacts instead of telling them to avoid every PEC
overlap. Absence of that unavailable check is not evidence of contact.

## Explicit cv05/cv15 requirements

A build-only probe of the actual baseline case source in Git confirms:

- cv15 accepts a registered wire moved two cells off ground while its geom
  dictionary remains unchanged, and reports the dictionary's old position.
- cv05 accepts a wire whose registered extent is halved, because its stack
  check does not inspect the feed. The baseline build-only path exits 0.

The shared case helper now reads the registered wire and the production
source-edge mapping. NU cv05 uses cumulative-coordinate `position_to_index`
and `wire_port_edge_span`; it does not substitute a uniform grid or z/dx.
The case requires a positive finite wire extent/impedance, at least one
live source edge, and endpoints on the independently identified ground and
patch planes at the actual feed column. Missing conductors and gaps fail.

Review found two additional false certificates in an intermediate helper:
cv05 could accept an intermediate sheet as its patch terminal, and cv15
could accept an entirely shorted source inside another PEC body. Specific
stack-plane targets and the live-edge condition now reject those cases.
The retained red control explicitly removes these two guards in memory;
it is a mutation experiment, not a claimed checkout of baseline source.

cv05 performs this check before its first Harminv/reference solve and reuses
the same checked ported model later. cv15 already checks before solving;
its check now observes actual registration rather than copied geom fields.
The cv05 dimensions tracked by #959, numerical gates, solve settings and
committed crossval result artifacts are unchanged. These are input-contract
checks, not new RF calibration or convergence evidence.

## Verification

- 48 API/contact and related preflight tests passed, including three axes,
  both ends, 0–3-cell gaps, negative extent, nearest-plane selection,
  off-column conductors and filament contacts.
- 37 case tests passed without FDTD: actual registration-only mutations,
  missing conductors, intermediate sheet, all-PEC source, baseline and
  sheet-plane falsifier controls, and refusal before the first solve.
- The complete example-fidelity/safety audit passed 206 tests after case
  edits stopped. A final wording-only clarification passed the three
  affected variants afterward. Its first run overlapped a source edit and reported a
  source-hash change; that preliminary teardown is not solver evidence.
- Five NU-unavailable message expectations were updated in three variants;
  all numerical snapshot fields remain unchanged. The new helper is
  explicitly classified as constructing no Simulation and doing no solve.

Logs, build-only baseline probe, mutation control and hashes are retained
alongside this note. CI on the final commit remains required.

Ruff passes on the changed library/tests and new helper. The two legacy
case scripts retain their existing 22/21 diagnostics with zero new ones,
verified against baseline source; unrelated formatting is unchanged.
