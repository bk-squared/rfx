# MSL attachment and assembly boundaries

2026-09-11. Implementation and verification checkpoint for the user's
approved policy: reject a port whose declared ground/trace do not meet the
realized conductors beyond ordinary lattice rounding. No physical or AD
acceptance tolerance is changed.

## Contract

The two z coordinates use the #931 sheet-plane rule, including lower-node
half-cell ties and refusal outside the grid. The source uses normal edges
between those bounding nodes. Its registered width centre is used for
Laplace normalization and trace extraction, including a narrow trace whose
rounded endpoints have a midpoint outside its footprint.

Validation reads geometry before any port clears edges. Each node of the
declared trace footprint, plus the actual extraction centre, must meet a
longitudinal conductor edge at both declared planes. Transverse crossbars,
partial trace coverage, normal metal inside the substrate, and intervening
conductor planes cannot certify attachment. Only explicit domain PEC faces
count as domain conductors; zero padding alone says nothing about a ground.

Lossy f0 sheet edges are observed without turning them into PEC. A vertical
lossy sheet crossing the driven Ez interval is also detected. The direct
coax transition does not implement f0 sheets and now refuses them instead
of dropping their impedance context. Preflight reports a blocking finding
while retaining independent clearance diagnostics. Runtime checks remain
active with `skip_preflight=True`.

## Post-assembly corner cases

- **Kottke volume boundaries:** a sphere at the trace plane can supply a
  staircase wall that the selected Kottke tensor releases. The uniform
  runner now observes tensor-frozen edges and surviving sheet/wire edges.
  It does not change the stepper's masks or freeze fractional Kottke edges.
  Preflight cannot certify a smoothing option selected only at `run()`.
- **Differentiable density reservation:** the existing six-face neighbour
  guard misses the diagonal `(i-1,j-1,k)` cell that also owns `Ez[i,j,k]`.
  MSL reservation now includes all four incident volume cells, with the
  actual periodic convention. A traced-density check verifies zero
  sensitivity in the reserved cell and retained sensitivity elsewhere.
- **Generated coax stub:** the shared stamper's axial padding wrote high
  conductivity into the MSL interval at an aligned junction and one extra
  layer at a half-cell tie. The direct transition now snaps its junction
  consistently and restores the registered epsilon/sigma arrays at and
  above that node. Below it, the coax pin, shell and dielectric remain.
  Standalone coax stamping is unchanged. This does not certify the caller's
  entire junction or its RF calibration.

## Tests and limits

`test_msl_realized_port_contract.py` exercises actual assembly and public
runner setup on uniform and nonuniform grids, all four MSL directions,
finite-thickness surfaces, boundary conditions, f0 sheets, Kottke selection,
density reservation, and mixed/direct-coax trace lookup. Build-only captures
stop before FDTD. The coax preservation test compares the entire registered
material region above the junction and checks pin/shell/PTFE below it.

An initial expanded CPU inventory finished with 710 passed, 10 failures,
2 setup errors and 9 expected failures. This was collected during edits and
is a failure inventory, not acceptance of a pinned final head. Missing
explicit grounds were repaired in the passivity, internal-probe and
offset-driver fixtures. Independent preflight diagnostics remain visible
when attachment blocks a run. The affected five-file run then passed its
103 execution/diagnostic cases; its remaining two sheet-census assertions
were updated to include the ground, and both passed in the subsequent
93-case source/geometry packet. Two additional instrument geometry checks
and one traced-density derivative check passed separately.

The historical coax attempt-3 model is now explicitly tested to reject its
displaced port. Its frozen geometry, numerical records and existing strict
expected failures remain history. `tests/_coax_msl_instrument_fixture.py`
is a separate aligned board with a square clearance aperture and connected
post. It is used only for instrumentation tests. Its conductor connectivity
is checked before stepping. All 27 ladder/standoff/payload checks passed,
including bit identity with optional ladder and flux outputs enabled.
Those 200-step records are intentionally unsettled; large condition numbers
are reported and they do not establish transmission, passivity or accuracy.

Logs for this checkpoint are retained in `attachment-checks/`, with hashes
in its manifest. The existing settled GPU AD, rotation and NU receipts
predate this final attachment work. Final GPU consumer checks, CI and PR
review are still required before closing #729.

Final local contract packet: 53 passed, including all attachment cases and
the f0 refusal inventory. Ruff and diff whitespace checks passed. Focused
read-only review of the three post-assembly fixes found no additional
confirmed defect; it did not run FDTD or qualify RF convergence.
