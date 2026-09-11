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
predate this final attachment work. At that checkpoint final GPU consumer checks, CI and PR review were still
required. The subsequent pinned GPU result follows below.

Final local contract packet: 53 passed, including all attachment cases and
the f0 refusal inventory. Ruff and diff whitespace checks passed. Focused
read-only review of the three post-assembly fixes found no additional
confirmed defect; it did not run FDTD or qualify RF convergence.

## Final pinned GPU consumer verification

VESSL 369367260440 ran commit `c2dd484e39780a8ac376ae39491f0573650097cd`
on an RTX 4090, from a clean self-contained checkout. Rotation, AD–FD and
NU gates all passed sequentially with unchanged acceptance thresholds.

- x/y rotation: max complex S difference `3.488e-6`; all four drive records
  settled below -100 dB. Both orientations flagged the same top two of
  twelve bins as unreliable, so this is a symmetry check, not qualification
  of those high-frequency bins.
- Fixed-band AD–FD: `g_ad=1.121405e-3`, `g_fd=1.120691e-3`, about 0.064%
  relative difference at the original `h=1e-3`, below the original 3% gate.
  All eight objective frequencies passed the reliability screen on the
  forward and both FD legs; worst settling was -124.82 dB. Actual field
  initialization receipts distinguish f32 AD from f64 FD fields.
- Existing NU patch gate: max `|S11|=0.9839`, in-band reactance crossing
  `7.7575 GHz`, in-band maximum `Re(Zin)=4420 ohm`. The fixture uses
  uniform-valued profiles and this does not certify graded propagation.

Raw V/I, S, wave amplitudes, conditioning, reliability, field dtypes and
full logs are in `gpu-attachment-final-369367260440/`. The full provider log
was saved and SHA-256 verified before terminal-run cleanup. An additional
38 local domain-fidelity/waveguide-aperture tests passed, covering the
other two existing #729 surfaces. PR #981 remains subject to CI and review.

The original other-surface reproductions also agree with their physical
spans: a commensurate 20 mm domain reports 20 cells and no quantization
finding; the WR-90 default aperture at dx=1 mm uses 23 cells (23 mm) and
`f_cutoff=6.512162173 GHz`, matching the discrete 23-cell formula.

## Final review: density reservation must include the modal fringe

A build-only review falsified the initial reservation beyond the trace
footprint: at the fringe source `(8,8,8)` its diagonal owner `(7,7,8)`
remained occupied even though a nonzero SourceSpec was injected there.
The core-interval checks passed, but the actual modal support was wider.

Reservation now uses the Laplace profile's `cell_indices`, which also
specify its resistive load support; the uniform mode retains its uniform
cross-section. Both the four incident owners and the existing Kottke
six-face-neighbour guard use that support. Static PEC clearing is unchanged.

Two fringe orientations and the outside-density derivative check failed
before the fix (3 failed, 2 core controls passed). Afterward all 6 targeted
checks passed, including an additional positive-width Kottke neighbour
outside the four-owner stencil. The fringe contains an actual nonzero
SourceSpec; remote density and its derivative remain observable. Focused
review confirmed the finding is resolved. Logs are in `attachment-checks/`.

This change only affects `pec_occupancy_override` reservation. The preceding
GPU packet exercises ordinary runs and `eps_override`, not that density
input. Its source receipt remains pinned to c2dd484e; those unchanged
numerical paths do not require another identical GPU solve. The density
reservation has its own traced-input and actual-source setup checks above,
not a new claim of spatial AD or RF optimization qualification.
