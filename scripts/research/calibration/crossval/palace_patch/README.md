# Palace FEM arbiter — X-band inset-fed patch S11

Third solver (**Palace**, AWS open-source FEM in the frequency domain)
alongside the two FDTD legs (rfx, openEMS).

**RETRACTION (#273):** the "rfx ~10.1 GHz / ~ -12 dB" dip this README
originally set out to arbitrate was a **miswired-frame artifact** — the
interior-ground frame of `../rfx_patch_inset_xband.py` (PR #272 post-mortem),
whose dx=197um "-30.25 dB @ 10.359 GHz" match collapses to -12.02 dB @
10.106 GHz at dx=98.5um and to -7.04 dB @ 9.250 GHz in the canonical frame
(`../rfx_patch_xband_canonical_frame.py`). There was no genuine ~0.85 GHz
solver disagreement to arbitrate. Final numbers (per
`../DESIGN_DOC_patch_student.md`, 2026-07-07):

**Level 1 — shielded (all-PEC box) resonance, fundamental mode:**

| solver   | method              | f0            |
|----------|---------------------|---------------|
| rfx      | FDTD + Harminv      | 9.131 GHz     |
| openEMS  | FDTD ring-down      | 9.194 GHz     |
| Palace   | FEM eigenmode       | 9.199 GHz     |

**Level 2 — radiating S11 (open boundary):**

| solver   | method              | dip freq      | dip depth  |
|----------|---------------------|---------------|------------|
| rfx      | FDTD (time)         | 9.250 GHz     | -7.04 dB   |
| openEMS  | FDTD (time)         | 9.2625 GHz    | -9.46 dB   |
| Palace   | FEM driven          | 9.05 GHz*     | -12.99 dB* |

\* Palace-driven caveat: the tight radiation box (~10 mm) with a first-order
absorbing boundary pulls the driven dip low; the shielded eigenmode of the
same mesh agrees with the FDTD legs at 9.199 GHz. Dip DEPTH is not comparable
across the three solvers while their loss models differ (see below). Evidence:
`../evidence/` + `../verdicts.json` (rows `coarse-dip-depth`,
`level1-shielded-agreement`, `level2-radiating-s11`).

## Design (identical to the FDTD runs)

From `../rfx_patch_inset_xband.py` (issue80 frame):

- Substrate RO4003C: `eps_r=3.38`, `h=0.787 mm`, **lossless by convention**
  (`LossTan=0.0` in `patch_s11.json`, matching the lossless FDTD legs; the
  arbitrated quantity is the dip FREQUENCY). Dip DEPTH is not comparable
  across solvers while loss models differ — the real board is
  tan-delta ~0.0027 @10 GHz (`../DESIGN_DOC_patch_student.md`); a lossy rerun
  changes depth, not the frequency arbitration. NOTE: the driven result on
  record (-12.99 dB @ 9.05 GHz) was produced with the pre-#273 `LossTan=1e-3`
  config; at that loss level the depth shifts slightly, the dip frequency
  negligibly.
- Patch: `L=8.595 mm` (x, resonant) x `W=10.129 mm` (y), PEC at `z=h`.
- 50-ohm feed strip `w=1.8 mm`, `12 mm` long before the patch edge.
- Inset depth `d=2.4 mm` (the rfx match point) through two etched notch
  slots, `gap=0.9 mm` each side.
- Ground: PEC on the substrate bottom face.
- Radiation box: `~10 mm` air margin laterally and above; outer faces (except
  the ground plane) are a first-order absorbing boundary.

## Files

- `mesh_patch.py` — Gmsh (OpenCASCADE) generator. Builds substrate + air
  volumes, imprints the zero-thickness PEC surfaces (ground, patch, feed,
  flanks) and the lumped-port rectangle, then tags physical groups whose
  integer tags are the Palace attributes:

  | attr | group           | Palace role              |
  |------|-----------------|--------------------------|
  | 1    | `substrate_vol` | Material eps_r=3.38      |
  | 2    | `air_vol`       | Material eps_r=1.0       |
  | 3    | `gnd`           | PEC                      |
  | 4    | `metal`         | PEC                      |
  | 5    | `port`          | LumpedPort (R=50, +Z)    |
  | 6    | `farfield`      | Absorbing (Order 1)      |

- `palace_patch.msh` — Gmsh 4.1 ASCII mesh (regenerate with the script).
- `patch_s11.json` — Palace driven config (8-12 GHz, 81 points, uniform).

## Regenerate the mesh

```bash
pip install gmsh
python mesh_patch.py            # writes palace_patch.msh
```

Current mesh: ~10.0k nodes, ~53.8k tets, ~11.1k boundary tris. All six
physical groups present and the domain boundary is closed (verified: no
ungrouped exterior faces). This is a moderate arbiter mesh; refine by
lowering `LC_MIN` / `LC_SUB` in `mesh_patch.py` if the FEM result is not
converged.

## Run Palace (do this once Palace is built)

```bash
# N = number of MPI ranks
mpirun -n 4 palace patch_s11.json
```

### Outputs

Palace writes to the `Problem.Output` directory `postpro/patch_s11/`:

- `port-S.csv` — the S-parameters vs frequency. Column `|S[1][1]| (dB)`
  is |S11| in dB; also `arg(S[1][1]) (deg.)`. **This is the comparison file.**
- `port-V.csv`, `port-I.csv` — port voltage/current (Zin if wanted).
- `paraview/` — field snapshots (from `SaveStep`) for eyes-on checks.
- `farfield.csv` — the far-field pattern samples.

### What to compare

1. Read the |S11|(f) dip frequency and depth from `port-S.csv`
   (result on record: **-12.99 dB @ 9.05 GHz**, `../evidence/palace_port-S.csv`).
2. Compare the dip FREQUENCY against the canonical-frame rfx 9.250 GHz and
   openEMS 9.2625 GHz (the old "rfx ~10.1 GHz" comparison target is retracted
   — miswired-frame artifact, see the header). Remember the Palace-driven
   absorbing-box caveat before reading the 0.2 GHz offset as physics.
3. Cross-check `Zin = 50*(1+S11)/(1-S11)` at the dip against the
   canonical-frame rfx JSON (`../evidence/rfx_cf_s11_dx197um.json`).
   Dip DEPTH stays a non-observable for cross-solver purposes until the loss
   models are unified.

## Open risks — flag for the first Palace run

- **Lumped-port orientation.** `Direction: "+Z"` assumes the port E-field
  points from the ground plane (z=0) up to the strip (z=h). If the excitation
  reads as reversed or S11 looks wrong-signed, flip to `-Z`. The port
  reference impedance is set to the design 50 ohm (`R=50`); the microstrip
  line's own Z0 is only nominally 50 ohm, so a small port mismatch ripple is
  expected and is not a physics error.
- **Absorbing boundary choice.** First-order absorbing (`Order: 1`) on a box
  only `~10 mm` (~0.3 lambda at 10 GHz) from the patch is approximate; a patch
  radiates broadside, so the truncation reflects some energy. If the dip
  frequency drifts with box size, increase `MARGIN_XY` / `AIR_ABOVE` in
  `mesh_patch.py`, or raise the absorbing order. This mirrors the openEMS MUR
  and rfx CPML truncations — all three approximate the open boundary.
- **Truncated dielectric.** The substrate is modelled full-footprint to the
  box wall (matching the rfx/openEMS "infinite ground + substrate" frame), so
  the substrate side faces sit on the absorbing boundary — an approximation of
  the finite-board reality, consistent across the three solvers.
- **Frequency units.** Palace driven frequencies are in **GHz**; the mesh is
  in **mm** via `L0 = 1e-3`. Confirm the sweep prints 8-12 GHz in the log.
- **Adaptive sweep option.** The config uses a uniform 81-point sweep. For
  speed, Palace also supports an adaptive fast-frequency-sweep sample
  (`"Type": "Adaptive"` with a tolerance) — see `examples/cpw/`.
- **Mesh convergence.** The current mesh resolves the 0.787 mm substrate with
  only ~1-2 element layers away from the metal (Palace uses Order-2 basis).
  Do a refinement check (halve `LC_SUB`/`LC_MIN`) before trusting the dip to
  better than ~0.1 GHz.
