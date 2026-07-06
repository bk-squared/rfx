# Palace FEM arbiter — X-band inset-fed patch S11

Third solver (**Palace**, AWS open-source FEM in the frequency domain) for the
S11 dip that the two FDTD codes disagree on:

| solver   | method            | dip freq   | dip depth |
|----------|-------------------|------------|-----------|
| rfx      | FDTD (time)       | ~10.1 GHz  | ~ -12 dB  |
| openEMS  | FDTD (time)       | ~9.26 GHz  | ~ -12 dB  |
| Palace   | FEM (frequency)   | **TBD**    | **TBD**   |

Palace solves the same geometry with an independent numerical method, so its
resonant frequency arbitrates the ~0.85 GHz disagreement.

## Design (identical to the FDTD runs)

From `scripts/crossval/rfx_patch_inset_xband.py` (issue80 frame):

- Substrate RO4003C: `eps_r=3.38`, `LossTan=1e-3`, `h=0.787 mm`.
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

1. Read the |S11|(f) dip frequency and depth from `port-S.csv`.
2. Compare against rfx ~10.1 GHz and openEMS ~9.26 GHz (both ~ -12 dB).
   - Palace near ~10.1 GHz supports rfx; near ~9.26 GHz supports openEMS.
3. Cross-check `Zin = 50*(1+S11)/(1-S11)` at the dip against the rfx JSON
   (`scripts/crossval/out_xband/xband_inset_2.4mm.json`).

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
```
