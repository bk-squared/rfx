# Palace FEM referee — cv07 Sheen microstrip LPF

Run tooling that produces the **independent-method referee** for the committed
cv07 rfx-vs-openEMS first-null split on the classic Sheen 1990 low-pass filter.

## What / why

cv07's committed cross-check
(`validation/crossval/_07_sheen_results/{rfx,openems}.json`) locks a ~9-10% first-S21-null
split: **rfx 7.218 GHz, openEMS 7.983 GHz** (both FDTD, dx~200 um / 4-5 substrate
cells). Both refs STAIRCASE the wide low-Z patch edges, so neither resolves the
open-end / step fringing exactly. Open-end fringing lengthens the resonator
electrically and pulls the transmission zero DOWN in frequency; which of the two
FDTD null positions is closer to the fringing-exact truth is not decidable from
two staircased FDTD runs.

**Palace** is a frequency-domain FEM solver on a **conformal tetrahedral mesh**
(no staircase), so it captures the fringing exactly and can referee. Run on the
SAME matched geometry (the exact domain frame of `07_sheen_lpf.py`) at two mesh
densities:

| mesh   | LC (mm) | tets    | ~DOF (order 2) | sweep            | VESSL run    |
|--------|---------|---------|----------------|------------------|--------------|
| coarse | 0.25    | 140,039 | 924,257        | 81 pt, 4-12 GHz  | 369367248550 |
| mid    | 0.18    | 373,388 | 2,464,361      | 51 pt, 6-9 GHz   | 369367248558 |

(These tet/DOF budgets match the proven cv06b notch referee's coarse ~143,812 /
mid ~376,802 meshes, which fit the 24 GB rtx4090. Both VESSL runs completed.)

**Result / verdict (frozen):** the referee revealed the premise was incomplete —
the Sheen stopband is a **DOUBLE transmission-zero (~7.0 AND ~8.0 GHz)**, not a
single null. Palace (conformal FEM) puts the two zeros at **7.032 & 8.048 GHz**
(mid mesh, parabolic; coarse->mid shift only +0.013/+0.022 GHz => converged).

| solver              | lower zero | upper zero | argmin "first null" | doublet Δ vs Palace |
|---------------------|-----------|-----------|---------------------|---------------------|
| Palace (FEM, mid)   | 7.032     | 8.048     | 8.048 (fragile)     | —                   |
| openEMS (FDTD)      | 7.031     | 7.995     | 7.983               | **0.66 %**          |
| rfx (FDTD, 200 µm)  | ~7.28     | (unresolved) | 7.218            | 4.64 %              |

openEMS resolves the SAME double-zero structure as the conformal referee (<~1 %);
rfx's coarser mesh + frequency sampling does NOT cleanly resolve the doublet (a
spurious extra dip + a shifted/merged central feature). The committed argmin
"first null" is **mesh-dependent** — it picks the marginally deeper member of the
near-equal-depth doublet, which flips from 7.0 GHz (Palace coarse) to 8.05 GHz
(Palace mid). So the ~10 % rfx-vs-openEMS "split" is largely a **comparator
artifact of a double-null**, not a physical single-null disagreement. `sides_with`
therefore names the structure-faithful match (**openEMS**); the argmin metric is
locked separately and labelled fragile. See the `referee` block of
`tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json` and the one-line
summary printed by the producer. **No analytic reference** is used: the Sheen
stepped-impedance transmission zeros have no clean fringing-free closed form, so
this referee is a strictly three-SOLVER comparison (rfx / openEMS / Palace).

The verdict is committed as evidence — the raw Palace `port-S.csv` arrays live in
`tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json` (dB -> LINEAR),
re-derived by `build_sheen_lpf_palace_referee.py`, gated by
`tests/test_sheen_lpf_palace_referee_gates.py`. Those survive a clean checkout;
this directory is the *provenance* that generated them.

## Geometry lock

`mesh_sheen.py` is hard-locked to the exact domain frame of
`validation/crossval/07_sheen_lpf.py` (propagation x, transverse y, stack z; the
Sheen board mapped rfx_x = Sheen_y), in mm:

    substrate  eps_r = 2.2    h = 0.794 mm    LOSSLESS (LossTan = 0)
    50-ohm feed  W = 2.413 mm
    wide patch  20.320 mm (transverse) x 2.540 mm (propagation), low-Z section
    input feed  x [0, 12.466]    centred y = 9.8565    (into -x absorber wall)
    output feed x [15.006, 27.472] centred y = 16.4635  (into +x absorber wall)
    ports       two 50-ohm lumped sheets (ground->strip, +Z) at x = 2.5, 24.972
    far box     first-order absorbing on all non-ground outer faces
    domain      27.472 x 26.320 x 3.794 mm  (3.0 mm air above the trace)

The port location only sets the (de-embedded) reference-plane phase; the FIRST-NULL
FREQUENCY of |S21| is a property of the filter transfer, invariant to it.

## Run order

1. **Mesh** (Gmsh; writes msh 2.2 for MFEM). `.msh` files are **regenerable and
   NOT committed** — regenerate into the (gitignored) `_artifacts/palace_sheen`
   WORK dir before solving:
   - coarse: `python mesh_sheen.py --out .../_artifacts/palace_sheen/palace_sheen.msh`
   - mid: `python mesh_sheen.py --lc-min 0.18 --lc-sub 0.21 --lc-max 1.20 \
       --out .../_artifacts/palace_sheen/palace_sheen_mid.msh`
     (the sqrt2 refinement — a pure CLI change, no source edit).
   Also copy the four `sheen_s21_*.json` config JSONs and `check_sparams.py` into
   the WORK dir (the YAML `cd`s there and references paths relative to it).
2. **Solve** on VESSL (`remilab-c0`, gpu-rtx4090, source-built Palace from the
   `microwave-energy` install):
   - `vessl run create -f vessl_palace_sheen_4090.yaml`   (coarse)
   - `vessl run create -f vessl_palace_sheen_mid.yaml`    (mid)
   - each YAML: dry-run -> 11-pt passivity probe + `check_sparams.py --gate` ->
     full sweep -> `check_sparams.py --summary`. Writes
     `postpro/sheen_{full,probe}_{4090,mid}/port-S.csv`.
3. **Fixture** (re-derives the committed JSON from the four CSVs):
   `python ../build_sheen_lpf_palace_referee.py --from-artifacts \
       --vessl-coarse <id> --vessl-mid <id>`
4. **Verdict** (one-liner from the committed fixture, no CSV needed):
   `python ../build_sheen_lpf_palace_referee.py`

### Output location

The two YAMLs pin
`WORK=/root/workspace/bk-workspace/rfx-oblique-rcs/scripts/diagnostics/_artifacts/palace_sheen`
(this worktree's gitignored artifacts tree — visible to the VESSL job because the
whole `personal-workspaces` NFS volume is mounted at `/root/workspace`) and
`cd $WORK` before solving, so the meshes, config JSONs, and `postpro/*/port-S.csv`
all live there at run time. The config JSONs reference the mesh / output paths
**relative to that WORK dir** (`"Mesh": "palace_sheen.msh"`,
`"Output": "postpro/sheen_full_4090"`).

## Failure lessons carried over from the cv06b notch lane

- **VESSL `run:` block is `dash`, not `bash`.** Heredocs / bashisms abort it. The
  shipped YAMLs are dash-safe: `set -eu`, no heredocs; `check_sparams.py` does the
  parsing/gating instead of inline shell.
- **A too-fine mesh OOMs the 24 GB rtx4090.** Both meshes here are sized to the
  proven notch coarse/mid tet budgets; the sqrt2 "mid" mesh is the convergence
  witness that fits.

## Files

| file                          | role                                             |
|-------------------------------|--------------------------------------------------|
| `mesh_sheen.py`               | Gmsh mesh generator (CLI `--lc-*` for coarse/mid) |
| `check_sparams.py`            | passivity `--gate` (exit 3) + first-null `--summary` |
| `sheen_s21_4090.json`         | Palace config, coarse full sweep (4-12 GHz)      |
| `sheen_s21_probe_4090.json`   | Palace config, coarse 11-pt passivity probe      |
| `sheen_s21_mid.json`          | Palace config, mid full sweep (6-9 GHz)          |
| `sheen_s21_probe_mid.json`    | Palace config, mid 11-pt passivity probe         |
| `vessl_palace_sheen_4090.yaml`| VESSL lane, coarse (dry-run -> probe -> sweep)   |
| `vessl_palace_sheen_mid.yaml` | VESSL lane, mid                                  |
