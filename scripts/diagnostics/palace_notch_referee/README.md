# Palace FEM referee — cv06b MSL open-stub notch (WP 1-B)

Run tooling that produced the **independent-method referee** for the committed
cv06b rfx-vs-openEMS notch split.

## What / why

cv06b's committed cross-check (`tests/fixtures/msl_notch_e4/msl_stub_notch_{rfx,openems}_dx50.json`)
locks a ~5.8% notch-frequency split: **rfx 3.6273 GHz, openEMS 3.4286 GHz,**
fringing-free analytic 3.69 GHz. Both refs are staircased FDTD, so neither
resolves the open-end fringing exactly — the old narrative *guessed* openEMS was
closer to the truth. That is a plausibility, not evidence.

**Palace** is a frequency-domain FEM solver on a **conformal tetrahedral mesh**
(no staircase), so it captures the open-end fringing exactly and can referee.
Ran on the SAME matched geometry at two mesh densities:

| mesh   | LC (mm) | tets    | sweep            | VESSL run    |
|--------|---------|---------|------------------|--------------|
| coarse | 0.12    | 143,812 | 101 pt, 2-7 GHz  | 369367246161 |
| mid    | 0.085   | 376,802 | 33 pt, 3.2-4 GHz | 369367246168 |

**Verdict:** Palace lands at ~3.631 GHz (parabolic notch) at BOTH densities
(convergence shift only -0.006 GHz / 0.16%) => **+0.1% from rfx, ~1.6% from
analytic, ~5.9% from openEMS. Palace SIDES WITH rfx.** openEMS's dx=50 µm
staircase notch is the OUTLIER, not the fringing truth.

The verdict is committed as evidence — the raw Palace `port-S.csv` arrays live in
`tests/fixtures/msl_notch_e4/msl_stub_notch_palace_referee.json` (dB -> LINEAR),
re-derived by `scripts/diagnostics/build_msl_notch_palace_referee.py`, gated by
`tests/test_msl_notch_palace_referee_gates.py`. Those survive a clean checkout;
this directory is the *provenance* that generated them.

## Geometry lock

`mesh_notch.py` is hard-locked to the matched cross-solver frame (mm), identical
to the two FDTD fixtures:

    substrate  eps_r = 3.66   h = 0.254 mm   LOSSLESS (LossTan = 0)
    trace      W = 0.6 mm, runs the full x-extent [0, 7]
    open stub  W = 0.6 mm, L = 12.0 mm, +y from the trace edge (x centred 3.5)
    ports      two 50-ohm lumped sheets (ground->strip) at x = 1.0 and x = 6.0
               => port-to-port line = 5 mm, 1 mm margin beyond each port
    far box    first-order absorbing on all non-ground outer faces

This is the sibling fixtures' `l_line=5mm, margin=1mm, stub=12mm, eps_r=3.66,
h_sub=254um, W=600um` frame, realised as conformal tets with the line run through
the port planes (both FDTD frames run the line through the absorber).

## Run order

1. **Mesh** (Gmsh; writes msh 2.2 for MFEM). `.msh` files are **regenerable and
   NOT committed** — regenerate before solving:
   - coarse: `python mesh_notch.py --out palace_notch.msh`
   - mid: edit `LC_MIN`, `LC_SUB` -> `0.085` and `LC_MAX` -> `0.7` (the sqrt2
     refinement — a 3-constant sed of `mesh_notch.py`), then
     `python mesh_notch.py --out palace_notch_mid.msh`.
2. **Solve** on VESSL (`remilab-c0`, gpu-rtx4090, source-built Palace from the
   `microwave-energy` install):
   - `vessl run create -f vessl_palace_notch_4090.yaml`   (coarse)
   - `vessl run create -f vessl_palace_notch_mid.yaml`    (mid)
   - each YAML: dry-run -> 11-pt passivity probe + `check_sparams.py --gate` ->
     full sweep -> `check_sparams.py --summary`. Writes
     `postpro/notch_{full,probe}_{4090,mid}/port-S.csv`.
3. **Fixture** (re-derives the committed JSON from the four CSVs):
   `python ../build_msl_notch_palace_referee.py --from-artifacts`
4. **Verdict** (one-liner from the committed fixture, no CSV needed):
   `python ../build_msl_notch_palace_referee.py`

### Output location (documented, not restructured)

The two YAMLs pin `WORK=.../scripts/diagnostics/_artifacts/palace_notch` (the
gitignored artifacts tree) and `cd $WORK` before solving, so the meshes, config
JSONs, and `postpro/*/port-S.csv` all live there at run time — that is where the
committed fixture was built from. The config JSONs here reference the mesh /
output paths **relative to that WORK dir** (`"Mesh": "palace_notch.msh"`,
`"Output": "postpro/notch_full_4090"`). To re-run, copy these config JSONs (and
the regenerated `.msh`) into the WORK dir, or edit the YAML `WORK=` / config paths
to point at wherever you place them. `mesh_notch.py`'s default `--out` writes the
mesh next to the script; pass `--out` to target the WORK dir.

## Failure lessons (both cost a relaunch)

- **VESSL `run:` block is `dash`, not `bash`.** Heredocs / bashisms abort it
  (run 369367246150 died on sh-syntax). The shipped YAMLs are dash-safe:
  `set -eu`, no heredocs, `check_sparams.py` does the parsing/gating instead of
  inline shell.
- **A fine LC/2 mesh OOMs the 24 GB rtx4090** (run 369367246167). Use the
  **sqrt2 "mid" mesh (LC 0.085)** as the convergence witness — it fits and the
  notch has already converged (shift 0.16%), so the finer mesh is unnecessary.

## Files

| file                          | role                                             |
|-------------------------------|--------------------------------------------------|
| `mesh_notch.py`               | Gmsh coarse-mesh generator (edit LC for mid)     |
| `check_sparams.py`            | passivity `--gate` (exit 3) + notch `--summary`  |
| `notch_s11_4090.json`         | Palace config, coarse full sweep (2-7 GHz)       |
| `notch_s11_probe_4090.json`   | Palace config, coarse 11-pt passivity probe      |
| `notch_s11_mid.json`          | Palace config, mid full sweep (3.2-4 GHz)        |
| `notch_s11_probe_mid.json`    | Palace config, mid 11-pt passivity probe         |
| `vessl_palace_notch_4090.yaml`| VESSL lane, coarse (dry-run -> probe -> sweep)   |
| `vessl_palace_notch_mid.yaml` | VESSL lane, mid                                  |
