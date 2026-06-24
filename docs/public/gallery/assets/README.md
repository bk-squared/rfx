# Gallery assets

This directory holds the **precomputed** artifacts the public gallery pages
serve. The public site never runs a solver; it serves these static files.

## How these are generated

```bash
# full validated runs (FDTD, GPU-heavy) — produced on VESSL
python scripts/precompute_gallery_artifacts.py --case all

# fast CPU smoke of the pipeline (placeholder artifacts, NOT validated)
python scripts/precompute_gallery_artifacts.py --case multilayer_fresnel --quick
```

Each `<case_id>/` bundle contains:

| File | Committed? | Notes |
|---|---|---|
| `manifest.json` | yes | provenance + validation tier/metric + asset list (with `served_url`, sha256) |
| `sparams.png` | yes | S-parameter magnitude plot (small thumbnail) |
| `smith.png` | yes | Smith chart of S11 (small thumbnail) |
| `sparams.s{N}p` | yes | Touchstone S-parameters |
| `sparams.json` | yes | machine-readable S-params for the interactive viewer (gitops) |
| `fields.mp4` / `*.npz` | **no** | large field media — git-ignored, synced separately (see `gallery-deploy/DEPLOYMENT.md`) |

## Validated vs smoke

A bundle whose `manifest.json` has `quick_smoke: true` is a **pipeline
placeholder**, not a validated result — `validation.passed` is `null` for those.
Replace them with validated artifacts produced on VESSL before treating any
gallery page as evidence.

## Served URL

Assets are served at `/rfx/gallery/assets/<case_id>/<file>` after the gitops
`build-astro.sh` static-sync step copies this tree into the Astro
`public/rfx/gallery/` directory (draft in `gallery-deploy/build-astro.snippet.sh`).
Gallery `.mdx` pages reference assets by that **absolute** URL.

## Large-binary policy

Large media (field animations, volume dumps) are committed to the **gitops**
repo and served from there; there is currently no NFS/LFS path wired for this
gallery. Keep committed samples small.
