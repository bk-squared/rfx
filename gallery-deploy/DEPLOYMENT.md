# DRAFT — apply in remilab-sites-gitops, NOT auto-deployed by the rfx export

This is the r02 runbook for shipping the precomputed rfx gallery to
`https://remilab.ai/rfx/gallery/`. The rfx side (pages + precompute pipeline +
asset bundles) lands in the rfx repo; everything in this directory is a draft
the professor applies in the **gitops** repo
(`/root/workspace/byungkwan-workspace/infra/remilab-sites-gitops`).

## Two flags before you start

1. **The export script's default gitops path is broken from this checkout.**
   You MUST pass `--gitops-root` explicitly:

   ```bash
   python scripts/export_public_docs_to_gitops.py \
     --gitops-root /root/workspace/byungkwan-workspace/infra/remilab-sites-gitops
   ```

2. **Do NOT use `gallery:` (or `cssclasses:`) frontmatter keys.** They trigger
   an unrelated internal-wiki Notion card-grid plugin and will mis-render the
   page. The gallery pages use only standard frontmatter (`title`,
   `description`, `sidebar.order`) and built-in Starlight components.

## Large-binary policy

Field animations and volume dumps are **committed into the gitops repo** under
`seed-pages/rfx/gallery/assets/<case>/` and served via the `build-astro.sh`
static-sync step (below). There is **no NFS/LFS path wired** for this gallery —
keep committed binaries small; do not commit multi-hundred-MB media.

---

## Step 1 — rfx repo: author + push the pages (CI manifest gate passes)

In the rfx repo (branch `feat/gallery-mvp`):

- Gallery pages live at `docs/public/gallery/{index,multilayer_fresnel,waveguide_wr90,patch_antenna}.mdx`.
- They are NOT added to `docs/public/site_map.json` (its CI gate
  `scripts/check_public_docs_manifest.py` only allows `rfx/guide*` slugs). They
  follow the `examples/`/`validation/` precedent: files exist + linked from
  `docs/public/index.mdx`.
- Confirm the gate still passes:

  ```bash
  python scripts/check_public_docs_manifest.py   # must print "site_map OK"
  ```

- Produce validated artifacts on VESSL (FDTD is GPU-heavy):

  ```bash
  python scripts/precompute_gallery_artifacts.py --case all
  # commit the small bundle files under docs/public/gallery/assets/<case>/
  ```

  (The MVP ships `--quick` smoke placeholders with `quick_smoke: true` /
  `validation.passed: null`; replace them with validated runs before citing.)

- Commit and push the rfx branch; open/merge the PR. CI manifest gate is green.

## Step 2 — gitops repo: component + sidebar + build-astro + refresh

In `remilab-sites-gitops`:

1. **Add the interactive viewer component** (optional, for interactivity):
   copy `RfxGalleryViewer.astro` to
   `deploy/obsidian-stack/astro-starlight-presets/public/components/RfxGalleryViewer.astro`.
   (The MVP pages do not import it; this only matters if you later add an
   interactive viewer to a seed page.)

2. **Add the sidebar entries** from `astro.config.sidebar.snippet.mjs` into the
   "rfx — FDTD Simulator" group in
   `deploy/obsidian-stack/astro-starlight-presets/public/astro.config.mjs`.

3. **Add the static-sync step** from `build-astro.snippet.sh` into
   `deploy/obsidian-stack/scripts/build-astro.sh`, right after the existing
   `rfx-api-generated` `sync_generated_static_subtree` call (~lines 307-311).

4. **Refresh the seed-pages from rfx** (note the `--gitops-root` flag from
   Flag 1):

   ```bash
   python scripts/export_public_docs_to_gitops.py \
     --gitops-root /root/workspace/byungkwan-workspace/infra/remilab-sites-gitops
   ```

   This copies `docs/public/gallery/` into
   `seed-pages/rfx/gallery/` (pages + the committed `assets/` tree).

5. **Verify the sidebar links resolve:**

   ```bash
   python3 deploy/obsidian-stack/scripts/check_rfx_sidebar_links.py
   # must print "sidebar OK: N RFX slugs resolve"
   ```

6. Commit and push the gitops repo.

## Step 3 — r02: pull + recreate the public Starlight container

On r02 (clean checkout):

```bash
git pull --ff-only
docker compose --env-file deploy/obsidian-stack/.env \
  -f deploy/obsidian-stack/docker-compose.yml \
  up -d --force-recreate starlight-public
# or: deploy/obsidian-stack/public_wiki.sh restart
```

## Step 4 — healthcheck + record SHA

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://remilab.ai/rfx/gallery/   # expect 200
curl -s -o /dev/null -w "%{http_code}\n" \
  https://remilab.ai/rfx/gallery/assets/multilayer_fresnel/sparams.png      # expect 200
git -C /root/workspace/byungkwan-workspace/infra/remilab-sites-gitops rev-parse HEAD
```

Record the deployed gitops SHA next to the healthcheck result.

---

## What is scaffold-only vs production-ready

| Item | State |
|---|---|
| Precompute pipeline + manifest + registry | production-ready (tested) |
| Gallery `.mdx` pages (built-in components, absolute URLs) | production-ready |
| Shipped asset bundles | **scaffold** — `--quick` smoke placeholders; need VESSL validated runs |
| `RfxGalleryViewer.astro` interactive viewer | **scaffold** — draft, must land in gitops; MVP pages don't import it |
| `build-astro.sh` static-sync step | **scaffold** — must be applied in gitops |
| Sidebar entries | **scaffold** — must be applied in gitops `astro.config.mjs` |
