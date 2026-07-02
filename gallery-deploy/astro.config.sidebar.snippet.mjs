// DRAFT — apply in remilab-sites-gitops, NOT auto-deployed by the rfx export.
//
// ADD these entries to the rfx sidebar in
//   deploy/obsidian-stack/astro-starlight-presets/public/astro.config.mjs
//
// The rfx sidebar is the group labeled "rfx — FDTD Simulator" whose `items`
// array holds bare slug strings ("rfx", "rfx/guide", ...) and labeled
// subgroups. Add a new subgroup like the one below (e.g. after the
// "Analysis & Validation" subgroup). Each slug must resolve to a seed-page
// file or `deploy/obsidian-stack/scripts/check_rfx_sidebar_links.py` fails.
//
// After `export_public_docs_to_gitops.py` refreshes seed-pages/rfx/, the
// gallery .mdx pages land at:
//   seed-pages/rfx/gallery/index.mdx            -> slug "rfx/gallery"
//   seed-pages/rfx/gallery/multilayer_fresnel.mdx -> slug "rfx/gallery/multilayer_fresnel"
//   seed-pages/rfx/gallery/waveguide_wr90.mdx   -> slug "rfx/gallery/waveguide_wr90"
//   seed-pages/rfx/gallery/patch_antenna.mdx    -> slug "rfx/gallery/patch_antenna"

{
  label: "Gallery",
  items: [
    "rfx/gallery",
    "rfx/gallery/multilayer_fresnel",
    "rfx/gallery/waveguide_wr90",
    "rfx/gallery/patch_antenna",
  ],
},

// Verify with:
//   python3 deploy/obsidian-stack/scripts/check_rfx_sidebar_links.py
