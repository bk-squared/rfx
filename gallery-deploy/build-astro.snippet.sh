# DRAFT — apply in remilab-sites-gitops, NOT auto-deployed by the rfx export.
#
# ADD this call to deploy/obsidian-stack/scripts/build-astro.sh, right after the
# existing rfx-api-generated static-sync block (currently around lines 307-311),
# inside the same `if [[ "${ASTRO_PRESET}" == "public" ]]; then ... fi` block.
#
# It mirrors the existing `sync_generated_static_subtree` call for
# `rfx/api/generated`: it copies the committed gallery assets out of the
# seed-pages tree into the Astro public/ tree so they are served as static
# files (the function skips .md/.mdx and index.html, copying binaries only).
#
# After this runs, an asset committed at
#   .../public/seed-pages/rfx/gallery/assets/<case>/sparams.png
# is served at
#   /rfx/gallery/assets/<case>/sparams.png
# which is exactly the absolute URL the rfx gallery .mdx pages reference.

  sync_generated_static_subtree \
    "${STACK_PRESET_ROOT}/public/seed-pages/rfx/gallery/assets" \
    "${ASTRO_WORKSPACE}/public/rfx/gallery/assets" \
    "rfx-gallery-assets"

# Notes:
#   * Place this immediately below the existing rfx-api-generated block:
#
#       sync_generated_static_subtree \
#         "${STACK_PRESET_ROOT}/public/seed-pages/rfx/api/generated" \
#         "${ASTRO_WORKSPACE}/public/rfx/api/generated" \
#         "rfx-api-generated"
#       # <-- add the rfx-gallery-assets block here, before the closing `fi`
#
#   * Large media (fields.mp4, *.npz) are committed to the gitops repo under the
#     same seed-pages/rfx/gallery/assets/<case>/ path and ride this same sync.
#     There is no NFS/LFS path wired; keep the committed binaries reasonable.
