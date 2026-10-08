Non-Box DC thin conductors now use the same admission rule on uniform and graded meshes.
By default, films with unknown footprint area or more than 1% area-equivalent length error are refused.
Use a finer in-plane mesh, a Box, or `snap="declared"` to accept an area finding explicitly.
Empty, multi-layer and ambiguous-normal shapes are refused on either snap setting.
Admitted off-node films occupy one layer, matching a Box sheet at the same coordinate.
Realized geometry reports the declared and occupied footprint areas of DC conductors.
