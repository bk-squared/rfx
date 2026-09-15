# Replacement text for `docs/public/guide/benchmarks.mdx`, lines 66-67

Owner of this file: the X-D migration agent (cv18, cv19). Owner of
`docs/public/**`: the docs agent. Apply there.

Both rows quote numbers the #931 regeneration moves. Take the digits from the
regenerated fixtures at ingest, not from here; what follows is the wording.

## Line 66 — cv18 (`18_wr90_iris_modematch`)

Whatever the row says about the iris being "exactly 2 coarse / 4 fine cells"
is still true of the DRAWING and is now also true of the REALIZATION. If the
row quotes a fine gate of 0.04 or an envelope of 0.0232, both are re-derived
by the regeneration and must be refreshed together with the fixture.

Add, in the row's note or the surrounding prose:

> Regenerated under the #931 lattice ownership contract. Before it, this case
> fed its mode-matching oracle the drawn iris thickness while the lattice
> realized one cell less; the numbers below are the first ones measured on a
> geometry that matches the oracle's input.

## Line 67 — cv19 (`19_wr90_iris_filter_aghanim`)

Any wording about compensated cell counts, an `(L_c + 1)` cavity leg or a
half-cell iris-thickness convention must go. Replace with:

> Regenerated under the #931 lattice ownership contract. The drawn cell counts
> are now the plain roundings — the case previously carried a +1 / -1
> compensation that existed only to cancel a realization in which a body's far
> face was never a wall — and the built filter is unchanged by the deletion
> (same realized wall planes, same cavities, same apertures).

If the row quotes the f0 gate (19 MHz) or the measured `d_f0` (+12.08 MHz),
refresh both from the regenerated fixture.
