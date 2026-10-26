### Changed — a uniform-mesh run with `conformal_pec=True` finds the edges inside a conductor once, before the time loop (#PR)

- The step used to compare three float32 weight grids with zero on every step to find the edges the conformal PEC zeroes. It now reads three boolean masks formed once before the loop; the zeroed edges are the same.
- On one RTX 3090 at 100 M cells a step on a `conformal_pec=True` board costs 18.88 ms instead of 19.73 ms. Boards without `conformal_pec` are unchanged.
- Fields, S-parameters and every-step peaks are bit-identical on the GPU (a microstrip board with a PEC block, a sphere in an open domain and a short-circuited WR-90 guide; 1000, 2000 and 4000 steps); gradients are bit-identical or move by one float32 unit of their peak.
- `rfx.stepping.conductors.apply_conformal_pec` takes boolean masks and refuses float weights; `rfx.geometry.conformal.apply_conformal_pec` keeps the weight form.
