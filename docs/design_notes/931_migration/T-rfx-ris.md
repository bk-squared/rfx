# Handoff from group T (tests) — `rfx/ris.py`

`rfx/ris.py` is in no group's ownership (the critic pass flagged it as
uncovered). Group T owns `tests/unit/misc/test_ris.py`, which reaches it, so
the measurement is recorded here for whoever takes the file.

State under the contract, measured 2026-09-07 on the `test_ris_build_sim` cell
(15 mm cell, `substrate_thickness = 1.5 mm`, `freq_range = (4e9, 8e9)`, which
auto-chooses `dx = 1.8737 mm`):

* `RISUnitCell._build_sim` draws its ground as `Box((0,0,0),(Lx,Ly,0))` and
  users draw reflectarray patches as zero-thickness Boxes. Both are already
  SHEET declarations under §1.5 — **no code change is needed for the
  classification**, and `pec_mask` comes back `None` as it should.
* **Gap 1 — the patch does not land on the laminate.** The patch is declared at
  `z = h_sub = 1.5 mm`, which is 0.80 of a cell above the substrate floor. Its
  sheet realizes on the nearest node, `z = 1.874 mm` — above the substrate top,
  not on it. `_build_sim` chooses `dx` from `freq_range` alone and does not
  preserve the laminate faces. Remedy: give `_build_sim` the `preserve_regions`
  treatment so `h_sub` is an exact number of cells, the same fix the patch
  examples take.
* **Gap 2 — the ground realizes buried.** With `include_cpml_pad_extension`
  (the default), the substrate is extended into the z pad, so the node plane
  the ground sheet lands on has laminate on BOTH sides. Assembly now says so:

      PEC sheet pec realizes on node plane 6 of axis z, and the cells on BOTH
      sides of that plane carry the same dielectric (eps_r = 34.54) over 81 of
      its footprint nodes. ... Nothing is re-sampled.

  Nothing is silently corrected — that is the contract working — but the
  realized cavity carries half a cell of laminate below the ground until the
  stack-up is drawn to the sheet plane, or the ground is placed on a plane the
  pad extension does not fill.

`tests/unit/misc/test_ris.py` is skipped module-wide (RIS deprecated pending a
Floquet redesign), so neither gap is exercised today. Both are written into
that module's docstring so the redesign inherits them.
