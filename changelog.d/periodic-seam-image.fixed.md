### Fixed — Conformal PEC image selection at periodic seams (#PR)

Conformal PEC bodies within one cell below a periodic seam now use the nearest
periodic image for cut-cell lengths and areas, including on graded cell tables.
The seam edge in the recorded sphere case now has free length 0.70 instead of
0.97. The defect changed reflected power by up to 2.4e-3 (about 2% of R) at λ/21
when the same periodic structure was translated by whole cells.
