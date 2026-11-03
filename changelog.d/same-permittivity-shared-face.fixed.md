### Fixed — shared faces within one dielectric use its bulk permittivity (#PR)

- Touching εr=4 shapes read 4.0 at on- and off-node shared faces with Yee
  and node tensors on uniform and graded grids, including traced geometry.
- An εr=4 Box spanning one periodic unit cell reads 4.0 at its shared face.
- Splitting the εr=10.2 cavity layer at z=4.95 or 5.0 mm leaves all Yee
  operands unchanged. At 200 periods, split/unsplit TM and TE errors are
  -0.057075% and -0.137781% against the analytic cavity.
- Gaps of at least half a cell are seen: a 0.45 mm gap on 0.5 mm cells reads
  Ez=1.047120, its one-Box value; a sub-half-cell 0.1 mm gap reads 4.0 (series 2.5).
- The measured outer seam cell reads 4.0 versus 3.4 for one Box.
