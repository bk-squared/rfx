### Fixed — shared faces within one dielectric use its bulk permittivity (#PR)

- Touching εr=4 shapes read 4.0 at on- and off-node shared faces with Yee
  and node tensors on uniform and graded grids, including traced geometry.
- Splitting the εr=10.2 cavity layer at z=4.95 or 5.0 mm leaves all Yee
  operands unchanged. At 200 periods, split/unsplit TM and TE errors are
  -0.057075% and -0.137781% against the analytic cavity.
- A 0.1 mm air gap on a 0.5 mm grid reads 4.0 (true-content series value 2.5);
  the measured outer seam cell reads 4.0 versus 3.4 for one Box.
