### Fixed — wire reference-plane S uses one wave definition (#PR)

With `reference_plane_cells` on every wire port, `run()` returns S of the line between the planes in one wave definition: S = B·A^-1 over both drive passes.
On the 16 mm thru with N=10 planes, |S11| is 0.0005–0.0025 (was 0.026–0.075 from the per-cell diagonal at N=3; 0.036–0.297 without the option, including feed transitions).
S21 moves at second order in the far termination's reflection, by at most 3.8e-4 in complex value; a partial opt-in is refused.
