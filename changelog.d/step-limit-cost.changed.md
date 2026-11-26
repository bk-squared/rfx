### Changed — a compiled call no longer runs the cell time-step check on the host unless a cell can fail it (#PR)

Admitted JIT calls now test possible cell Courant refusals on the device before
running the unchanged ordered host check. On an RTX 4090, the staged check cost
25–44 ms per call and now costs 0.90–1.10 ms (161³ forward, 97³ gradient). On a 60³
design-loop grid the check adds 0.86 ms to a 9.5 ms call (9.0 %) at 200 steps and
0.75 ms to 36.4 ms (2.1 %) at 800 steps.
On CPU `jit(grad)` at 81³, the first call after a program without the check cost
246 / 388 ms more at 100 / 200 steps; the conditional removes that extra cost.
With a batched predicate, `vmap` still calls the host once per lane.
