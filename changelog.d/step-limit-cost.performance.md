### Performance — Cell time-step checks (#PR)

Admitted JIT calls now test possible cell Courant refusals on the device before
running the unchanged ordered host check. On an RTX 4090, the staged check cost
25–44 ms per call and now costs about 1.1–1.4 ms. On a small GPU design-loop scene,
the added cost is SMALL_SCENE_MS ms (SMALL_SCENE_PCT %).
On CPU `jit(grad)` at 81³, the first call after a program without the check cost
246 / 388 ms more at 100 / 200 steps; the conditional removes that extra cost.
With a batched predicate, `vmap` still calls the host once per lane.
