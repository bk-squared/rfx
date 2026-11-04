### Changed — uniform multi-device runs step through the one-device time step on each slab (#PR)

- `run(devices=...)` and `forward(distributed=True)` on a uniform grid now call the same time step as a one-device run,
  wrapped per x slab, instead of a separate multi-device loop. Fields, probes, port waves, S-parameters, flux and far
  field are bit-identical to the previous multi-device results. Closed-box TF/SF keeps the previous loop for now.
