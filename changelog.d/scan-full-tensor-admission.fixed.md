### Fixed — Direct S-matrix scans and uniform runs now refuse unsupported full-tensor smoothing by name (#PR)

- The lumped/wire S-matrix scan called directly, and `rfx.runners.run_uniform` called directly, with `boundary="upml"` and `subpixel_smoothing="full_tensor"` ran the full tensor's diagonal without its off-diagonal part (S moved up to 2.9e-3 on a synthetic eps_r 4 sphere); they now refuse by name, as `run()` and `forward()` do.
- A traced `eps_r` under `full_tensor` on the direct scan is refused by name too.
- `run()`, `forward()` and every other admitted case are unchanged.
