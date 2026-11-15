### Fixed — A closed-box plane wave beside a soft source: run() refused what forward() ran, and declaration order decided admission (#PR)

- `run()`, `forward()` and `add_tfsf_source` now admit `add_source` beside a closed-box
  TF/SF plane wave in either declaration order; the source-first order, refused at
  declaration before, now also runs on `run(devices=...)`. On one device,
  `run(compute_s_params=False)` and `forward()` return the same fields bit for bit.
- A lumped or wire port (`add_port`) beside a plane wave stays refused in either order,
  with one message, before the model changes.
- An explicit S request no impedance port can answer (`run(compute_s_params=True)`,
  `forward(port_s11_freqs=...)`) is refused by one check on both entries, with the same
  text apart from the entry name.
- Waveguide refusals now name soft sources, which stay refused beside a waveguide port.
