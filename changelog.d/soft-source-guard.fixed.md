### Fixed — Admit soft sources beside closed-box plane waves (#PR)

Closed-box TF/SF plane waves now accept `add_source` in either declaration
order. Fields from `run(compute_s_params=False)` and `forward()` agree.
Lumped and wire ports still refuse the combination before changing the model,
with one message explaining that ports load cells and record a port response.
Waveguide refusals now explicitly name soft sources, which remain unsupported
beside a waveguide drive.
