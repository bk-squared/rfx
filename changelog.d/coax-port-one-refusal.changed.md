### Changed — `run()` and `forward()` refuse a coaxial port from the one lane check (#PR)

- A `Simulation` with `add_coaxial_port()` is still refused by `run()` and `forward()` before the first time step. The refusal now comes from the lane check both entries share, so it names the lane ("a coaxial port (add_coaxial_port) is not carried by the uniform run() lane") instead of "add_coaxial_port() is not wired into Simulation.run()". The advice on `compute_coaxial_line_reflection`, `compute_coaxial_two_port` and `compute_s_matrix(method=...)` stays.
- An explicit S-parameter request (`run(compute_s_params=True)`, `forward(port_s11_freqs=...)`) is checked first: with a coaxial port and no `add_port()` port it now raises `ValueError` instead of `NotImplementedError`; with both it raises `NotImplementedError` as before. Both messages name the coaxial calculators.
- `solver='adi'` keeps its own `ValueError`. No number changes.
