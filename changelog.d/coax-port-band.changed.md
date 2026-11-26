### Changed — A coaxial port declares the bottom of its band, and its probe planes are placed from that band (#PR)

- `add_coaxial_port(freq_min=...)` declares the lowest frequency the port measures (default `freq_max/10`); a value outside (0, `freq_max`) is refused.
- The port's first probe plane, plane spacing, source clearance and shortest line now follow from the band `freq_min` to `freq_max` and the line's TE11 and TE21 fields. A shorter line is refused, naming the cells and millimetres to add or a higher `freq_min`; a non-uniform grid is refused.
- `run(s_param_freqs=...)` and `forward(port_s11_freqs=...)` refuse a concrete bin below the port's `freq_min` or above `freq_max`. `run()`'s default band is `linspace(freq_min, freq_max, 50)`: unchanged while `freq_min` keeps its default.
