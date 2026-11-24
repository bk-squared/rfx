### Fixed — plot_time_series and the dashboard draw each probe at its sampling time (#PR)

Plots drew row n at n·dt, but E probes are sampled at (n + 1)·dt and H probes at (n + ½)·dt.
E curves were drawn one step early and H curves half a step early.
`components=` names each column's field; without it every column is drawn as an E probe.
