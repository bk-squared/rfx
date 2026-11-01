### Changed — Reject mixed waveguide drives on every run and forward lane (#PR)

Uniform `forward()` and graded `run()`/`forward()` now refuse a waveguide port beside a lumped, wire or passive port, a point source (`add_source`) or a TFSF source before building fields, using the same message as uniform `run()`:
"Waveguide ports are not supported together with lumped ports or TFSF".
