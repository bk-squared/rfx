### Changed — Reject mixed waveguide drives on every run and forward lane (#PR)

Graded `run()` and uniform/graded `forward()` now refuse a waveguide port together with lumped or wire ports or TFSF before building fields, using the same message as uniform `run()`:
"Waveguide ports are not supported together with lumped ports or TFSF".
