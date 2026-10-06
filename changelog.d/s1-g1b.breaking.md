Soft sources on realized PEC edges now raise before stepping on uniform,
graded, and multi-device paths, even with `skip_preflight=True`. This includes
volume, sheet, and wire conductors and internal H sources whose entire curl
loop is PEC. Move the source off the conductor or use a port. Probes remain
advisory, and port-generated drives retain their edge-clearing behavior.
