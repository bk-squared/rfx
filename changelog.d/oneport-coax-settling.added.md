### Added — Coax one-port settling witness (#PR)

Coax one-port reflection results now include `settling_db` and `settling_witness`
for short, open, and matched loads, with the same −40 dB check as the two-port.
Differentiable `eps_scale` runs return `nan` with an undetermined witness.
