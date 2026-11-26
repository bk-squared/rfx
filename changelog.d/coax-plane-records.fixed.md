### Fixed — The coax calculators no longer keep their probe planes at every time step (#PR)

- `compute_coaxial_line_reflection`, `compute_coaxial_two_port` and `compute_coax_msl_transition` kept every coax probe plane in full at every step, only to form one voltage per plane for the settling witness. GPU memory grew by about 195 B per (transverse cell × step): 12.3 GB at 19 000 steps and 24.1 GB at 38 000 steps on 57 × 57 transverse cells.
- Each coax plane's voltage is now sampled inside the step, and only that voltage is kept. S11 and S21 are unchanged bit for bit.
- The transition's microstrip planes still keep their records (see Known limitations).
- The coax source waveforms are still stored for the whole record: device memory still grows by 2.9 kB per step on the coax one-port above (see Known limitations).
