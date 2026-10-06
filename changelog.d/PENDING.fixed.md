Line ports now refuse open stubs behind the port when any odd quarter-wave resonance is inside or within a factor of 1.5 of the requested band, including with preflight bypassed (#1512).
Outside that interval, preflight reports the stub length, estimated permittivity and resonances; start the strip at the port plane to remove it.
Measured in #1512: a 2 mm overhang gave |S11| −2.8 dB / |S21| −4.6 dB; at 0 mm, |S11| was < −36 dB.
