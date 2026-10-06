Line ports now refuse realized open stubs when any odd quarter-wave resonance is inside or within a factor of 1.5 of the requested band, including with preflight bypassed (#1512).
Findings quote realized and declared overhang, estimated permittivity and resonances; zero realized overhang is silent.
The message prints the port’s realized grid-node coordinate: start the signal strip at that coordinate (the port’s grid node), so it covers the port node and nothing behind it.
Measured in #1512: a 2 mm overhang gave |S11| −2.8 dB / |S21| −4.6 dB; at 0 mm, |S11| was < −36 dB.
