Coax two-port and coax-to-MSL transition S parameters now use the built
coax line impedance from the solved TEM profile when referring waves to
the registered port impedance. Results report that value in `line_impedances`.
For the SMA line at 0.4 mm, the 50-ohm |S11| envelope changes from -30.88 dB
to -13.70 dB. End resistor targets are unchanged.
