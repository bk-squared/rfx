### Fixed — the coax two-port and coax-to-microstrip transition refer their waves to the line the mesh builds, not the drawn radii (#PR)

- The 50-ohm S parameters now use the built coax line impedance, taken from the solved TEM profile on the stamped mask, when referring waves to the registered port impedance. `line_impedances` reports that value.
- For the SMA line (pin 0.635 mm, shell 2.055 mm, eps_r 2.1) at 0.4 mm cells, the built line is 40.55 ohm against the drawn 48.59 ohm. The 50-ohm |S11| envelope moves from -30.88 dB to -13.70 dB.
- End resistor targets are unchanged.
