Oblique Bloch TF/SF source construction now warns when the pulse amplitude at
the cutoff exceeds −60 dB relative to f0, including the default waveform.
The warning reports the angle, bandwidth, cutoff frequency and level, and
advises narrowing the bandwidth or tapering the probe time-series tail before
a DFT to reduce record-length dependence from the non-decaying cutoff component.
