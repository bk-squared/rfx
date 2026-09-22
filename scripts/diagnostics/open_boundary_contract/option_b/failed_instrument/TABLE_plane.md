Normal-incidence plane-wave box, 0.2–6 GHz. Cells contain recorded values for the requested configurations.

Maximum clean-reference |R| (dB).

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -5.895 | -5.905 | -5.924 | -5.994 | -6.207 | -1.342 | 11.36 |
| 8 | -13.21 | -13.22 | -13.23 | -13.26 | -13.35 | -13.68 | -1.964 |
| 16 | -24.06 | -24.07 | -24.08 | -24.15 | -24.33 | -24.96 | -26.24 |

End energy / post-source peak (dB), maximum across recorded drives. TS = truncation-suspect under the specified −40 dB witness. No post-source samples means the source table remains above its recording threshold at the last sample.

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -129.8 | -110.9 | -84.63 | -80.21 | -64.61 | -46.17 | -34.18; TS |
| 8 | -127.5 | -123.9 | -107.4 | -84.01 | -68.48 | -58.53 | -48.3 |
| 16 | -118.7 | -112.9 | -105.3 | -103.6 | -99.04 | -62.04 | -42.69 |

Worst recorded probe settling (dB): last 5% maximum / whole-record maximum; worst across probes and drives.

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -137.1 | -131.3 | -96.17 | -88.76 | -85.61 | -56.83 | -54.67 |
| 8 | -156 | -141.8 | -130.9 | -106.6 | -86.95 | -72.38 | -57.75 |
| 16 | -136.6 | -142.5 | -142 | -136.3 | -136.3 | -99.15 | -72.71 |

Reflection spectra: [per-arm CSV directory](results/plane/). Interpolated threshold crossings and sample counts: [CROSSINGS_plane.csv](CROSSINGS_plane.csv). An empty crossing list means no crossing within the sampled band; it is not a measured cutoff frequency. The comparison formula is 0.05 s / (2π ε0).

Maximum absolute difference of the two mirror-probe complex reflection records (dimensionless).

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | 1.008 | 1.007 | 1.006 | 0.9998 | 0.9825 | 0.9182 | 0.8962 |
| 8 | 0.4721 | 0.4719 | 0.4715 | 0.4699 | 0.4654 | 0.4497 | 0.4033 |
| 16 | 0.1415 | 0.1413 | 0.1411 | 0.1401 | 0.1374 | 0.1282 | 0.1056 |

All per-drive energies and source-end indices: [WITNESSES_2.csv](WITNESSES_2.csv).

Conclusion: leader fills after the sweep.
