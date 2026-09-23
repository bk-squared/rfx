# B0c recorded cavity data

Vacuum box 24 x 20 x 4 mm; x faces declared PMC, y/z faces PEC, no absorber. Ez differentiated Gaussian at (7,6,2) mm, f0=8.5 GHz, fractional bandwidth 0.8; Ez probe at (17,13,2) mm. The tables contain the realized grids, returned mode records and observed operator calls.

Raw JSON, NPZ and log paths are relative to this directory. Error and Q are the repository Harminv return fields; error is dimensionless. The fixed selection windows are 6.8-8.2 GHz for (0,1) and 9.0-10.7 GHz for (1,1), selecting nearest the declared analytic value within each window. A missing candidate is recorded as no returned mode in that window. Full returned modes, including amplitude, phase and decay, are retained in each JSON.

Harminv uses the mean-subtracted probe record from max(source end, one quarter of the record), f_min=5 GHz, f_max=12 GHz, min_Q=1, max_modes=24, sv_threshold=1e-3 and automatic decimation. Source end is the last sample above 1e-5 of the sampled source peak.

The derived wall distance is a'=1/sqrt((2*f11/c)^2-(1/b)^2), with c=299792458 m/s and b=20 mm. This is labelled derived for every entry, including entries whose trace contains no PMC call.

| Entry | dx mm | Nodes x/y/z | Node spans mm | Pads lo/hi x/y/z | Steps | dt ps | Source/probe indices | Run id | Evidence |
|---|---:|---|---|---|---:|---:|---|---|---|
| run-cpu | 1 | [25, 21, 5] | 24/20/4 | 0/0/0/0/0/0 | 4096 | 1.90657486953 | [7, 6, 2] / [17, 13, 2] | 369367263573 | [witness/run-cpu__1.json](witness/run-cpu__1.json) |
| run-cpu | 0.5 | [49, 41, 9] | 24/20/4 | 0/0/0/0/0/0 | 8192 | 0.953287434766 | [14, 12, 4] / [34, 26, 4] | 369367263573 | [witness/run-cpu__0p5.json](witness/run-cpu__0p5.json) |
| run-cpu | 0.25 | [97, 81, 17] | 24/20/4 | 0/0/0/0/0/0 | 16384 | 0.476643717383 | [28, 24, 8] / [68, 52, 8] | 369367263573 | [witness/run-cpu__0p25.json](witness/run-cpu__0p25.json) |
| forward-cpu | 1 | [25, 21, 5] | 24/20/4 | 0/0/0/0/0/0 | 4096 | 1.90657486953 | [7, 6, 2] / [17, 13, 2] | 369367263574 | [witness/forward-cpu__1.json](witness/forward-cpu__1.json) |
| forward-cpu | 0.5 | [49, 41, 9] | 24/20/4 | 0/0/0/0/0/0 | 8192 | 0.953287434766 | [14, 12, 4] / [34, 26, 4] | 369367263574 | [witness/forward-cpu__0p5.json](witness/forward-cpu__0p5.json) |
| forward-cpu | 0.25 | [97, 81, 17] | 24/20/4 | 0/0/0/0/0/0 | 16384 | 0.476643717383 | [28, 24, 8] / [68, 52, 8] | 369367263574 | [witness/forward-cpu__0p25.json](witness/forward-cpu__0p25.json) |
| sweep-cpu | 1 | [25, 21, 5] | 24/20/4 | 0/0/0/0/0/0 | 4096 | 1.90657486953 | [7, 6, 2] / [17, 13, 2] | 369367263576 | [witness/sweep-cpu__1.json](witness/sweep-cpu__1.json) |
| sweep-cpu | 0.5 | [49, 41, 9] | 24/20/4 | 0/0/0/0/0/0 | 8192 | 0.953287434766 | [14, 12, 4] / [34, 26, 4] | 369367263576 | [witness/sweep-cpu__0p5.json](witness/sweep-cpu__0p5.json) |
| sweep-cpu | 0.25 | [97, 81, 17] | 24/20/4 | 0/0/0/0/0/0 | 16384 | 0.476643717383 | [28, 24, 8] / [68, 52, 8] | 369367263576 | [witness/sweep-cpu__0p25.json](witness/sweep-cpu__0p25.json) |
| nonuniform-cpu | 1 | [25, 21, 5] | 24/20/4 | 0/0/0/0/0/0 | 4096 | 1.90657486953 | [7, 6, 2] / [17, 13, 2] | 369367263577 | [witness/nonuniform-cpu__1.json](witness/nonuniform-cpu__1.json) |
| nonuniform-cpu | 0.5 | [49, 41, 9] | 24/20/4 | 0/0/0/0/0/0 | 8192 | 0.953287434765 | [14, 12, 4] / [34, 26, 4] | 369367263577 | [witness/nonuniform-cpu__0p5.json](witness/nonuniform-cpu__0p5.json) |
| nonuniform-cpu | 0.25 | [97, 81, 17] | 24/20/4 | 0/0/0/0/0/0 | 16384 | 0.476643717383 | [28, 24, 8] / [68, 52, 8] | 369367263577 | [witness/nonuniform-cpu__0p25.json](witness/nonuniform-cpu__0p25.json) |
| run-gpu | 1 | [25, 21, 5] | 24/20/4 | 0/0/0/0/0/0 | 4096 | 1.90657486953 | [7, 6, 2] / [17, 13, 2] | 369367263578 | [witness/run-gpu__1.json](witness/run-gpu__1.json) |
| run-gpu | 0.5 | [49, 41, 9] | 24/20/4 | 0/0/0/0/0/0 | 8192 | 0.953287434766 | [14, 12, 4] / [34, 26, 4] | 369367263578 | [witness/run-gpu__0p5.json](witness/run-gpu__0p5.json) |
| run-gpu | 0.25 | [97, 81, 17] | 24/20/4 | 0/0/0/0/0/0 | 16384 | 0.476643717383 | [28, 24, 8] / [68, 52, 8] | 369367263578 | [witness/run-gpu__0p25.json](witness/run-gpu__0p25.json) |

| Entry / dx mm | Operator calls in trace order | Fast update observed | Preflight stdout | Energy peak / final J; final/peak dB |
|---|---|---|---|---|
| run-cpu / 1 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/run-cpu__1.preflight.stdout.txt](witness/run-cpu__1.preflight.stdout.txt) | 3.29653829789e-21 / 3.0899761932e-21; -0.281029922 dB |
| run-cpu / 0.5 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/run-cpu__0p5.preflight.stdout.txt](witness/run-cpu__0p5.preflight.stdout.txt) | 1.97238122204e-22 / 1.90520043896e-22; -0.150501863 dB |
| run-cpu / 0.25 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/run-cpu__0p25.preflight.stdout.txt](witness/run-cpu__0p25.preflight.stdout.txt) | 1.20722145339e-23 / 1.19252552693e-23; -0.0531926041 dB |
| forward-cpu / 1 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/forward-cpu__1.preflight.stdout.txt](witness/forward-cpu__1.preflight.stdout.txt) | 3.29653769205e-21 / 3.0899761932e-21; -0.281029124 dB |
| forward-cpu / 0.5 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/forward-cpu__0p5.preflight.stdout.txt](witness/forward-cpu__0p5.preflight.stdout.txt) | 1.97238147448e-22 / 1.90520195357e-22; -0.150498966 dB |
| forward-cpu / 0.25 | M(x_hi,x_lo); P(xyz); F(y_hi,y_lo,z_hi,z_lo) | False | [witness/forward-cpu__0p25.preflight.stdout.txt](witness/forward-cpu__0p25.preflight.stdout.txt) | 1.20722090119e-23 / 1.19252505361e-23; -0.0531923413 dB |
| sweep-cpu / 1 | P(xyz) | False | [witness/sweep-cpu__1.preflight.stdout.txt](witness/sweep-cpu__1.preflight.stdout.txt) | 1.05271051799e-21 / 9.74894753947e-22; -0.333512287 dB |
| sweep-cpu / 0.5 | P(xyz) | False | [witness/sweep-cpu__0p5.preflight.stdout.txt](witness/sweep-cpu__0p5.preflight.stdout.txt) | 6.36800834437e-23 / 6.07691993252e-23; -0.203201095 dB |
| sweep-cpu / 0.25 | P(xyz) | False | [witness/sweep-cpu__0p25.preflight.stdout.txt](witness/sweep-cpu__0p25.preflight.stdout.txt) | 3.92133210768e-24 / 3.82600694476e-24; -0.106878714 dB |
| nonuniform-cpu / 1 | M(x_hi,x_lo); P(xyz) | False | [witness/nonuniform-cpu__1.preflight.stdout.txt](witness/nonuniform-cpu__1.preflight.stdout.txt) | 3.29653991348e-21 / 3.08997679904e-21; -0.281031199 dB |
| nonuniform-cpu / 0.5 | M(x_hi,x_lo); P(xyz) | False | [witness/nonuniform-cpu__0p5.preflight.stdout.txt](witness/nonuniform-cpu__0p5.preflight.stdout.txt) | 1.97238109582e-22 / 1.90520043896e-22; -0.150501585 dB |
| nonuniform-cpu / 0.25 | M(x_hi,x_lo); P(xyz) | False | [witness/nonuniform-cpu__0p25.preflight.stdout.txt](witness/nonuniform-cpu__0p25.preflight.stdout.txt) | 1.2072202701e-23 / 1.19252505361e-23; -0.053190071 dB |
| run-gpu / 1 | B(xyz); update_he_fast | True | [witness/run-gpu__1.preflight.stdout.txt](witness/run-gpu__1.preflight.stdout.txt) | NOT MEASURED |
| run-gpu / 0.5 | B(xyz); update_he_fast | True | [witness/run-gpu__0p5.preflight.stdout.txt](witness/run-gpu__0p5.preflight.stdout.txt) | NOT MEASURED |
| run-gpu / 0.25 | B(xyz); update_he_fast | True | [witness/run-gpu__0p25.preflight.stdout.txt](witness/run-gpu__0p25.preflight.stdout.txt) | NOT MEASURED |

P(axes) = apply_pec; F(faces) = apply_pec_faces; M(faces) = apply_pmc_faces; B(axes) = precompute_coeffs with pec_axes. The source locations and callers of each event are in the raw JSON. `operator_measurement.json` records actual zero planes from the same source operators applied to nonzero fields and vacuum coefficients at all three array shapes.

Tangential E zero planes: P/F at index 0 and N-1; B zeros the tangential E ca/cb coefficients at 0 and N-1. Tangential H zero planes: M at index 0 and N-2. At x faces these H samples are x=dx/2 and x=24 mm-dx/2. P and z_hi F additionally zero Ez at the high-z ghost plane; B has no such Ez zero plane.

Instrumentation check: the plain and observed 64-step CPU probe arrays are byte-for-byte equal for run, forward, sweep and constant-dz nonuniform entry points; maximum absolute difference is 0 V/m in each. Evidence: `instrumentation_verification.json`, `smoke/*__plain.npz` and `smoke/*__v2.npz`.

The initial GPU run used JAX_PLATFORMS=cuda. All three arms raised in jax.debug.callback because the CPU backend was unavailable; the original exceptions, traces and provider log remain in `witness/run-gpu*` and `jobs/run-gpu/`.

Conclusion: leader fills.
