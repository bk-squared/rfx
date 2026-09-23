cv06b: 30 mm line; 12 mm stub; 63.5 µm cells; 7 GHz maximum excitation frequency; 2 drives; 100 frequency bins; 20 periods.
cv20: 10 mm line; 50 µm cells; 5 GHz maximum excitation frequency; 2 drives; 30 frequency bins; 12 periods.

| fixture | baseline/continued run ID | inset1 run ID | inset1 launches | inset1 solve calls |
| --- | --- | --- | --- | --- |
| cv06b | 369367262634 | 369367262640 | 1 | 2 |
| cv20 | 369367262635 | 369367262641 | 1 | 2 |

Commands: [COMMANDS.md](COMMANDS.md). GPU commands: [vessl_cv06b.yaml](vessl_cv06b.yaml), [vessl_cv20.yaml](vessl_cv20.yaml).

A1: trace-edge occupancy = Ex PEC OR Ey PEC; outermost occupancy 0; occupancy 1 in each remaining x absorber cell.

| fixture | initial A1 (0/1) | final A1 (0/1) | dry read-backs | dry time steps | final x lo (µm) | final x hi (µm) | lo adjustment (cells) | hi adjustment (cells) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv06b | 0 | 1 | 3 | 0 | -444.5 | 34508 | 0 | 1 |
| cv20 | 1 | 1 | 1 | 0 | -350 | 14350 | 0 | 0 |

| fixture | z index | position | x index | Ex PEC | Ey PEC |
| --- | --- | --- | --- | --- | --- |
| cv06b | 4 | lo outer | 0 | 0 | 0 |
| cv06b | 4 | lo adjacent | 1 | 1 | 1 |
| cv06b | 4 | hi adjacent | 551 | 0 | 1 |
| cv06b | 4 | hi outer | 552 | 0 | 0 |
| cv20 | 5 | lo outer | 0 | 0 | 0 |
| cv20 | 5 | lo adjacent | 1 | 1 | 1 |
| cv20 | 5 | hi adjacent | 295 | 0 | 1 |
| cv20 | 5 | hi outer | 296 | 0 | 0 |
| cv20 | 6 | lo outer | 0 | 0 | 0 |
| cv20 | 6 | lo adjacent | 1 | 1 | 1 |
| cv20 | 6 | hi adjacent | 295 | 0 | 1 |
| cv20 | 6 | hi outer | 296 | 0 | 0 |

FACT read-back; numerical comparisons: verification.json.

| fixture | grid shape | dx (µm) | x absorbers | baseline edge extents x/y | eps_r/mu_r/sigma changed entries | FACT mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| cv06b | [553, 280, 37] | 63.5 | [[0, 7], [545, 552]] | [[8, 542], [8, 543]] | [0, 0, 0] | 0 |
| cv20 | [297, 66, 45] | 50 | [[0, 7], [289, 296]] | [[8, 287], [8, 288], [8, 287], [8, 288]] | [0, 0, 0] | 0 |

**EDGES_inset1.md**

Readback: `rfx.simulation.run`; time-stepping calls = 0. Indices are zero-based.
Trace-edge occupancy = Ex PEC OR Ey PEC. A1 = occupancy 0 at x indices 0 and nx-1; occupancy 1 at all other x absorber indices, on every recorded trace plane.

A1 = 0; time-stepping calls = 0.

**cv06b_dry_00**

grid = [553, 280, 37]; dx = 63.5 µm; dt = 1.21067504215e-13 s; steps = 23600.
x absorber indices = [0, 7], [545, 552].
source x indices received = [39]; point-probe x indices received = [99, 115, 131, 147, 163, 388, 404, 420, 436, 452].
pec_mask present = 0; pec_edge_masks present = 1.
trace declared bounds (m) = [[-0.0004445, 0.001016, 0.000254], [0.0344445, 0.0016159999999999998, 0.000254]]; realized z indices = [4].
readback = `cv06b_dry_00/inset1/assembly_received_00.json`.

| port | source x index | reference x index | probe x indices | source x (mm) | reference x (mm) |
| --- | --- | --- | --- | --- | --- |
| msl_0 | 39 | 99 | [99, 115, 131, 147, 163] | 2 | 5.7785 |
| msl_1 | 512 | 452 | [452, 436, 420, 404, 388] | 32 | 28.194 |

z index = 4; z = 254 µm; y index = 29; y = 1333.5 µm; eps_r sample z index = 3; z = 190.5 µm.

| edge | first x | last x | centre count | centre x-lo count | centre x-hi count | width x-lo count | width x-hi count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | 1 | 549 | 549 | 7 | 5 | 70 | 50 |
| y | 1 | 550 | 550 | 7 | 6 | 63 | 54 |

| x index | x (mm) | Ex PEC | Ey PEC | pec_mask | eps_r below |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.508 | 0 | 0 | 0 | 3.66000008583 |
| 1 | -0.4445 | 1 | 1 | 0 | 3.66000008583 |
| 2 | -0.381 | 1 | 1 | 0 | 3.66000008583 |
| 3 | -0.3175 | 1 | 1 | 0 | 3.66000008583 |
| 4 | -0.254 | 1 | 1 | 0 | 3.66000008583 |
| 5 | -0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 6 | -0.127 | 1 | 1 | 0 | 3.66000008583 |
| 7 | -0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 8 | 0 | 1 | 1 | 0 | 3.66000008583 |
| 9 | 0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 10 | 0.127 | 1 | 1 | 0 | 3.66000008583 |
| 11 | 0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 541 | 33.8455 | 1 | 1 | 0 | 3.66000008583 |
| 542 | 33.909 | 1 | 1 | 0 | 3.66000008583 |
| 543 | 33.9725 | 1 | 1 | 0 | 3.66000008583 |
| 544 | 34.036 | 1 | 1 | 0 | 3.66000008583 |
| 545 | 34.0995 | 1 | 1 | 0 | 3.66000008583 |
| 546 | 34.163 | 1 | 1 | 0 | 3.66000008583 |
| 547 | 34.2265 | 1 | 1 | 0 | 3.66000008583 |
| 548 | 34.29 | 1 | 1 | 0 | 3.66000008583 |
| 549 | 34.3535 | 1 | 1 | 0 | 3.66000008583 |
| 550 | 34.417 | 0 | 1 | 0 | 3.66000008583 |
| 551 | 34.4805 | 0 | 0 | 0 | 3.66000008583 |
| 552 | 34.544 | 0 | 0 | 0 | 3.66000008583 |

| array | total flags |
| --- | --- |
| pec_mask | 0 |
| pec_edge_x | 7191 |
| pec_edge_y | 6840 |
| pec_edge_z | 0 |

A1 = 0; time-stepping calls = 0.

**cv06b_dry_01**

grid = [553, 280, 37]; dx = 63.5 µm; dt = 1.21067504215e-13 s; steps = 23600.
x absorber indices = [0, 7], [545, 552].
source x indices received = [39]; point-probe x indices received = [99, 115, 131, 147, 163, 388, 404, 420, 436, 452].
pec_mask present = 0; pec_edge_masks present = 1.
trace declared bounds (m) = [[-0.0004445, 0.001016, 0.000254], [0.03447625, 0.0016159999999999998, 0.000254]]; realized z indices = [4].
readback = `cv06b_dry_01/inset1/assembly_received_00.json`.

| port | source x index | reference x index | probe x indices | source x (mm) | reference x (mm) |
| --- | --- | --- | --- | --- | --- |
| msl_0 | 39 | 99 | [99, 115, 131, 147, 163] | 2 | 5.7785 |
| msl_1 | 512 | 452 | [452, 436, 420, 404, 388] | 32 | 28.194 |

z index = 4; z = 254 µm; y index = 29; y = 1333.5 µm; eps_r sample z index = 3; z = 190.5 µm.

| edge | first x | last x | centre count | centre x-lo count | centre x-hi count | width x-lo count | width x-hi count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | 1 | 549 | 549 | 7 | 5 | 70 | 50 |
| y | 1 | 550 | 550 | 7 | 6 | 63 | 54 |

| x index | x (mm) | Ex PEC | Ey PEC | pec_mask | eps_r below |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.508 | 0 | 0 | 0 | 3.66000008583 |
| 1 | -0.4445 | 1 | 1 | 0 | 3.66000008583 |
| 2 | -0.381 | 1 | 1 | 0 | 3.66000008583 |
| 3 | -0.3175 | 1 | 1 | 0 | 3.66000008583 |
| 4 | -0.254 | 1 | 1 | 0 | 3.66000008583 |
| 5 | -0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 6 | -0.127 | 1 | 1 | 0 | 3.66000008583 |
| 7 | -0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 8 | 0 | 1 | 1 | 0 | 3.66000008583 |
| 9 | 0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 10 | 0.127 | 1 | 1 | 0 | 3.66000008583 |
| 11 | 0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 541 | 33.8455 | 1 | 1 | 0 | 3.66000008583 |
| 542 | 33.909 | 1 | 1 | 0 | 3.66000008583 |
| 543 | 33.9725 | 1 | 1 | 0 | 3.66000008583 |
| 544 | 34.036 | 1 | 1 | 0 | 3.66000008583 |
| 545 | 34.0995 | 1 | 1 | 0 | 3.66000008583 |
| 546 | 34.163 | 1 | 1 | 0 | 3.66000008583 |
| 547 | 34.2265 | 1 | 1 | 0 | 3.66000008583 |
| 548 | 34.29 | 1 | 1 | 0 | 3.66000008583 |
| 549 | 34.3535 | 1 | 1 | 0 | 3.66000008583 |
| 550 | 34.417 | 0 | 1 | 0 | 3.66000008583 |
| 551 | 34.4805 | 0 | 0 | 0 | 3.66000008583 |
| 552 | 34.544 | 0 | 0 | 0 | 3.66000008583 |

| array | total flags |
| --- | --- |
| pec_mask | 0 |
| pec_edge_x | 7191 |
| pec_edge_y | 6840 |
| pec_edge_z | 0 |

A1 = 1; time-stepping calls = 0.

**cv06b_dry_02**

grid = [553, 280, 37]; dx = 63.5 µm; dt = 1.21067504215e-13 s; steps = 23600.
x absorber indices = [0, 7], [545, 552].
source x indices received = [39]; point-probe x indices received = [99, 115, 131, 147, 163, 388, 404, 420, 436, 452].
pec_mask present = 0; pec_edge_masks present = 1.
trace declared bounds (m) = [[-0.0004445, 0.001016, 0.000254], [0.034508000000000004, 0.0016159999999999998, 0.000254]]; realized z indices = [4].
readback = `cv06b_dry_02/inset1/assembly_received_00.json`.

| port | source x index | reference x index | probe x indices | source x (mm) | reference x (mm) |
| --- | --- | --- | --- | --- | --- |
| msl_0 | 39 | 99 | [99, 115, 131, 147, 163] | 2 | 5.7785 |
| msl_1 | 512 | 452 | [452, 436, 420, 404, 388] | 32 | 28.194 |

z index = 4; z = 254 µm; y index = 29; y = 1333.5 µm; eps_r sample z index = 3; z = 190.5 µm.

| edge | first x | last x | centre count | centre x-lo count | centre x-hi count | width x-lo count | width x-hi count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | 1 | 550 | 550 | 7 | 6 | 70 | 60 |
| y | 1 | 551 | 551 | 7 | 7 | 63 | 63 |

| x index | x (mm) | Ex PEC | Ey PEC | pec_mask | eps_r below |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.508 | 0 | 0 | 0 | 3.66000008583 |
| 1 | -0.4445 | 1 | 1 | 0 | 3.66000008583 |
| 2 | -0.381 | 1 | 1 | 0 | 3.66000008583 |
| 3 | -0.3175 | 1 | 1 | 0 | 3.66000008583 |
| 4 | -0.254 | 1 | 1 | 0 | 3.66000008583 |
| 5 | -0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 6 | -0.127 | 1 | 1 | 0 | 3.66000008583 |
| 7 | -0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 8 | 0 | 1 | 1 | 0 | 3.66000008583 |
| 9 | 0.0635 | 1 | 1 | 0 | 3.66000008583 |
| 10 | 0.127 | 1 | 1 | 0 | 3.66000008583 |
| 11 | 0.1905 | 1 | 1 | 0 | 3.66000008583 |
| 541 | 33.8455 | 1 | 1 | 0 | 3.66000008583 |
| 542 | 33.909 | 1 | 1 | 0 | 3.66000008583 |
| 543 | 33.9725 | 1 | 1 | 0 | 3.66000008583 |
| 544 | 34.036 | 1 | 1 | 0 | 3.66000008583 |
| 545 | 34.0995 | 1 | 1 | 0 | 3.66000008583 |
| 546 | 34.163 | 1 | 1 | 0 | 3.66000008583 |
| 547 | 34.2265 | 1 | 1 | 0 | 3.66000008583 |
| 548 | 34.29 | 1 | 1 | 0 | 3.66000008583 |
| 549 | 34.3535 | 1 | 1 | 0 | 3.66000008583 |
| 550 | 34.417 | 1 | 1 | 0 | 3.66000008583 |
| 551 | 34.4805 | 0 | 1 | 0 | 3.66000008583 |
| 552 | 34.544 | 0 | 0 | 0 | 3.66000008583 |

| array | total flags |
| --- | --- |
| pec_mask | 0 |
| pec_edge_x | 7201 |
| pec_edge_y | 6849 |
| pec_edge_z | 0 |

A1 = 1; time-stepping calls = 0.

**cv20_dry_00**

grid = [297, 66, 45]; dx = 50 µm; dt = 9.53287434766e-14 s; steps = 25177.
x absorber indices = [0, 7], [289, 296].
source x indices received = [48]; point-probe x indices received = [98, 109, 120, 131, 142, 154, 165, 176, 187, 198].
pec_mask present = 1; pec_edge_masks present = 1.
trace declared bounds (m) = [[-0.00035, 0.000908, 0.000254], [0.01435, 0.0015079999999999998, 0.000304]]; realized z indices = [5, 6].
readback = `cv20_dry_00/inset1/assembly_received_00.json`.

| port | source x index | reference x index | probe x indices | source x (mm) | reference x (mm) |
| --- | --- | --- | --- | --- | --- |
| msl_0 | 48 | 98 | [98, 109, 120, 131, 142] | 2 | 4.5 |
| msl_1 | 248 | 198 | [198, 187, 176, 165, 154] | 12 | 9.5 |

z index = 5; z = 250 µm; y index = 32; y = 1200 µm; eps_r sample z index = 4; z = 200 µm.

| edge | first x | last x | centre count | centre x-lo count | centre x-hi count | width x-lo count | width x-hi count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | 1 | 294 | 294 | 7 | 6 | 91 | 78 |
| y | 1 | 295 | 295 | 7 | 7 | 84 | 84 |

| x index | x (mm) | Ex PEC | Ey PEC | pec_mask | eps_r below |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.4 | 0 | 0 | 0 | 3.66000008583 |
| 1 | -0.35 | 1 | 1 | 1 | 3.66000008583 |
| 2 | -0.3 | 1 | 1 | 1 | 3.66000008583 |
| 3 | -0.25 | 1 | 1 | 1 | 3.66000008583 |
| 4 | -0.2 | 1 | 1 | 1 | 3.66000008583 |
| 5 | -0.15 | 1 | 1 | 1 | 3.66000008583 |
| 6 | -0.1 | 1 | 1 | 1 | 3.66000008583 |
| 7 | -0.05 | 1 | 1 | 1 | 3.66000008583 |
| 8 | 0 | 1 | 1 | 1 | 3.66000008583 |
| 9 | 0.05 | 1 | 1 | 1 | 3.66000008583 |
| 10 | 0.1 | 1 | 1 | 1 | 3.66000008583 |
| 11 | 0.15 | 1 | 1 | 1 | 3.66000008583 |
| 285 | 13.85 | 1 | 1 | 1 | 3.66000008583 |
| 286 | 13.9 | 1 | 1 | 1 | 3.66000008583 |
| 287 | 13.95 | 1 | 1 | 1 | 3.66000008583 |
| 288 | 14 | 1 | 1 | 1 | 3.66000008583 |
| 289 | 14.05 | 1 | 1 | 1 | 3.66000008583 |
| 290 | 14.1 | 1 | 1 | 1 | 3.66000008583 |
| 291 | 14.15 | 1 | 1 | 1 | 3.66000008583 |
| 292 | 14.2 | 1 | 1 | 1 | 3.66000008583 |
| 293 | 14.25 | 1 | 1 | 1 | 3.66000008583 |
| 294 | 14.3 | 1 | 1 | 1 | 3.66000008583 |
| 295 | 14.35 | 0 | 1 | 0 | 3.66000008583 |
| 296 | 14.4 | 0 | 0 | 0 | 3.66000008583 |

z index = 6; z = 300 µm; y index = 32; y = 1200 µm; eps_r sample z index = 5; z = 250 µm.

| edge | first x | last x | centre count | centre x-lo count | centre x-hi count | width x-lo count | width x-hi count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x | 1 | 294 | 294 | 7 | 6 | 91 | 78 |
| y | 1 | 295 | 295 | 7 | 7 | 84 | 84 |

| x index | x (mm) | Ex PEC | Ey PEC | pec_mask | eps_r below |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.4 | 0 | 0 | 0 | 3.66000008583 |
| 1 | -0.35 | 1 | 1 | 0 | 3.66000008583 |
| 2 | -0.3 | 1 | 1 | 0 | 3.66000008583 |
| 3 | -0.25 | 1 | 1 | 0 | 3.66000008583 |
| 4 | -0.2 | 1 | 1 | 0 | 3.66000008583 |
| 5 | -0.15 | 1 | 1 | 0 | 3.66000008583 |
| 6 | -0.1 | 1 | 1 | 0 | 3.66000008583 |
| 7 | -0.05 | 1 | 1 | 0 | 3.66000008583 |
| 8 | 0 | 1 | 1 | 0 | 3.66000008583 |
| 9 | 0.05 | 1 | 1 | 0 | 3.66000008583 |
| 10 | 0.1 | 1 | 1 | 0 | 3.66000008583 |
| 11 | 0.15 | 1 | 1 | 0 | 3.66000008583 |
| 285 | 13.85 | 1 | 1 | 0 | 3.66000008583 |
| 286 | 13.9 | 1 | 1 | 0 | 3.66000008583 |
| 287 | 13.95 | 1 | 1 | 0 | 3.66000008583 |
| 288 | 14 | 1 | 1 | 0 | 3.66000008583 |
| 289 | 14.05 | 1 | 1 | 0 | 3.66000008583 |
| 290 | 14.1 | 1 | 1 | 0 | 3.66000008583 |
| 291 | 14.15 | 1 | 1 | 0 | 3.66000008583 |
| 292 | 14.2 | 1 | 1 | 0 | 3.66000008583 |
| 293 | 14.25 | 1 | 1 | 0 | 3.66000008583 |
| 294 | 14.3 | 1 | 1 | 0 | 3.66000008583 |
| 295 | 14.35 | 0 | 1 | 0 | 3.66000008583 |
| 296 | 14.4 | 0 | 0 | 0 | 3.66000008583 |

| array | total flags |
| --- | --- |
| pec_mask | 3528 |
| pec_edge_x | 7644 |
| pec_edge_y | 7080 |
| pec_edge_z | 3835 |

**TABLE.md**

S = public result.S; S_raw = public result.S_raw. Column power = sum over output ports of magnitude squared.
Delta = inset1 minus reference. Phase delta = arg(S_inset1 * conj(S_reference)), degrees.
Ring-down = 10 log10(last-10%-mean(Ez squared) / peak(Ez squared)), maximum over the 10 point probes, per drive.
S arithmetic: complex128. S_raw arithmetic: stored dtype. Passivity correction: public diagnostic, linear amplitude units.

**cv06b**

| quantity | baseline | continued | inset1 |
| --- | --- | --- | --- |
| frequency bins | 100 | 100 | 100 |
| time-stepping calls | 2 | 2 | 2 |
| ring-down drive 0 (dB) | -97.1130932934 | -57.8492224149 | -57.8539747857 |
| ring-down drive 1 (dB) | -97.314104117 | -56.9939780023 | -56.9938357998 |
| recomputed ring-down drive 0 (dB) | -97.1130932934 | -57.8492224149 | -57.8539747857 |
| recomputed ring-down drive 1 (dB) | -97.314104117 | -56.9939780023 | -56.9938357998 |
| max abs(S12-S21) | 0.00054364729611 | 0.0366171284646 | 0.0366163441782 |
| max raw abs(S12-S21) | 0.000543647271115 | 0.0388318970799 | 0.0388302616775 |
| max column power, raw | 1.00654220581 | 1.04516410828 | 1.04516255856 |
| max column power, corrected | 0.999984822306 | 0.999984806345 | 0.999984801558 |
| max passivity correction | 0.00568488007411 | 0.025357555598 | 0.0253532640636 |

**cv06b: inset1-vs-baseline**

| S entry | delta unit | bins | finite bins | max abs delta | mean abs delta | min signed delta | max signed delta | mean signed delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S11 | linear magnitude | 100 | 100 | 0.0163214179424 | 0.00211460141165 | -0.0163214179424 | 0.0149759023153 | -0.000387904861591 |
| S11 | dB | 100 | 100 | 1.61381160558 | 0.0899621258278 | -1.61381160558 | 0.819677403134 | -0.0203695083865 |
| S11 | phase (deg) | 100 | 100 | 10.7314421733 | 0.805557868261 | -10.7314421733 | 6.72634258466 | -0.134967594985 |
| S12 | linear magnitude | 100 | 100 | 0.0217413073682 | 0.00174312537859 | -0.0217413073682 | 0.00441172666142 | -0.00140334335314 |
| S12 | dB | 100 | 100 | 0.194546919818 | 0.0177568895297 | -0.194546919818 | 0.0540812781181 | -0.0133443747407 |
| S12 | phase (deg) | 100 | 100 | 9.73504737395 | 0.385049969955 | -9.73504737395 | 2.51252779749 | -0.0996730985117 |
| S21 | linear magnitude | 100 | 100 | 0.0121438420982 | 0.00138216293318 | -0.0121438420982 | 0.00370945602458 | -0.000964172242085 |
| S21 | dB | 100 | 100 | 0.107735795427 | 0.0144758296399 | -0.107735795427 | 0.0540818930378 | -0.00904177851032 |
| S21 | phase (deg) | 100 | 100 | 9.59471544865 | 0.291871235628 | -2.46910842851 | 9.59471544865 | 0.0721498451482 |
| S22 | linear magnitude | 100 | 100 | 0.0116762123615 | 0.00111074543292 | -0.0116762123615 | 0.00626390923287 | -0.000495641174607 |
| S22 | dB | 100 | 100 | 0.697107182365 | 0.0459901713172 | -0.697107182365 | 0.549556236858 | -0.0102714031693 |
| S22 | phase (deg) | 100 | 100 | 7.7137597239 | 0.658634935076 | -7.7137597239 | 7.51213330912 | -0.0748822839887 |

| array | changed entries |
| --- | --- |
| eps_r_changed_entries | 0 |
| mu_r_changed_entries | 0 |
| sigma_changed_entries | 0 |
| pec_mask_changed_entries | 0 |
| pec_edge_x_changed_entries | 150 |
| pec_edge_y_changed_entries | 135 |
| pec_edge_z_changed_entries | 0 |

**cv06b: inset1-vs-continued**

| S entry | delta unit | bins | finite bins | max abs delta | mean abs delta | min signed delta | max signed delta | mean signed delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S11 | linear magnitude | 100 | 100 | 1.62464256319e-06 | 2.02520297497e-07 | -1.62464256319e-06 | 1.4817364625e-06 | -6.75607780692e-10 |
| S11 | dB | 100 | 100 | 0.000176529845806 | 8.615360412e-06 | -0.000176529845806 | 0.000103927654678 | -3.62289431991e-07 |
| S11 | phase (deg) | 100 | 100 | 0.00248995768894 | 8.60859924292e-05 | -0.00248995768894 | 0.000452384688799 | -5.60815184232e-05 |
| S12 | linear magnitude | 100 | 100 | 1.29335368637e-06 | 1.35344690484e-07 | -1.29335368637e-06 | 9.86880305476e-07 | 3.52248046399e-08 |
| S12 | dB | 100 | 100 | 1.63393161259e-05 | 1.80199470837e-06 | -1.15893437131e-05 | 1.63393161259e-05 | 5.81781175482e-07 |
| S12 | phase (deg) | 100 | 100 | 0.000143646069466 | 1.89285820509e-05 | -5.23650014873e-05 | 0.000143646069466 | 8.22031816678e-06 |
| S21 | linear magnitude | 100 | 100 | 2.13916418346e-06 | 1.99170999486e-07 | -2.13916418346e-06 | 1.52625229999e-06 | 9.10065892687e-09 |
| S21 | dB | 100 | 100 | 2.69140075773e-05 | 2.50116035726e-06 | -2.69140075773e-05 | 1.40287737405e-05 | -6.32732893614e-08 |
| S21 | phase (deg) | 100 | 100 | 0.000613561226627 | 3.30573017082e-05 | -0.000613561226627 | 0.000181905864741 | -5.38071063507e-06 |
| S22 | linear magnitude | 100 | 100 | 2.92462421583e-06 | 2.5743873819e-07 | -2.42434357497e-06 | 2.92462421583e-06 | 4.94898303866e-10 |
| S22 | dB | 100 | 100 | 0.000190607011085 | 1.05384179638e-05 | -0.000150706576289 | 0.000190607011085 | -2.43948030899e-07 |
| S22 | phase (deg) | 100 | 100 | 0.00136504579666 | 6.46592827873e-05 | -0.000306480786227 | 0.00136504579666 | 3.41789744532e-05 |

| array | changed entries |
| --- | --- |
| eps_r_changed_entries | 0 |
| mu_r_changed_entries | 0 |
| sigma_changed_entries | 0 |
| pec_mask_changed_entries | 0 |
| pec_edge_x_changed_entries | 20 |
| pec_edge_y_changed_entries | 18 |
| pec_edge_z_changed_entries | 0 |

Notch: S21; 3-point parabola in log magnitude using spectral_features.refined_extremum(transform="log").

| quantity | baseline | continued | inset1 |
| --- | --- | --- | --- |
| sampled frequency (GHz) | 3.754545408 | 3.754545408 | 3.754545408 |
| 3-point frequency (GHz) | 3.75864590339 | 3.75855758135 | 3.7585575814 |
| sampled depth (dB) | -39.3781660566 | -39.4174177674 | -39.4174446814 |
| 3-point depth (dB) | -39.4351352244 | -39.4722105269 | -39.4722375419 |
| 3-point shift (bins) | 0.0644359790909 | 0.0630480694444 | 0.0630480702369 |

**cv20**

| quantity | baseline | continued | inset1 |
| --- | --- | --- | --- |
| frequency bins | 30 | 30 | 30 |
| time-stepping calls | 2 | 2 | 2 |
| ring-down drive 0 (dB) | -98.4414383625 | -63.4187372246 | -63.422860271 |
| ring-down drive 1 (dB) | -101.282353436 | -55.0880096561 | -55.0896763306 |
| recomputed ring-down drive 0 (dB) | -98.4414383625 | -63.4187372246 | -63.422860271 |
| recomputed ring-down drive 1 (dB) | -101.282353436 | -55.0880096561 | -55.0896763306 |
| max abs(S12-S21) | 6.51959920121e-05 | 0.0508016516239 | 0.0508016860668 |
| max raw abs(S12-S21) | 8.2175038936e-05 | 0.0515043719816 | 0.0515044080528 |
| max column power, raw | 1.00091327572 | 1.02775159333 | 1.02775159342 |
| max column power, corrected | 0.999990702741 | 0.999967731405 | 0.999967731314 |
| max passivity correction | 0.00126932118836 | 0.0226544867371 | 0.0226544649257 |

**cv20: inset1-vs-baseline**

| S entry | delta unit | bins | finite bins | max abs delta | mean abs delta | min signed delta | max signed delta | mean signed delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S11 | linear magnitude | 30 | 30 | 0.0106563835714 | 0.000945822123482 | -0.0106563835714 | 0.00352959394506 | -0.000520867246138 |
| S11 | dB | 30 | 30 | 1.27772356023 | 0.145785209606 | -1.27772356023 | 0.415746170895 | -0.0616610731915 |
| S11 | phase (deg) | 30 | 30 | 12.7935960693 | 1.8594622769 | -12.7935960693 | 5.3743411731 | -0.480104776394 |
| S12 | linear magnitude | 30 | 30 | 0.0241703642729 | 0.00189263019078 | -0.0241703642729 | 0.00012325304787 | -0.0018746852599 |
| S12 | dB | 30 | 30 | 0.213187554449 | 0.0165944116518 | -0.213187554449 | 0.00107311632873 | -0.016438278579 |
| S12 | phase (deg) | 30 | 30 | 1.97695155084 | 0.184179015954 | -0.966170664825 | 1.97695155084 | 0.0324584398641 |
| S21 | linear magnitude | 30 | 30 | 0.0101622179241 | 0.0010126678578 | -0.0101622179241 | 0.000123964818634 | -0.00100440353655 |
| S21 | dB | 30 | 30 | 0.0890064224923 | 0.00884556352213 | -0.0890064224923 | 0.00108000129892 | -0.00877356343553 |
| S21 | phase (deg) | 30 | 30 | 0.965466779977 | 0.0964565907036 | -0.965466779977 | 0.500188930771 | -0.0074967378572 |
| S22 | linear magnitude | 30 | 30 | 0.00819138066361 | 0.00087468121783 | -0.00430112117982 | 0.00819138066361 | -0.000176419534553 |
| S22 | dB | 30 | 30 | 0.868048354877 | 0.125292154925 | -0.516451337307 | 0.868048354877 | -0.0482128018654 |
| S22 | phase (deg) | 30 | 30 | 11.0240805922 | 1.50139925475 | -5.20336708429 | 11.0240805922 | -0.0335997494203 |

| array | changed entries |
| --- | --- |
| eps_r_changed_entries | 0 |
| mu_r_changed_entries | 0 |
| sigma_changed_entries | 0 |
| pec_mask_changed_entries | 168 |
| pec_edge_x_changed_entries | 364 |
| pec_edge_y_changed_entries | 336 |
| pec_edge_z_changed_entries | 182 |

**cv20: inset1-vs-continued**

| S entry | delta unit | bins | finite bins | max abs delta | mean abs delta | min signed delta | max signed delta | mean signed delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S11 | linear magnitude | 30 | 30 | 8.21310115195e-08 | 6.76962115005e-09 | -8.21310115195e-08 | 1.28127402438e-08 | -3.74250850062e-09 |
| S11 | dB | 30 | 30 | 1.06088561687e-05 | 1.00193936194e-06 | -1.06088561687e-05 | 1.65227656979e-06 | -4.60773791033e-07 |
| S11 | phase (deg) | 30 | 30 | 9.22492529488e-05 | 8.37672736233e-06 | -9.22492529488e-05 | 1.29808041313e-05 | -3.68520350074e-06 |
| S12 | linear magnitude | 30 | 30 | 1.37757018459e-07 | 8.75999303096e-09 | -4.27252111468e-09 | 1.37757018459e-07 | 8.38023788511e-09 |
| S12 | dB | 30 | 30 | 1.2300790081e-06 | 7.74231594076e-08 | -3.72274875475e-08 | 1.2300790081e-06 | 7.41155008824e-08 |
| S12 | phase (deg) | 30 | 30 | 1.02933273007e-05 | 9.60044668072e-07 | -1.02933273007e-05 | 5.23548403838e-06 | 1.0471147637e-08 |
| S21 | linear magnitude | 30 | 30 | 2.32947183765e-08 | 2.80741282073e-09 | -2.32947183765e-08 | 4.78576756002e-09 | -1.22571154707e-09 |
| S21 | dB | 30 | 30 | 2.03893837758e-07 | 2.45561848911e-08 | -2.03893837758e-07 | 4.17339726458e-08 | -1.07889801188e-08 |
| S21 | phase (deg) | 30 | 30 | 1.22746730227e-05 | 9.68021138542e-07 | -1.22746730227e-05 | 1.19443868065e-07 | -9.24419284286e-07 |
| S22 | linear magnitude | 30 | 30 | 6.0957292311e-08 | 6.04414763401e-09 | -6.0957292311e-08 | 9.5145110679e-09 | -2.63180406704e-09 |
| S22 | dB | 30 | 30 | 6.14740506322e-06 | 8.50752003601e-07 | -6.14740506322e-06 | 1.33817443171e-06 | -3.02881714731e-07 |
| S22 | phase (deg) | 30 | 30 | 2.80532722604e-05 | 4.84817756542e-06 | -1.39333105074e-06 | 2.80532722604e-05 | 4.61888003668e-06 |

| array | changed entries |
| --- | --- |
| eps_r_changed_entries | 0 |
| mu_r_changed_entries | 0 |
| sigma_changed_entries | 0 |
| pec_mask_changed_entries | 36 |
| pec_edge_x_changed_entries | 78 |
| pec_edge_y_changed_entries | 48 |
| pec_edge_z_changed_entries | 26 |

Beta: 20_msl_phase_referee.py::_analytic_beta_witness; eps_eff: _hammerstad_jensen_eps_eff; height: _h_dielectric_under_strip.
Signed deviation (%) = 100 * (Re(beta_fitted) / beta_HJ - 1); gated band 3.0–4.5 GHz.

| quantity | baseline | continued | inset1 |
| --- | --- | --- | --- |
| width (µm) | 600 | 600 | 600 |
| height (µm) | 250 | 250 | 250 |
| eps_r | 3.66 | 3.66 | 3.66 |
| eps_eff | 2.87297022632 | 2.87297022632 | 2.87297022632 |
| c0 (m/s) | 299800000 | 299800000 | 299800000 |

| quantity | baseline | continued | inset1 |
| --- | --- | --- | --- |
| gated bins | 9 | 9 | 9 |
| signed beta deviation min (%) | 1.31074048263 | 0.802181063571 | 0.801804623743 |
| signed beta deviation max (%) | 1.32077172895 | 1.17177027336 | 1.17191263606 |
| signed beta deviation mean (%) | 1.31679693122 | 0.997930731363 | 0.997936807669 |

fitted beta (rad/m)

| bin | f (GHz) | HJ beta (rad/m) | baseline | continued | inset1 |
| --- | --- | --- | --- | --- | --- |
| 17 | 3.13793103448 | 111.469793579 | 112.936080933 | 112.363983154 | 112.363563538 |
| 18 | 3.29310344828 | 116.982036118 | 118.525794983 | 117.970741272 | 117.971176147 |
| 19 | 3.44827586207 | 122.494278658 | 124.109535217 | 123.601478577 | 123.601104736 |
| 20 | 3.60344827586 | 128.006521197 | 129.691650391 | 129.272949219 | 129.272659302 |
| 21 | 3.75862068966 | 133.518763737 | 135.282241821 | 134.880264282 | 134.880187988 |
| 22 | 3.91379310345 | 139.031006277 | 140.86618042 | 140.4584198 | 140.458602905 |
| 23 | 4.06896551724 | 144.543248816 | 146.437835693 | 146.151672363 | 146.151901245 |
| 24 | 4.22413793103 | 150.055491356 | 152.024734497 | 151.813796997 | 151.81401062 |
| 25 | 4.37931034483 | 155.567733895 | 157.616867065 | 157.294326782 | 157.294662476 |

signed beta deviation (%)

| bin | f (GHz) | HJ beta (rad/m) | baseline | continued | inset1 |
| --- | --- | --- | --- | --- | --- |
| 17 | 3.13793103448 | 111.469793579 | 1.31541228071 | 0.802181063571 | 0.801804623743 |
| 18 | 3.29310344828 | 116.982036118 | 1.31965463752 | 0.845176906277 | 0.845548651805 |
| 19 | 3.44827586207 | 122.494278658 | 1.31863837003 | 0.903878883962 | 0.903573693927 |
| 20 | 3.60344827586 | 128.006521197 | 1.31644011371 | 0.989346487579 | 0.989120001467 |
| 21 | 3.75862068966 | 133.518763737 | 1.32077172895 | 1.01970727342 | 1.01965013243 |
| 22 | 3.91379310345 | 139.031006277 | 1.31997472539 | 1.02668718397 | 1.02681888515 |
| 23 | 4.06896551724 | 144.543248816 | 1.31074048263 | 1.11276283067 | 1.11292117901 |
| 24 | 4.22413793103 | 150.055491356 | 1.31234326945 | 1.17177027336 | 1.17191263606 |
| 25 | 4.37931034483 | 155.567733895 | 1.31719677256 | 1.10986567945 | 1.11008146543 |

angle(S21) (deg)

| bin | f (GHz) | HJ beta (rad/m) | baseline | continued | inset1 |
| --- | --- | --- | --- | --- | --- |
| 17 | 3.13793103448 | 111.469793579 | -32.3730883681 | -32.3083462611 | -32.3083461626 |
| 18 | 3.29310344828 | 116.982036118 | -33.9740528327 | -33.9701750994 | -33.9701750979 |
| 19 | 3.44827586207 | 122.494278658 | -35.5747878502 | -35.6199589201 | -35.6199591654 |
| 20 | 3.60344827586 | 128.006521197 | -37.1755377767 | -37.0972162265 | -37.097216324 |
| 21 | 3.75862068966 | 133.518763737 | -38.7760426933 | -38.7067625889 | -38.7067626488 |
| 22 | 3.91379310345 | 139.031006277 | -40.3759131301 | -40.4913218958 | -40.4913226248 |
| 23 | 4.06896551724 | 144.543248816 | -41.9756566828 | -41.9495236448 | -41.9495245518 |
| 24 | 4.22413793103 | 150.055491356 | -43.5754749143 | -43.3452920855 | -43.3452926043 |
| 25 | 4.37931034483 | 155.567733895 | -45.174245359 | -45.3254980903 | -45.3254999132 |

| analytic witness exception count | baseline | continued | inset1 |
| --- | --- | --- | --- |
| RuntimeError | 0 | 0 | 0 |

**Recorded diagnostics**

local cv06b dry attempt 0: matplotlib cache

~~~text
Could not save font_manager cache NO_MUTATION: os.remove ('/root/workspace/bk-workspace/.801-measure/msl_inset/mpl_config/fontlist-v3.11.0.json.matplotlib-lock', -1)
~~~

| quantity | value |
| --- | --- |
| completed delete operations | 0 |
| lock file size (bytes) | 0 |

| cv20 arm | analytic witness RuntimeError count | exact diagnostic field |
| --- | --- | --- |
| baseline | 0 | cv20_reduced.json: variants.baseline.analytic_beta.exception |
| continued | 0 | cv20_reduced.json: variants.continued.analytic_beta.exception |
| inset1 | 0 | cv20_reduced.json: variants.inset1.analytic_beta.exception |

| quantity | value |
| --- | --- |
| new GPU runs | 2 |
| YAML relaunches | 0 |
| run_id.txt append commands | 1 |
| GitHub posts | 0 |
| git commands | 0 |
| pre-existing file edits | 0 |
| completed delete operations | 0 |
| FACT mismatches | 0 |

**Files; sizes in bytes**

| file | bytes |
| --- | --- |
| COMMANDS.md | 14644 |
| EDGES_inset1.md | 11386 |
| REPORT.md | 33487 |
| TABLE.md | 14588 |
| cv06b/inset1/A1_received_00.json | 299 |
| cv06b/inset1/A1_received_01.json | 299 |
| cv06b/inset1/assembly_received_00.json | 225165474 |
| cv06b/inset1/assembly_received_00.npz | 164141 |
| cv06b/inset1/assembly_received_01.json | 225165546 |
| cv06b/inset1/assembly_received_01.npz | 164141 |
| cv06b/inset1/builder_copy.py | 2001 |
| cv06b/inset1/diagnostics.json | 115996 |
| cv06b/inset1/diagnostics.npz | 11754 |
| cv06b/inset1/preflight.json | 3758 |
| cv06b/inset1/preflight.txt | 3736 |
| cv06b/inset1/run.log | 21056 |
| cv06b/inset1/s.npz | 3802 |
| cv06b/inset1/settings.json | 3907 |
| cv06b/inset1/solve_return_00.json | 342 |
| cv06b/inset1/solve_return_01.json | 342 |
| cv06b/inset1/status.json | 280 |
| cv06b/inset1/witness_series_00.npz | 827426 |
| cv06b/inset1/witness_series_01.npz | 826528 |
| cv06b/provenance.json | 1817 |
| cv06b/summary.json | 310 |
| cv06b_dry_00/inset1/A1_received_00.json | 299 |
| cv06b_dry_00/inset1/assembly_received_00.json | 225165473 |
| cv06b_dry_00/inset1/assembly_received_00.npz | 164131 |
| cv06b_dry_00/inset1/builder_copy.py | 1999 |
| cv06b_dry_00/inset1/preflight.json | 3760 |
| cv06b_dry_00/inset1/preflight.txt | 3738 |
| cv06b_dry_00/inset1/run.log | 11957 |
| cv06b_dry_00/inset1/settings.json | 3885 |
| cv06b_dry_00/inset1/status.json | 247 |
| cv06b_dry_00/provenance.json | 1816 |
| cv06b_dry_00/summary.json | 275 |
| cv06b_dry_01/inset1/A1_received_00.json | 299 |
| cv06b_dry_01/inset1/assembly_received_00.json | 225165474 |
| cv06b_dry_01/inset1/assembly_received_00.npz | 164131 |
| cv06b_dry_01/inset1/builder_copy.py | 1999 |
| cv06b_dry_01/inset1/preflight.json | 3760 |
| cv06b_dry_01/inset1/preflight.txt | 3738 |
| cv06b_dry_01/inset1/run.log | 11957 |
| cv06b_dry_01/inset1/settings.json | 3887 |
| cv06b_dry_01/inset1/status.json | 247 |
| cv06b_dry_01/provenance.json | 1816 |
| cv06b_dry_01/summary.json | 275 |
| cv06b_dry_02/inset1/A1_received_00.json | 299 |
| cv06b_dry_02/inset1/assembly_received_00.json | 225165474 |
| cv06b_dry_02/inset1/assembly_received_00.npz | 164141 |
| cv06b_dry_02/inset1/builder_copy.py | 1999 |
| cv06b_dry_02/inset1/preflight.json | 3758 |
| cv06b_dry_02/inset1/preflight.txt | 3736 |
| cv06b_dry_02/inset1/run.log | 11951 |
| cv06b_dry_02/inset1/settings.json | 3907 |
| cv06b_dry_02/inset1/status.json | 247 |
| cv06b_dry_02/provenance.json | 1816 |
| cv06b_dry_02/summary.json | 275 |
| cv06b_reduced.json | 219238 |
| cv20/inset1/A1_received_00.json | 595 |
| cv20/inset1/A1_received_01.json | 595 |
| cv20/inset1/assembly_received_00.json | 20868 |
| cv20/inset1/assembly_received_00.npz | 33454 |
| cv20/inset1/assembly_received_01.json | 20983 |
| cv20/inset1/assembly_received_01.npz | 33454 |
| cv20/inset1/builder_copy.py | 1001 |
| cv20/inset1/diagnostics.json | 36138 |
| cv20/inset1/diagnostics.npz | 7418 |
| cv20/inset1/preflight.json | 4527 |
| cv20/inset1/preflight.txt | 4463 |
| cv20/inset1/run.log | 26859 |
| cv20/inset1/s.npz | 2503 |
| cv20/inset1/settings.json | 3640 |
| cv20/inset1/solve_return_00.json | 342 |
| cv20/inset1/solve_return_01.json | 341 |
| cv20/inset1/status.json | 277 |
| cv20/inset1/witness_series_00.npz | 869867 |
| cv20/inset1/witness_series_01.npz | 867527 |
| cv20/provenance.json | 1826 |
| cv20/summary.json | 307 |
| cv20_dry_00/inset1/A1_received_00.json | 595 |
| cv20_dry_00/inset1/assembly_received_00.json | 20868 |
| cv20_dry_00/inset1/assembly_received_00.npz | 33454 |
| cv20_dry_00/inset1/builder_copy.py | 1001 |
| cv20_dry_00/inset1/preflight.json | 4527 |
| cv20_dry_00/inset1/preflight.txt | 4463 |
| cv20_dry_00/inset1/run.log | 14248 |
| cv20_dry_00/inset1/settings.json | 3640 |
| cv20_dry_00/inset1/status.json | 243 |
| cv20_dry_00/provenance.json | 1825 |
| cv20_dry_00/summary.json | 271 |
| cv20_reduced.json | 159799 |
| events_import.json | 318 |
| measure.py | 18575 |
| mpl_config/fontlist-v3.11.0.json | 26961 |
| mpl_config/fontlist-v3.11.0.json.matplotlib-lock | 0 |
| reduce.py | 9178 |
| render_edges.py | 4063 |
| report.py | 4989 |
| run_id.txt | 37 |
| selection.json | 720 |
| verification.json | 6844 |
| verify.py | 6228 |
| vessl_cv06b.yaml | 821 |
| vessl_cv06b_369367262640.log | 26622 |
| vessl_cv06b_369367262640.txt | 1745 |
| vessl_cv20.yaml | 827 |
| vessl_cv20_369367262641.log | 33460 |
| vessl_cv20_369367262641.txt | 1751 |
