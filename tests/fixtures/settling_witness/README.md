Frozen records copied without cropping or downsampling from
rfx-archive `rfx/records/20261002-settling-witness-1426-audit/` @ `5ea04b56`
(`measure.py`, `analyze.py`, and the npz files there).

| File | Bytes | Read bins | Identification freq_max | Source end |
|---|---:|---|---:|---:|
| ntff_worst.npz | 12,929 | 3 GHz | 5 GHz | 520 |
| waveguide_worst.npz | 48,408 | 17 bins, 8.4–11.6 GHz | 11.6 GHz | 503 |

Read-bin provenance: `tests/locks/test_ntff_directivity_validation_battery.py`
`hw9mm` and `tests/oracle/test_waveguide_chain_battery_closure.py`
`measure_closure_witness`. The contract reads
`selected_record`, `metadata_json`, and the original time step.
