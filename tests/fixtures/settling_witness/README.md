Frozen records copied without cropping or downsampling from
`/private/tmp/claude-501/-Users-byungkwankim-rfx-research-rfx-ports/6a795cf5-0882-4b1b-9af5-4758368329b0/scratchpad/dc_check/`.

| File | Bytes | Read bins | Identification freq_max | Source end |
|---|---:|---|---:|---:|
| ntff_worst.npz | 12,929 | 3 GHz | 5 GHz | 520 |
| waveguide_worst.npz | 48,408 | 17 bins, 8.4–11.6 GHz | 11.6 GHz | 503 |

Read-bin provenance: `tests/locks/test_ntff_directivity_validation_battery.py`
and `tests/_waveguide_chain_battery_fixture.py`. The contract reads
`selected_record`, `metadata_json`, and the original time step.
