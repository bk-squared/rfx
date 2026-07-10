# S-Parameter Dump Replay Harness

This guide defines the compact raw-dump schema used for E3 S-parameter
evidence. It is validation infrastructure: it does not make a solver claim by
itself, but it lets a reviewer recompute a production S-matrix from raw phasors
without calling the production extractor under review.

## Schema

Use `rfx.PortDumpMetadata` plus `save_port_vi_dump_npz(...)` to write a compact
`.npz` file with these fields:

| Field | Required | Meaning |
|---|---:|---|
| `metadata_json` | yes | JSON object with schema version, commit hash, geometry/material/grid/boundary setup, port definitions, waveform, `dt`, phase convention, current convention, and reference-plane convention |
| `freqs_hz` | yes | Frequency grid, shape `(n_freqs,)` |
| `voltages` | yes | Raw complex V phasors, shape `(n_driven, n_ports, n_freqs)` |
| `currents` | yes | Raw complex I phasors, same shape as `voltages` |
| `port_impedances_ohm` | yes | Scalar or per-port impedance vector |
| `port_names` | yes | Port names in S-matrix order |
| `driven_port_indices` | yes | Which physical port was driven for each raw dump row |
| `production_smatrix` | recommended | Production output to compare against, shape `(n_ports, n_ports, n_freqs)` |
| `reference_plane_offsets_m` | optional | Per-port raw-plane to reported-reference-plane offset |
| `propagation_constants` | optional | Per-port/frequency propagation constants used for reference-plane shifts |

The S-matrix convention is always:

```text
S[receiver_port, driven_port, frequency_index]
```

## Replay formula

The independent replay uses the power-wave split, role-selected per port to
match the production lumped decomposer (issue #308):

```text
a      = (V + Z0 I) / (2 sqrt(Z0))     # incident, at the driven port
b      = (V - Z0 I) / (2 sqrt(Z0))     # reflected, at the driven port
b_recv = -(V + Z0 I) / (2 sqrt(Z0))    # arriving, at a passive receive port
S[driven, driven, f] = b[driven, f] / a[driven, f]
S[recv,   driven, f] = b_recv[recv, f] / a[driven, f]   (recv != driven)
```

The receive channel is the production receive wave `(V_fdtd - Z0 I)` expressed
in this schema's into-DUT voltage convention (`V = -V_fdtd`); its overall sign
is pinned empirically by the DC falsifier on the canonical 2-port thru
(S21(DC) -> +1).

By default current is positive into the DUT. If a dump uses current positive
out of the DUT, record `current_convention="positive_out_of_dut"`; replay flips
the sign before wave decomposition.

Reference-plane shifts are explicit metadata. If `d > 0` means the reported
reference plane is farther into the DUT than the raw measurement plane, replay
applies:

```text
a_ref = a_raw exp(-gamma d)
b_ref = b_raw exp(+gamma d)
```

## APIs and CLI

```python
from rfx import (
    PortDumpMetadata,
    save_port_vi_dump_npz,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    compare_replayed_smatrix,
)

save_port_vi_dump_npz(
    "port_dump.npz",
    voltages=v_raw,
    currents=i_raw,
    freqs=freqs_hz,
    port_impedances=50.0,
    metadata=PortDumpMetadata(commit_hash="..."),
    port_names=("in", "out"),
    production_smatrix=S_prod,
)

dump = load_port_vi_dump_npz("port_dump.npz")
replayed = replay_smatrix_from_port_vi_dump(dump)
report = compare_replayed_smatrix(replayed, type("P", (), {
    "s_params": dump.production_smatrix,
    "freqs": dump.freqs,
})())
```

Command-line replay:

```bash
python scripts/diagnostics/replay_port_vi_dump.py port_dump.npz --write-json replay.json
```

## Promotion rule

A port family can cite dump replay as E3 only when the dump was produced from a
real simulation artifact, the replay script passes against production under a
stated tolerance, and the same claim also states the mesh/frequency/geometry
envelope. Synthetic algebra tests of this harness are E0 for the harness, not
physics validation of a port family.
