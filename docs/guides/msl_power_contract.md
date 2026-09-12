# MSL S references, power, and replay

`compute_msl_s_matrix` returns power-wave S at each port's first voltage
probe plane. With the measured voltage V, inward-oriented current I, and
positive real analytic reference R for each port, its wave definitions are

    a = (V + R I) / (2 sqrt(R))
    b = (V - R I) / (2 sqrt(R))
    B = S A

`result.reference_impedances` contains these R values in `port_names` order.
They are the analytic Hammerstad–Jensen references used by the extractor,
not `result.Z0` (a fitted diagnostic) or `add_msl_port(impedance=...)`
(source/load resistance). This API currently supports positive real
references; it does not implement complex-reference renormalization.

For a passive network expressed in power-consistent port variables,
`S.conj().T @ S <= identity` at each frequency. Equivalently, the largest
singular value is at most one. Column power tests check individual drives;
they can miss excess power under simultaneous coherent excitation.
The normalization guarantees the algebraic identity
`abs(a)**2 - abs(b)**2 = real(V * conj(I))`. Interpreting that quantity as
electromagnetic power additionally requires V/I to represent the power
carried by the fields. Peak rather than RMS phasors introduce a shared 1/2
factor; it cancels in S and power ratios.

This distinction matters for quasi-TEM microstrip probes. A uniform line
section with a dominant mode is the intended port model. Evanescent or
additional propagating modes, strong longitudinal fields, finite records,
and discretization can invalidate or bias the reduction from fields to
one V/I pair. A large reflection or a deep notch alone does not establish
that this model failed. The distinction between equivalent-circuit power,
modal fields, and reference impedances is discussed by Williams in
[Traveling Waves and Power Waves](https://www.nist.gov/system/files/documents/2017/05/09/MicrowaveCircuitTheory-proof.pdf).

`enforce_passivity=True` clips the assembled S singular values and preserves
changed raw values in `S_raw` with `passivity_correction`. Use
`enforce_passivity=False` to diagnose the raw extraction. Projection,
`reliable`, `probe_clearance`, and `cond_a` do not certify RF accuracy.
The equal-reference cv06b records retained with issue #726 still show raw
coherent power gain above one; this normalization change does not explain
or remove that residual.

## Compatibility

Legacy MSL extraction returned voltage-wave S. For unchanged measured V/I
and references, the new power matrix satisfies

    S_power[i, j] = sqrt(R[j] / R[i]) * S_voltage[i, j]

All equal-reference entries and every diagonal retain the same mathematical
definition. Internally, relative scales `sqrt(R[0])/sqrt(R[i])` leave equal
references exactly unscaled. For unequal references, transmission and
`cond_a` change to the power basis; the passivity projection now uses the
appropriate Euclidean power metric. Already projected legacy voltage S
cannot in general be repaired by rescaling it: recover the raw V/I or raw
voltage S first.

## Saved records

`raw_3probe_dump_path` produces `rfx.msl_nprobe_dump` v4, with power-wave
metadata and `s_reference_impedances_ohm`. V/I remain measured phasors;
`production_smatrix` is the unprojected result. Its assembly marker
distinguishes the full multi-drive solve from the legacy single-ratio
fallback. A fallback includes returning waves at receiving ports and is
not a substitute for a well-conditioned multi-drive solve.

`rfx.validation.load_port_vi_dump_npz` and
`replay_smatrix_from_port_vi_dump` replay v4 in the power basis and v3 in
its historical voltage basis. Replay is at the recorded probe planes;
no feed-plane translation or current sign change is inferred. A v3 record
without actual S references is rejected: source/load resistance and fitted
Z0 are insufficient substitutes. Such records need a separately verified
historical reconstruction. `scripts/diagnostics/replay_port_vi_dump.py`
provides the same replay as a command-line comparison.

The normalization regressions use independently prescribed passive,
reciprocal two-port waves with unequal references through the public
extraction path, plus saved real-field v3 replay. They test wave algebra,
dump compatibility, and differentiation; they do not constitute a new
unequal-port FDTD accuracy benchmark.
