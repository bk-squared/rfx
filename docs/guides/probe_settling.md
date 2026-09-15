# Recorded-probe settling

`run()` and `forward()` expose `settling_db` and `settling_witness` from
the user-probe time series already recorded by the simulation:

```python
from rfx.api._sparams import settling_verdict

result = sim.forward(n_steps=1000)
print(settling_verdict(result.settling_db))  # pass, fail, or absent
print(result.settling_witness)
```

For each usable record, the common scorer compares mean power over the
last tenth with peak power over the whole record, using `|u|²` for real
or complex samples. The result is the largest ratio over the selected
records, in dB. The existing comparison is `pass` at or below -40 dB.

No registered user probes, no retained samples, fewer than ten samples,
or no record above the storage-format coverage floor produce `absent`.
`settling_db` is then `None`; this is never a passing result. The floor
is the existing smallest-normal-amplitude criterion required to represent
the -40 dB decision. The witness names skipped records. Automatic source
probes and library-internal port probes cannot substitute for user probes.
Any selected record containing NaN or Inf makes the diagnostic unavailable,
even if another record decays. Invalid records are named separately from
underflow skips. Amplitude scaling before squaring prevents finite large
or small records from overflowing or underflowing the power calculation.

This is a probe-record decay diagnostic for ring-down measurements. A
passing record does not certify all spatial modes, global field energy,
source purity, RF accuracy or continuous-wave steady-state convergence.
Probe placement and the excitation/record window remain part of the
measurement design.

An eager forward call emits the existing aggregate failure/absence
warning when NTFF or field-DFT output is requested. Bare probe results
still carry the diagnostic; driver-internal calls keep their own warning
ownership. Reading the forward properties repeatedly does not emit the
warning again.

Forward diagnostics are host calculations on concrete records, not
differentiable objectives. During tracing they report absence; inspect
the concrete result outside the JIT/AD objective. Numeric probe metadata
preserves the selected columns and component labels without introducing
string leaves into the JAX result tree. The selected provenance remains
valid if the simulation's registrations later change.

This does not make every field of a complete `ForwardResult` JAX-compatible;
the existing host `Grid` object, for example, cannot be a JIT output leaf.
Return the numeric record and its selection metadata from a jitted function,
then reconstruct the diagnostic result on the host:

```python
import jax
from rfx.api._spec import ForwardResult

@jax.jit
def recorded_forward(eps):
    result = sim.forward(eps_override=eps, n_steps=1000)
    return result.time_series, result.settling_probe_info

record, selection = recorded_forward(eps)
diagnostic_result = ForwardResult(
    time_series=record, settling_probe_info=selection,
)
print(diagnostic_result.settling_witness)
```

The forward properties are computed from the record when read and are
not fields in `ForwardResult._asdict()`. Export them explicitly when
retaining an evidence artifact:

```python
diagnostic = {
    "settling_db": result.settling_db,
    "settling_witness": result.settling_witness,
}
```
