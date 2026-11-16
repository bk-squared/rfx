Preflight now reports `not_judged_traced` when an advisory check cannot judge
traced inputs, naming the check and the declaration or forward input fields.
Other advisory checks continue after a JAX tracer conversion failure.
Named rfx refusals and unrelated validator errors keep their existing handling.
Repeated skips in the same check produce one finding per input-name set.
Strict preflight escalates these not-judged findings along with every other issue;
`run()` and `forward()` continue to use non-strict preflight.
