### Changed — Record-end warnings name the outputs read from each unguarded record (#PR)

Concrete `run()` and `forward()` results now warn once per group of records that read the
same outputs, naming those outputs: S-parameters, NTFF far fields, field-DFT planes, flux
monitors and current-moment far fields as applicable. An S warning no longer hides a
field-record warning; current-moment-only runs without probes also warn. Verdicts and
returned outputs are unchanged.
