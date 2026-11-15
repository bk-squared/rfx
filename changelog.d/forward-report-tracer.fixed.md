After differentiating a graded-mesh `forward(ringdown=...)`, reading
`result.ringdown.report` no longer fails with `UnexpectedTracerError`.
The report keeps the declared nominal mesh and port widths as host data,
while the completed S-parameters retain their mesh derivatives.
