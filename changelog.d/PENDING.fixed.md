### Fixed — port drives and RLC use realized edge materials (#PENDING)

- Ports sharing an edge include its final summed load; uniform single- and multi-device drives now agree.
- Ports at a periodic material seam use the wrapped edge permittivity.
- Graded RLC elements on an MSL feed edge include the MSL load in their update denominator.
