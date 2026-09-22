### Fixed — a multi-device run no longer stages the whole domain on the first device of every process (#PENDING)

- `sim.run(devices=...)` rasterized and staged the whole domain on the first device of
  every process and kept those arrays through the run, kept preflight's cached assembly
  alive, and gave every device slab-size zero arrays for Debye/Lorentz materials the model
  did not have. On two RTX 4090 nodes (one process per GPU) that cost 169–173 bytes per
  whole-domain cell on each GPU, against 72–98 for the same model on one GPU, so splitting a
  model across GPUs ran out of memory before a single GPU did.
- Each device now receives only its own slab (state created on its device, material and
  PEC-mask slabs staged per addressable shard), the whole-domain setup arrays and
  preflight's cached assembly are released before time stepping, and absent Debye/Lorentz
  materials use one placeholder value per device. On two RTX 4090 nodes a 324 M-cell PEC box
  that runs out of memory on one 4090, and on two 4090 nodes without this change, now runs at
  11.7 GB per GPU (36 bytes per whole-domain cell).
- Not changed: when one process drives several devices, the time loop still captures the
  material, PEC-mask, dispersion and CPML arrays, which the compiler keeps as whole-domain
  constants on every device; one process per GPU avoids it.
- Results are bit-identical to before (SHA256 of the probe trace and all six field
  components, eight models on two devices).
