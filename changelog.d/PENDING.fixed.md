### Fixed — a multi-device run no longer needs more memory on one GPU than a one-GPU run of the same model (#PENDING)

- `sim.run(devices=...)` used to rasterize and stage the whole domain on the first
  device of every process and keep those arrays for the entire run, and every device
  carried slab-size zero arrays for Debye/Lorentz materials the model did not have:
  the first GPU held about 142 bytes per cell of the whole domain at the start of the
  time loop, against 16 for a one-device run, so splitting a model across GPUs ran out
  of memory before a single GPU did.
- Each device now receives only its own slab (state created on its device, material
  and PEC-mask slabs staged one addressable shard at a time), the whole-domain setup
  arrays and preflight's cached assembly are released before time stepping, and
  absent Debye/Lorentz materials use one placeholder value per device. Measured on two
  CPU devices: 18.7 bytes per whole-domain cell on each device (PEC box), 21.9 (CPML).
- Results are bit-identical to before (SHA256 of the probe trace and all six field
  components, eight models on two devices).
