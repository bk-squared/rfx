Graded multi-device models with CPML and no Debye or Lorentz poles now run the
same one-device time step on each slab as PEC models. Results are unchanged:
probes and fields equal one device bit for bit (2 and 3 devices, up to 1600
steps). Per-device memory in the first step is 0.51 (2 devices) and 0.37
(3 devices) of one device, 1.38 and 1.36 times the previous multi-device step.
Dispersive CPML keeps the previous step.
