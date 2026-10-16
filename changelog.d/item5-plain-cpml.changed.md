Graded multi-device CPML models without Debye or Lorentz poles now use the
wrapped one-device step on each slab. The CPML call preserves the native slab
psi layout and one-device boundary operands. Dispersive CPML retains the legacy
step.
