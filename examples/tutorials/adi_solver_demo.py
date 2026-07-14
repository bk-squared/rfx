"""ADI solver -- trade wavelength-scale accuracy for a larger timestep.

The default ``solver="yee"`` is an explicit scheme: its timestep is bounded by
the Courant (CFL) limit, which is set by the *smallest* cell in the mesh.  A
thin layer -- a substrate a few cells thick, a coating, a narrow gap -- forces a
tiny timestep on the *whole* domain even though the wavelength is large.  That
is the stiff-mesh problem.

The ``solver="adi"`` path (Alternating-Direction-Implicit, the Zheng-Chen-Zhang
two-sub-step 3D scheme) is *unconditionally stable*: it stays bounded at any
timestep.  ``adi_cfl_factor`` sets how far past the standard limit you push.
Stability is free; accuracy is not.  The scheme adds a temporal dispersion error
that grows with the timestep (roughly as ``dt^2``), so a large factor is a
throughput setting, not an accuracy setting.

This tutorial resonates a closed vacuum PEC cavity -- an exact analytic
oracle -- and reads TE101 three ways: explicit Yee, ADI at ``adi_cfl_factor=2``
(the accuracy setting), and ADI at ``adi_cfl_factor=5`` (the default throughput
setting).  It prints the frequency error each way so the accuracy-vs-timestep
trade is visible rather than asserted.

``f_mnp = (c/2)*sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``.

Scope, stated plainly:
  - ADI is an EXPERIMENTAL solver lane.  Its 3D accuracy is validated against
    this analytic cavity (``tests/test_review_tier1_validation_battery.py``
    holds a 2% eigenfrequency gate at ``adi_cfl_factor=2``, ~15 cells per
    wavelength).  Its *throughput advantage on a genuinely stiff mesh* is not
    yet demonstrated -- do not read this demo as a speed claim.
  - Use ``adi_cfl_factor <= 2`` for quantitative wavelength-scale results.  The
    default ``5.0`` is a stiff-mesh throughput default and carries a visible
    wavelength-scale error (shown below).  Large factors stay stable but are
    quantitative only for features much coarser than the timestep.
  - ``solver="adi"`` differentiates end-to-end (``jax.grad`` flows through the
    tridiagonal solve); this demo does not exercise gradients.

Run as::

    python examples/tutorials/adi_solver_demo.py
"""

from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.harminv import HarminvMode, harminv


C0 = 299_792_458.0

# A closed rectangular PEC cavity: a, b, d are the box edges; TE101 is a
# length (a,d) mode.  dx resolves the ~40 mm free-space wavelength at ~27 cells,
# comfortably inside the documented ~15-cells/wavelength envelope.
A = 24.0e-3
B = 12.0e-3
D = 36.0e-3
DX = 1.5e-3
FREQ_MAX = 12.0e9

M, N, P = 1, 0, 1
F_TE101 = (C0 / 2.0) * np.sqrt((M / A) ** 2 + (N / B) ** 2 + (P / D) ** 2)

# Same physical record length for every solver so the Harminv frequency
# resolution (which scales as 1/record-length) is identical across the three.
RECORD_NS = 6.9e-9


def build_cavity(solver: str, adi_cfl_factor: float) -> Simulation:
    """Closed vacuum PEC cavity with an off-centre TE101 source and probe."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(A, B, D),
        dx=DX,
        boundary=BoundarySpec.uniform("pec"),
        solver=solver,
        adi_cfl_factor=adi_cfl_factor,
    )
    # Off-centre positions couple to TE101 (Ey is a TE101 field component here).
    sim.add_source(
        (0.37 * A, 0.41 * B, 0.20 * D),
        component="ey",
        waveform=GaussianPulse(f0=F_TE101, bandwidth=0.9),
    )
    sim.add_probe((0.63 * A, 0.58 * B, 0.80 * D), component="ey")
    return sim


def run_and_read(label: str, solver: str, adi_cfl_factor: float) -> HarminvMode:
    """Run one solver, quote its preflight verbatim, and read TE101."""
    sim = build_cavity(solver, adi_cfl_factor)
    print(f"\n[{label}] preflight (quoted verbatim, not suppressed):")
    sim.preflight(strict=False)

    # Convert the fixed physical record into a step count for this solver's dt.
    # ADI's dt is adi_cfl_factor times larger than the 2D CFL limit, so the same
    # physical record needs proportionally fewer steps.
    probe_result = sim.run(n_steps=8, compute_s_params=False, skip_preflight=True)
    dt = float(probe_result.dt)
    n_steps = int(round(RECORD_NS / dt))

    result = sim.run(n_steps=n_steps, compute_s_params=False, skip_preflight=True)
    trace = np.asarray(result.time_series)[:, 0]
    ring_down = trace[len(trace) // 4 :]
    ring_down = ring_down - np.mean(ring_down)
    modes = harminv(
        ring_down, float(result.dt), 0.70 * F_TE101, 1.50 * F_TE101,
        min_Q=1.0, max_modes=12,
    )
    physical = [m for m in modes if m.Q > 1.0 and m.amplitude > 1e-9] or list(modes)
    if not physical:
        raise RuntimeError(f"[{label}] Harminv returned no cavity mode")
    te101 = min(physical, key=lambda m: abs(m.freq - F_TE101))
    err = 100.0 * (te101.freq - F_TE101) / F_TE101
    print(f"[{label}] dt={dt:.3e}s  {n_steps} steps  record={n_steps*dt*1e9:.2f} ns")
    print(f"[{label}] TE101 = {te101.freq/1e9:.4f} GHz   error = {err:+.2f}%   "
          f"Q(window) = {te101.Q:.0f}")
    return te101


def main() -> None:
    lam = C0 / F_TE101
    print("=" * 70)
    print("ADI solver demo -- vacuum PEC cavity TE101")
    print("=" * 70)
    print(f"Analytic TE101      : {F_TE101/1e9:.4f} GHz")
    print(f"Mesh                : dx = {DX*1e3:.2f} mm  "
          f"({A/DX:.0f} x {B/DX:.0f} x {D/DX:.0f} cells, "
          f"~{lam/DX:.0f} cells / free-space wavelength)")

    yee = run_and_read("yee", "yee", 5.0)          # adi_cfl_factor ignored for yee
    adi2 = run_and_read("adi cfl=2 (accuracy)", "adi", 2.0)
    adi5 = run_and_read("adi cfl=5 (default)", "adi", 5.0)

    def e(mode: HarminvMode) -> float:
        return 100.0 * (mode.freq - F_TE101) / F_TE101

    print("\n" + "-" * 70)
    print("Summary -- accuracy is traded for timestep, stability is not:")
    print(f"  explicit Yee                : {e(yee):+.2f}%  (CFL-limited dt)")
    print(f"  ADI, adi_cfl_factor = 2     : {e(adi2):+.2f}%  (2x dt, accuracy setting)")
    print(f"  ADI, adi_cfl_factor = 5     : {e(adi5):+.2f}%  (5x dt, default throughput)")
    print("-" * 70)
    print("Both ADI runs are stable at their enlarged timestep -- that is the")
    print("unconditional-stability property.  The cfl=5 error is the documented")
    print("wavelength-scale cost of the default; drop to <=2 when the resonance")
    print("frequency, not the timestep, is what you need to be accurate.")
    print("ADI is an experimental lane; its stiff-mesh throughput advantage is a")
    print("separate, not-yet-demonstrated claim.")


if __name__ == "__main__":
    main()
