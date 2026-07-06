"""Stage-0 sample: demonstrate the magnitude-blindness of the current
differentiable_material_fit loss (the "S-param proxy" self-normalization).

This is a DIAGNOSIS sample, not a fix. It isolates the exact defect the file
header (L8-16) confesses to, with numbers, so we can agree on the research
direction before touching the fitter.

The current forward() (rfx/differentiable_material_fit.py L474-484) builds the
simulated S-matrix by dividing each probe spectrum by its OWN peak magnitude:

    mag = jnp.abs(s_raw[i])
    safe_max = jnp.maximum(jnp.max(mag), 1e-30)
    s_sim = s_sim.at[i, i, :].set(s_raw[i] / safe_max)

so |s_sim| always peaks at 1.0, regardless of the material's true response
level. Meanwhile s_meas is compared at its TRUE scale. We show two consequences:

  (A) SCALE-INVARIANCE: two materials whose responses differ by a constant
      factor (e.g. |S21| peak 0.6 vs 0.06 -- a 20 dB difference in insertion
      loss) produce the IDENTICAL normalized s_sim, hence the identical loss.
      => the true magnitude / insertion-loss level is not identifiable.

  (B) IRREDUCIBLE FLOOR: because s_sim is pinned to peak 1 while a real
      measurement is not, the magnitude term of the loss has a floor the
      optimizer can never drive out, even at the true poles.
"""

from __future__ import annotations

import numpy as np

from rfx.differentiable_material_fit import sparam_loss


def self_normalize(s_raw: np.ndarray) -> np.ndarray:
    """Replicate the current forward()'s per-probe self-normalization."""
    mag = np.abs(s_raw)
    safe_max = max(float(np.max(mag)), 1e-30)
    return s_raw / safe_max


def synthetic_s21(freqs_hz: np.ndarray, peak: float, f0: float, bw: float) -> np.ndarray:
    """A toy complex S21: Lorentzian magnitude bump * linear phase (delay)."""
    mag = peak / (1.0 + ((freqs_hz - f0) / bw) ** 2)
    phase = -2.0 * np.pi * freqs_hz * 0.2e-9  # 0.2 ns group delay
    return mag * np.exp(1j * phase)


def main() -> None:
    freqs = np.linspace(1e9, 10e9, 64)

    # "Measured" truth: a device with peak |S21| = 0.60 (about -4.4 dB).
    s_meas = synthetic_s21(freqs, peak=0.60, f0=5e9, bw=1.5e9)

    # Two candidate materials whose raw probe responses share the SAME shape
    # but differ in overall level by 10x (a 20 dB insertion-loss difference).
    s_raw_correct = synthetic_s21(freqs, peak=0.60, f0=5e9, bw=1.5e9)  # matches truth
    s_raw_wrong = synthetic_s21(freqs, peak=0.06, f0=5e9, bw=1.5e9)    # 20 dB off

    print("=" * 72)
    print("(A) SCALE-INVARIANCE OF THE CURRENT SELF-NORMALIZED LOSS")
    print("=" * 72)
    print(f"  measured  peak|S21| = {np.max(np.abs(s_meas)):.3f}  "
          f"({20*np.log10(np.max(np.abs(s_meas))):+.1f} dB)")
    print(f"  cand-correct peak    = {np.max(np.abs(s_raw_correct)):.3f}  "
          f"({20*np.log10(np.max(np.abs(s_raw_correct))):+.1f} dB)")
    print(f"  cand-wrong   peak    = {np.max(np.abs(s_raw_wrong)):.3f}  "
          f"({20*np.log10(np.max(np.abs(s_raw_wrong))):+.1f} dB)  <-- 20 dB too low")

    # CURRENT pipeline: s_sim = self_normalize(s_raw); compare to raw s_meas.
    loss_correct = float(sparam_loss(self_normalize(s_raw_correct), s_meas))
    loss_wrong = float(sparam_loss(self_normalize(s_raw_wrong), s_meas))
    print("\n  CURRENT loss (self-normalized s_sim vs true-scale s_meas):")
    print(f"    loss(correct material) = {loss_correct:.6e}")
    print(f"    loss(20 dB-wrong material) = {loss_wrong:.6e}")
    print(f"    difference = {abs(loss_correct - loss_wrong):.3e}  "
          f"-> the loss CANNOT tell them apart (magnitude unidentifiable)")

    print("\n" + "=" * 72)
    print("(B) IRREDUCIBLE MAGNITUDE FLOOR AT THE TRUE POLES")
    print("=" * 72)
    # Even the *correct* material gives nonzero magnitude loss purely because
    # s_sim is pinned to peak 1.0 while the measurement peaks at 0.60.
    mag_floor = float(np.mean((np.abs(self_normalize(s_raw_correct)) - np.abs(s_meas)) ** 2))
    print(f"  mag-loss at the TRUE material (should be ~0) = {mag_floor:.6e}")
    print("  -> nonzero because self-normalized |s_sim| peaks at 1.0, not 0.60.")

    print("\n" + "=" * 72)
    print("PROPOSED FIX DIRECTION (Stage 1): drop self-normalization; compare a")
    print("physically-scaled S-parameter (proper port de-embed / ratio to the")
    print("injected incident wave) against the measurement directly.")
    print("=" * 72)
    # Sanity check that a *non-normalized* loss DOES separate the candidates.
    loss_correct_fix = float(sparam_loss(s_raw_correct, s_meas))
    loss_wrong_fix = float(sparam_loss(s_raw_wrong, s_meas))
    print(f"  fixed loss(correct)  = {loss_correct_fix:.6e}   (~0, recovers truth)")
    print(f"  fixed loss(20 dB-wrong) = {loss_wrong_fix:.6e}   (large, penalized)")
    ratio = loss_wrong_fix / max(loss_correct_fix, 1e-30)
    print(f"  separation ratio = {ratio:.3e}  -> magnitude now identifiable")


if __name__ == "__main__":
    main()
