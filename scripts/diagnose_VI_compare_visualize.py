"""Visualize v_a(t), v_b(t), i_a(t), i_b(t) and their differences to find when
A and B.2 diverge in time domain.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("/tmp/VI_compare_A_B2.npz")
v_a, v_b = D["v_a"], D["v_b"]
i_a, i_b = D["i_a"], D["i_b"]
dt = float(D["dt"])
n = v_a.shape[0]
t = np.arange(n) * dt * 1e9  # ns

print(f"n_steps = {n}, dt = {dt*1e12:.3f} ps, t span = {t[-1]:.2f} ns")

# Find first nonzero region in v_a (first pulse arrival)
threshold = 1e-3 * np.abs(v_a).max()
first_nonzero = np.argmax(np.abs(v_a) > threshold)
print(f"First nonzero t = {t[first_nonzero]:.3f} ns")

# Statistics in different time windows
for label, t_lo, t_hi in [
    ("first pulse (0-2ns)", 0.0, 2.0),
    ("transit window (2-10ns)", 2.0, 10.0),
    ("ringdown 10-30ns", 10.0, 30.0),
    ("late tail 30-117ns", 30.0, 117.0),
]:
    mask = (t >= t_lo) & (t < t_hi)
    if not mask.any():
        continue
    va_max = np.abs(v_a[mask]).max()
    vb_max = np.abs(v_b[mask]).max()
    diff_max = np.abs(v_a[mask] - v_b[mask]).max()
    print(f"  {label}: |v_a|_max={va_max:.3e}, |v_b|_max={vb_max:.3e}, "
          f"|diff|_max={diff_max:.3e}")

fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)

ax = axes[0]
ax.plot(t, v_a, label="V_a (Setup A, mask)", color="C0", lw=0.7)
ax.plot(t, v_b, label="V_b (Setup B.2, boundary)", color="C1", lw=0.7, ls="--")
ax.set_xlabel("t (ns)"); ax.set_ylabel("V"); ax.legend(loc="upper right")
ax.set_title("V(t) at probe (full window)")
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)

ax = axes[1]
ax.plot(t, v_a - v_b, label="V_a - V_b", color="C2", lw=0.7)
ax.set_xlabel("t (ns)"); ax.set_ylabel("V_a - V_b"); ax.legend(loc="upper right")
ax.set_title("V_a − V_b time-domain difference (zoom 0-30 ns)")
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)

ax = axes[2]
ax.plot(t, i_a, label="I_a", color="C0", lw=0.7)
ax.plot(t, i_b, label="I_b", color="C1", lw=0.7, ls="--")
ax.set_xlabel("t (ns)"); ax.set_ylabel("I"); ax.legend(loc="upper right")
ax.set_title("I(t) at probe (zoom 0-30 ns)")
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)

ax = axes[3]
ax.plot(t, i_a - i_b, label="I_a - I_b", color="C2", lw=0.7)
ax.set_xlabel("t (ns)"); ax.set_ylabel("I_a - I_b"); ax.legend(loc="upper right")
ax.set_title("I_a − I_b difference (zoom 0-30 ns)")
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig("/tmp/VI_compare_A_B2.png", dpi=120)
print(f"\nSaved /tmp/VI_compare_A_B2.png")

# Look at first reflection arrival in detail
fig2, axes = plt.subplots(2, 1, figsize=(11, 6))
ax = axes[0]
ax.plot(t, v_a, label="V_a", color="C0", lw=0.8)
ax.plot(t, v_b, label="V_b", color="C1", lw=0.8, ls="--")
ax.set_xlim(0.5, 2.5)
ax.set_xlabel("t (ns)"); ax.set_ylabel("V"); ax.legend()
ax.set_title("V(t) zoom on first incident+reflected pulse")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t, i_a, label="I_a", color="C0", lw=0.8)
ax.plot(t, i_b, label="I_b", color="C1", lw=0.8, ls="--")
ax.set_xlim(0.5, 2.5)
ax.set_xlabel("t (ns)"); ax.set_ylabel("I"); ax.legend()
ax.set_title("I(t) zoom on first pulse")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/VI_compare_A_B2_zoom.png", dpi=120)
print(f"Saved /tmp/VI_compare_A_B2_zoom.png")
