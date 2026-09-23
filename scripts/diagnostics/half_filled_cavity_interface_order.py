"""How far a dielectric interface sits from where it was drawn (#1210).

FIXTURE. A 24 mm PEC cube, the lower half (z < 12 mm) filled with eps_r = 4,
air above. Modes with E transverse to z are TE-to-z: E_t(x, y, z) = e_t(x, y)
f(z) with k_t^2 = (m*pi/a)^2 + (n*pi/b)^2 the transverse eigenvalue of the PEC
cross-section, and f(z) = sin(k1 z) below the interface, sin(k2 (d - z)) above.
E_t is tangential to the z = 12 mm plane, so E_t and H_t are both continuous
there; with one mu that is f and df/dz continuous, giving the transverse
resonance

    k1 * cot(k1 * H) = -k2 * cot(k2 * (d - H)),
    k_i = sqrt(eps_i * (w/c)^2 - k_t^2)    (imaginary -> cot becomes -coth)

HOW TO RUN

    python scripts/diagnostics/half_filled_cavity_interface_order.py

It prints the analytic root of each named mode, then the FDTD reading at
dx = 1 mm and dx = 0.5 mm on two arms: the per-component edge average this
branch installs, and the cell-owned rule it replaces (emulated by patching
rfx.core.yee.edge_averaged_materials to return the owning cell, with the JAX
compilation cache cleared between arms -- without that clear both arms return
the first arm's numbers).

The committed gate on one of these modes is
tests/oracle/test_half_filled_cavity_interface_order.py.
"""
import sys
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # the repo root, wherever it is checked out
import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402
import rfx.core.yee as Y                                # noqa: E402
from rfx import GaussianPulse, Simulation               # noqa: E402
from rfx.harminv import harminv                         # noqa: E402

C0 = 299792458.0
L = 24e-3          # cube side (a = b = d)
H = 12e-3          # fill height
EPSR = 4.0

_TRUE = Y.edge_averaged_materials
_OWNED = lambda eps, sig, per=(False, False, False): ((eps,) * 3, (sig,) * 3)

MODES = {"TE10": (1, 0), "TE11": (1, 1), "TE20": (2, 0)}


def _kt(m, n):
    return np.sqrt((m * np.pi / L) ** 2 + (n * np.pi / L) ** 2)


def _residual(freq, kt):
    k0 = 2 * np.pi * freq / C0

    def branch(eps, length):
        ksq = eps * k0 ** 2 - kt ** 2
        if ksq > 0:
            k = np.sqrt(ksq)
            return k / np.tan(k * length)  # k*cot(k*L)
        # Evanescent: k = -j*alpha, and k*cot(k*L) = +alpha*coth(alpha*L).
        # (The sign here is the whole derivation: -alpha*coth moves every
        # root of the fixture, and put the first version of this script on a
        # higher mode than the one it named.)
        alpha = np.sqrt(-ksq)
        return alpha / np.tanh(alpha * length)

    return branch(EPSR, H) + branch(1.0, L - H)


def analytic(m, n, f_hi=12e9):
    kt = _kt(m, n)
    # Start just above the DIELECTRIC cutoff: below it both regions are
    # evanescent and no resonance exists. Between the two cutoffs the air
    # half is evanescent, which is where the lowest roots of this fixture
    # live -- the band the first version of this script skipped.
    f_lo = C0 * kt / (2 * np.pi * np.sqrt(EPSR)) * 1.0001
    fs = np.linspace(f_lo, f_hi, 20000)
    vals = np.array([_residual(f, kt) for f in fs])
    for i in range(len(fs) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0 and abs(a - b) < 1e4:
            return brentq(_residual, fs[i], fs[i + 1], args=(kt,))
    return float("nan")


def measure(dx, rule, steps, fwin):
    Y.edge_averaged_materials = rule
    jax.clear_caches()
    f0 = 0.5 * (fwin[0] + fwin[1])
    sim = Simulation(freq_max=3 * f0, domain=(L, L, L), dx=dx, boundary="pec")
    sim.add_source((L * 0.23, L * 0.31, L * 0.41), "ex", amplitude_kind="field",
                   waveform=GaussianPulse(f0=f0, bandwidth=1.0))
    sim.add_probe((L * 0.73, L * 0.64, L * 0.29), "ex")
    r0 = sim.forward(n_steps=2, skip_preflight=True)
    shape, dt = tuple(r0.grid.shape), float(r0.grid.dt)
    zc = (np.arange(shape[2]) + 0.5) * dx
    eps = jnp.broadcast_to(
        jnp.asarray(np.where(zc < H, EPSR, 1.0).astype(np.float32))[None, None, :],
        shape)
    res = sim.forward(eps_override=eps, n_steps=steps, skip_preflight=True,
                      checkpoint=False)
    Y.edge_averaged_materials = _TRUE
    jax.clear_caches()
    modes = [m for m in harminv(np.asarray(res.time_series[:, 0])[steps // 4:],
                                dt, *fwin) if m.Q > 30]
    if not modes:
        return float("nan")
    return max(modes, key=lambda m: m.amplitude).freq


if __name__ == "__main__":
    for name, (m, n) in MODES.items():
        f_an = analytic(m, n)
        win = (f_an * 0.94, f_an * 1.10)
        print(f"{name}  analytic {f_an / 1e9:.4f} GHz")
        for dx, steps in ((1e-3, 6000), (0.5e-3, 12000)):
            for arm, rule in (("edge-avg  ", _TRUE), ("cell-owned", _OWNED)):
                f = measure(dx, rule, steps, win)
                print(f"    dx={dx * 1e3:.2f}mm {arm} f={f / 1e9:.4f} GHz  "
                      f"rel={abs(f - f_an) / f_an:.3e}", flush=True)
