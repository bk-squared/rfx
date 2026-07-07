"""Exact Mie series oracle: RCS of a perfectly conducting (PEC) sphere.

Provides the monostatic (backscatter) RCS and the bistatic pattern used by
the committed sphere fixture (``fixture.json``) and its gate test
(``tests/test_rcs_mie_fixture.py``).

Convention (Bohren & Huffman, e^{-i w t}):
  Riccati-Bessel:  psi_n(x)  = x j_n(x),      psi_n'(x)  = j_n(x) + x j_n'(x)
                   xi_n(x)   = x h_n^(1)(x),  h_n^(1) = j_n + i y_n
  PEC (m -> inf) Mie coefficients:
                   a_n = psi_n'(x) / xi_n'(x)
                   b_n = psi_n(x)  / xi_n(x)
  Backscatter efficiency (B&H eq. 4.61/4.77):
                   sigma_b/(pi a^2) = (1/x^2) | sum_{n>=1} (2n+1)(-1)^n (a_n - b_n) |^2
  Bistatic amplitude functions (B&H eq. 4.74), scattering angle t
  (0 = forward, pi = backscatter):
      S1(t) = sum (2n+1)/(n(n+1)) [a_n pi_n(cos t) + b_n tau_n(cos t)]  ("H-plane")
      S2(t) = sum (2n+1)/(n(n+1)) [a_n tau_n(cos t) + b_n pi_n(cos t)]  ("E-plane")
      sigma(t)/(pi a^2) = (4/x^2) |S(t)|^2

The oracle is validated INDEPENDENTLY of rfx by ``validate_oracle()`` (four
witnesses, all pure assertions):
  (a) Rayleigh limit:      sigma/(pi a^2) -> 9 (ka)^4      as ka -> 0
  (b) geometric optics:    sigma/(pi a^2) -> 1              as ka -> inf
  (c) series convergence:  term-doubling changes the ka~1 value by < 1e-6
  (d) bistatic bridge:     (4/x^2)|S1(pi)|^2 == backscatter formula
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _riccati(n_max, x):
    """Return psi, psi', xi, xi' arrays for n = 1..n_max at scalar x."""
    n = np.arange(0, n_max + 1)
    jn = spherical_jn(n, x)                     # j_n(x)
    jnp_ = spherical_jn(n, x, derivative=True)  # j_n'(x)
    yn = spherical_yn(n, x)                     # y_n(x)
    ynp_ = spherical_yn(n, x, derivative=True)  # y_n'(x)

    hn = jn + 1j * yn        # h_n^(1)(x)
    hnp_ = jnp_ + 1j * ynp_  # h_n^(1)'(x)

    psi = x * jn             # psi_n = x j_n
    psip = jn + x * jnp_     # psi_n' = j_n + x j_n'
    xi = x * hn              # xi_n = x h_n^(1)
    xip = hn + x * hnp_      # xi_n' = h_n^(1) + x h_n^(1)'

    # drop n=0 (Mie sum starts at n=1)
    return psi[1:], psip[1:], xi[1:], xip[1:]


def mie_pec_coeffs(x, n_max):
    """PEC Mie coefficients a_n, b_n for n=1..n_max at size parameter x=ka."""
    psi, psip, xi, xip = _riccati(n_max, x)
    a_n = psip / xip
    b_n = psi / xi
    return a_n, b_n


def backscatter_rcs_over_pi_a2(x, n_max=None):
    """Normalized monostatic RCS sigma_b/(pi a^2) for a PEC sphere at x=ka.

    Wiscombe rule-of-thumb term count if n_max is None:
    n = x + 4.05 x^(1/3) + 2.
    """
    if n_max is None:
        n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 1
        n_max = max(n_max, 1)
    a_n, b_n = mie_pec_coeffs(x, n_max)
    n = np.arange(1, n_max + 1)
    terms = (2 * n + 1) * ((-1) ** n) * (a_n - b_n)
    S = np.sum(terms)
    return (1.0 / x ** 2) * np.abs(S) ** 2


def pi_tau(n_max, cos_t):
    """Angular functions pi_n, tau_n for n=1..n_max at scalar cos_t."""
    pin = np.zeros(n_max + 1)
    taun = np.zeros(n_max + 1)
    pin[0] = 0.0
    pin[1] = 1.0
    taun[1] = cos_t * pin[1]  # tau_1 = 1*cos*pi_1 - 2*pi_0 = cos
    for n in range(2, n_max + 1):
        pin[n] = ((2 * n - 1) / (n - 1)) * cos_t * pin[n - 1] \
            - (n / (n - 1)) * pin[n - 2]
        taun[n] = n * cos_t * pin[n] - (n + 1) * pin[n - 1]
    return pin[1:], taun[1:]


def mie_S1_S2(x, theta, n_max=20):
    """Far-field amplitude functions S1, S2 at scattering angle theta."""
    a_n, b_n = mie_pec_coeffs(x, n_max)
    n = np.arange(1, n_max + 1)
    fac = (2 * n + 1) / (n * (n + 1))
    pin, taun = pi_tau(n_max, np.cos(theta))
    S1 = np.sum(fac * (a_n * pin + b_n * taun))
    S2 = np.sum(fac * (a_n * taun + b_n * pin))
    return S1, S2


def bistatic_over_pi_a2(x, theta, plane, n_max=20):
    """Bistatic sigma(theta)/(pi a^2).

    plane='H' -> S1 (E perp scattering plane); plane='E' -> S2 (E in
    scattering plane). theta is the SCATTERING angle: 0 = forward,
    pi = backscatter.
    """
    S1, S2 = mie_S1_S2(x, theta, n_max)
    S = S1 if plane == "H" else S2
    return (4.0 / x ** 2) * np.abs(S) ** 2


def validate_oracle():
    """Self-check the oracle against four independent witnesses (asserts).

    Raises AssertionError on any failure; returns a dict of the measured
    witness values on success.
    """
    # (a) Rayleigh limit: sigma/(pi a^2) -> 9 (ka)^4 as ka -> 0
    ka = 0.1
    val = backscatter_rcs_over_pi_a2(ka, n_max=8)
    rayleigh = 9.0 * ka ** 4
    rel_a = abs(val - rayleigh) / rayleigh
    assert rel_a < 0.02, (
        f"Rayleigh witness failed at ka={ka}: computed {val:.6e} vs "
        f"9(ka)^4={rayleigh:.6e}, rel err {rel_a:.3e} >= 0.02"
    )

    # (b) geometric-optics limit: window-mean over ka in [18, 22] -> 1
    n_terms = int(np.ceil(22.0 + 4.05 * 22.0 ** (1 / 3) + 2)) + 5
    kas = np.linspace(18, 22, 41)
    go_mean = float(np.mean(
        [backscatter_rcs_over_pi_a2(k, n_max=n_terms) for k in kas]
    ))
    assert abs(go_mean - 1.0) < 0.1, (
        f"GO witness failed: window-mean sigma/(pi a^2) over ka in [18,22] "
        f"= {go_mean:.4f}, expected 1.0 +/- 0.1"
    )

    # (c) series convergence at ka ~ 1: doubling n_max changes < 1e-6
    conv = {}
    for ka_c in (0.9997, 1.0):
        v1 = backscatter_rcs_over_pi_a2(ka_c, n_max=10)
        v2 = backscatter_rcs_over_pi_a2(ka_c, n_max=20)
        dd = abs(v2 - v1)
        assert dd < 1e-6, (
            f"Convergence witness failed at ka={ka_c}: "
            f"|n_max 10 -> 20 change| = {dd:.3e} >= 1e-6"
        )
        conv[ka_c] = dd

    # (d) bistatic-formula bridge: (4/x^2)|S(pi)|^2 == backscatter formula
    x = 0.9997
    back_formula = backscatter_rcs_over_pi_a2(x, n_max=20)
    back_bi_H = bistatic_over_pi_a2(x, np.pi, "H")
    back_bi_E = bistatic_over_pi_a2(x, np.pi, "E")
    assert np.allclose([back_bi_H, back_bi_E], back_formula, rtol=1e-9), (
        f"Bistatic bridge witness failed at ka={x}: backscatter formula "
        f"{back_formula:.9f} vs S1(pi) {back_bi_H:.9f} / S2(pi) {back_bi_E:.9f}"
    )

    return {
        "rayleigh_rel_err": rel_a,
        "go_window_mean": go_mean,
        "convergence_abs_change": conv,
        "bistatic_bridge_value": back_formula,
    }


if __name__ == "__main__":
    witnesses = validate_oracle()
    print("Mie PEC-sphere oracle — all witnesses PASS")
    for k, v in witnesses.items():
        print(f"  {k}: {v}")
