"""Single-mode spatial-stagger counterexample; no FDTD or gate changes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main():
    zc = 50.0
    rows = []
    for kh in (0.2, 0.1, 0.05):
        # V at x=0; the stored H/current at the SAME Yee index is at h/2.
        v = 1.0 + 0j
        i_right = np.exp(-0.5j * kh) / zc
        i_left = np.exp(0.5j * kh) / zc
        old_gamma = (v - zc * i_right) / (v + zc * i_right)
        centred_i = (i_left + i_right) / 2
        centred_gamma = (v - zc * centred_i) / (v + zc * centred_i)
        np.testing.assert_allclose(old_gamma, 1j * np.tan(kh / 4), atol=1e-14)
        np.testing.assert_allclose(centred_gamma, np.tan(kh / 4)**2, atol=1e-14)
        rows.append(dict(k_dx=kh, same_index_reflection=abs(old_gamma),
                         centred_reflection=abs(centred_gamma)))

    # Do not attribute every passivity excess to this staggering. For a
    # single mode and real positive Zc, the cross terms in V I* are imaginary.
    checks = []
    for gamma in (0j, 0.4 * np.exp(0.7j), np.exp(0.7j)):
        kh = 0.2
        v = 1 + gamma
        i_right = (np.exp(-0.5j * kh) - gamma * np.exp(0.5j * kh)) / zc
        real_vi = float(np.real(v * np.conj(i_right)))
        expected = float(np.cos(kh / 2) * (1 - abs(gamma)**2) / zc)
        np.testing.assert_allclose(real_vi, expected, atol=1e-15)
        g = (v - zc * i_right) / (v + zc * i_right)
        checks.append(dict(modal_reflection_magnitude=float(abs(gamma)),
                           real_vi=real_vi, same_index_reflection=float(abs(g))))
    output = dict(
        scope='analytic single mode only; H interpolation is a proposed correction, not applied to production',
        matched_wave=rows, passivity_limitation=checks,
        conclusion='same-index sampling gives O(dx) false reflection for a matched wave; symmetric H averaging gives O(dx^2). This alone does not explain the observed RF power excess.',
    )
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
