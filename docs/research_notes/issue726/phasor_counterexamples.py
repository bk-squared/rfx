"""Exact single-mode algebra, not FDTD or a calibration of the RF fixture."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64
import numpy as np

from rfx.api._sparams import _msl_wave_split_reliability, msl_solve_s_from_waves


def main():
    # Real modal impedance Zc, distinct real reference impedance Zr. All
    # fields are the sum of ONLY forward/backward waves of this one mode.
    zc, zr = 60.0, 50.0
    beta_x = np.array([0.0, np.pi / 2])
    a_mode = np.exp(-1j * beta_x)
    b_mode = 0.6 * np.exp(1j * beta_x)
    v = a_mode + b_mode
    i = (a_mode - b_mode) / zc
    a_ref, b_ref = (v + zr * i) / 2, (v - zr * i) / 2
    with enable_x64():
        s_ref, _ = msl_solve_s_from_waves([[a_ref]], [[b_ref]])
    gamma_c = b_mode / a_mode
    mismatch = (zc - zr) / (zc + zr)
    expected = (mismatch + gamma_c) / (1 + mismatch * gamma_c)
    full_reflection = np.exp(2j * beta_x)
    full_ref = (mismatch + full_reflection) / (1 + mismatch * full_reflection)
    np.testing.assert_allclose(abs(full_ref), 1.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(s_ref)[0, 0], expected, rtol=1e-13, atol=1e-13)

    # A reciprocal, lossless two-port with a perfect transmission zero in
    # the middle bin. S*S^H=I; independently matched drives give A=I.
    # Neither a small incident drive nor a singular wave solve occurs.
    f = np.array([3.0e9, 3.7e9, 4.4e9])
    reflection = np.array([0.3, 1.0, 0.3])
    transmission = 1j * np.sqrt(1 - reflection**2)
    planted = np.array([[reflection, transmission], [transmission, reflection]])
    a = np.repeat(np.eye(2, dtype=complex)[:, :, None], len(f), axis=2)
    b = planted.copy()  # B=S*A=S; leading axes are (port, drive)
    v_all = (a + b).transpose(1, 0, 2).reshape(4, len(f))
    i_all = ((a - b) / zr).transpose(1, 0, 2).reshape(4, len(f))
    mask = _msl_wave_split_reliability(v_all, i_all, f).reshape(2, 2, -1).all(axis=0)
    with enable_x64():
        solved, cond = msl_solve_s_from_waves(
            [[a[p, d] for p in range(2)] for d in range(2)],
            [[b[p, d] for p in range(2)] for d in range(2)],
        )
    solved = np.asarray(solved)
    identity_error = max(float(np.max(abs(
        planted[:, :, k] @ planted[:, :, k].conj().T - np.eye(2)))) for k in range(len(f)))
    np.testing.assert_allclose(solved, planted, rtol=1e-13, atol=1e-13)
    assert identity_error < 1e-13
    output = dict(
        scope="single-mode analytic phasors; no fields evolved; no physical gate changed",
        reference_impedance=dict(
            zc_ohm=zc, zr_ohm=zr, beta_x_rad=beta_x.tolist(),
            modal_reflection_magnitude=abs(gamma_c).tolist(),
            reported_reflection_magnitude=abs(np.asarray(s_ref)[0, 0]).tolist(),
            reported_reflection_db=(20 * np.log10(abs(np.asarray(s_ref)[0, 0]))).tolist(),
            full_reflection_magnitude=abs(full_ref).tolist(),
            limitation="real positive impedance mismatch preserves unit magnitude for a perfect reflector",
        ),
        exact_transmission_zero=dict(
            freqs_hz=f.tolist(), current_reliable_mask=mask.tolist(),
            cond_a=np.asarray(cond).tolist(), max_s_error=float(np.max(abs(solved - planted))),
            unitarity_error=identity_error,
            interpretation="low signal alone does not establish matrix corruption; this does not certify noisy FDTD data",
        ),
    )
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
