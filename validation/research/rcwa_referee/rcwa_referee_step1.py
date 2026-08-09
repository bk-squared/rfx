"""RCWA referee, step 1 of #491: wire grcwa and prove it against its own
canonical example and against closed-form physics. NO rfx comparison here
(step 2 is blocked on the #265 drive-isolation question; see README.md).

Comparator-first build rule (workspace research/CLAUDE.md): an external
reference implementation must first reproduce its own tool's documented
known-good result, recorded here BEFORE any geometry/config change.

TOOL
  grcwa 0.1.2 (PyPI, pure numpy/autograd RCWA, Weiliang Jin,
  https://github.com/weiliangjinca/grcwa), installed into the rfx venv via
  `uv pip install --python .venv/bin/python grcwa` (see INSTALL.md).
  Conventions: c = 1, time dependence exp(-i omega t), freq in units of c/L.

REPRODUCE-GATE ARTIFACTS (recorded 2026-08-09, this checkout)
  Source example : tests/test_rcwa.py::test_rcwa from the grcwa 0.1.2 sdist
                   (https://files.pythonhosted.org/packages/a9/4d/d79d7dfcf73bb402890bc573ad482036d524f39d1868383da7c4a47c2a4b/grcwa-0.1.2.tar.gz)
                   -- vacuum / patterned eps=12 circular-pillar layer
                   (r=0.4, thick 0.2, period 0.1x0.1, nG=101 requested ->
                   101 actual, 100x100 grid) / vacuum, freq=1,
                   theta=pi/18, phi=pi/9.
  Expected number: T_p = 0.85249901083265 and T_s = 0.83900479939861,
                   published in the shipped test as an S4 cross-check with
                   rel tolerance 1e-3 (`tolS4`).
  Reproduced     : full shipped suite (test_rcwa.py + test_kbloch.py):
                   10 passed in 8.51s;
                   in-script rerun (Section A below, installed package):
                   T_p = 0.85317304967237 (rel err 7.91e-4),
                   T_s = 0.83965428139193 (rel err 7.74e-4) -- inside the
                   shipped 1e-3 gate.
  Log paths      : logs/20260809_grcwa012_shipped_tests.log   (pytest run)
                   logs/20260809_rcwa_referee_step1_run.log   (this script)

DELTAS FROM THE CANONICAL EXAMPLE
  Section A: none -- geometry, truncation, angles, excitation are verbatim
             from the shipped test (only re-expressed as a function).
  Sections B/C: new geometries, each gated by an oracle INDEPENDENT of
             grcwa: closed-form Fresnel/Airy (computed in this file) and
             lossless energy conservation.

HAND-PORTED SANITY CHECKS (external scripts get NO rfx preflight; the
rfx-side equivalents are the preflight excitation/energy checks):
  - nonzero excitation reaches the detector (T > 0 asserted everywhere);
  - R, T within [0, 1+eps] (passivity analogue);
  - lossless energy conservation |R+T-1| as a witness on every run;
  - propagating-order count printed, and nonzero diffracted (|m|>=1)
    power asserted in Section C so the Fourier machinery is provably
    exercised (a homogeneous slab does not exercise it).

Run:  .venv/bin/python validation/research/rcwa_referee/rcwa_referee_step1.py
Exits nonzero on any gate failure.
"""

import sys

import numpy as np

import grcwa

FAILURES = []


def check(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Section A -- reproduce-gate: grcwa's own canonical shipped test, verbatim
# geometry (tests/test_rcwa.py::test_rcwa of the 0.1.2 sdist).
# --------------------------------------------------------------------------
def section_a():
    print("== Section A: reproduce-gate (grcwa 0.1.2 tests/test_rcwa.py::test_rcwa) ==")
    print(f"  grcwa imported from: {grcwa.__file__}")

    nG = 101
    L1, L2 = [0.1, 0.0], [0.0, 0.1]
    Nx = Ny = 100
    freq, theta, phi = 1.0, np.pi / 18, np.pi / 9
    pthick = 0.2

    epgrid = np.ones((Nx, Ny), dtype=float)
    x0 = np.linspace(0.0, 1.0, Nx)
    y0 = np.linspace(0.0, 1.0, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    epgrid[(x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.4**2] = 12.0

    def run(p_amp, s_amp):
        obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(pthick, Nx, Ny)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Init_Setup(Pscale=1.0, Gmethod=0)
        obj.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
        obj.GridLayer_geteps(epgrid.flatten())
        return obj.RT_Solve(normalize=0)

    tolS4 = 1e-3  # the tolerance the shipped test itself uses vs S4
    expected = {"p": 0.85249901083265, "s": 0.83900479939861}
    for pol, (p, s) in {"p": (1, 0), "s": (0, 1)}.items():
        R, T = run(p, s)
        rel = abs(T - expected[pol]) / expected[pol]
        check(
            f"canonical T_{pol} vs published S4 number",
            rel < tolS4 and T > 0,
            f"T={T:.14f} expected={expected[pol]} rel_err={rel:.3e} (gate {tolS4})",
        )


# --------------------------------------------------------------------------
# Section B -- analytic bridge: homogeneous slab at normal incidence vs
# closed-form Fresnel/Airy. RCWA is exact for uniform layers, so agreement
# must be near machine precision.
# --------------------------------------------------------------------------
def airy_slab(eps_slab, d, freq, eps_sub=1.0):
    """Closed-form R,T for vacuum / slab(eps_slab, thickness d) / eps_sub,
    normal incidence, wavelength 1/freq (c=1). Lossless => T = 1-R."""
    n1, n2, n3 = 1.0, np.sqrt(eps_slab), np.sqrt(eps_sub)
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    ph = np.exp(2j * (2 * np.pi * freq * n2) * d)
    r = (r12 + r23 * ph) / (1 + r12 * r23 * ph)
    R = abs(r) ** 2
    return R, 1.0 - R


def slab_rcwa(eps_slab, d, freq, eps_sub=1.0, nG=11):
    obj = grcwa.obj(nG, [0.1, 0.0], [0.0, 0.1], freq, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Add_LayerUniform(d, eps_slab)
    obj.Add_LayerUniform(1.0, eps_sub)
    obj.Init_Setup()
    obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    return obj.RT_Solve(normalize=1)  # normalize: output medium may be eps_sub


def section_b():
    print("== Section B: homogeneous eps_r=4 slab vs closed-form Fresnel/Airy ==")
    worst = 0.0
    for freq in (0.7, 1.0, 1.3):
        for d in (0.1, 0.125, 0.25, 0.3):
            R, T = slab_rcwa(4.0, d, freq)
            Ra, Ta = airy_slab(4.0, d, freq)
            worst = max(worst, abs(R - Ra), abs(T - Ta))
            note = ""
            if freq == 1.0 and d == 0.25:
                note = "  <- half-wave null (R=0 exactly)"
                check("half-wave null R=0", R < 1e-14, f"R={R:.3e}")
            print(
                f"    freq={freq} d={d}: R_rcwa={R:.15f} R_airy={Ra:.15f} "
                f"|dR|={abs(R - Ra):.2e} |R+T-1|={abs(R + T - 1):.2e}{note}"
            )
            check(f"passivity/energy f={freq} d={d}", 0 <= R <= 1 + 1e-12 and T > 0 and abs(R + T - 1) < 1e-12, f"R={R:.6f} T={T:.6f}")
    # quarter-wave antireflection null: eps=4 coating (n=2 = sqrt(n0*ns)) on
    # eps=16 substrate, d = lambda/(4 n2) = 0.125 at freq=1 -> R = 0 exactly.
    R, T = slab_rcwa(4.0, 0.125, 1.0, eps_sub=16.0)
    Ra, Ta = airy_slab(4.0, 0.125, 1.0, eps_sub=16.0)
    worst = max(worst, abs(R - Ra), abs(T - Ta))
    print(f"    quarter-wave AR on eps=16: R_rcwa={R:.3e} R_airy={Ra:.3e} T_rcwa={T:.15f}")
    check("quarter-wave AR null R=0", R < 1e-14 and abs(T - Ta) < 1e-12, f"R={R:.3e} |dT|={abs(T - Ta):.2e}")
    check("max |R_rcwa - R_analytic| near machine precision", worst < 1e-12, f"max|delta|={worst:.3e} (gate 1e-12)")
    return worst


# --------------------------------------------------------------------------
# Section C -- periodic sanity: lossless lamellar grating, sum of diffracted
# order powers must equal 1, and |m|>=1 orders must carry real power
# (homogeneous slabs never exercise the Fourier factorization).
# --------------------------------------------------------------------------
def section_c():
    print("== Section C: lamellar grating energy conservation (lossless) ==")
    nG_req, theta, freq = 101, np.pi / 18, 1.0
    period, fill, eps_hi, thick = 1.5, 0.5, 12.0, 0.3
    Nx, Ny = 1024, 16
    epgrid = np.ones((Nx, Ny))
    xc = (np.arange(Nx) + 0.5) / Nx
    epgrid[xc < fill, :] = eps_hi

    for pol, (p, s) in {"p": (1, 0), "s": (0, 1)}.items():
        obj = grcwa.obj(nG_req, [period, 0.0], [0.0, 0.1], freq, theta, 0.0, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(thick, Nx, Ny)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Init_Setup()
        obj.MakeExcitationPlanewave(p, 0.0, s, 0.0, order=0)
        obj.GridLayer_geteps(epgrid.flatten())
        Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
        R, T = float(np.sum(Ri)), float(np.sum(Ti))

        kx, ky = np.array(obj.kx), np.array(obj.ky)
        om = 2 * np.pi * freq
        prop = kx**2 + ky**2 < om**2
        G = np.array(obj.G)
        diffracted = float(np.sum(np.abs(Ri[(G[:, 0] != 0) & prop])) + np.sum(np.abs(Ti[(G[:, 0] != 0) & prop])))
        print(f"    pol={pol}: nG={obj.nG} propagating orders={int(prop.sum())} R={R:.12f} T={T:.12f}")
        for m in np.unique(G[prop, 0]):
            sel = (G[:, 0] == m) & prop
            print(f"      order m={m:+d}: R_m={float(np.sum(Ri[sel])):.12f} T_m={float(np.sum(Ti[sel])):.12f}")
        check(f"energy conservation ({pol}-pol)", abs(R + T - 1) < 1e-10 and T > 0, f"|R+T-1|={abs(R + T - 1):.3e} (gate 1e-10)")
        check(f"Fourier machinery exercised ({pol}-pol)", diffracted > 1e-3, f"power in |m|>=1 propagating orders = {diffracted:.6f}")


def main():
    print(f"grcwa referee step 1 -- numpy {np.__version__}, python {sys.version.split()[0]}")
    section_a()
    section_b()
    section_c()
    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} gate(s)): {FAILURES}")
        return 1
    print("RESULT: ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
