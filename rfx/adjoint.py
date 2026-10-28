"""Settled-spectrum reciprocity adjoint of the uniform production scan (#1424).

Discrete overlap
----------------

Use the production DFT convention `F = dt sum_n f[n] exp(-i w n dt)`.
E is sampled after the update, at `(n+1)dt`; the design hook observes
`E[n]` and `curl H[n+1/2]`. For each bin, eliminating H gives a symmetric
uniform interior electric operator L with `L E = q/Cb`, where q is an
additive E increment with the post-update timestamp. Reciprocity gives
`dY_m = sum_j L^-1[m,j] (dCa_j Epre_j + dCb_j curlH_j)/Cb_j`.
For JAX's complex cotangent b, `dJ = Re sum_m b_m dY_m` (no conjugation).
An ordinary second forward run with `DFT(q_m)=Cb_m b_m` produces A on the
design edges. Therefore

dJ/dCa_j = Re sum_bins A_j Epre_j / Cb_j
dJ/dCb_j = Re sum_bins A_j curlH_j / Cb_j.

Both field factors include dt; the source DFT targets b, not b*dt.
JAX differentiates the production edge averaging and Ca/Cb arithmetic
outside the custom VJP. Public adjoint admission refuses every supplied
`design_sigma_override` (#1424: on lossless design cells the conductivity gradient has no settled value in a finite record; static sensitivity remains although the field decays, and this adjoint returns settled-spectrum gradients only).
Fixed material conductivity remains allowed; its dCa/deps reaches gCa.
Finite record endpoint terms are omitted: the result is a settled-spectrum
gradient. No time tape or transposed time sweep is part of F2.

Injection and wavelets
----------------------

Inject the monitor's E component at its exact Yee array index, after the
design update and before the ordinary source/monitor stage. A point is a
single selected pixel of an E DFT plane; a plane injects each of its pixels.
Interior uniform dual volumes are V=dx^3: reciprocal density sources use
b/V and the overlap uses V, so these factors cancel. CPML and boundary
monitor pixels are excluded from admission. The half-step current spectrum
is `exp(+i w dt/2)` times the post-update increment spectrum; equivalently
q uses the E timestamp and no additional phase. Epre uses the same timestamp
as q, giving its explicit one-step delay relative to post-update E.

For each monitor use compact early real wavelets formed from positive and
negative complex exponentials with a smooth envelope. Take the compact
discrete difference of each carrier to enforce zero deposited DC. Their sampled
DFTs form A (positive carriers) and B (negative carriers). Solve
`(A-B conj(A)^-1 conj(B)) c = b-B conj(A)^-1 conj(b)` and inject
`2 Re sum_k c_k difference(envelope exp(+i w_k t))`. These are Nf-by-Nf solves;
including the conjugate carriers makes arbitrary complex targets reachable
with real fields. Monitors must share identical bins. Reject empty, duplicate,
DC/Nyquist/out-of-band bins.
The remaining run lets the wavelet response decay. Storage is design
edges × components × bins, plus monitor bins and ordinary field carries.

Gates, measurements and the refusal table: issue #1424 and PR #1430.
"""
from rfx.measurement.wavelet import _wavelet_basis, _wavelet_coefficients


def admit_forward_adjoint(sim, *, distributed, ringdown, design_box, design_eps,
                          other_overrides, port_s11_freqs, design_sigma=None):
    """Refuse inputs outside the first implementation before lane dispatch."""
    if design_sigma is not None:
        raise NotImplementedError(
            "gradient='adjoint' refuses design_sigma_override (#1424): "
            "on lossless design cells the gradient with respect to conductivity has no "
            "settled value in a finite record (a static sensitivity remains although "
            "the field decays), and this adjoint returns settled-spectrum gradients only")
    reason = None
    if distributed:
        reason = "distributed execution"
    elif any(p is not None for p in
             (sim._dx_profile, sim._dy_profile, sim._dz_profile)):
        reason = "graded meshes"
    elif ringdown is not None:
        reason = "ring-down"
    elif (any(p.impedance != 0 or p.extent is not None for p in sim._ports)
          or any(getattr(sim, p) for p in
                 ("_waveguide_ports", "_msl_ports", "_coaxial_ports",
                  "_floquet_ports", "_lumped_rlc"))
          or port_s11_freqs is not None):
        reason = "ports and lumped circuits"
    elif sim._tfsf is not None:
        reason = "TFSF sources"
    elif any(m.debye_poles or m.lorentz_poles for m in sim._materials.values()):
        reason = "Debye/Lorentz materials"
    elif any(m.chi3 != 0 for m in sim._materials.values()):
        reason = "Kerr materials"
    elif sim._current_moments is not None:
        reason = "current moments"
    elif sim._solver != "yee":
        reason = "non-Yee solvers"
    elif any(v is not None for v in other_overrides):
        reason = "overrides other than design eps"
    elif design_box is None or design_eps is None:
        reason = "calls without design_box and design_eps_override"
    if reason is not None:
        raise NotImplementedError(f"gradient='adjoint' does not support {reason}")


# Lazy compatibility exports avoid a cycle with the moved implementation.
_LAZY_EXPORTS = {
    "_plane_slice": "rfx.stepping.adjoint_scan",
    "design_adjoint_scan": "rfx.stepping.adjoint_scan",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value
