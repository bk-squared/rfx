# MSL calculation contracts: established facts and open questions

Audited against main87927064 and the evidence branch. This is a calculation
review, not a new support declaration. The original MSL first-probe S remains
the target observable; a source-model response is a separate diagnostic.

| Contract | Code / direct check | Status and consequence |
|---|---|---|
| The declared trace-width core spans actual ground-to-trace edges | `msl_cross_section_span`, conductor anchor checks; PR981 | Fixed. The normal span excludes the upper bounding node's outgoing edge. Wrong conductor heights are rejected. This does not require the Laplace profile's lateral fringe columns to contact the trace. |
| V and I refer to the same plane and time | `msl_h_plane_stencil`, collocation and the retained temporal correction; PR986 | Spatial mismatch fixed. Actual periodic Yee traveling/standing waves verify the temporal sign in f32/f64. This is not a general finite-window error bound. |
| S uses one power metric with known references | `_msl_power_wave_scales`, full A/B construction; PR987 | Fixed for positive real references. Fitted Z0 and load R are not substituted for the analytic S references. |
| The electric source/load update respects work | `setup_msl_port` → `make_msl_port_sources` → `update_e` → source addition | Direct electric-substep test closes source work, storage change and ohmic loss to roundoff. No negative load or duplicate fold found in CV06b. |
| A stated R is a matched scalar load for arbitrary fields | scalar sigma is normalized against the supplied Ez profile but applied to Ex/Ey/Ez | Not established. The R identity holds on that profile; additional field components dissipate additional nonnegative power. |
| The waveform parameter is the stated Thevenin voltage | actual added force is e*u, without sigma_port multiplying u | The voltage wording does not match the operation. U=R*N*u for the shaped source. This scalar drive-amplitude distinction alone cannot explain S gain in linear B A^-1. |
| Source work is represented by probe-0 centre-line V and H contour I | source-conjugate voltage is a weighted source-region midpoint projection | Not established. They are different functionals at different locations. The667 source-work observation measures the distinction without changing either source or MSL S; its two-channel sampled response satisfies the passive bound while the first-plane response does not. |
| The Laplace source is a complete single propagating mode | current default creates Ez sources and no magnetic source; source support is cropped laterally | Not established. Naming the profile a mode does not prove a full Maxwell mode or purely modal illumination. The retired eigenmode/J+M path is blocked and must not simply be enabled. |
| Extra power is generated inside the passive DUT box | independent six-face field Gram on633 | Not supported by these records. Unrepresented lateral incoming flux largely balances the apparent two-port excess. Finite-window residuals are not globally bounded. |
| Missing PEC continuation alone resolves the power defect | isolated647 continuation comparison | Refuted as a complete fix at the recorded duration: local improvement, worse full-band gain and poorer power closure. No production fixture change adopted. |
| Source enters CPML or normalization includes unexecuted cells | actual source/load support and profile loops | Refuted for CV06b. All72 cells per port are interior and included by both loops.32 fringe cells lack an upper trace contact;11.59% is a profile norm fraction, not an RF power fraction. |

For the shaped source, let e be its supplied Ez profile and
N=sum(volume*e^2). The source builder and electric update imply

    epsilon*(E_next-E_old)/dt + sigma_total*E_mid = curl(H) + e*u
    E_mid = (E_next+E_old)/2
    sigma_port = 1/(R*N)
    Q = sum(volume*e*Ez_mid)
    P_source = u*Q
    V_proj = Q/N
    I_source = N*u

The uniform electric-substep check includes all three E components in the
load loss. The supplied-profile contribution is V_proj^2/R; Cauchy–Schwarz
makes the remaining load dissipation nonnegative. This yields a distinct
source-model work port. It does not equate V_proj to a centre-line voltage
at a different cross-section and does not certify first-plane MSL S.

For canonical power waves a=(V+R*I)/(2*sqrt(R)) and positive real S references, the unprojected V/I wave algebra gives

    A.H*A - B.H*B = (V.H*I + I.H*V)/2
    identity - S.H*S = A^-H * ((V.H*I + I.H*V)/2) * A^-1

Consequently an invertible drive solve preserves the inertia of the V/I
power form under a change of positive real references. Refitting the
analytic reference impedance cannot repair a negative power direction.
Neither can passivity clipping establish that the recorded V/I variables
represent the intended electromagnetic ports.

The667 field measurement follows the explicit source-work
identity above. It retains the633 source/DUT, records exact post-source
Ez point samples and base waveforms, forms midpoint variables, and reports
source-model S beside unchanged first-plane MSL S. A passive source-model
answer must not be substituted for the desired MSL answer. Any later
change to the port's physical meaning or reference plane requires a
separate decision.

Release placement follows ROADMAP: the published1.8 work package focuses
on waveguide and lumped/wire; MSL+#726 belongs to the former1.9 work package.
Current main includes the breaking931 realization contract and is documented
under Unreleased2.0.0. The version number and the all-family scientific
chain-closure milestone are distinct. Confirmed correctness defects should
be repaired before their affected capabilities are advertised; a broad
new mode-port capability cannot be presumed complete from these diagnostics.

The v1.8.0 source already contains the uncollocated H read, unequal-reference
voltage waves, and the explicitly noted non-continuation of boundary-touching
PEC traces. These are not newly introduced by931. The geometry migration does
change thin-conductor realizations and invalidates old fixture assumptions,
which is a separate, substantial source of requalification work.


Explicit user constraint: existing port autodiff must remain intact. Any
production repair must preserve currently differentiable port/input paths,
run/forward primal agreement, finite meaningful nonzero sensitivities and
agreement with an independent finite-difference referee. An RF-only result
cannot justify detaching the tape, silently freezing a design-dependent
quantity, or dropping existing AD support. Source-work observers here are
analysis tools only and do not replace the production differentiable path.

The existing MSL material derivative is of a fixed-launch function:
`F(eps_override; registered_model)`. Grid/geometry, the registered waveform,
Laplace profile, added port conductivity and analytic reference impedance
stay fixed. The electric source increment is nevertheless
`Cb(eps_override, sigma_total) * e * u`: **Cb must remain differentiable**.
The source builder reads overridden material coefficients for that factor.
Re-solving a material-dependent profile in finite differences while freezing
it only under AD would compare different functions (the earlier #483 defect).
Run/forward primal parity must compare the same raw observable; the default
concrete run can apply passivity projection while the override path omits it.

`tests/unit/ports/test_msl_source_work.py` now checks the actual source/load
and electric substep for uniform and prescribed nonuniform profiles, four
propagation directions and float32/float64. The independent midpoint equation
checks electric storage, source work and all three components' positive load
loss. JIT reverse AD with respect to permittivity is compared with float64
finite differences of that independent equation. These 16 local checks do
not replace the existing full-run AD–FD qualification or establish RF accuracy.
The [in-memory falsifiers](source-work-gate-falsifiers.json), reproducible with
`verify_source_work_gate.py --out <new-file>`, reject detaching Cb, an extra
conductivity multiplier, a reversed force and a source outside the declared
support. The production functions are never edited by this verifier.
