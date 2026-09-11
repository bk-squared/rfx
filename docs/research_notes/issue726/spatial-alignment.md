# A separate uniform-lane sampling defect identified during #726

Status: source diagnosis and an analytic falsifier. Numerical correction is
pending the user's choice. The paired VESSL run has not been restarted.

The Yee layout deliberately places Ez at (x_i, y_j, z_(k+1/2)), Hy at
(x_(i+1/2), y_j, z_(k+1/2)), and Hz at
(x_(i+1/2), y_(j+1/2), z_k). In the uniform MSL route the driver registers
the voltage and both current-plane components using the same pxs[0]. The
runner maps each to the same integer propagation index without a
component-specific adjustment, and the scan directly accumulates
field[index,...]. The subsequent V/I assembly applies the half-time-step
phase correction, but no longitudinal spatial interpolation or alignment.

Therefore the voltage and current entering the split are not measured at
the same physical x plane. Reversing the port direction does not reverse
the underlying Yee offset. The y-propagating coordinate permutation has the
corresponding offset in y. The concrete source chain is:

- `rfx/api/_sparams.py`: Ez and the two H DFT registrations at pxs[0];
  time-phase correction and wave assembly in `compute_msl_s_matrix`.
- `rfx/runners/uniform.py`: component-independent conversion of the DFT
  coordinate to a grid index.
- `rfx/simulation.py`: direct DFT accumulation of each field[index,...].
- `rfx/core/yee.py`: forward propagation-axis differences in the H update.
- `rfx/geometry/smoothing.py`: the E-node coordinate convention.

This is not malformed user geometry for preflight to reject. It is a
sampling/observable question inside the extraction route. The current API
reference plane is nominally the E-probe plane; it does not establish
spatially co-located V/I.

For a matched forward wave, after correcting temporal staggering, set V=1
at x=0 and I_right=exp(-j k dx/2)/Zc at the current's sample. With Zref=Zc,
the current split returns

    Gamma_same_index = j tan(k dx/4),

rather than zero. Averaging the two bracketing H/current samples at
x=+/-dx/2 gives

    Gamma_centred = tan^2(k dx/4).

The proposed operation thus removes the first-order spurious reflection,
leaving a second-order interpolation error in this smooth single-mode
example. `yee_spatial_stagger.py` reproduces the mesh-halving rates without
FDTD or changes to any production gate. A nonuniform extension would need
actual bracketing H coordinates and interpolation weights, not a blanket
arithmetic mean; both run and forward routes would need verification.

Crucial limit: for one mode, real positive Zc and resolved k dx,

    Re(V I_right*) = cos(k dx/2) (|a|^2 - |b|^2) / Zc.

Thus this staggering alone does not prove the cause of the measured power
excess. In particular, an ideal perfect reflector still gives unit reflection
magnitude. No RF error magnitude or causal attribution is claimed here.
