# Measured tables

TE air / lossless epsilon_r = 4 slab / air; declared slab thickness 7.49481145 mm; f0 = 5 GHz.
Native public S11 and phase: absent (`s_params=None`, `freqs=None`).
The following are diagnostic `extract_floquet_modes` values from additional plane averages retained by an observer, outside the public run.
The observer uses the same-time E/H samples and -j DFT kernel of `update_floquet_dft`; no Yee spatial/time staggering correction is added.
Source plane z = 14.9896229 mm; ahead plane z = 22.48443435 mm; slab front z = 37.47405725 mm.
Analytic phase: exp(+jwt), air reference at the named plane, with exp(-2j*k0*cos(theta)*distance) translation.
Analytic slab formula: r01*(1-exp(-2j*kz1*d))/(1-r01^2*exp(-2j*kz1*d)); r01=(cos(theta)-sqrt(4-sin(theta)^2))/(cos(theta)+sqrt(4-sin(theta)^2)).
17 bins span the full declared 4–6 GHz port band; CSV retains every numerical value and signed magnitude/phase difference.

## full_theta0_r1

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.607186 | -3.904081 | 0.580703 | -21.429232 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.620398 | -86.782516 | 0.600000 | -90.000000 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.468017 | -152.201111 | 0.580703 | -158.570768 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.634759 | 66.199821 | 0.580703 | 50.570768 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.658038 | 6.712728 | 0.600000 | 0.000000 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.466804 | -34.199909 | 0.580703 | -50.570768 | 0.580703 | -50.570768 |

## full_theta15_r1

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.596154 | -4.060193 | 0.595942 | -13.794961 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.620005 | -88.998276 | 0.616486 | -80.204110 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.479969 | -153.374420 | 0.599210 | -146.569130 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.630733 | 68.226204 | 0.595942 | 55.751699 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.648169 | 6.971133 | 0.616486 | 6.729214 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.455617 | -35.695686 | 0.599210 | -42.249141 | 0.580703 | -50.570768 |

## full_theta30_r1

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.559929 | -4.604838 | 0.642719 | 8.342084 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.621461 | -95.968994 | 0.666206 | -51.695940 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.517148 | -156.677612 | 0.654496 | -111.556966 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.620194 | 74.821175 | 0.642719 | 70.695914 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.615672 | 7.864203 | 0.666206 | 26.246347 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.421009 | -40.966656 | 0.654496 | -18.026223 | 0.580703 | -50.570768 |

## full_theta45_r1

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.486744 | -5.892331 | 0.722967 | 42.694958 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.634532 | -108.552330 | 0.748306 | -7.066655 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.583130 | -161.560013 | 0.743849 | -56.445664 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.610577 | 87.664116 | 0.722967 | 93.606647 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.549628 | 9.919396 | 0.748306 | 56.572955 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.362531 | -53.492264 | 0.743849 | 19.921868 | 0.580703 | -50.570768 |

## full_theta0_r2

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.575027 | -7.605233 | 0.580703 | -21.429232 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.655886 | -88.524048 | 0.600000 | -90.000000 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.487644 | -161.292847 | 0.580703 | -158.570768 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.589387 | 63.467449 | 0.580703 | 50.570768 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.674004 | 3.207066 | 0.600000 | 0.000000 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.480843 | -48.563698 | 0.580703 | -50.570768 | 0.580703 | -50.570768 |

## full_theta30_r2

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.525446 | -9.012032 | 0.642719 | 8.342084 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.658032 | -97.485008 | 0.666206 | -51.695940 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.538243 | -164.288666 | 0.654496 | -111.556966 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.571388 | 72.252136 | 0.642719 | 70.695914 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.632910 | 3.752132 | 0.666206 | 26.246347 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.446712 | -57.161587 | 0.654496 | -18.026223 | 0.580703 | -50.570768 |

## full_theta0_r4

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.557750 | -9.980189 | 0.580703 | -21.429232 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.669118 | -89.203987 | 0.600000 | -90.000000 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.511006 | -164.857590 | 0.580703 | -158.570768 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.565050 | 61.560562 | 0.580703 | 50.570768 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.678214 | 1.632561 | 0.600000 | 0.000000 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.507155 | -54.606640 | 0.580703 | -50.570768 | 0.580703 | -50.570768 |

## full_theta30_r4

| Plane | GHz | diagnostic magnitude | phase deg | analytic at reported angle magnitude | phase deg | analytic normal magnitude | phase deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_plane | 4 | 0.507188 | -11.857625 | 0.642719 | 8.342084 | 0.580703 | -21.429232 |
| source_plane | 5 | 0.671639 | -98.088882 | 0.666206 | -51.695940 | 0.600000 | -90.000000 |
| source_plane | 6 | 0.560730 | -167.238953 | 0.654496 | -111.556966 | 0.580703 | -158.570768 |
| ahead_plane | 4 | 0.544771 | 70.425873 | 0.642719 | 70.695914 | 0.580703 | 50.570768 |
| ahead_plane | 5 | 0.637478 | 1.909363 | 0.666206 | 26.246347 | 0.600000 | 0.000000 |
| ahead_plane | 6 | 0.479125 | -63.508698 | 0.654496 | -18.026223 | 0.580703 | -50.570768 |

## TFSF, same slab, rotated normal from z to x

`Simulation.forward`, angle_deg=30, polarization=ez, modulated_gaussian bandwidth=0.15.
`rfx.probes.oblique_reflection_magnitude`: mean(abs(total_DFT-incident_DFT))/mean(abs(incident_DFT)); +j kernel, full 3200-step record.
The source uses a Bloch phase at f0. Only the 5 GHz row is at the injected 30-degree angle; other rows retain the requested-angle analytic comparison.
Phase is not supplied by this magnitude method. The cited oracle uses a thick-slab front-face gate; this quarter-wave slab uses the full record.

| GHz | measured magnitude | analytic TE 30 deg | analytic normal | difference dB |
|---|---:|---:|---:|---:|
| 4 | 0.599304 | 0.642719 | 0.580703 | -0.607484 |
| 4.125 | 0.617320 | 0.647706 | 0.585288 | -0.417349 |
| 4.25 | 0.633772 | 0.652075 | 0.589231 | -0.247291 |
| 4.375 | 0.646263 | 0.655846 | 0.592545 | -0.127853 |
| 4.5 | 0.654830 | 0.659035 | 0.595241 | -0.055606 |
| 4.625 | 0.660859 | 0.661656 | 0.597328 | -0.010464 |
| 4.75 | 0.665820 | 0.663719 | 0.598814 | 0.027458 |
| 4.875 | 0.670686 | 0.665233 | 0.599704 | 0.070904 |
| 5 | 0.675677 | 0.666206 | 0.600000 | 0.122616 |
| 5.125 | 0.680324 | 0.666641 | 0.599704 | 0.176476 |
| 5.25 | 0.683545 | 0.666540 | 0.598814 | 0.218815 |
| 5.375 | 0.683909 | 0.665903 | 0.597328 | 0.231744 |
| 5.5 | 0.680049 | 0.664727 | 0.595241 | 0.197941 |
| 5.625 | 0.671392 | 0.663007 | 0.592545 | 0.109159 |
| 5.75 | 0.658938 | 0.660735 | 0.589231 | -0.023666 |
| 5.875 | 0.645613 | 0.657903 | 0.585288 | -0.163788 |
| 6 | 0.635578 | 0.654496 | 0.580703 | -0.254764 |

## Energy witness

E = 0.5*dx^3*sum_interior(eps0*eps_r*sum(abs(E_i)^2)+mu0*mu_r*sum(abs(H_i)^2)); CPML cells excluded.
Post-source begins at 7*tau = t0+4*tau. The ratio is 10*log10(E_final/max(E_post_source)). Complex TFSF fields use their envelope norm.
Rows at or above -40 dB carry the required truncation-suspect qualifier; no S11 accuracy verdict is assigned from them.

| case | subcase | post-source peak J | final J | ratio dB |
|---|---|---:|---:|---:|
| full_theta0_r1 |  | 1.28302698e-22 | 8.75868257e-26 | -31.657970 |
| full_theta15_r1 |  | 1.28302698e-22 | 8.75868257e-26 | -31.657970 |
| full_theta30_r1 |  | 1.28302698e-22 | 8.75868257e-26 | -31.657970 |
| full_theta45_r1 |  | 1.28302698e-22 | 8.75868257e-26 | -31.657970 |
| full_theta0_r2 |  | 8.63066921e-24 | 4.52385259e-26 | -22.805360 |
| full_theta30_r2 |  | 8.63066921e-24 | 4.52385259e-26 | -22.805360 |
| full_theta0_r4 |  | 5.77088533e-25 | 2.29015642e-26 | -14.013773 |
| full_theta30_r4 |  | 5.77088533e-25 | 2.29015642e-26 | -14.013773 |
| full_forward_theta30_r1 |  | 1.28302673e-22 | 8.75747278e-26 | -31.658569 |
| ris_theta30_r1 |  | 5.02414151e-24 | 8.94436256e-26 | -17.495125 |
| tfsf_theta30_r1 | vacuum | 5.39793913e-26 | 5.15699251e-27 | -10.198315 |
| tfsf_theta30_r1 | slab | 1.02671585e-22 | 2.56179305e-27 | -46.029062 |

Conclusion: leader fills.
