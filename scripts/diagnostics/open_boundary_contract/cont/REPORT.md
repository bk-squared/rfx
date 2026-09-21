# 801
GaussianPulse: f0 = 8.5 GHz; bandwidth = 1.6. Substrate: h = 0.787 mm; eps_r = 3.38; sigma = 0 S/m; ground/patch = PEC. Record = 150 periods; field precision = 32 bits.


```json
{
  "run_id": 369367262579,
  "variant_a_runs": 4,
  "variant_b_runs": 0,
  "cpu_smoke_steps": 200,
  "cpu_smoke_exit_code": 0,
  "build_check_exit_code": 0,
  "reduce_exit_code": 0,
  "verification_exit_code": 1,
  "verification_first_arm": "n2_pad10_cpml4",
  "verification_failed_expression": "assert np.array_equal(rates, r[\"rates_per_step\"])",
  "A1_held": 1,
  "A2_held": 1,
  "FACT_discrepancies_observed": 0,
  "n4_baseline_region_fraction_count": 0,
  "n4_baseline_max_cell_count": 0,
  "n4_baseline_distance_count": 0,
  "new_run_id_file": "cont/run_id.txt",
  "pre_existing_run_id_file_writes": 0
}
```
## verification.txt


```text
Traceback (most recent call last):
  File "<string>", line 22, in <module>
AssertionError
```
## TABLE.md

| arm | variant | settling_dB | worst_rate_per_step | interior | x_lo | x_hi | y_lo | y_hi | z_lo | z_hi | edge_overlap | corner_overlap | max_E_cell | d_domain_cells | d_absorber_cells | d_PEC_cells |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_pad10_cpml4 | baseline | 0 | 0.00255 | 0.517 | 0.00989 | 0.069 | 0.000884 | 0.389 | 0.00134 | 0.000365 | 0.0127 | 2.51e-05 | (120, 90, 15) | 4 | 0 | 1.73 |
| n2_pad0_cpml4 | baseline | 0 | 0.000768 | 0.529 | 0.0141 | 0.144 | 0.0204 | 0.266 | 0.00201 | 0.000744 | 0.0238 | 5.36e-05 | (80, 50, 15) | 4 | 0 | 1.73 |
| n3_pad10_cpml6 | baseline | 0 | 0.000584 | 0.508 | 0.0126 | 0.182 | 0.0153 | 0.259 | 0.00168 | 0.000687 | 0.0211 | 4.74e-05 | (180, 135, 22) | 6 | 0 | 1.73 |
| n4_pad10_cpml8 | baseline | -43.30 | -1.93e-4 | null | null | null | null | null | null | null | null | null | null | null | null | null |
| n2_pad10_cpml4 | a | -44.8059531 | -0.000393936134 | 0.994135566 | 0.000455499653 | 9.75122771e-05 | 0.000833268721 | 0.000357648322 | 0 | 0.00392056303 | 0.000194985592 | 4.95677043e-06 | (63, 40, 16) | 16 | 12 | 1 |
| n2_pad0_cpml4 | a | -43.0371213 | -0.000381817276 | 0.993302523 | 0.000291988171 | 3.61844644e-05 | 0.00157915118 | 0.00154143402 | 0 | 0.00301308371 | 0.000234687969 | 9.47946154e-07 | (43, 20, 16) | 16 | 12 | 1 |
| n3_pad10_cpml6 | a | -44.7963324 | -0.000265549678 | 0.99691184 | 0.000261994296 | 4.69880088e-05 | 0.00043411851 | 0.000301600753 | 0 | 0.00193374846 | 0.000107937477 | 1.77250686e-06 | (94, 60, 23) | 23 | 17 | 2 |
| n4_pad10_cpml8 | a | -44.8170826 | -0.000199167943 | 0.997628458 | 8.01003337e-05 | 9.94879962e-05 | 0.000247797318 | 0.00018242549 | 0 | 0.00170189262 | 5.88590607e-05 | 9.79506476e-07 | (126, 80, 31) | 31 | 23 | 2 |

## Four probe rates


```json
[
  {
    "arm": "n2_pad10_cpml4",
    "variant": "a",
    "grid": [
      125,
      95,
      41
    ],
    "steps": 13330,
    "rates_per_step": [
      -0.0003939361335434324,
      -0.00040419168425671437,
      -0.00040569903387328485,
      -0.0004070948774261883
    ],
    "settling_db": -44.80595312254415,
    "worst_rate_per_step": -0.0003939361335434324
  },
  {
    "arm": "n2_pad0_cpml4",
    "variant": "a",
    "grid": [
      85,
      55,
      41
    ],
    "steps": 13330,
    "rates_per_step": [
      -0.00038181727628329985,
      -0.00038485740513128015,
      -0.0003895779893017447,
      -0.0003923743743445862
    ],
    "settling_db": -43.037121335525114,
    "worst_rate_per_step": -0.00038181727628329985
  },
  {
    "arm": "n3_pad10_cpml6",
    "variant": "a",
    "grid": [
      187,
      142,
      61
    ],
    "steps": 19994,
    "rates_per_step": [
      -0.0002655496775842091,
      -0.00026787184245422073,
      -0.0002715649201494669,
      -0.00026891312718956646
    ],
    "settling_db": -44.796332388343636,
    "worst_rate_per_step": -0.0002655496775842091
  },
  {
    "arm": "n4_pad10_cpml8",
    "variant": "a",
    "grid": [
      249,
      189,
      81
    ],
    "steps": 26659,
    "rates_per_step": [
      -0.00019916794348066782,
      -0.000201667666272444,
      -0.00019923855751587498,
      -0.00020098420747165113
    ],
    "settling_db": -44.81708256486837,
    "worst_rate_per_step": -0.00019916794348066782
  }
]
```
## Normalized profiles


```json
[
  {
    "arm": "n2_pad10_cpml4",
    "x": {
      "first8": [
        9.910538913378724e-17,
        1.3646788683296652e-07,
        0.00019515753345237348,
        0.00035366563736227995,
        0.0003450579606282087,
        0.00033368672209267616,
        0.00032076499836323705,
        0.00030553514408103596
      ],
      "last8": [
        6.294125058832424e-05,
        6.24626737006293e-05,
        6.258933469936934e-05,
        6.331588877247766e-05,
        6.492449779293318e-05,
        4.7858871576433104e-05,
        3.018406847959595e-07,
        0.0
      ]
    },
    "y": {
      "first8": [
        5.504399313238144e-16,
        2.0103820437825302e-07,
        0.00033277563687373837,
        0.0006116699470853555,
        0.0006397243456313067,
        0.0006705650063329776,
        0.0006981488518871448,
        0.0007285166194348034
      ],
      "last8": [
        0.0003149850592755385,
        0.00029235835427728047,
        0.0002720291132620004,
        0.0002553997939899586,
        0.00024086881755476045,
        0.00016164962630466813,
        2.6665870409791324e-07,
        0.0
      ]
    }
  },
  {
    "arm": "n2_pad0_cpml4",
    "x": {
      "first8": [
        4.317735887086446e-17,
        5.939946601812368e-08,
        0.00010333911290514131,
        0.00021838403947896365,
        0.00022202580167311015,
        0.00022720017781172763,
        0.0002341852274004588,
        0.0002431539817922224
      ],
      "last8": [
        3.292319734866736e-05,
        3.074846413916216e-05,
        2.8889702112283885e-05,
        2.7370860829263524e-05,
        2.6074192750265393e-05,
        1.8560815561353404e-05,
        8.42763596260246e-08,
        0.0
      ]
    },
    "y": {
      "first8": [
        3.08727475727078e-16,
        2.2356650044835777e-07,
        0.000664495439752193,
        0.0010205396067951387,
        0.0010981067334552546,
        0.0011935465260423184,
        0.0013148076600352765,
        0.001475654185118297
      ],
      "last8": [
        0.0012741455969877686,
        0.0011948408349939956,
        0.0011343125069541576,
        0.0010893398188776952,
        0.0010516202920386288,
        0.00059255461251322,
        1.4806126932726273e-06,
        0.0
      ]
    }
  },
  {
    "arm": "n3_pad10_cpml6",
    "x": {
      "first8": [
        1.1924191832010163e-14,
        5.603150803839157e-10,
        8.585086213218224e-07,
        5.0420728947099854e-05,
        0.000122620238065356,
        0.00013332397777748427,
        0.0001332955804691734,
        0.00013267385588562134
      ],
      "last8": [
        2.8898726238935465e-05,
        2.5382326343150296e-05,
        2.2303575551864536e-05,
        1.907838768474849e-05,
        1.0414692164714483e-05,
        6.602985218344752e-07,
        9.305795447009316e-10,
        0.0
      ]
    },
    "y": {
      "first8": [
        2.714599919972129e-14,
        8.575497974082127e-10,
        1.2556756582646736e-06,
        8.568761973396348e-05,
        0.00019266370339265332,
        0.00020776384007573543,
        0.00020811518059733473,
        0.00020883918094270247
      ],
      "last8": [
        0.0001402959730204472,
        0.00013762900067602288,
        0.0001354629538157895,
        0.00012675022769560355,
        6.825436337391666e-05,
        2.5098545942715555e-06,
        5.653633337716921e-10,
        0.0
      ]
    }
  },
  {
    "arm": "n4_pad10_cpml8",
    "x": {
      "first8": [
        4.816596379306758e-13,
        3.5359098958587563e-10,
        2.130371455888224e-09,
        5.666996334900245e-07,
        1.0210417371404331e-05,
        2.5599014385055105e-05,
        3.1041948257043095e-05,
        3.095445750181514e-05
      ],
      "last8": [
        3.691501989698045e-05,
        3.403590357123839e-05,
        2.666290862956178e-05,
        1.09235454681172e-05,
        7.72640093314595e-07,
        3.3197142615547743e-09,
        2.69423257096646e-10,
        0.0
      ]
    },
    "y": {
      "first8": [
        7.634936430606233e-13,
        5.634661708817115e-10,
        4.898993019131883e-09,
        1.861066503840074e-06,
        2.985291340377989e-05,
        6.858867597440593e-05,
        8.49453466558152e-05,
        8.866328475121192e-05
      ],
      "last8": [
        6.139385785414327e-05,
        5.935405856834112e-05,
        5.0910640512888034e-05,
        2.636900129593397e-05,
        3.201332495106727e-06,
        7.34829746892305e-09,
        2.3738844966823214e-10,
        0.0
      ]
    }
  }
]
```
w = eps0 * eps_r * (ex^2 + ey^2 + ez^2) + mu0 * mu_r * (hx^2 + hy^2 + hz^2).
eps0 = 8.8541878128e-12 F/m; mu0 = 1.25663706212e-6 H/m; component interpolation = 0.
Profile normalization = axis-summed w / total w; plot range = [-8, 0] in log10(w / max w).

## variant_note.txt


```json
{
  "run_id": 369367262579,
  "variants_run": [
    "a"
  ],
  "variant_b_runs": 0,
  "A1_held": 1,
  "A2_held": 1,
  "readback_site": "rfx.simulation.run: pec_mask and pec_edge_masks input arguments",
  "readback_files": [
    "cont/n2_pad10_cpml4_a/assembly_used.npz",
    "cont/n2_pad0_cpml4_a/assembly_used.npz",
    "cont/n3_pad10_cpml6_a/assembly_used.npz",
    "cont/n4_pad10_cpml8_a/assembly_used.npz"
  ],
  "mask_modifying_monkeypatches": 0,
  "records": [
    {
      "arm": "n2_pad10_cpml4",
      "variant": "a",
      "steps": 13330,
      "pec_mask_sha256": "023a3060fc67a873463a1fc19b7f1f05c104c8ad972808b2e719872bda1512d2",
      "ground_row_mask_as_used": {
        "grid": [
          125,
          95,
          41
        ],
        "face_pads": {
          "x_lo": 4,
          "x_hi": 4,
          "y_lo": 4,
          "y_hi": 4,
          "z_lo": 4,
          "z_hi": 4
        },
        "ground_z": 14,
        "substrate_z": 15,
        "mid_x": 62,
        "mid_y": 47,
        "pec_cells_total": 12447,
        "pec_absorber_cells": {
          "x_lo": 380,
          "x_hi": 380,
          "y_lo": 500,
          "y_hi": 500,
          "z_lo": 0,
          "z_hi": 0
        },
        "ground_absorber_cells": {
          "x_lo": 380,
          "x_hi": 380,
          "y_lo": 500,
          "y_hi": 500
        },
        "x_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "x_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "ground_x_all_ones": 1,
        "ground_y_all_ones": 1,
        "ground_plane_all_ones": 1,
        "eps_r_x_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_x_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ]
      },
      "run_assembly_calls": 1,
      "manual_mask_same_object_as_solve": 0,
      "solve_mask_same_object_as_run_assembly": 1,
      "pec_edge_counts": {
        "x": 24938,
        "y": 24946,
        "z": 12496
      },
      "pec_edge_sha256": {
        "x": "e5de1984a79b0d74e3b38faa0208251f79ff3fb34059a8197eda887894a3fc20",
        "y": "0b7afd2d5f5b5cb7ffadb722163d03fde997a6cf6a79287eff1171126a7c4071",
        "z": "fc7f278b9250d7fea4393c19580066e7d674dbe00af5cc5b002114777f201f48"
      },
      "materials_equal_to_baseline": {
        "eps_r": 1,
        "mu_r": 1,
        "sigma": 1
      },
      "changed_pec_cells": 1899,
      "changed_pec_z_indices": [
        14
      ]
    },
    {
      "arm": "n2_pad0_cpml4",
      "variant": "a",
      "steps": 13330,
      "pec_mask_sha256": "bdac610d1d4c84dcfe4398371cce65fca56a95124c5166c4464c2c4c527d5a59",
      "ground_row_mask_as_used": {
        "grid": [
          85,
          55,
          41
        ],
        "face_pads": {
          "x_lo": 4,
          "x_hi": 4,
          "y_lo": 4,
          "y_hi": 4,
          "z_lo": 4,
          "z_hi": 4
        },
        "ground_z": 14,
        "substrate_z": 15,
        "mid_x": 42,
        "mid_y": 27,
        "pec_cells_total": 5247,
        "pec_absorber_cells": {
          "x_lo": 220,
          "x_hi": 220,
          "y_lo": 340,
          "y_hi": 340,
          "z_lo": 0,
          "z_hi": 0
        },
        "ground_absorber_cells": {
          "x_lo": 220,
          "x_hi": 220,
          "y_lo": 340,
          "y_hi": 340
        },
        "x_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "x_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "ground_x_all_ones": 1,
        "ground_y_all_ones": 1,
        "ground_plane_all_ones": 1,
        "eps_r_x_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_x_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ]
      },
      "run_assembly_calls": 1,
      "manual_mask_same_object_as_solve": 0,
      "solve_mask_same_object_as_run_assembly": 1,
      "pec_edge_counts": {
        "x": 10538,
        "y": 10546,
        "z": 5296
      },
      "pec_edge_sha256": {
        "x": "598f5e739ffc58d09c7c64922af42beb5a2ce21f50b0dd26229139cba94cf72e",
        "y": "f13016662f98a7b191efe37634a4113b68f348686936992120f0f2a4e22754ab",
        "z": "ced1c2bf439b2ace06aff308026b2dd43ad872a5aa8096f12c4129cd391de41a"
      },
      "materials_equal_to_baseline": {
        "eps_r": 1,
        "mu_r": 1,
        "sigma": 1
      },
      "changed_pec_cells": 1179,
      "changed_pec_z_indices": [
        14
      ]
    },
    {
      "arm": "n3_pad10_cpml6",
      "variant": "a",
      "steps": 19994,
      "pec_mask_sha256": "a25f04e3fc450ec4266409af4ab63011c64f6ea57afdf3c711b8010a5452497c",
      "ground_row_mask_as_used": {
        "grid": [
          187,
          142,
          61
        ],
        "face_pads": {
          "x_lo": 6,
          "x_hi": 6,
          "y_lo": 6,
          "y_hi": 6,
          "z_lo": 6,
          "z_hi": 6
        },
        "ground_z": 21,
        "substrate_z": 22,
        "mid_x": 93,
        "mid_y": 71,
        "pec_cells_total": 27841,
        "pec_absorber_cells": {
          "x_lo": 852,
          "x_hi": 852,
          "y_lo": 1122,
          "y_hi": 1122,
          "z_lo": 0,
          "z_hi": 0
        },
        "ground_absorber_cells": {
          "x_lo": 852,
          "x_hi": 852,
          "y_lo": 1122,
          "y_hi": 1122
        },
        "x_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "x_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "ground_x_all_ones": 1,
        "ground_y_all_ones": 1,
        "ground_plane_all_ones": 1,
        "eps_r_x_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_x_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ]
      },
      "run_assembly_calls": 1,
      "manual_mask_same_object_as_solve": 0,
      "solve_mask_same_object_as_run_assembly": 1,
      "pec_edge_counts": {
        "x": 55748,
        "y": 55760,
        "z": 27914
      },
      "pec_edge_sha256": {
        "x": "ccb8007368af4bfa797615ca9f77ff779d428f4d8b19bcdbdea95ca178dddb66",
        "y": "f63b9f6156c5a14e581d5f5e404ba2ffddfc38b8452d7166a98088c11599f8c1",
        "z": "83053fbf47d827b73575d1419469bc6ba9b621b9e89b355d608e166ea9df16b8"
      },
      "materials_equal_to_baseline": {
        "eps_r": 1,
        "mu_r": 1,
        "sigma": 1
      },
      "changed_pec_cells": 4108,
      "changed_pec_z_indices": [
        21
      ]
    },
    {
      "arm": "n4_pad10_cpml8",
      "variant": "a",
      "steps": 26659,
      "pec_mask_sha256": "b6359cf8ca57e0f0a2bacab708f8625148f98bde10f62b0cae633a127a59d1d9",
      "ground_row_mask_as_used": {
        "grid": [
          249,
          189,
          81
        ],
        "face_pads": {
          "x_lo": 8,
          "x_hi": 8,
          "y_lo": 8,
          "y_hi": 8,
          "z_lo": 8,
          "z_hi": 8
        },
        "ground_z": 28,
        "substrate_z": 29,
        "mid_x": 124,
        "mid_y": 94,
        "pec_cells_total": 49349,
        "pec_absorber_cells": {
          "x_lo": 1512,
          "x_hi": 1512,
          "y_lo": 1992,
          "y_hi": 1992,
          "z_lo": 0,
          "z_hi": 0
        },
        "ground_absorber_cells": {
          "x_lo": 1512,
          "x_hi": 1512,
          "y_lo": 1992,
          "y_hi": 1992
        },
        "x_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "x_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_first8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "y_last8": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "ground_x_all_ones": 1,
        "ground_y_all_ones": 1,
        "ground_plane_all_ones": 1,
        "eps_r_x_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_x_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_first8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ],
        "eps_r_y_last8": [
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918,
          3.380000114440918
        ]
      },
      "run_assembly_calls": 1,
      "manual_mask_same_object_as_solve": 0,
      "solve_mask_same_object_as_run_assembly": 1,
      "pec_edge_counts": {
        "x": 98786,
        "y": 98802,
        "z": 49446
      },
      "pec_edge_sha256": {
        "x": "4dbe8f5ba13ffa151fee90221b7b24826a1625b0d4f41e3883bf6952103bd08f",
        "y": "ec702ff3651df819b37ecf7263092fc7c71cce6e555ba94abb574cc7c6201eb6",
        "z": "920446e045528494cb65f8eac25b3377c228ee0fa2dca0e2eafb6f6b53819fa0"
      },
      "materials_equal_to_baseline": {
        "eps_r": 1,
        "mu_r": 1,
        "sigma": 1
      },
      "changed_pec_cells": 7157,
      "changed_pec_z_indices": [
        28
      ]
    }
  ]
}
```

## build_check.txt


```text
{"builder_lateral_bound_changes": 4, "devices": ["cpu:0"], "x64": 0}
{"arm": "n2_pad10_cpml4", "variant": "baseline"}
  [PREFLIGHT] 'pec' z-extent 393.5µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Gap between PEC structures: 787µm = 2.0 cells along z — coupling may be under-resolved.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox lo face at -39.35µm is 39.35µm past the x-lo absorber boundary at 0mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 39.35µm to 39.35µm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.329mm (drawn 3.896mm -> 4.289mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 5.115mm and 5.509mm (drawn 5.076mm -> 5.47mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[0] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-lo CPML face (declared 39.35µm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-hi CPML face (declared 39.35µm from it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-lo CPML face (declared 39.35µm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-hi CPML face (declared 39.35µm from it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 4, "x_lo": 4, "y_hi": 4, "y_lo": 4, "z_hi": 4, "z_lo": 4}, "grid": [125, 95, 41], "ground_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0}, "ground_plane_all_ones": 0, "ground_x_all_ones": 0, "ground_y_all_ones": 0, "ground_z": 14, "mid_x": 62, "mid_y": 47, "pec_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 10548, "substrate_z": 15, "x_first8": [0, 0, 0, 0, 1, 1, 1, 1], "x_last8": [1, 1, 1, 0, 0, 0, 0, 0], "y_first8": [0, 0, 0, 0, 1, 1, 1, 1], "y_last8": [1, 1, 1, 0, 0, 0, 0, 0]}
{"arm": "n2_pad10_cpml4", "variant": "a"}
  [PREFLIGHT] 'pec' z-extent 393.5µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Gap between PEC structures: 787µm = 2.0 cells along z — coupling may be under-resolved.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox lo face at -1.968mm is 1.968mm past the x-lo absorber boundary at 0mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 39.35µm to 1.968mm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.329mm (drawn 3.896mm -> 4.289mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 5.115mm and 5.509mm (drawn 5.076mm -> 5.47mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 4 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 4 listed): geometry[0] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[0] 'pec' (volume) y: extent 37.78mm, worst face residual 393.5µm (1.04% of the extent); geometry[0] 'pec' (volume) x: extent 49.58mm, worst face residual 393.5µm (0.79% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the x-lo CPML face (declared 1.968mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the x-hi CPML face (declared 1.967mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the y-lo CPML face (declared 1.968mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the y-hi CPML face (declared 1.967mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 4, "x_lo": 4, "y_hi": 4, "y_lo": 4, "z_hi": 4, "z_lo": 4}, "grid": [125, 95, 41], "ground_absorber_cells": {"x_hi": 380, "x_lo": 380, "y_hi": 500, "y_lo": 500}, "ground_plane_all_ones": 1, "ground_x_all_ones": 1, "ground_y_all_ones": 1, "ground_z": 14, "mid_x": 62, "mid_y": 47, "pec_absorber_cells": {"x_hi": 380, "x_lo": 380, "y_hi": 500, "y_lo": 500, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 12447, "substrate_z": 15, "x_first8": [1, 1, 1, 1, 1, 1, 1, 1], "x_last8": [1, 1, 1, 1, 1, 1, 1, 1], "y_first8": [1, 1, 1, 1, 1, 1, 1, 1], "y_last8": [1, 1, 1, 1, 1, 1, 1, 1]}
{"arm": "n2_pad0_cpml4", "variant": "baseline"}
  [PREFLIGHT] 'pec' z-extent 393.5µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Gap between PEC structures: 787µm = 2.0 cells along z — coupling may be under-resolved.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox lo face at -39.35µm is 39.35µm past the x-lo absorber boundary at 0mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 39.35µm to 39.35µm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.329mm (drawn 3.896mm -> 4.289mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 5.115mm and 5.509mm (drawn 5.076mm -> 5.47mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[0] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-lo CPML face (declared 39.35µm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-hi CPML face (declared 39.35µm from it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-lo CPML face (declared 39.35µm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-hi CPML face (declared 39.35µm from it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 4, "x_lo": 4, "y_hi": 4, "y_lo": 4, "z_hi": 4, "z_lo": 4}, "grid": [85, 55, 41], "ground_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0}, "ground_plane_all_ones": 0, "ground_x_all_ones": 0, "ground_y_all_ones": 0, "ground_z": 14, "mid_x": 42, "mid_y": 27, "pec_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 4068, "substrate_z": 15, "x_first8": [0, 0, 0, 0, 1, 1, 1, 1], "x_last8": [1, 1, 1, 0, 0, 0, 0, 0], "y_first8": [0, 0, 0, 0, 1, 1, 1, 1], "y_last8": [1, 1, 1, 0, 0, 0, 0, 0]}
{"arm": "n2_pad0_cpml4", "variant": "a"}
  [PREFLIGHT] 'pec' z-extent 393.5µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Gap between PEC structures: 787µm = 2.0 cells along z — coupling may be under-resolved.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox lo face at -1.968mm is 1.968mm past the x-lo absorber boundary at 0mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 39.35µm to 1.968mm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.329mm (drawn 3.896mm -> 4.289mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 5.115mm and 5.509mm (drawn 5.076mm -> 5.47mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 4 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 4 listed): geometry[0] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 393.5µm, worst face residual 39.35µm (10.00% of the extent); geometry[0] 'pec' (volume) y: extent 22.04mm, worst face residual 393.5µm (1.79% of the extent); geometry[0] 'pec' (volume) x: extent 33.84mm, worst face residual 393.5µm (1.16% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the x-lo CPML face (declared 1.968mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the x-hi CPML face (declared 1.967mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the y-lo CPML face (declared 1.968mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -5 cell(s) of clearance from the y-hi CPML face (declared 1.968mm past it), and that face has 4 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 4, "x_lo": 4, "y_hi": 4, "y_lo": 4, "z_hi": 4, "z_lo": 4}, "grid": [85, 55, 41], "ground_absorber_cells": {"x_hi": 220, "x_lo": 220, "y_hi": 340, "y_lo": 340}, "ground_plane_all_ones": 1, "ground_x_all_ones": 1, "ground_y_all_ones": 1, "ground_z": 14, "mid_x": 42, "mid_y": 27, "pec_absorber_cells": {"x_hi": 220, "x_lo": 220, "y_hi": 340, "y_lo": 340, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 5247, "substrate_z": 15, "x_first8": [1, 1, 1, 1, 1, 1, 1, 1], "x_last8": [1, 1, 1, 1, 1, 1, 1, 1], "y_first8": [1, 1, 1, 1, 1, 1, 1, 1], "y_last8": [1, 1, 1, 1, 1, 1, 1, 1]}
{"arm": "n3_pad10_cpml6", "variant": "baseline"}
  [PREFLIGHT] 'pec' z-extent 262.3µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] 'pec' z-extent 262.3µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox lo face at -26.23µm is 26.23µm past the x-lo absorber boundary at 0mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 26.23µm to 26.23µm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.197mm (drawn 3.909mm -> 4.171mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 4.984mm and 5.247mm (drawn 4.958mm -> 5.22mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 2 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 2 listed): geometry[0] 'pec' (volume) z: extent 262.3µm, worst face residual 26.23µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 262.3µm, worst face residual 26.23µm (10.00% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-lo CPML face (declared 26.23µm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the x-hi CPML face (declared 26.23µm from it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-lo CPML face (declared 26.23µm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes 0 cell(s) of clearance from the y-hi CPML face (declared 26.23µm from it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 6, "x_lo": 6, "y_hi": 6, "y_lo": 6, "z_hi": 6, "z_lo": 6}, "grid": [187, 142, 61], "ground_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0}, "ground_plane_all_ones": 0, "ground_x_all_ones": 0, "ground_y_all_ones": 0, "ground_z": 21, "mid_x": 93, "mid_y": 71, "pec_absorber_cells": {"x_hi": 0, "x_lo": 0, "y_hi": 0, "y_lo": 0, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 23733, "substrate_z": 22, "x_first8": [0, 0, 0, 0, 0, 0, 1, 1], "x_last8": [1, 0, 0, 0, 0, 0, 0, 0], "y_first8": [0, 0, 0, 0, 0, 0, 1, 1], "y_last8": [1, 0, 0, 0, 0, 0, 0, 0]}
{"arm": "n3_pad10_cpml6", "variant": "a"}
  [PREFLIGHT] 'pec' z-extent 262.3µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] 'pec' z-extent 262.3µm = 1.0 cells — below 1 cell resolution. A PEC shape passed to sim.add() is a VOLUME (realized with both faces and a shorted interior, lattice ownership contract #931 §1.2), and a volume thinner than one local cell is refused at assembly (§1.5) rather than snapped to a plane. If this is foil (a ground plane, patch, trace), declare a SHEET: a zero-thickness Box via add(), or add_thin_conductor(shape) — realized on the node plane nearest its mid-plane, with the normal E edge through it live. Otherwise resolve the thickness with a finer local cell.
  [PREFLIGHT] Material 'pec' (geometry entry #0, Box) extends into CPML region along x-axis: bbox hi face at 47.48mm is 1.836mm past the x-hi absorber boundary at 45.65mm. 2 geometry entries cross the x-axis absorber (worst shown; overshoot 26.23µm to 1.836mm); per-entry index, face and overshoot in this finding's loc. CPML modifies field updates — geometry inside the absorber is physically meaningless (issue #61).
  [PREFLIGHT] all dielectric(s) ['ro4003c'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
  [PREFLIGHT] 2 PEC volume(s) are exactly ONE cell thick along some axis and are realized as a filled slab with walls on BOTH faces (every E edge between the two faces is shorted; lattice ownership contract #931 §1.2): geometry[0] 'pec' (Box) is one cell thick along z [z: walls at 3.935mm and 4.197mm (drawn 3.909mm -> 4.171mm)]; geometry[2] 'pec' (Box) is one cell thick along z [z: walls at 4.984mm and 5.247mm (drawn 4.958mm -> 5.22mm)]. If this is foil (a ground plane, patch or trace), declare it as a SHEET — add_thin_conductor(shape), or a zero-thickness Box via add() — which realizes on ONE node plane with the normal E edge through it live; a slab and a sheet are different conductors on this lattice. If it is a plate, an iris or a wall drawn one cell thick on purpose, no action is needed (report-only). COVERAGE: examined 2 PEC volume(s) on the uniform lane through rfx.boundaries.pec.realized_wall_planes. STALE IF: realized_wall_planes on the named entry's own edge masks returns planes more than one index apart.
  [PREFLIGHT] 4 conductor-Box design edge(s) sit off-lattice by more than 0.5% of their extent (worst 4 listed): geometry[0] 'pec' (volume) z: extent 262.3µm, worst face residual 26.23µm (10.00% of the extent); geometry[2] 'pec' (volume) z: extent 262.3µm, worst face residual 26.23µm (10.00% of the extent); geometry[0] 'pec' (volume) y: extent 37.51mm, worst face residual 262.3µm (0.70% of the extent); geometry[0] 'pec' (volume) x: extent 49.32mm, worst face residual 262.3µm (0.53% of the extent). OBSERVED: distance from each declared face to its nearest E-node on this run's own node coordinates; a PEC volume's face realizes on the nearest node plane and a sheet footprint on the nodes it covers (lattice ownership contract #931 §1.1/§1.3). Reported residuals describe declared-face alignment only. A sheet's extent can change by more than this nearest-node residual, and an extent change involves both faces. Read the realized bounds from fidelity_report(); frequency sensitivity depends on the mode and the affected dimension. COST (measured, #703): a uniform-mesh sweep rounded ONE substrate thickness by 8-10% across three 'convergence' points — three different boards solved under one name; the same campaign's board survived at dx=50um only because every patterned dimension happened to be an exact multiple of 50um. REMEDY: for a PEC VOLUME choose dx commensurate with its dimensions, slide the lattice origin onto the worst face, or (non-uniform lane) place mesh nodes on its faces; for a SHEET see sheet_effective_size -- a node on a sheet's edge is not the fix. COVERAGE: examined 6 axis extent(s) on 2 conductor Box declaration(s) on the uniform lane; 0 sheet normal axis/axes reported by sheet_plane_realized instead; 0 non-Box conductor entr(y/ies) skipped (no analytic face coordinates). STALE IF: |face - nearest node| on the run's node coordinates does not reproduce the printed residuals.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -7 cell(s) of clearance from the x-lo CPML face (declared 1.836mm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -7 cell(s) of clearance from the x-hi CPML face (declared 1.836mm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -7 cell(s) of clearance from the y-lo CPML face (declared 1.836mm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
  [PREFLIGHT] Conductor 'pec' (geometry entry #0, Box) realizes -7 cell(s) of clearance from the y-hi CPML face (declared 1.836mm past it), and that face has 6 absorbing layer(s). Both at once is a measured growth class (issue #801): on the isolated-patch rig at dx = h/3 with a laterally padded domain, 6 layers with the ground flush against the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and 12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) and 6 layers with the conductors pulled 2 cells clear settled (-42.8 dB). Either remedy removed it in every arm measured. These are the EDGES OF THE MEASURED REGION, not a stability bound: the mechanism is not established, which is why this advises rather than refuses. The measured co-factor was the lateral padding, and the one unpadded arm (4 layers, conductor at the face) settled, so this may over-warn on an unpadded domain.
{"preflight_findings_count": 10, "preflight_exceptions": 0}
{"eps_r_x_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_x_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_first8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "eps_r_y_last8": [3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918, 3.380000114440918], "face_pads": {"x_hi": 6, "x_lo": 6, "y_hi": 6, "y_lo": 6, "z_hi": 6, "z_lo": 6}, "grid": [187, 142, 61], "ground_absorber_cells": {"x_hi": 852, "x_lo": 852, "y_hi": 1122, "y_lo": 1122}, "ground_plane_all_ones": 1, "ground_x_all_ones": 1, "ground_y_all_ones": 1, "ground_z": 21, "mid_x": 93, "mid_y": 71, "pec_absorber_cells": {"x_hi": 852, "x_lo": 852, "y_hi": 1122, "y_lo": 1122, "z_hi": 0, "z_lo": 0}, "pec_cells_total": 27841, "substrate_z": 22, "x_first8": [1, 1, 1, 1, 1, 1, 1, 1], "x_last8": [1, 1, 1, 1, 1, 1, 1, 1], "y_first8": [1, 1, 1, 1, 1, 1, 1, 1], "y_last8": [1, 1, 1, 1, 1, 1, 1, 1]}
{"selected_variants": ["a"], "A1_all_three": 1}
```

## Commands
Output records: [commands.json](commands.json). Remote output: [vessl_369367262579.log](vessl_369367262579.log).

### 1


```sh
cat /root/workspace/bk-workspace/.801-measure/BRIEF_conductor_continuation.md
```
output_file: /root/workspace/bk-workspace/.801-measure/BRIEF_conductor_continuation.md
exit_code: 0

### 2


```sh
cat /root/workspace/bk-workspace/CLAUDE.md
```
output_file: /root/workspace/bk-workspace/CLAUDE.md
exit_code: 0

### 3


```sh
ls -la /root/workspace/bk-workspace/.801-measure
```
output_file: cont/commands.json; entry = 3
exit_code: 0

### 4


```sh
cat /root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md
```
output_file: /root/workspace/bk-workspace/_configs/.claude/rules/vessl-jobs.md
exit_code: 0

### 5


```sh
gh issue view 801 --repo bk-squared/rfx --comments
```
output_file: cont/issue_comments.txt (second retrieval below)
exit_code: 0

### 6


```sh
cat /root/workspace/bk-workspace/.801-measure/vessl_801_dump.yaml
```
output_file: /root/workspace/bk-workspace/.801-measure/vessl_801_dump.yaml
exit_code: 0

### 7


```sh
mkdir /root/workspace/bk-workspace/.801-measure/cont
```
output_file: cont/commands.json; entry = 7
exit_code: 0

### 8


```sh
cat /root/workspace/bk-workspace/.801-measure/dump_fields.py
```
output_file: cont/commands.json; entry = 8
exit_code: 0

### 9


```sh
cat /root/workspace/bk-workspace/.801-measure/run_arms.py
```
output_file: cont/commands.json; entry = 9
exit_code: 0

### 10


```sh
cat /root/workspace/bk-workspace/.801-measure/src-main/tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py
```
output_file: cont/commands.json; entry = 10
exit_code: 0

### 11


```sh
cat /root/workspace/bk-workspace/.801-measure/reduce_dumps.py /root/workspace/bk-workspace/.801-measure/leader_replot.py
```
output_file: cont/commands.json; entry = 11
exit_code: 0

### 12


```sh
gh issue view 801 --repo bk-squared/rfx --comments > /root/workspace/bk-workspace/.801-measure/cont/issue_comments.txt
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/issue_comments.txt
exit_code: 0

### 13


```sh
sed -n '120,280p' /root/workspace/bk-workspace/.801-measure/src-main/tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py
```
output_file: cont/commands.json; entry = 13
exit_code: 0

### 14


```sh
rg -n '_assemble_materials|pec_mask|def run\(|def _build_grid|preflight' /root/workspace/bk-workspace/.801-measure/src-main/rfx/simulation.py
```
output_file: cont/commands.json; entry = 14
exit_code: 0

### 15


```sh
cat /root/workspace/bk-workspace/.801-measure/dumps/TABLE.md /root/workspace/bk-workspace/.801-measure/result_main.json
```
output_file: cont/commands.json; entry = 15
exit_code: 0

### 16


```sh
rg --files -g AGENTS.md -g CLAUDE.md -g '*summary.json' -g '*meta.json' /root/workspace/bk-workspace/.801-measure
```
output_file: cont/commands.json; entry = 16
exit_code: 0

### 17


```sh
rg -n 'def |pad_x|pad_y|pec_mask|extend' /root/workspace/bk-workspace/.801-measure/src-main/rfx/simulation_materials.py /root/workspace/bk-workspace/.801-measure/src-main/rfx/grid.py
```
output_file: cont/commands.json; entry = 17
exit_code: 2

### 18


```sh
rg -n 'def _assemble_materials|def _build_grid|class Simulation|def run\(' /root/workspace/bk-workspace/.801-measure/src-main/rfx -g '*.py'
```
output_file: cont/commands.json; entry = 18
exit_code: 0

### 19


```sh
tail -n 12 /root/workspace/bk-workspace/.801-measure/cont/issue_comments.txt
```
output_file: cont/commands.json; entry = 19
exit_code: 0

### 20


```sh
vessl run list
```
output_file: cont/commands.json; entry = 20
exit_code: 0

### 21


```sh
sed -n '88,260p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_compile.py
```
output_file: cont/commands.json; entry = 21
exit_code: 0

### 22


```sh
rg -n '_assemble_materials|pec_mask=|from rfx.simulation|import.*run|extend.*pad|pad.*extend' /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_execute.py /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_compile.py
```
output_file: cont/commands.json; entry = 22
exit_code: 0

### 23


```sh
sed -n '3476,3650p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_execute.py
```
output_file: cont/commands.json; entry = 23
exit_code: 0

### 24


```sh
sed -n '1,70p' /root/workspace/bk-workspace/.801-measure/reduce_dumps.py
```
output_file: cont/commands.json; entry = 24
exit_code: 0

### 25


```json
{
  "tool": "apply_patch",
  "action": "Add File",
  "path": "/root/workspace/bk-workspace/.801-measure/cont_build_check.py"
}
```
output_file: cont/commands.json; entry = 25

### 26


```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 /root/workspace/bk-workspace/rfx/.venv/bin/python -B /root/workspace/bk-workspace/.801-measure/cont_build_check.py > /root/workspace/bk-workspace/.801-measure/cont/build_check.txt 2>&1
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/build_check.txt
session_id: 81605

### 27


```sh
sed -n '3780,3945p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_execute.py
```
output_file: cont/commands.json; entry = 27
exit_code: 0

### 28


```sh
sed -n '1,145p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/runners/uniform.py
```
output_file: cont/commands.json; entry = 28
exit_code: 0

### 29


```sh
rg -n 'run_fn|\brun\(|pec_mask|simulation import' /root/workspace/bk-workspace/.801-measure/src-main/rfx/runners/uniform.py
```
output_file: cont/commands.json; entry = 29
exit_code: 0

### 30


```sh
tail -n 8 /root/workspace/bk-workspace/.801-measure/cont/build_check.txt
```
output_file: cont/commands.json; entry = 30
exit_code: 0

### 31


```json
{
  "tool": "write_stdin",
  "session_id": 81605,
  "chunk_id": "3ee09b",
  "wall_time_seconds": 1.833e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 31
exit_code: 0
session_id: 81605

### 32


```sh
rg '^\{' /root/workspace/bk-workspace/.801-measure/cont/build_check.txt
```
output_file: cont/commands.json; entry = 32
exit_code: 0

### 33


```sh
sed -n '1901,1948p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/simulation.py
```
output_file: cont/commands.json; entry = 33
exit_code: 0

### 34


```sh
sed -n '778,818p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/runners/uniform.py
```
output_file: cont/commands.json; entry = 34
exit_code: 0

### 35


```sh
sed -n '336,365p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/runners/uniform.py
```
output_file: cont/commands.json; entry = 35
exit_code: 0

### 36


```sh
sed -n '260,325p' /root/workspace/bk-workspace/.801-measure/src-main/rfx/api/_compile.py
```
output_file: cont/commands.json; entry = 36
exit_code: 0

### 37


```json
{
  "tool": "apply_patch",
  "action": "Add File",
  "path": "/root/workspace/bk-workspace/.801-measure/cont_run.py"
}
```
output_file: cont/commands.json; entry = 37

### 38


```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 /root/workspace/bk-workspace/rfx/.venv/bin/python -B /root/workspace/bk-workspace/.801-measure/cont_run.py n2_pad0_cpml4 --n-steps 200 --out-label _cpu_smoke > /root/workspace/bk-workspace/.801-measure/cont/cpu_smoke.txt 2>&1
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/cpu_smoke.txt
session_id: 24109

### 39


```json
{
  "tool": "apply_patch",
  "action": "Add File",
  "path": "/root/workspace/bk-workspace/.801-measure/vessl_801_cont.yaml"
}
```
output_file: cont/commands.json; entry = 39

### 40


```sh
PYTHONDONTWRITEBYTECODE=1 /root/workspace/bk-workspace/rfx/.venv/bin/python -B -c 'from pathlib import Path
import subprocess
p = Path("/root/workspace/bk-workspace/.801-measure/vessl_801_cont.yaml")
s = p.read_text().split("run: |-\n", 1)[1]
block = "\n".join(line[2:] for line in s.splitlines()) + "\n"
assert "<<" not in block
assert "|" not in block
r = subprocess.run(["sh", "-n"], input=block, text=True, check=True)
print("sh -n exit_code =", r.returncode)
print(block, end="")
'
```
output_file: cont/commands.json; entry = 40
exit_code: 0

### 41


```json
{
  "tool": "write_stdin",
  "session_id": 24109,
  "chunk_id": "85f8a8",
  "wall_time_seconds": 1.672e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 41
exit_code: 0
session_id: 24109

### 42


```sh
cat /root/workspace/bk-workspace/.801-measure/cont/cpu_smoke.txt
```
output_file: cont/commands.json; entry = 42
exit_code: 0

### 43


```sh
ls -l /root/workspace/bk-workspace/.801-measure/cont/_cpu_smoke
```
output_file: cont/commands.json; entry = 43
exit_code: 0

### 44


```sh
vessl run create -f /root/workspace/bk-workspace/.801-measure/vessl_801_cont.yaml > /root/workspace/bk-workspace/.801-measure/cont/submit.txt 2>&1
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/submit.txt
session_id: 86760

### 45


```json
{
  "tool": "write_stdin",
  "session_id": 86760,
  "chunk_id": "26ed94",
  "wall_time_seconds": 3.225e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 45
exit_code: 0
session_id: 86760

### 46


```sh
cat /root/workspace/bk-workspace/.801-measure/cont/submit.txt
```
output_file: cont/commands.json; entry = 46
exit_code: 0

### 47


```sh
printf '%s\n' '369367262579' >> /root/workspace/bk-workspace/.801-measure/cont/run_id.txt
```
output_file: cont/commands.json; entry = 47
exit_code: 0

### 48


```sh
vessl run read 369367262579
```
output_file: cont/commands.json; entry = 48
exit_code: 0

### 49


```sh
sleep 60
```
output_file: cont/commands.json; entry = 49
session_id: 12775

### 50


```json
{
  "tool": "apply_patch",
  "action": "Add File",
  "path": "/root/workspace/bk-workspace/.801-measure/cont_reduce.py"
}
```
output_file: cont/commands.json; entry = 50

### 51


```json
{
  "tool": "write_stdin",
  "session_id": 12775,
  "chunk_id": "fca29b",
  "wall_time_seconds": 2.594e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 51
exit_code: 0
session_id: 12775

### 52


```sh
vessl run read 369367262579
```
output_file: cont/commands.json; entry = 52
exit_code: 0

### 53


```sh
find /root/workspace/bk-workspace/.801-measure/cont -maxdepth 2 -type f -printf '%P %s\n'
```
output_file: cont/commands.json; entry = 53
exit_code: 0

### 54


```sh
vessl run logs 369367262579 --tail 100000 > /root/workspace/bk-workspace/.801-measure/cont/vessl_369367262579.log 2>&1
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/vessl_369367262579.log
session_id: 10491

### 55


```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /root/workspace/bk-workspace/rfx/.venv/bin/python -B /root/workspace/bk-workspace/.801-measure/cont_reduce.py > /root/workspace/bk-workspace/.801-measure/cont/reduce.txt 2>&1
```
output_file: /root/workspace/bk-workspace/.801-measure/cont/reduce.txt
session_id: 26059

### 56


```json
{
  "tool": "write_stdin",
  "session_id": 10491,
  "chunk_id": "c42288",
  "wall_time_seconds": 1.823e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 56
exit_code: 0
session_id: 10491

### 57


```json
{
  "tool": "write_stdin",
  "session_id": 26059,
  "chunk_id": "6a419a",
  "wall_time_seconds": 2.064e-06,
  "exit_code": 0,
  "original_token_count": 0
}
```
output_file: cont/commands.json; entry = 57
exit_code: 0
session_id: 26059

### 58


```sh
cat /root/workspace/bk-workspace/.801-measure/cont/TABLE.md
```
output_file: cont/commands.json; entry = 58
exit_code: 0

### 59


```sh
head -n 28 /root/workspace/bk-workspace/.801-measure/cont/vessl_369367262579.log
```
output_file: cont/commands.json; entry = 59
exit_code: 0

### 60


```sh
cat /root/workspace/bk-workspace/.801-measure/cont/reduce.txt
```
output_file: cont/commands.json; entry = 60
exit_code: 0

### 61


```sh
PYTHONDONTWRITEBYTECODE=1 /root/workspace/bk-workspace/rfx/.venv/bin/python -B -c 'from pathlib import Path
import json
base = Path("/root/workspace/bk-workspace/.801-measure")
out = base / "cont"
arms = ["n2_pad10_cpml4", "n2_pad0_cpml4", "n3_pad10_cpml6", "n4_pad10_cpml8"]
build = [json.loads(line) for line in (out / "build_check.txt").read_text().splitlines() if line.startswith("{")]
selection = build[-1]
records = []
for arm in arms:
    row = json.loads((out / (arm + "_a") / "result.json").read_text())
    records.append({key: row[key] for key in ("arm", "variant", "steps", "pec_mask_sha256",
        "ground_row_mask_as_used", "run_assembly_calls", "manual_mask_same_object_as_solve",
        "solve_mask_same_object_as_run_assembly", "pec_edge_counts", "pec_edge_sha256",
        "materials_equal_to_baseline", "changed_pec_cells", "changed_pec_z_indices")})
note = dict(run_id=369367262579, variants_run=["a"], variant_b_runs=0, A1_held=selection["A1_all_three"],
    A2_held=int(all(r["run_assembly_calls"] >= 1 and r["manual_mask_same_object_as_solve"] == 0
                   and r["solve_mask_same_object_as_run_assembly"] == 1 for r in records)),
    readback_site="rfx.simulation.run: pec_mask and pec_edge_masks input arguments",
    readback_files=["cont/" + arm + "_a/assembly_used.npz" for arm in arms],
    mask_modifying_monkeypatches=0, records=records)
with (out / "variant_note.txt").open("x") as f:
    json.dump(note, f, indent=2)
    f.write("\n")
print(json.dumps(dict(A1_held=note["A1_held"], A2_held=note["A2_held"], variant_a_runs=4, variant_b_runs=0)))
'
```
output_file: cont/commands.json; entry = 61
exit_code: 0

### 62


```json
{
  "tool": "view_image",
  "path": "/root/workspace/bk-workspace/.801-measure/cont/n3_pad10_cpml6_a/slices.png",
  "output_file": "cont/n3_pad10_cpml6_a/slices.png"
}
```
output_file: cont/n3_pad10_cpml6_a/slices.png

### 63


```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /root/workspace/bk-workspace/rfx/.venv/bin/python -B -c 'from pathlib import Path
import hashlib
import json
import sys
import numpy as np
base = Path("/root/workspace/bk-workspace/.801-measure")
sys.path.insert(0, str(base / "src-main"))
from tests.oracle import test_lossless_open_domain_ringdown_does_not_grow as oracle
out = base / "cont"
arms = ["n2_pad10_cpml4", "n2_pad0_cpml4", "n3_pad10_cpml6", "n4_pad10_cpml8"]
records = []
for arm in arms:
    d = out / (arm + "_a")
    expected = ["result.json", "final_state.npz", "assembly_used.npz", "time_series.npz",
                "summary.json", "profiles.json", "slices.png", "profiles.png"]
    assert all((d / f).is_file() and (d / f).stat().st_size > 0 for f in expected)
    r = json.loads((d / "result.json").read_text())
    with np.load(d / "time_series.npz") as f:
        ts = f["time_series"]
    rates = oracle._late_time_log_rate_per_step(ts)
    settling = oracle._settling_db(ts)
    assert np.array_equal(rates, r["rates_per_step"])
    assert settling == r["settling_db"]
    assert r["num_periods"] == oracle.NUM_PERIODS == 150
    assert r["steps"] == len(ts)
    assert len(rates) == 4
    with np.load(d / "final_state.npz") as f:
        assert len(f.files) == 6
        assert all(f[k].dtype == np.float32 and tuple(f[k].shape) == tuple(r["grid"]) for k in f.files)
    records.append(dict(arm=arm, expected_files=len(expected), found_files=sum((d/f).is_file() for f in expected),
                        field_arrays=6, field_bits=32, steps=r["steps"], probes=len(rates),
                        rate_max_abs_difference=float(np.max(np.abs(np.asarray(rates)-r["rates_per_step"]))),
                        settling_abs_difference_dB=abs(settling-r["settling_db"])))
log = (out / "vessl_369367262579.log").read_text()
hashes = {}
for rel in ["cont_run.py", "cont_build_check.py", "src-main/rfx/boundaries/cpml.py", "src-main/rfx/grid.py"]:
    digest = hashlib.sha256((base/rel).read_bytes()).hexdigest()
    assert digest in log
    hashes[rel] = digest
assert "workload completed" in log
assert len((out / "TABLE.md").read_text().splitlines()) == 10
print(json.dumps(dict(run_id=369367262579, records=records, provenance_sha256=hashes,
                      baseline_rows=4, variant_rows=4), indent=2))
' > /root/workspace/bk-workspace/.801-measure/cont/verification.txt 2>&1
```
output_file: 0
exit_code: 1

### 64


```sh
cat /root/workspace/bk-workspace/.801-measure/cont/verification.txt
```
output_file: cont/commands.json; entry = 64
exit_code: 0

### 65


```json
{
  "tool": "apply_patch",
  "action": "Add File",
  "path": "/root/workspace/bk-workspace/.801-measure/cont_report.py"
}
```
output_file: cont/commands.json; entry = 65

### commands.json creation
tool: apply_patch; action: Add File; path: cont/commands.json; output: {}

### REPORT.md creation


```sh
PYTHONDONTWRITEBYTECODE=1 /root/workspace/bk-workspace/rfx/.venv/bin/python -B /root/workspace/bk-workspace/.801-measure/cont_report.py
```
output: cont/REPORT.md 99539 bytes; exit_code: 0

## Remote run block


```sh
set -eu
base=/root/workspace/bk-workspace/.801-measure
cat "$base/PROVENANCE.txt"
sha256sum "$base/cont_run.py" "$base/cont_build_check.py" "$base/src-main/rfx/boundaries/cpml.py" "$base/src-main/rfx/grid.py"
python -m pip install -q "scipy>=1.11" pytest
python -B "$base/cont_run.py" n2_pad10_cpml4 --require-gpu
python -B "$base/cont_run.py" n2_pad0_cpml4 --require-gpu
python -B "$base/cont_run.py" n3_pad10_cpml6 --require-gpu
python -B "$base/cont_run.py" n4_pad10_cpml8 --require-gpu
```
output_file: cont/vessl_369367262579.log

## Files

| path | bytes |
|---|---:|
| cont/TABLE.md | 1766 |
| cont/_cpu_smoke/assembly_used.npz | 3068550 |
| cont/_cpu_smoke/final_state.npz | 4601638 |
| cont/_cpu_smoke/result.json | 3311 |
| cont/_cpu_smoke/time_series.npz | 3476 |
| cont/_mpl_reduce/fontlist-v3.11.0.json | 26961 |
| cont/baseline_sources.json | 535 |
| cont/build_check.txt | 56916 |
| cont/commands.json | 201652 |
| cont/cpu_smoke.txt | 4501 |
| cont/issue_comments.txt | 59002 |
| cont/n2_pad0_cpml4_a/assembly_used.npz | 3068550 |
| cont/n2_pad0_cpml4_a/final_state.npz | 4601638 |
| cont/n2_pad0_cpml4_a/profiles.json | 4483 |
| cont/n2_pad0_cpml4_a/profiles.png | 65123 |
| cont/n2_pad0_cpml4_a/result.json | 3332 |
| cont/n2_pad0_cpml4_a/slices.png | 93938 |
| cont/n2_pad0_cpml4_a/summary.json | 2168 |
| cont/n2_pad0_cpml4_a/time_series.npz | 213556 |
| cont/n2_pad10_cpml4_a/assembly_used.npz | 7791750 |
| cont/n2_pad10_cpml4_a/final_state.npz | 11686438 |
| cont/n2_pad10_cpml4_a/profiles.json | 6682 |
| cont/n2_pad10_cpml4_a/profiles.png | 62568 |
| cont/n2_pad10_cpml4_a/result.json | 3336 |
| cont/n2_pad10_cpml4_a/slices.png | 107200 |
| cont/n2_pad10_cpml4_a/summary.json | 2163 |
| cont/n2_pad10_cpml4_a/time_series.npz | 213556 |
| cont/n3_pad10_cpml6_a/assembly_used.npz | 25918454 |
| cont/n3_pad10_cpml6_a/final_state.npz | 38876494 |
| cont/n3_pad10_cpml6_a/profiles.json | 10075 |
| cont/n3_pad10_cpml6_a/profiles.png | 63366 |
| cont/n3_pad10_cpml6_a/result.json | 3356 |
| cont/n3_pad10_cpml6_a/slices.png | 261082 |
| cont/n3_pad10_cpml6_a/summary.json | 2165 |
| cont/n3_pad10_cpml6_a/time_series.npz | 320180 |
| cont/n4_pad10_cpml8_a/assembly_used.npz | 60992806 |
| cont/n4_pad10_cpml8_a/final_state.npz | 91488022 |
| cont/n4_pad10_cpml8_a/profiles.json | 13471 |
| cont/n4_pad10_cpml8_a/profiles.png | 65554 |
| cont/n4_pad10_cpml8_a/result.json | 3349 |
| cont/n4_pad10_cpml8_a/slices.png | 273944 |
| cont/n4_pad10_cpml8_a/summary.json | 2156 |
| cont/n4_pad10_cpml8_a/time_series.npz | 426820 |
| cont/reduce.txt | 6723 |
| cont/run_id.txt | 13 |
| cont/submit.txt | 163 |
| cont/variant_note.txt | 13703 |
| cont/verification.txt | 90 |
| cont/vessl_369367262579.log | 24009 |
| cont_build_check.py | 7303 |
| cont_run.py | 6297 |
| cont_reduce.py | 8056 |
| cont_report.py | 5559 |
| vessl_801_cont.yaml | 1014 |
| cont/REPORT.md | 99539 |
