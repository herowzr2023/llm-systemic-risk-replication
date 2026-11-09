# Table A6. Forecast Losses for Direct One-Trading-Day-Ahead Predictions

| Bank | MSE no-text | MSE text | MSE Δ% | MAE no-text | MAE text | MAE Δ% | MSE wins | MAE wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PAB | 22.06 ± 8.07 | 10.52 ± 3.22 | 52.33 | 3.84 ± 0.83 | 2.52 ± 0.39 | 34.28 | 20/20 | 20/20 |
| NBCB | 1213.14 ± 720.29 | 254.14 ± 112.54 | 79.05 | 28.03 ± 9.90 | 12.33 ± 2.81 | 56.02 | 19/20 | 19/20 |
| SPDB | 222.18 ± 170.88 | 45.79 ± 23.99 | 79.39 | 12.42 ± 4.61 | 5.24 ± 1.41 | 57.78 | 20/20 | 20/20 |
| HB | 16.27 ± 6.99 | 4.31 ± 0.89 | 73.49 | 3.44 ± 0.76 | 1.62 ± 0.17 | 52.82 | 20/20 | 20/20 |
| CMBC | 187.65 ± 261.69 | 42.36 ± 24.87 | 77.43 | 9.33 ± 5.42 | 5.09 ± 1.56 | 45.44 | 17/20 | 17/20 |
| CMB | 20.05 ± 6.20 | 14.13 ± 6.73 | 29.49 | 3.71 ± 0.65 | 2.82 ± 0.66 | 24.07 | 15/20 | 17/20 |
| BON | 992.71 ± 619.08 | 100.37 ± 45.24 | 89.89 | 22.01 ± 8.17 | 7.35 ± 1.21 | 66.60 | 19/20 | 19/20 |
| CIB | 538.52 ± 271.05 | 139.70 ± 41.90 | 74.06 | 19.02 ± 4.88 | 9.30 ± 1.37 | 51.10 | 20/20 | 20/20 |
| BOB | 17.98 ± 7.00 | 14.19 ± 4.13 | 21.07 | 3.40 ± 0.68 | 2.90 ± 0.38 | 14.72 | 12/20 | 14/20 |
| BCM | 656.35 ± 655.26 | 94.74 ± 48.27 | 85.57 | 18.04 ± 9.72 | 7.60 ± 1.68 | 57.85 | 18/20 | 18/20 |
| ICBC | 26.09 ± 11.07 | 10.51 ± 3.22 | 59.70 | 4.25 ± 0.98 | 2.56 ± 0.40 | 39.83 | 20/20 | 20/20 |
| CCB | 5.39 ± 1.38 | 4.00 ± 1.10 | 25.77 | 1.87 ± 0.26 | 1.55 ± 0.20 | 17.15 | 17/20 | 17/20 |
| BOC | 34.96 ± 35.38 | 8.98 ± 2.75 | 74.30 | 4.37 ± 2.64 | 2.39 ± 0.36 | 45.45 | 15/20 | 14/20 |
| CNCB | 16.10 ± 23.31 | 10.69 ± 3.78 | 33.63 | 2.66 ± 1.74 | 2.61 ± 0.51 | 1.77 | 8/20 | 7/20 |

Note: For each seed, losses are computed over the 416 common target dates; entries report the mean ± sample SD across 20 paired seeds (ddof = 1). MSE is reported in units of 10⁻⁴ and MAE in units of 10⁻². Δ% = 100 × (1 − mean loss_text / mean loss_no-text), computed from unrounded mean losses with no-text as the denominator. A zero denominator is reported as NA; small positive denominators and negative reductions are retained. Wins count seeds whose text loss is strictly lower than no-text loss; ties are excluded and counts are out of 20.
