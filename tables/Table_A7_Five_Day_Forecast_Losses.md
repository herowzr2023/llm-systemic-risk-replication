# Table A7. Forecast Losses for Direct Five-Trading-Day-Ahead Predictions

| Bank | MSE no-text | MSE text | MSE Δ% | MAE no-text | MAE text | MAE Δ% | MSE wins | MAE wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PAB | 28.22 ± 34.17 | 11.87 ± 3.41 | 57.92 | 3.91 ± 2.31 | 2.77 ± 0.41 | 29.21 | 12/20 | 12/20 |
| NBCB | 1103.70 ± 977.40 | 510.42 ± 141.97 | 53.75 | 24.13 ± 12.73 | 18.22 ± 2.77 | 24.48 | 13/20 | 10/20 |
| SPDB | 346.88 ± 293.34 | 78.26 ± 31.84 | 77.44 | 14.37 ± 6.93 | 6.78 ± 1.25 | 52.83 | 19/20 | 18/20 |
| HB | 21.05 ± 16.02 | 6.86 ± 1.51 | 67.41 | 3.61 ± 1.68 | 2.06 ± 0.24 | 42.92 | 17/20 | 17/20 |
| CMBC | 201.73 ± 227.39 | 80.78 ± 52.44 | 59.95 | 9.55 ± 5.76 | 6.98 ± 2.30 | 26.94 | 12/20 | 12/20 |
| CMB | 15.89 ± 4.08 | 27.81 ± 26.26 | -75.07 | 3.26 ± 0.48 | 3.92 ± 1.83 | -20.01 | 9/20 | 11/20 |
| BON | 800.75 ± 655.88 | 524.45 ± 460.22 | 34.51 | 20.42 ± 9.97 | 16.03 ± 7.27 | 21.51 | 12/20 | 12/20 |
| CIB | 685.74 ± 753.70 | 244.46 ± 75.91 | 64.35 | 19.44 ± 11.79 | 12.38 ± 2.19 | 36.32 | 13/20 | 13/20 |
| BOB | 24.60 ± 9.68 | 23.13 ± 3.74 | 6.01 | 3.82 ± 0.76 | 3.76 ± 0.31 | 1.51 | 9/20 | 10/20 |
| BCM | 285.05 ± 301.18 | 137.79 ± 69.73 | 51.66 | 11.35 ± 5.66 | 9.25 ± 2.25 | 18.46 | 12/20 | 12/20 |
| ICBC | 29.05 ± 13.14 | 15.70 ± 2.57 | 45.96 | 4.35 ± 1.11 | 3.13 ± 0.26 | 28.00 | 19/20 | 19/20 |
| CCB | 10.66 ± 8.24 | 9.68 ± 4.02 | 9.21 | 2.53 ± 0.96 | 2.46 ± 0.55 | 2.92 | 9/20 | 9/20 |
| BOC | 42.78 ± 61.94 | 42.73 ± 21.50 | 0.12 | 4.59 ± 3.63 | 5.39 ± 1.63 | -17.46 | 5/20 | 5/20 |
| CNCB | 25.55 ± 9.58 | 34.95 ± 33.67 | -36.77 | 4.03 ± 0.91 | 4.27 ± 2.08 | -5.97 | 9/20 | 10/20 |

Note: For each seed, losses are computed over the 416 common target dates; entries report the mean ± sample SD across 20 paired seeds (ddof = 1). MSE is reported in units of 10⁻⁴ and MAE in units of 10⁻². Δ% = 100 × (1 − mean loss_text / mean loss_no-text), computed from unrounded mean losses with no-text as the denominator. A zero denominator is reported as NA; small positive denominators and negative reductions are retained. Wins count seeds whose text loss is strictly lower than no-text loss; ties are excluded and counts are out of 20.
