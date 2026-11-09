# Table 6. Secondary aggregate losses for direct forecasts

| Model | h1 MSE (×10⁻⁴) | h1 MAE (×10⁻²) | h5 MSE (×10⁻³) | h5 MAE (×10⁻²) |
| --- | --- | --- | --- | --- |
| Informer | 3.96 | 1.46 | 1.78 | 3.17 |
| Persistence | 4.60 | 1.62 | 2.17 | 3.35 |
| AR(1) | 4.58 | 1.62 | 2.11 | 3.35 |
| GBR | 4.52 | 1.59 | 2.15 | 3.61 |
| PatchTST | 7.89 | 2.08 | 2.75 | 3.80 |

MSE and MAE are computed over target dates within each group, then averaged across the 20 groups.
Values are rounded to two decimals in the indicated units. Each horizon retains its own target-date sample.
