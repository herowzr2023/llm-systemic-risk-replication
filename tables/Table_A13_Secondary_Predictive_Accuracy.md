# Table A13. Secondary predictive-accuracy statistics

| h | Loss | Persistence | AR(1) | GBR | PatchTST |
| --- | --- | --- | --- | --- | --- |
| 1-day | MAE | -3.432\*\*\* | -3.467\*\*\* | -3.147\*\*\* | -9.317\*\*\* |
| 1-day | MSE | -1.847\* | -1.847\* | -2.106\*\* | -6.634\*\*\* |
| 5-day | MAE | -1.764\* | -1.847\* | -4.667\*\*\* | -5.670\*\*\* |
| 5-day | MSE | -2.412\*\* | -2.279\*\* | -3.783\*\*\* | -4.895\*\*\* |

For each target date, losses are averaged across the 20 groups. The loss differential is Informer minus comparator.
With n target dates, d̄ is the mean differential and γ₀ = Σ(dₜ − d̄)² / n. DM = d̄ / √(γ₀/n) × √((n − 1)/n). Two-sided probabilities use Student-t with n − 1 degrees of freedom.
Negative DM statistics favor Informer. Stars denote unadjusted p-values: \* p < 0.10, \*\* p < 0.05 and \*\*\* p < 0.01. DM values are rounded to three decimals; significance uses unrounded statistics.
