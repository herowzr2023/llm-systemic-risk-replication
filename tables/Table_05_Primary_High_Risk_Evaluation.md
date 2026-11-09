# Table 5. Primary high-risk evaluation of direct forecasts: rolling 252-observation q90

| h | Specification | HR MAE | Recall | Precision | F1 | AP | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Without text | 0.1068 | 0.7066 | NA | 0.6668 | 0.8380 | 0.0443 |
| 1 | With text | 0.0604 | 0.7909 | 0.8172 | 0.7868 | 0.8742 | 0.0249 |
| 5 | Without text | 0.1240 | 0.5804 | NA | 0.6295 | 0.8462 | 0.0236 |
| 5 | With text | 0.0968 | 0.6519 | NA | 0.6834 | 0.8522 | 0.0253 |

Note. SR = −100 × raw CoVaR. Each cutoff is the linear 90th percentile of the latest 252 observed SR values through and including the forecast origin. Actual high-risk days and forecast alarms use ≥ the same cutoff. HR MAE uses actual high-risk days; AP ranks prediction minus cutoff with tied scores grouped. Metrics are calculated per seed, averaged over 20 paired seeds within each bank, then averaged equally over 14 banks. NA denotes an undefined denominator or one-class AP in any required seed, propagated through both means.
