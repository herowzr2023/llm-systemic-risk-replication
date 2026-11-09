# Table A10. Rolling-q95 robustness

| h | Spec | HR-MAE | Recall | F1 | AP | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | no_text | 0.109474 | 0.637547 | 0.600399 | 0.803232 | 0.029678 |
| 1 | text | 0.064815 | 0.716343 | 0.726538 | 0.838057 | 0.017289 |
| 5 | no_text | 0.136239 | 0.430810 | 0.501488 | 0.797131 | 0.015201 |
| 5 | text | 0.108257 | 0.494485 | 0.565814 | 0.796961 | 0.015087 |

Note. SR = -100 × raw CoVaR. Each q95 cutoff is the linearly interpolated 95th percentile of the latest 252 observed SR values through and including the forecast origin. Actual high-risk days and forecast alarms use ≥ the same cutoff. HR-MAE uses realized high-risk days. Recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN), and FPR = FP/(FP+TN). AP is noninterpolated average precision of forecast-minus-cutoff margins, with tied scores grouped. Conditional errors without realized events, rates with zero denominators, and one-class AP are NA. Metrics are calculated within each seed, averaged over 20 paired seeds within each bank, then averaged equally over 14 banks. Any undefined required seed or bank makes the corresponding summary NA; available subsets do not replace strict means.
