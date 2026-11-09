# Table A9. Supplementary rolling-q90 metrics

| h | Spec | HR-MSE | HR-Bias | FPR | Precision |
| --- | --- | --- | --- | --- | --- |
| 1 | no_text | 0.036850 | -0.074156 | 0.044250 | NA |
| 1 | text | 0.008467 | -0.030988 | 0.024927 | 0.817235 |
| 5 | no_text | 0.042074 | -0.106870 | 0.023609 | NA |
| 5 | text | 0.022418 | -0.078949 | 0.025350 | NA |

Note. SR = -100 × raw CoVaR. Each q90 cutoff is the linearly interpolated 90th percentile of the latest 252 observed SR values through and including the forecast origin. Actual high-risk days and forecast alarms use ≥ the same cutoff. HR-MSE and HR-Bias use realized high-risk days; signed error is forecast minus realization. FPR = FP/(FP+TN) and Precision = TP/(TP+FP). Conditional errors without realized events and rates with zero denominators are NA. Metrics are calculated within each seed, averaged over 20 paired seeds within each bank, then averaged equally over 14 banks. Any undefined required seed or bank makes the corresponding summary NA; available subsets do not replace strict means.
