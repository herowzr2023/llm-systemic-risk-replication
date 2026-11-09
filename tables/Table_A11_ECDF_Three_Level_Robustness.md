# Table A11. Recent-history ECDF three-level robustness

| h | Spec | Accuracy | Macro-F1 | Ordinal MAE | High recall | High FPR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | no_text | 0.564054 | 0.533780 | 0.511976 | 0.673311 | 0.213973 |
| 1 | text | 0.673051 | 0.661620 | 0.345905 | 0.755136 | 0.158131 |
| 5 | no_text | 0.566286 | 0.541521 | 0.511521 | 0.560707 | 0.117341 |
| 5 | text | 0.586204 | 0.573467 | 0.471274 | 0.616147 | 0.132570 |

Note. SR = -100 × raw CoVaR. At each forecast origin, actual and predicted SR are ranked separately against the latest 252 observed SR values through and including that origin. The ECDF count is the number of history values ≤ SR. Counts below 84 give grade 0, counts 84–167 give grade 1, and counts 168–252 give grade 2. Accuracy is the fraction of matching grades; ordinal MAE is mean absolute grade distance. Macro-F1 averages the three count-based class F1 scores within a seed and is NA unless all three actual classes have positive support. High recall and High FPR treat grade 2 as the positive class; zero denominators are NA. Metrics are calculated within each seed, averaged over 20 paired seeds within each bank, then averaged equally over 14 banks. Any undefined required seed or bank makes the corresponding summary NA; available subsets do not replace strict means.
