# Table A2-2. Held-out classification scores and 95% confidence intervals

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | --- | --- | --- | --- |
| Llama-3-Chinese-8B-Instruct-v3-emotion | 0.806[0.781,0.831] | 0.806[0.782,0.831] | 0.806[0.781,0.831] | 0.805[0.781,0.830] |
| Llama-3-Chinese-8B-Instruct-v2 | 0.692[0.663,0.720] | 0.712[0.685,0.739] | 0.692[0.663,0.720] | 0.695[0.667,0.723] |
| Llama-3-Chinese-8B-Instruct-v3 | 0.719[0.690,0.746] | 0.723[0.696,0.751] | 0.719[0.690,0.746] | 0.720[0.691,0.747] |
| Llama-3-Chinese-8B-Instruct | 0.704[0.675,0.732] | 0.706[0.677,0.735] | 0.704[0.675,0.732] | 0.704[0.675,0.733] |
| Llama-3-Chinese-8B | 0.506[0.474,0.536] | 0.507[0.471,0.543] | 0.506[0.474,0.536] | 0.475[0.441,0.508] |
| Llama-3-8B-Instruct | 0.698[0.669,0.726] | 0.697[0.668,0.726] | 0.698[0.669,0.726] | 0.692[0.662,0.721] |
| Llama-3-8B | 0.413[0.383,0.443] | 0.548[0.500,0.597] | 0.413[0.383,0.443] | 0.366[0.334,0.399] |
| ROBERTa | 0.626[0.596,0.656] | 0.626[0.596,0.657] | 0.626[0.596,0.656] | 0.621[0.591,0.652] |
| Chinese-ROBERTa-wwm-ext-large | 0.700[0.672,0.729] | 0.699[0.672,0.729] | 0.700[0.672,0.729] | 0.698[0.670,0.727] |
| Chinese-ROBERTa-wwm-ext | 0.697[0.668,0.724] | 0.698[0.669,0.726] | 0.697[0.668,0.724] | 0.696[0.667,0.723] |
| Chinese-BERT-wwm-ext | 0.685[0.656,0.714] | 0.688[0.659,0.717] | 0.685[0.656,0.714] | 0.683[0.653,0.712] |
| Chinese-BERT-wwm | 0.623[0.593,0.653] | 0.628[0.598,0.659] | 0.623[0.593,0.653] | 0.619[0.588,0.650] |

Note: Cells report held-out scores and marginal 95% confidence intervals: the 2.5th–97.5th percentiles of 10,000 bootstrap resamples of the 1,000 paired reference and predicted labels, preserving sample size. Classifiers and finalized reference labels remain fixed. Intervals reflect held-out sampling uncertainty, not retraining or label development, and do not test model rankings.
