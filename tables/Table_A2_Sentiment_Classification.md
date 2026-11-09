# Table A2. Sentiment-Classification Performance: Weighted and Macro Summaries

## Panel A. Training set (n = 7,000)

| Model | Accuracy¹ | Weighted P | Weighted F1 | Macro P | Macro R | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3-Chinese-8B-Instruct-v3-emotion | 0.976 | 0.976 | 0.976 | 0.975 | 0.975 | 0.975 |
| Llama-3-Chinese-8B-Instruct-v2 | 0.689 | 0.711 | 0.692 | 0.696 | 0.694 | 0.688 |
| Llama-3-Chinese-8B-Instruct-v3 | 0.723 | 0.726 | 0.723 | 0.717 | 0.719 | 0.716 |
| Llama-3-Chinese-8B-Instruct | 0.729 | 0.732 | 0.730 | 0.722 | 0.725 | 0.723 |
| Llama-3-Chinese-8B | 0.526 | 0.531 | 0.494 | 0.531 | 0.490 | 0.474 |
| Llama-3-8B-Instruct | 0.712 | 0.711 | 0.707 | 0.710 | 0.696 | 0.699 |
| Llama-3-8B | 0.423 | 0.573 | 0.374 | 0.544 | 0.447 | 0.371 |
| ROBERTa | 0.667 | 0.667 | 0.663 | 0.667 | 0.652 | 0.655 |
| Chinese-ROBERTa-wwm-ext-large | 0.721 | 0.721 | 0.720 | 0.718 | 0.714 | 0.715 |
| Chinese-ROBERTa-wwm-ext | 0.706 | 0.707 | 0.705 | 0.704 | 0.699 | 0.700 |
| Chinese-BERT-wwm-ext | 0.700 | 0.702 | 0.697 | 0.702 | 0.687 | 0.691 |
| Chinese-BERT-wwm | 0.656 | 0.658 | 0.651 | 0.658 | 0.640 | 0.644 |

## Panel B. Held-out test set (n = 1,000)

| Model | Accuracy¹ | Weighted P | Weighted F1 | Macro P | Macro R | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3-Chinese-8B-Instruct-v3-emotion | 0.806 | 0.806 | 0.805 | 0.806 | 0.802 | 0.804 |
| Llama-3-Chinese-8B-Instruct-v2 | 0.692 | 0.712 | 0.695 | 0.698 | 0.694 | 0.690 |
| Llama-3-Chinese-8B-Instruct-v3 | 0.719 | 0.723 | 0.720 | 0.715 | 0.717 | 0.714 |
| Llama-3-Chinese-8B-Instruct | 0.704 | 0.706 | 0.704 | 0.698 | 0.700 | 0.699 |
| Llama-3-Chinese-8B | 0.506 | 0.507 | 0.475 | 0.509 | 0.473 | 0.457 |
| Llama-3-8B-Instruct | 0.698 | 0.697 | 0.692 | 0.696 | 0.681 | 0.683 |
| Llama-3-8B | 0.413 | 0.548 | 0.366 | 0.525 | 0.445 | 0.368 |
| ROBERTa | 0.626 | 0.626 | 0.621 | 0.624 | 0.611 | 0.614 |
| Chinese-ROBERTa-wwm-ext-large | 0.700 | 0.699 | 0.698 | 0.698 | 0.692 | 0.694 |
| Chinese-ROBERTa-wwm-ext | 0.697 | 0.698 | 0.696 | 0.695 | 0.690 | 0.691 |
| Chinese-BERT-wwm-ext | 0.685 | 0.688 | 0.683 | 0.687 | 0.675 | 0.678 |
| Chinese-BERT-wwm | 0.623 | 0.628 | 0.619 | 0.628 | 0.608 | 0.612 |

Note: P = precision; R = recall. Weighted metrics use reference-label class frequencies; macro metrics weight classes equally. ¹ In this single-label three-class setting, accuracy equals weighted recall and micro precision, recall and F1, so these values appear once. Scores use the same final pre-2012 split and three-decimal rounding. Macro and micro summaries are aggregation checks, not independent replications.
