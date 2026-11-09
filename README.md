# Replication package

Python code and inputs for reproducing the manuscript and appendix figures and tables. Methods and model specifications are described in the paper.

## Package information

| Item | Information |
| --- | --- |
| Compilation date | 2026-11-01 |
| Author | Zhongrui Wu |
| Contact | herowzr@gmail.com |

## Reference environment and estimated runtime

The reference workstation is an Intel Core i9-13900KF with 24 cores and 32 logical processors, 64 GB RAM configured as 2 x 32 GB DDR5-5600, one NVIDIA GeForce RTX 4090 with 24 GB VRAM and a 2 TB-class NVMe SSD. The reference training environment uses Windows 11 Pro for Workstations 64-bit, build 26200, NVIDIA driver 610.88, Python 3.12 and CUDA 12.4 with PyTorch 2.6.0. On Windows, sentiment-model training runs in WSL (Windows Subsystem for Linux). Training uses one GPU and one model at a time.

The 1,120 bank-level Informer fits are estimated at approximately **33.04 hours** sequentially on one RTX 4090. Allow **2 days** for the complete bank-level training run. Sentiment-classifier training and evaluation are estimated at approximately **1 day**, full-corpus sentiment inference and daily aggregation at **4–6 days**, and the complete sentiment workflow at approximately **7 days**. These are planning estimates for the reference configuration.

## Package structure

| Path | Contents |
| --- | --- |
| `data/` | Direct plotting inputs, saved predictions and forum-coverage aggregates. |
| `data-raw/` | Bank daily series, CoVaR history, sentiment diagnostics and anonymous classification labels. |
| `Figure_*.py`, `Table_*.py` | Figure and table replication scripts. |
| `run_all_replications.py` | Entry point for all computational replications. |
| `code/` | Data preparation and model training code. |
| `plots/` | Generated figures. |
| `tables/` | Generated tables in CSV and Markdown. |
| `DATA_AVAILABILITY.md` | Source-data access, copyright and redistribution information. |

## Run all replications

Run from the replication-package directory.

```bash
python -m pip install -r requirements.txt
python run_all_replications.py
```

The entry point runs 16 steps using the supplied inputs and saved predictions and writes three figures and 20 table outputs. Model training is available through the scripts in `code/`.

The conceptual diagrams and configuration or definition tables are Fig. 1, Fig. 3, Tables 1–2, A1, A5 and A8.

## Figure environments

Use these environments to reproduce the figures pixel for pixel.

| Figures | Python | Matplotlib | Other packages |
| --- | --- | --- | --- |
| Figure 2 | 3.14.6 | 3.11.1 | NumPy 2.5.1, Pillow 12.3.0; Times New Roman font |
| Figures A1–A2 | 3.12.14 | 3.10.7 | NumPy 2.3.5, pandas 3.0.1 |

The root `requirements.txt` specifies the Figure 2 environment. Run Figure 2 in that environment.

```bash
python Figure_02_CrossBank_Diagnostics.py
```

Run Figures A1–A2 in a separate Python 3.12 environment.

```bash
python -m pip install matplotlib==3.10.7 numpy==2.3.5 pandas==3.0.1
python Figure_A1_Forecast_Paths.py
python Figure_A2_Forecast_Paths.py
```

The all-in-one entry point uses the same selected Python interpreter for every script. Use the corresponding figure environments above for pixel-identical rendering.

## Individual replications

Run an individual script with `python <script>` from the package directory. Shared input names below refer to files in `data/` unless a different directory is shown.

| Figure or table | Script | Input |
| --- | --- | --- |
| Figure 2 | `Figure_02_CrossBank_Diagnostics.py` | `monthly_median_range.csv` |
| Figures A1–A2 | `Figure_A1_Forecast_Paths.py`, `Figure_A2_Forecast_Paths.py` | `figure_A1_predictions.csv`, `figure_A2_predictions.csv` |
| Table 3 | `Table_03_CrossBank_Summary.py` | `predictions_1day_20pairs.csv`, `predictions_5day_20pairs.csv` |
| Table 4 | `Table_04_Bank_Level_Forecast_Results.py` | Same paired prediction files |
| Table 5 | `Table_05_Primary_High_Risk_Evaluation.py` | Paired predictions and `data-raw/All_CoVar_no_other_diff.csv` |
| Table 6 | `Table_06_Secondary_Aggregate_Losses.py` | `aggregate_predictions_h1_20groups.csv`, `aggregate_predictions_h5_20groups.csv` |
| Table A2 | `Table_A2_Sentiment_Classification.py` | `sentiment_classification_predictions.csv` |
| Table A2-1 | `Table_A2_1_Class_Distribution.py` | `data-raw/sentiment_classification/sentiment_labels.csv` and `split_ids.json` |
| Table A2-2 | `Table_A2_2_Heldout_Confidence_Intervals.py` | `sentiment_classification_predictions.csv` |
| Tables A3–A4 | `Table_A3_Sentiment_Descriptive_Statistics.py`, `Table_A4_Lagged_Sentiment_Risk_Correlations.py` | ICBC series in `data-raw/full_period/with_sentiment/` and sector series in `data-raw/sentiment_diagnostics/` |
| Tables A6–A7 | `Table_A6_A7_Bank_Level_Forecast_Losses.py` | Paired prediction files |
| Tables A9–A11 | `Table_A9_A11_High_Risk_Robustness.py` | Paired predictions and CoVaR history |
| Tables A12–A13 | `Table_A12_A13_Secondary_Aggregate_Results.py` | Aggregate prediction files |
| Table A14 | `Table_A14_Forum_Coverage.py` | Four aggregate inputs in `data/forum_coverage/` |

For example, reproduce Table 4 with the following command.

```bash
python Table_04_Bank_Level_Forecast_Results.py
```

## Code

`code/` contains scripts for data preparation, bank-level and aggregate forecasting, sentiment-classifier training, and large language model fine-tuning, prediction and inference. The methods and model settings are described in the paper.

Training seeds are recorded in `code/training_seeds.csv` and `code/aggregate_training_seeds.csv`. Each bank-level experiment uses the same seed for the models with and without sentiment.

## Informer citation

The implementation in `code/informer/` is adapted from [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020). The upstream Apache-2.0 license and attribution are retained in `code/informer/LICENSE` and `code/informer/NOTICE`.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(12), 11106–11115. [doi:10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325).
