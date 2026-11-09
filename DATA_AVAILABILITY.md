# Data availability

The complete post-level forum archive is not redistributed with this replication package because of third-party copyright and redistribution-rights considerations. The source is [Eastmoney Guba](https://guba.eastmoney.com/); source content is subject to the rights of its respective owners and the platform's [terms of service](https://about.eastmoney.com/home/protocol).

## Sentiment classification

The public sentiment-classification inputs contain 10,000 final labels in `data-raw/sentiment_classification/sentiment_labels.csv` (`sample_id,final_label`), the ordered 7,000/2,000/1,000 split in `split_ids.json`, and 96,000 saved train/test predictions in `data/sentiment_classification_predictions.csv` (`model,split,sample_id,true_label,predicted_label`). Anonymous sample IDs link these files. The package contains no bank-title text, raw model-generated text, source-post identifiers or mapping to forum records.

These public inputs reproduce Tables A2, A2-1 and A2-2 without source text. Text-based classifier training and inference require a separate CSV supplied by the researcher through `--labels`, with columns `sample_id,bank,title,final_label` and sample IDs matching the public fixed split. Researchers must obtain the bank-title text separately, subject to source access and reuse permissions. The training scripts are described briefly in [README.md](README.md#code).

## Forum coverage

The public inputs in `data/forum_coverage/` contain bank-by-subperiod post counts, active-day counts, distinct-author-label counts, calendar-period metadata, archive inventory totals and separate marginal frequency distributions for views and comments. They contain no post identifiers, individual publication dates, author labels, titles, text or URLs. View and comment distributions are not linked at the post level.

`Table_A14_Forum_Coverage.py` reproduces all four Table A14 panels from these aggregate inputs. It calculates calendar exposure, zero-post days and active-day percentages, and reconstructs record-level linear quantiles from the marginal frequencies. Post, active-day and distinct-author-label counts are provided as aggregate statistics. `code/prepare_forum_coverage_data.py` generates the aggregate inputs for researchers with access to the retained source records.

Researchers seeking source forum records should consult the source platform and obtain any required access and reuse permissions. Any provision of post-level records for independent verification is subject to the applicable rights and permissions. This package does not grant redistribution rights to source forum content.
