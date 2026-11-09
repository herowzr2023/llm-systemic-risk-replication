"""Prepare the final Figure 2 monthly table from the packaged daily data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / 'data-raw/full_period/with_sentiment'
BANKS = [
    ('000001', 'ping_an_bank', '平安银行'),
    ('002142', 'bank_of_ningbo', '宁波银行'),
    ('600000', 'shanghai_pudong_development_bank', '浦发银行'),
    ('600015', 'huaxia_bank', '华夏银行'),
    ('600016', 'china_minsheng_bank', '民生银行'),
    ('600036', 'china_merchants_bank', '招商银行'),
    ('601009', 'bank_of_nanjing', '南京银行'),
    ('601166', 'industrial_bank', '兴业银行'),
    ('601169', 'bank_of_beijing', '北京银行'),
    ('601328', 'bank_of_communications', '交通银行'),
    ('601398', 'industrial_and_commercial_bank_of_china', '工商银行'),
    ('601939', 'china_construction_bank', '建设银行'),
    ('601988', 'bank_of_china', '中国银行'),
    ('601998', 'china_citic_bank', '中信银行'),
]
SERIES = [
    ('Emotional_Skew', 'sentiment_direction'),
    ('Total_Count', 'attention_post_count'),
    ('bank_CoVaR', 'systemic_risk'),
    ('risk_spillover', 'dy_connectedness_to'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    output = output.resolve()
    if output.is_relative_to((PACKAGE / 'data-raw').resolve()):
        raise ValueError('The output must be outside data-raw.')
    inputs = []
    for code, slug, bank in sorted(BANKS, key=lambda item: item[2]):
        path = DATA / f'{slug}_{code}_daily_with_sentiment.csv'
        before = sha(path)
        frame = pd.read_csv(path)
        frame['date'] = pd.to_datetime(frame['date'], errors='raise')
        assert not frame['date'].duplicated().any(), bank
        aliases = [alias for _, alias in SERIES]
        values = frame.set_index('date')[aliases].apply(pd.to_numeric, errors='raise').sort_index()
        values.columns = ['Emotional_Skew', 'Total_Count', bank + '_CoVaR', 'risk_spillover']
        assert np.isfinite(values.to_numpy()).all(), bank
        inputs.append((bank, path, before, values))
    assert len(inputs) == 14
    assert set(DATA.glob('*.csv')) == {path for _, path, _, _ in inputs}
    # The observed-date union covers every trading day in the common May 2010–January 2023 sample.
    calendar = pd.DatetimeIndex(sorted(set().union(*(values.index for _, _, _, values in inputs))))
    end = pd.Timestamp('2023-01-31')
    records = []
    for bank, path, before, values in inputs:
        means, stds = values.mean(), values.std(ddof=1)
        assert np.isfinite(stds).all() and (stds > 0).all(), bank
        dates = calendar[(calendar >= values.index.min()) & (calendar <= end)]
        filled = values.reindex(dates).ffill()
        assert np.isfinite(filled.to_numpy()).all(), bank
        actual_dates = values.index[values.index <= end]
        assert np.array_equal(filled.loc[actual_dates].to_numpy(), values.loc[actual_dates].to_numpy())
        monthly = ((filled - means) / stds).resample('ME').mean()
        for column, (metric, _) in zip(values.columns, SERIES):
            for date, value in monthly[column].items():
                records.append(dict(bank=bank, metric=metric, month=date.strftime('%Y-%m-%d'), monthly_mean=value))
        assert sha(path) == before
    long = pd.DataFrame(records)
    wide = {metric: long[long.metric.eq(metric)].pivot(index='month', columns='bank', values='monthly_mean')
            for metric, _ in SERIES}
    common = sorted(set.intersection(*(set(frame.dropna().index) for frame in wide.values())))
    assert len(common) == 153 and common[0] == '2010-05-31' and common[-1] == '2023-01-31'
    summaries = []
    for metric, _ in SERIES:
        rows = wide[metric].loc[common]
        assert (rows.count(axis=1) == 14).all()
        for month, row in rows.iterrows():
            minimum, median, maximum = float(row.min()), float(row.median()), float(row.max())
            assert minimum <= median <= maximum
            summaries.append(dict(month=month, metric=metric, median=median, minimum=minimum,
                                  maximum=maximum, n_banks=int(row.count())))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output, index=False, encoding='utf-8-sig', float_format='%.17g', lineterminator='\r\n')
    assert all(sha(path) == before for _, path, before, _ in inputs)
    print(json.dumps({'output': str(output), 'months': len(common), 'rows': len(summaries),
                      'sha256': sha(output), 'source_files_unchanged': True}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PACKAGE / 'data/monthly_median_range.csv',
                        help='Final monthly CSV (default: data/monthly_median_range.csv in this package).')
    prepare(parser.parse_args().output)


if __name__ == '__main__':
    main()
