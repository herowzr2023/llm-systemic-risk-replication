"""Reproduce Table A4 using previous-observation sentiment and current risk.

Pearson correlations use a two-sided test and the exact beta null distribution.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

from Table_A3_Sentiment_Descriptive_Statistics import INDICATORS, ROOT, read_observations, resolve_input_paths


STEM = "Table_A4_Lagged_Sentiment_Risk_Correlations"
CAPTION = "Table A4. Correlations Between Previous-Observation-Lagged Sentiment Indicators and Subsequent Risk Measures"
RISKS = (("ICBC", "systemic_risk", "SR proxy"),
         ("ICBC", "dy_connectedness_to", "DY spillover To"),
         ("Banking sector", "systemic_risk", "SR proxy"))
OUTPUT_COLUMNS = ("Index", *(f"{scope} {title}" for scope, _, title in RISKS))
NOTE = ("Correlations use the SR series employed in the analysis and DY spillover To. "
        "*, ** and *** denote nominal two-sided Pearson-correlation p-values below 0.10, "
        "0.05 and 0.01, respectively; these descriptive tests are not multiplicity-adjusted.")


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction with modified Lentz steps."""
    tiny = 1e-300

    def nonzero(value):
        return value if abs(value) >= tiny else math.copysign(tiny, value)

    c = 1.0
    d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0))
    result = d
    for m in range(1, 401):
        twice = 2 * m
        numerator = m * (b - m) * x / ((a + twice - 1) * (a + twice))
        d = 1.0 / nonzero(1.0 + numerator * d)
        c = nonzero(1.0 + numerator / c)
        result *= d * c
        numerator = -(a + m) * (a + b + m) * x / ((a + twice) * (a + twice + 1))
        d = 1.0 / nonzero(1.0 + numerator * d)
        c = nonzero(1.0 + numerator / c)
        change = d * c
        result *= change
        if abs(change - 1.0) <= 3e-14:
            return result
    raise ValueError("Incomplete-beta evaluation did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def pearson_test(x, y):
    count = len(x)
    if count != len(y) or count < 2:
        raise ValueError("Pearson correlation requires at least two paired observations")
    x_mean, y_mean = math.fsum(x) / count, math.fsum(y) / count
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    x_norm, y_norm = math.hypot(*x_centered), math.hypot(*y_centered)
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant series")
    r = max(-1.0, min(1.0, math.fsum((left / x_norm) * (right / y_norm)
                                    for left, right in zip(x_centered, y_centered))))
    if count == 2:
        return r, 1.0, count
    a = (count - 2) / 2
    # Choose the smaller beta tail to preserve both tiny p-values and r near zero.
    if r * r < 3.0 / (count + 3):
        p = 1.0 - regularized_beta(r * r, 0.5, a)
    else:
        p = regularized_beta((1.0 - r) * (1.0 + r), a, 0.5)
    return r, max(0.0, min(1.0, p)), count


def significance_stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def calculate_table(data_directory: Path | None = None):
    observations, samples = {}, {}
    for scope, path in resolve_input_paths(data_directory).items():
        columns = [column for column, _ in INDICATORS]
        columns += [risk for risk_scope, risk, _ in RISKS if risk_scope == scope]
        observations[scope] = read_observations(path, columns)
        series = observations[scope]
        if not series:
            raise ValueError(f"{path.name}: no observations")
        samples[scope] = (len(series), series[0]["date"], series[-1]["date"])
    rows, details = [], []
    for indicator, title in INDICATORS:
        row = {"Index": title}
        for scope, risk, risk_title in RISKS:
            series = observations[scope]
            pairs = [(previous[indicator], current[risk]) for previous, current in zip(series, series[1:])
                     if previous[indicator] is not None and current[risk] is not None]
            r, p, count = pearson_test([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            stars = significance_stars(p)
            row[f"{scope} {risk_title}"] = f"{r:.3f}{stars}"
            details.append({"Indicator": title, "Scope": scope, "Risk": risk_title,
                            "R": r, "P": p, "N": count, "Stars": stars})
        rows.append(row)
    return rows, samples, details


def write_outputs(rows, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{STEM}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {CAPTION}", "", "| " + " | ".join(OUTPUT_COLUMNS) + " |",
             "| " + " | ".join("---" for _ in OUTPUT_COLUMNS) + " |"]
    lines += ["| " + " | ".join(row[column] for column in OUTPUT_COLUMNS) + " |" for row in rows]
    lines += ["", f"Note: {NOTE}", ""]
    md_path = output_directory / f"{STEM}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Table A4 from full-period sentiment and risk.")
    parser.add_argument("--data-dir", type=Path,
                        help="Read icbc_daily_sentiment.csv and banking_sector_daily_sentiment.csv from this directory; defaults use full_period for ICBC and sentiment_diagnostics for the sector.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Directory for the Table A4 CSV and Markdown outputs.")
    arguments = parser.parse_args()
    try:
        rows, samples, details = calculate_table(arguments.data_dir)
        outputs = write_outputs(rows, arguments.output_dir)
    except (OSError, ValueError, OverflowError, csv.Error, UnicodeError) as error:
        print(f"Table A4 failed: {error}", file=sys.stderr)
        return 1
    for scope, (count, first, last) in samples.items():
        print(f"{scope}: {count:,} observations, {first} to {last}.")
    for result in details:
        print(f"{result['Indicator']}, {result['Scope']}, {result['Risk']}: "
              f"r={result['R']:.12g}, p={result['P']:.12g}, n={result['N']}.")
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
