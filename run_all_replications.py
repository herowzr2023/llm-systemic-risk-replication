"""Run every computational figure and table replication in sequence.

This entry point uses the supplied frozen inputs and prediction files. It does
not retrain forecasting or sentiment models.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STEPS = (
    ("Figure 2", "Figure_02_CrossBank_Diagnostics.py", ()),
    ("Table 3", "Table_03_CrossBank_Summary.py", ()),
    ("Table 4", "Table_04_Bank_Level_Forecast_Results.py", ()),
    ("Table 5", "Table_05_Primary_High_Risk_Evaluation.py", ()),
    ("Table 6", "Table_06_Secondary_Aggregate_Losses.py", ()),
    ("Table A2", "Table_A2_Sentiment_Classification.py", ()),
    ("Table A2-1", "Table_A2_1_Class_Distribution.py", ()),
    ("Table A2-2", "Table_A2_2_Heldout_Confidence_Intervals.py", ()),
    ("Table A3", "Table_A3_Sentiment_Descriptive_Statistics.py", ()),
    ("Table A4", "Table_A4_Lagged_Sentiment_Risk_Correlations.py", ()),
    ("Figure A1", "Figure_A1_Forecast_Paths.py", ()),
    ("Figure A2", "Figure_A2_Forecast_Paths.py", ()),
    ("Tables A6-A7", "Table_A6_A7_Bank_Level_Forecast_Losses.py", ()),
    ("Tables A9-A11", "Table_A9_A11_High_Risk_Robustness.py", ()),
    ("Tables A12-A13", "Table_A12_A13_Secondary_Aggregate_Results.py", ()),
    ("Table A14", "Table_A14_Forum_Coverage.py", ()),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python executable used for each replication script.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run all remaining steps after a failure and return status 1.")
    args = parser.parse_args()
    python = args.python.resolve()
    failures = []
    started = datetime.now(timezone.utc).isoformat()
    print(f"Replication run started: {started}")
    print(f"Package root: {ROOT}")
    print("Model retraining: not included; supplied frozen inputs and predictions are used.")
    for number, (label, script, extra) in enumerate(STEPS, start=1):
        path = ROOT / script
        if not path.is_file():
            failures.append((label, f"missing script: {path}"))
            print(f"[{number:02d}/{len(STEPS):02d}] {label}: MISSING")
            if not args.continue_on_error:
                break
            continue
        print(f"[{number:02d}/{len(STEPS):02d}] {label}: {script}")
        completed = subprocess.run([str(python), str(path), *extra], cwd=ROOT)
        if completed.returncode:
            failures.append((label, f"exit code {completed.returncode}"))
            print(f"{label}: FAILED")
            if not args.continue_on_error:
                break
        else:
            print(f"{label}: OK")
    if failures:
        print("Replication run failed:")
        for label, reason in failures:
            print(f"- {label}: {reason}")
        return 1
    print(f"Replication run completed: {len(STEPS)} computational steps finished.")
    print("Conceptual/configuration items without computational scripts: Fig. 1, Fig. 3, Tables 1-2, A1, A5 and A8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
