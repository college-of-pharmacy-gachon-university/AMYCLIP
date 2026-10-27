#!/usr/bin/env python3
"""Compare regenerated example scores with the committed checkpoint result."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    expected = pd.read_csv(args.expected)
    observed = pd.read_csv(args.observed)
    if list(expected.columns) != list(observed.columns):
        raise SystemExit("FAIL: output columns differ")
    if len(expected) != len(observed):
        raise SystemExit("FAIL: output row counts differ")
    numeric = expected.select_dtypes(include=[np.number]).columns
    difference = np.abs(expected[numeric].to_numpy() - observed[numeric].to_numpy())
    maximum = float(difference.max(initial=0.0))
    if maximum > args.tolerance:
        raise SystemExit(f"FAIL: maximum numeric difference {maximum:.3g}")
    print(f"PASS: checkpoint inference reproduced; maximum difference={maximum:.3g}")


if __name__ == "__main__":
    main()
