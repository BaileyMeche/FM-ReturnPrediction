#!/usr/bin/env python3
"""
Test script that replicates Lewellen (2014) Table 1 exactly as shown in the image.
All values below are hard-coded to demonstrate the final table format.
"""

import pandas as pd
import numpy as np

def replicate_table_1_test() -> pd.DataFrame:
    """
    Return a DataFrame that matches the Lewellen (2014) Table 1 exactly.
    Columns are a 2-level MultiIndex:
        [("All stocks", [Avg, Std, N]),
         ("All-but-tiny stocks", [Avg, Std, N]),
         ("Large stocks", [Avg, Std, N])]
    Rows are the variables in the same order shown in the table.
    """
    # Row labels as they appear in the published table:
    row_labels = [
        "Return (%)",
        "LogSize_{-1}",
        "LogB/M_{-1}",
        "Return_{-2,-12}",
        "LogIssues_{-1,-36}",
        "Accruals_{yr-1}",
        "ROA_{yr-1}",
        "LogAG_{yr-1}",
        "DY_{-1,-12}",
        "LogReturn_{-13,-36}",
        "LogIssues_{-1,-12}",
        "Beta_{-1,-36}",
        "StdDev_{-1,-12}",
        "Turnover_{-1,-12}",
        "Debt/Price_{yr-1}",
        "Sales/Price_{yr-1}",
    ]

    # Columns as a MultiIndex for [subset, statistic]
    col_tuples = [
        ("All stocks", "Avg"), ("All stocks", "Std"), ("All stocks", "N"),
        ("All-but-tiny stocks", "Avg"), ("All-but-tiny stocks", "Std"), ("All-but-tiny stocks", "N"),
        ("Large stocks", "Avg"), ("Large stocks", "Std"), ("Large stocks", "N"),
    ]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["Subset", "Statistic"])

    # Hard-coded table values row by row (matching the image exactly)
    # Each row has 9 values: [AllStocks: Avg, Std, N,  AllButTiny: Avg, Std, N,  Large: Avg, Std, N].
    data = [
        [ 1.27, 14.79, 3955,  1.12,  9.84, 1706,  1.03,  8.43,  876],  # Return (%)
        [ 4.63,  1.93, 3955,  6.38,  1.18, 1706,  7.30,  0.90,  876],  # LogSize_{-1}
        [-0.51,  0.84, 3955, -0.73,  0.73, 1706, -0.81,  0.71,  876],  # LogB/M_{-1}
        [ 0.13,  0.48, 3955,  0.20,  0.41, 1706,  0.19,  0.36,  876],  # Return_{-2,-12}
        [ 0.11,  0.25, 3519,  0.10,  0.22, 1583,  0.09,  0.21,  837],  # LogIssues_{-1,-36}
        [-0.02,  0.10, 3656, -0.02,  0.08, 1517, -0.03,  0.07,  778],  # Accruals_{yr-1}
        [ 0.01,  0.14, 3896,  0.05,  0.08, 1679,  0.06,  0.07,  865],  # ROA_{yr-1}
        [ 0.12,  0.26, 3900,  0.15,  0.22, 1680,  0.14,  0.20,  865],  # LogAG_{yr-1}
        [ 0.02,  0.02, 3934,  0.02,  0.02, 1702,  0.03,  0.02,  875],  # DY_{-1,-12}
        [ 0.24,  0.58, 3417,  0.23,  0.46, 1556,  0.25,  0.41,  828],  # LogReturn_{-13,-36}
        [ 0.04,  0.12, 3953,  0.03,  0.10, 1706,  0.03,  0.10,  876],  # LogIssues_{-1,-12}
        [ 0.96,  0.55, 3720,  1.06,  0.50, 1639,  1.05,  0.46,  854],  # Beta_{-1,-36}
        [ 0.15,  0.08, 3954,  0.11,  0.04, 1706,  0.09,  0.03,  876],  # StdDev_{-1,-12}
        [ 0.08,  0.08, 3666,  0.10,  0.08, 1635,  0.09,  0.08,  857],  # Turnover_{-1,-12}
        [ 0.83,  1.59, 3908,  0.64,  1.16, 1677,  0.61,  1.09,  864],  # Debt/Price_{yr-1}
        [ 2.53,  3.56, 3905,  1.59,  1.95, 1677,  1.37,  1.52,  865],  # Sales/Price_{yr-1}
    ]

    table_1 = pd.DataFrame(data, index=row_labels, columns=columns)
    return table_1


def main():
    table_1 = replicate_table_1_test()
    print(table_1)
    # Optionally, write to CSV or Excel for further checks:
    # table_1.to_csv("table_1_test.csv", float_format="%.2f")

if __name__ == "__main__":
    main()
