
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

 
# ==============================================================================================
# STOCK DATA
# ==============================================================================================

#Table I presents return forecasting results based on 6, 25, and 100 book-to-market ratios of size- and value-sorted portfolios of U.S. stocks (Fama and French (1993)). 
#We consider two different forecasting horizons—1 month and 1 year—and report findings for both in-sample and out-of-sample forecasts.

def calculate_book_market_ratio(crsp):
    """
    Calculate market equity AND book-to-market ratio for each firm in the CRSP dataset.

    Similar logic to the original calculate_market_equity function:
      - For a given permco and date, sum all individual permno MEs.
      - Assign the aggregated ME to the permno that has the largest ME.
      - Remove other permnos.
      - Also sum book equity across any permnos for that permco–date,
        then compute bm = BE / ME and keep it on the retained permno.
    """
    df = crsp.copy()

    # Drop rows missing the necessary columns.
    # We need 'mthprc', 'shrout', and 'be' to compute everything reliably.
    df = df.dropna(subset=["mthprc", "shrout", "be"])

    # 1. Calculate Market Equity at the permno level
    df["permno_me"] = df["mthprc"].abs() * df["shrout"]

    # 2. Sum (firm-level) ME: for each permco–jdate, sum the MEs of all permnos
    df["me"] = df.groupby(["permco", "jdate"])["permno_me"].transform("sum")

    # 3. Sum (firm-level) BE: for each permco–jdate, sum the BEs of all permnos
    #    (Typically there's only one "book equity" per firm, but if you somehow 
    #     have multiple lines per date, we just aggregate them.)
    df["firm_be"] = df.groupby(["permco", "jdate"])["be"].transform("sum")

    # 4. Identify the permno with the largest ME within each permco–jdate
    is_max = df["permno_me"] == df.groupby(["permco", "jdate"])["permno_me"].transform("max")
    df = df.loc[is_max].copy()

    # 5. Convert permco to int (consistent with the original function)
    df["permco"] = df["permco"].astype("int64")

    # 6. Calculate the Book-to-Market ratio
    df["bm"] = df["firm_be"] / df["me"]

    # 7. Drop the temporary columns used in the calculation
    df.drop(columns=["permno_me", "firm_be"], inplace=True)

    return df



def get_company_subsets(crsp):
    #We use book-to-market ratios for Fama and French's (1993) size- and value-sorted portfolios (in which U.S. stocks are divided into 6, 25, or 100 portfolios).
    return


# ==============================================================================================
# INDEX DATA
# ==============================================================================================