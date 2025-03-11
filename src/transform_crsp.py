
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


def calculate_market_equity(crsp):
    """
    Calculate market equity for each firm in the CRSP dataset.

    There were cases when the same firm (permco) had two or more securities
    (permno) on the same date. For the purpose of ME for the firm, we
    aggregated all ME for a given permco, date. This aggregated ME was assigned
    to the CRSP permno that has the largest ME. We remove all other permnos
    except the one with the largest ME.

    """
    df = crsp.copy()
    df = df.dropna(subset=["mthprc", "shrout"])
    df["permno_me"] = df["mthprc"].abs() * df["shrout"] # Calculate Market Equity for each permno.

    df['me'] = df.groupby(['permco', 'jdate'])['permno_me'].transform('sum') # Compute the aggregated (firm-level) market equity
    is_max = df['permno_me'] == df.groupby(['permco', 'jdate'])['permno_me'].transform('max') # Find the permno with the largest market cap
    df = df.loc[is_max].copy() # Keep only the permno with the largest market cap

    df['permco'] = df['permco'].astype('int64')
    df = df.drop(columns=['permno_me'])

    return df


# ==============================================================================================
# INDEX DATA
# ==============================================================================================



