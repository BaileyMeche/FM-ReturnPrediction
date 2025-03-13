"""
Transformations for CRSP Data
==============================

This module provides transformation utilities for CRSP market data. Its main
function calculates firm-level market equity (ME) by aggregating the market equity
of multiple securities (permnos) associated with the same firm (permco) on a given
date. The process involves computing individual security market equity, summing
these values at the firm level, and then retaining the security with the largest
market equity to represent the firm.

Function:
---------
- calculate_market_equity(crsp):
    Calculates the market equity for each firm in the CRSP dataset by first computing
    the market equity for each security as the product of the absolute monthly price
    and shares outstanding, aggregating these values by firm and date, and retaining the
    security (permno) with the highest individual market equity as the representative
    for the firm.

Usage:
------
This transformation is designed to prepare CRSP stock data for further analysis by
ensuring that each firm is represented by a single, aggregate market equity value,
which is essential for cross-sectional and time-series analyses in financial research.
"""


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

from utils import load_cache_data


# ==============================================================================================
# GLOBAL CONFIGURATION
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
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
    df = df.dropna(subset=["prc", "shrout"])
    df["permno_me"] = df["prc"].abs() * df["shrout"]  # Calculate Market Equity for each permno.

    # Compute aggregated (firm-level) market equity
    df['me'] = df.groupby(['permco', 'jdate'])['permno_me'].transform('sum')
    
    # Sort by firm, date, then descending market equity and tie-break by permno (ascending)
    df = df.sort_values(['permco', 'jdate', 'permno_me', 'permno'], ascending=[True, True, False, True])
    
    # Drop duplicates for each firm and date, keeping only the first (largest ME, tie broken by permno)
    df = df.drop_duplicates(subset=['permco', 'jdate'], keep='first').copy()
    
    df["permco"] = df["permco"].astype("int64")
    df.drop(columns=["permno_me"], inplace=True)

    return df


