
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



# ==============================================================================================
# TREASURIES DATA
# ==============================================================================================

def calc_runness(rawdata):
    """
    Calculate runness for the securities issued in 1980 or later.

    This is due to the following condition of Gurkaynak, Sack, and Wright (2007):
        iv) Exclude on-the-run issues and 1st off-the-run issues
        for 2,3,5, 7, 10, 20, 30 years securities issued in 1980 or later.
    """
    data = rawdata.copy()
    data.columns = data.columns.str.upper()
    
    def _calc_runness(df):
        temp = df.sort_values(by=["CALDT", "original_maturity", "TDATDT"])
        next_temp = (
            temp.groupby(["CALDT", "original_maturity"])["TDATDT"].rank(
                method="first", ascending=False
            )
            - 1
        )
        return next_temp

    data = data.assign(original_maturity=np.round(data["TMATDT"] - data["TDATDT"]))
    data_run_ = data[data["CALDT"] >= "1980"]
    runs = _calc_runness(data_run_)
    data["run"] = 0
    data.loc[data_run_.index, "run"] = runs
    return data


def process_wrds_treasury_data(
    rawdata: pd.DataFrame,
    keys_extra: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Processes WRDS Treasury data into a standardized format, renaming columns, 
    calculating TTM, dirty price, and annualizing yields.

    Parameters:
    -----------
    rawdata : pd.DataFrame
        Raw DataFrame from WRDS with columns such as CALDT, TDBID, TDASK, ...
    keys_extra : list of str, optional
        Additional columns from `rawdata` to keep in the output.

    Returns:
    --------
    pd.DataFrame
        Processed data with standardized columns:
        ['type', 'quote date', 'issue date', 'maturity date', 'ttm', 'accrual fraction',
         'cpn rate', 'bid', 'ask', 'price', 'accrued int', 'dirty price', 'ytm', ...].
    """
    if keys_extra is None:
        keys_extra = []

    DAYS_YEAR = 365.25
    FREQ = 2  # semi-annual

    data = rawdata.copy()
    data.columns = data.columns.str.upper()
    data.sort_values('TMATDT', inplace=True)
    data.set_index('KYTREASNO', inplace=True)

    # Subset columns
    keep_cols = [
        'CALDT', 'TDBID', 'TDASK', 'TDNOMPRC', 'TDACCINT', 'TDYLD', 'TDATDT',
        'TMATDT', 'TCOUPRT', 'ITYPE', 'TDDURATN', 'TDPUBOUT', 'TDTOTOUT'
    ] + keys_extra
    data = data[keep_cols]

    # Map integer codes to descriptive bond types
    dict_type = {
        1: 'bond',
        2: 'note',
        4: 'bill',
        11: 'TIPS note',
        12: 'TIPS bond'
    }
    data['ITYPE'] = data['ITYPE'].replace(dict_type)

    # Rename columns to standardized names
    data.rename(
        columns={
            'CALDT': 'quote date',
            'TDATDT': 'issue date',
            'TMATDT': 'maturity date',
            'TCOUPRT': 'cpn rate',
            'TDTOTOUT': 'total size',
            'TDPUBOUT': 'public size',
            'TDDURATN': 'duration',
            'ITYPE': 'type',
            'TDBID': 'bid',
            'TDASK': 'ask',
            'TDNOMPRC': 'price',
            'TDACCINT': 'accrued int',
            'TDYLD': 'ytm'
        },
        inplace=True
    )

    # Convert dates to datetime
    data['quote date'] = pd.to_datetime(data['quote date'])
    data['issue date'] = pd.to_datetime(data['issue date'])
    data['maturity date'] = pd.to_datetime(data['maturity date'])

    # Time-to-maturity in years
    data['ttm'] = (data['maturity date'] - data['quote date']).dt.days.astype(float) / DAYS_YEAR

    # Dirty price = price + accrued interest
    data['dirty price'] = data['price'] + data['accrued int']

    # Convert duration from days to years
    data['duration'] /= 365.0

    # Convert sizes to face value in dollars
    data['total size'] *= 1e6
    data['public size'] *= 1e6

    # Annualize YTM (assumed input is in daily logs or something) for semiannual compounding
    def to_semiannual_ytm(x: float) -> float:
        return (np.exp(x * DAYS_YEAR / FREQ) - 1.0) * FREQ

    data['ytm'] = data['ytm'].apply(to_semiannual_ytm)

    # Accrual fraction
    data['accrual fraction'] = data['accrued int'] / (data['cpn rate'] / FREQ)
    idx_na = data['accrual fraction'].isna()
    # If not available, approximate from fractional part of TTM
    data.loc[idx_na, 'accrual fraction'] = 1.0 - (data.loc[idx_na, 'ttm'] - np.round(data.loc[idx_na, 'ttm'])) * FREQ

    data = calc_runness(data)

    # Final organization
    standard_keys = [
        'type', 'quote date', 'issue date', 'maturity date', 'ttm',
        'accrual fraction', 'cpn rate', 'bid', 'ask', 'price',
        'accrued int', 'dirty price', 'ytm', 'run'
    ]
    data = data[standard_keys + keys_extra]

    return data
