"""
This module pulls and saves data on fundamentals from CRSP and Compustat.
It pulls fundamentals data from Compustat needed to calculate
book equity, and the data needed from CRSP to calculate market equity.

Note: This code uses the new CRSP CIZ format. Information
about the differences between the SIZ and CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

For more information about variables in CRSP, see:
https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
I don't think this is updated for the new CIZ format, though.

Here is some information about the old SIZ CRSP format:
https://wrds-www.wharton.upenn.edu/documents/1095/CRSP_Flat_File_formats_and_notes.pdf

The following is an outdated programmer's guide to CRSP:
https://wrds-www.wharton.upenn.edu/documents/400/CRSP_Programmers_Guide.pdf


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

# ==============================================================================================
# GLOBAL CONFIGURATION
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def get_CRSP_columns(wrds_username=WRDS_USERNAME, table_schema="crsp", table_name="msf_v2"):
    """Get all column names from CRSP monthly stock file (CIZ format)."""

    sql_query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = {table_schema}
        AND table_name = {table_name}
        ORDER BY ordinal_position;
    """
    
    db = wrds.Connection(wrds_username=wrds_username)
    columns = db.raw_sql(sql_query)
    db.close()
    
    return columns
 
# ==============================================================================================
# STOCK DATA
# ==============================================================================================
"""
More information about the CRSP US Stock & Indexes Database can be found here:
https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_&_Indexes_Database_Data_Descriptions_Guide.pdf
"""

def pull_CRSP_stock(
    start_date=None, end_date=None, freq="D", wrds_username=WRDS_USERNAME
):
    """Pull necessary CRSP monthly or daily stock data to
    compute Fama-French factors. Use the new CIZ format.

    Notes
    -----
    
    ## Cumulative Adjustment Factors (CFACPR and CFACSHR)
    https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/

    In the legacy format, CRSP provided two data series, CFACPR and CFACSHR for
    cumulative adjustment factors for price and share respectively. In the new CIZ
    data format, these two data series are no longer provided, at least in the
    initial launch, per CRSP.

    WRDS understands the importance of these two variables to many researchers and
    we prepared a sample code that researchers can use to recreate the series using
    the raw adjustment factors. However, we need to caution users that the results
    of our sample code do not line up with the legacy CFACPR and CFACSHR completely.
    While it generates complete replication in 95% of the daily observations, we do
    observe major differences in the tail end. We do not have an explanation from
    CRSP about the remaining 5%, hence we urge researchers to use caution. Please
    contact CRSP support (Support@crsp.org) if you would like to discuss the issue
    of missing cumulative adjustment factors in the new CIZ data.

    For now, it's close enough to just let
    market_cap = mthprc * shrout

    """
    if start_date is None:
        start_date = "01/01/1959"
    if end_date is None:
        end_date = datetime.today().strftime("%m/%d/%Y")
    if freq == "M":
        table = "msf_v2"
        date_col = "mthcaldt"
    elif freq == "D":
        table = "dsf_v2"
        date_col = "dlycaldt"
    else:
        raise ValueError("freq must be either 'D' or 'M'.")
    
    sql_query = f"""
        SELECT 
            permno, permco, mthcaldt, 
            issuertype, securitytype, securitysubtype, sharetype, 
            usincflg, 
            primaryexch, conditionaltype, tradingstatusflg,
            mthret, mthretx, shrout, mthprc
        FROM 
            crsp.{table}
        WHERE 
            mthcaldt >= '{start_date}' AND mthcaldt <= '{end_date}'
        """

    db = wrds.Connection(wrds_username=wrds_username)
    crsp = db.raw_sql(sql_query, date_cols=[date_col])
    db.close()

    # change variable format to int
    crsp[["permco", "permno"]] = crsp[["permco", "permno"]].astype(int)

    # Line up date to be end of month
    crsp["jdate"] = crsp[date_col] + MonthEnd(0)

    return crsp



# ==============================================================================================
# INDEX DATA
# ==============================================================================================
"""
More information about the CRSP US Stock & Indexes Database can be found here:
https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_&_Indexes_Database_Data_Descriptions_Guide.pdf
"""

def pull_CRSP_index(
    start_date=None, end_date=None, wrds_username=WRDS_USERNAME
):
    """
    Pulls the monthly CRSP index files from crsp_a_indexes.msix:
    (Monthly)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix)
    """
    # Pull index files
    if start_date is None:
        start_date = "01/01/1959"
    if end_date is None:
        end_date = datetime.today().strftime("%m/%d/%Y")
    query = f"""
        SELECT * 
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(query, date_cols=["month", "caldt"])
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def pull_constituents(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    db = wrds.Connection(wrds_username=wrds_username)

    df_constituents = db.raw_sql(f""" 
    SELECT *
    from crsp_m_indexes.dsp500list_v2 
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """)

    # Convert string columns to datetime if they aren't already
    df_constituents["mbrstartdt"] = pd.to_datetime(df_constituents["mbrstartdt"])
    df_constituents["mbrenddt"] = pd.to_datetime(df_constituents["mbrenddt"])

    return df_constituents



# ==============================================================================================
# TREASURIES DATA
# ==============================================================================================
"""
More information about the CRSP US Treasury Database can be found here:
https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Treasury_Database_Guide_for_SAS_ASCII_EXCEL_R.pdf
"""

def pull_CRSP_treasuries(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME, include_descr=True
):
    """
    Pulls the CRSP treasuries issue descriptions and daily time series quotes from crsp_m_treasuries.tfz_dly
    https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/monthly-update/treasuries/daily-time-series/
    """
    query = f"""
        SELECT *
        FROM crsp_m_treasuries.tfz_dly
        WHERE caldt >= '{start_date}' AND caldt <= '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])

    if include_descr:
        keys = tuple(df['kytreasno'])
        query = f"""
            SELECT *
            FROM crsp_m_treasuries.tfz_iss WHERE kytreasno IN {keys}
        """
        df_info = db.raw_sql(query)
        df = df.merge(df_info, on='kytreasno')
    
    db.close()

    return df


def pull_CRSP_treasury_info(keys_list: List[int] = None,
                            wrds_username: str = WRDS_USERNAME):
    """
    Pulls the CRSP treasuries issue descriptions from crsp_m_treasuries.tfz_iss
    https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/monthly-update/treasuries/daily-time-series/
    """
    if keys_list is not None:
        keys = tuple(keys_list)
        query = f"""
            SELECT *
            FROM crsp_m_treasuries.tfz_iss WHERE kytreasno IN {keys}
        """
    else:
        query = f"""
            SELECT *
            FROM crsp_m_treasuries.tfz_iss
        """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query)
    db.close()

    return df


def pull_CRSP_yield_curve(
    start_date=START_DATE, end_date=END_DATE, freq='D', wrds_username=WRDS_USERNAME
):
    """
    Pulls the daily CRSP yield curve data from crsp_m_treasuries.tfz_dly_ft.
    Highlight the performance of single treasury issues at fixed maturity horizons.
    Seven groups of indexes: 30-year, 20-year, 10-year, 7-year, 5-year, 2-year, or 1-year target maturity.
    Index creates a sophisticated bond yield curve, allowing the selection of data items referenced by returns, prices and duration.

    https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_m_treasuries/tfz_dly_ft/
    """
    if freq == "D":
        table = "tfz_dly_ft"
    elif freq == "M":
        table = "tfz_mth_ft"
    else:
        raise ValueError("freq must be either 'D' or 'M'.")
    
    query = f"""
        SELECT *
        FROM crsp_m_treasuries.{table}
        WHERE caldt >= '{start_date}' AND caldt <= '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    df = df.pivot_table(index='caldt',values='tdytm', columns='kytreasnox')
    df = df.rename(columns={2000003:1, 2000004:2, 2000005:5, 2000006:7, 2000007:10, 2000008:20, 2000009:30})
    df.columns = [1,2,5,7,10,20,30]

    return df


def pull_CRSP_zero_bonds(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls the monthly CRSP zero coupon bonds (Fama Bliss) from crsp_m_treasuries.tfz_dly_zc.

    Data begin in 1952
    Contain artificial discount bonds with one to five years to maturity, constructed after first extracting the term structure from a filtered subset of the available bonds.

    It gives prices on **zero coupon bonds** with maturities of 1 through 5 years.
    * These are prices per $1 face value on bonds that only pay principal.
    * Such bonds can be created from treasuries by stripping out their coupons.
    * In essence, you can consider these prices as the discount factors $Z$, for maturity intervals 1 through 5 years.

    To calculate the spot rates, you can use the formula:
    px = pull_CRSP_zero_bonds(wrds_username=WRDS_USERNAME)
    spots = -np.log(px)/px.columns

    https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/monthly-update/treasuries/fama-bliss-discount-bonds-monthly-only/
    """

    query = f"""
        SELECT *
        FROM crsp_m_treasuries.tfz_mth_fb
        WHERE caldt >= '{start_date}' AND caldt <= '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()

    fb = fb.rename(columns={'mcaldt':'date','tmnomprc':'price','tmytm':'yld'})
    fb = fb.pivot_table(values='price',index='date',columns='kytreasnox')
    fb = fb.rename(columns={2000047:1, 2000048:2, 2000049:3, 2000050:4, 2000051:5})
    fb.columns.name = 'maturity'
    
    return df


def risk_CRSP_free_rate(start_date=START_DATE, end_date=END_DATE, freq="D", wrds_username=WRDS_USERNAME):
    """
    Contain one- and three-month risk free rates for use in pricing and macroeconomic models.
    Monthly:
        Begin in 1925
        Contain one- and three-month risk free rates for use in pricing and macroeconomic models
    Daily:
        Begin in 1961
        Four-week, 13-week, and 26-week rates
        Provides lending and borrowing rates derived from bid, ask, and bid/ask average prices.
    """
    if freq == "D":
        table = "tfz_dly_rf2"
        date_col = "caldt"
        value_col = "tdyld"
        col_dict = {2000061:'rf_4w',2000062:'rf_13w', 2000063:'rf_26w'}
    elif freq == "M":
        table = "tfz_mth_rf"
        date_col = "mcaldt"
        value_col = "tmytm"
        col_dict = {2000001:'rf_1m',2000002:'rf_3m'}
    else:
        raise ValueError("freq must be either 'D' or 'M'.")
    
    query = f"""
        SELECT *
        FROM crsp_m_treasuries.{table}
        WHERE {date_col} >= '{start_date}' AND {date_col} <= '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=[date_col])
    db.close()

    df = df.pivot_table(index=date_col,values=value_col, columns='kytreasnox')
    df = df.rename(columns=col_dict)

    return df
    
        
# ==============================================================================================
# LOAD SAVED DATA
# ==============================================================================================

def load_CRSP_stock(data_dir=RAW_DATA_DIR, freq="D"):
    if freq == "D":
        path = Path(data_dir) / f"CRSP_stock_D.parquet"
    elif freq == "M":
        path = Path(data_dir) / f"CRSP_stock_M.parquet"
    else:
        raise ValueError("freq must be either 'D' or 'M'.")
    crsp = pd.read_parquet(path)
    return crsp


def load_CRSP_index(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "CRSP_MSIX.parquet")

def load_CRSP_index_constituents(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "df_sp500_constituents.parquet")

def load_CRSP_treasuries(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "CRSP_treasuries.parquet")

def load_CRSP_yield_curve(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "CRSP_yield_curve.parquet")

def load_CRSP_zero_bonds(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "CRSP_zero_bonds.parquet")

def load_CRSP_treasury_info(data_dir=RAW_DATA_DIR):
    return pd.read_parquet(data_dir / "CRSP_treasuries_info.parquet")

def _demo():
    crsp_d = load_CRSP_stock(data_dir=RAW_DATA_DIR, freq="D")
    crsp_m = load_CRSP_stock(data_dir=RAW_DATA_DIR, freq="M")
    df_msix = load_CRSP_index(data_dir=RAW_DATA_DIR)
    constituents = load_CRSP_index_constituents(data_dir=RAW_DATA_DIR)
    treasuries = load_CRSP_treasuries(data_dir=RAW_DATA_DIR)
    treasury_info = load_CRSP_treasury_info(data_dir=RAW_DATA_DIR)
    yield_curve = load_CRSP_yield_curve(data_dir=RAW_DATA_DIR)
    zero_bonds = load_CRSP_zero_bonds(data_dir=RAW_DATA_DIR)


if __name__ == "__main__":

    crsp_m = pull_CRSP_stock(start_date=START_DATE, end_date=END_DATE, freq='M', wrds_username=WRDS_USERNAME)
    crsp_m.to_parquet(RAW_DATA_DIR / "CRSP_stock_m.parquet")

    crsp_d = pull_CRSP_stock(start_date=START_DATE, end_date=END_DATE, freq='D', wrds_username=WRDS_USERNAME)
    crsp_d.to_parquet(RAW_DATA_DIR / "CRSP_stock_d.parquet")

    df_msix = pull_CRSP_index(start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME)
    df_msix.to_parquet(RAW_DATA_DIR / "CRSP_MSIX.parquet")

    constituents = pull_constituents(start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME)
    constituents.to_parquet(RAW_DATA_DIR / "df_sp500_constituents.parquet")

    treasuries = pull_CRSP_treasuries(start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME)
    treasuries.to_parquet(RAW_DATA_DIR / "CRSP_treasuries.parquet")

    treasury_info = pull_CRSP_treasury_info(wrds_username=WRDS_USERNAME)
    treasury_info.to_parquet(RAW_DATA_DIR / "CRSP_treasuries_info.parquet")

    yield_curve_m = pull_CRSP_yield_curve(start_date=START_DATE, end_date=END_DATE, freq='M', wrds_username=WRDS_USERNAME)
    yield_curve_m.to_parquet(RAW_DATA_DIR / "CRSP_yield_curve_m.parquet")

    yield_curve_d = pull_CRSP_yield_curve(start_date=START_DATE, end_date=END_DATE, freq='D', wrds_username=WRDS_USERNAME)
    yield_curve_d.to_parquet(RAW_DATA_DIR / "CRSP_yield_curve_d.parquet")

    zero_bonds = pull_CRSP_zero_bonds(start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME)
    zero_bonds.to_parquet(RAW_DATA_DIR / "CRSP_zero_bonds.parquet")


