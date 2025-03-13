"""
This module pulls and saves data on fundamentals from CRSP and Compustat.
It pulls fundamentals data from Compustat needed to calculate book equity, and the data needed from CRSP to calculate market equity.

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
from utils import (
    _cache_filename,
    _hash_cache_filename,
    _file_cached,
    _read_cached_data,
    _write_cache_data,
    _flatten_dict_to_str,
    _tickers_to_tuple,
    _format_tuple_for_sql_list,
    _save_cache_data,
    load_cache_data,
)

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
"""
More information about the CRSP US Stock & Indexes Database can be found here:
https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_&_Indexes_Database_Data_Descriptions_Guide.pdf
"""

description_crsp = {
    "permno": "Permanent Number - A unique identifier assigned by CRSP to each security.",
    "permco": "Permanent Company - A unique company identifier assigned by CRSP that remains constant over time for a given company.",
    "ticker": "Ticker - The stock ticker symbol for the security.",
    "mthcaldt": "Calendar Date - The date for the monthly data observation.",
    "dlycaldt": "Calendar Date - The date for the daily data observation.",
    "issuertype": "Issuer Type - Classification of the issuer, such as corporate or government.",
    "securitytype": "Security Type - General classification of the security, e.g., stock or bond.",
    "securitysubtype": "Security Subtype - More specific classification of the security within its type.",
    "sharetype": "Share Type - Classification of the equity share type, e.g., common stock, preferred stock.",
    "usincflg": "U.S. Incorporation Flag - Indicator of whether the company is incorporated in the U.S.",
    "primaryexch": "Primary Exchange - The primary stock exchange where the security is listed.",
    "conditionaltype": "Conditional Type - Indicator of any conditional issues related to the security.",
    "tradingstatusflg": "Trading Status Flag - Indicator of the trading status of the security, e.g., active, suspended.",
    "mthret": "Monthly Return - The total return of the security for the month, including dividends.",
    "mthretx": "Monthly Return Excluding Dividends - The return of the security for the month, excluding dividends.",
    "mthprc": "Monthly Price - The price of the security at the end of the month.",
    "dlyret": "Daily Return - The total return of the security for the day, including dividends.",
    "dlyretx": "Daily Return Excluding Dividends - The return of the security for the day, excluding dividends.",
    "dlyprc": "Daily Price - The price of the security at the end of the day.",
    "shrout": "Shares Outstanding - The number of outstanding shares of the security.",
    "naics": "NAICS - The North American Industry Classification System code for the company.",
    "siccd": "SIC - The Standard Industrial Classification code for the company.",
}


def pull_CRSP_stock(
    wrds_username: str = WRDS_USERNAME,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    freq: str = "D",
    filter_by: str = None,  # "permno", "permco", or "ticker"
    filter_value: Union[str, List[str], None] = None,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    file_name: str = None,
    hash_file_name: bool = False,
    file_type: str = None,
) -> pd.DataFrame:
    """
    Pull CRSP monthly or daily stock data (new CIZ format) with optional filters and caching.

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

    Parameters
    ----------
    wrds_username : str
        WRDS username.
    start_date : str, pd.Timestamp, or None
        Start date (default '1959-01-01' if None).
    end_date : str, pd.Timestamp, or None
        End date (default is today's date if None).
    freq : {'D', 'M'}
        Frequency of the data. 'D' for daily, 'M' for monthly.
    filter_by : {None, 'permno', 'permco', 'ticker'}, optional
        If provided, the column to filter on.
    filter_value : str or list of str, optional
        If provided, the value(s) to filter. Must be used in conjunction with filter_by.
    data_dir : pathlib.Path or None, optional
        Directory for caching the data.
    file_name : str, optional
        If provided, save/read the data under this file name.
    hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name.
    file_type : str, optional
        File type for caching. Default 'parquet'.

    Returns
    -------
    pd.DataFrame
        CRSP stock data, with an added 'jdate' column set to end-of-month if freq='D' or 'M'.
    """
    # Handle date defaults
    if start_date is None:
        start_date = "1959-01-01"
    elif isinstance(start_date, (pd.Timestamp, datetime)):
        start_date = start_date.strftime("%Y-%m-%d")

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    elif isinstance(end_date, (pd.Timestamp, datetime)):
        end_date = end_date.strftime("%Y-%m-%d")

    # Determine which CRSP table/columns to query
    if freq.upper() == "M":
        table = "msf_v2"
        date_col = "mthcaldt"
        tot_ret_col = "mthret"
        prc_ret_col = "mthretx"
        prc_col = "mthprc"
    elif freq.upper() == "D":
        table = "dsf_v2"
        date_col = "dlycaldt"
        tot_ret_col = "dlyret"
        prc_ret_col = "dlyretx"
        prc_col = "dlyprc"
    else:
        raise ValueError("freq must be either 'D' or 'M'.")

    # Convert filter_value to a tuple for SQL
    filter_value_tuple = _tickers_to_tuple(filter_value)

    # Build or derive the cache file name
    if file_name is None:
        filters = {
            "start_date": start_date,
            "end_date": end_date}
        if filter_by is not None and filter_value is not None:
            filters[filter_by] = filter_value
        filter_str = _flatten_dict_to_str(filters)
        if hash_file_name:
            cache_paths = _hash_cache_filename(f"crsp_{table}", filter_str, data_dir)
        else:
            cache_paths = _cache_filename(f"crsp_{table}", filter_str, data_dir)
        
        # Check if file is cached
        cached_fp = _file_cached(cache_paths)
    else:
        if not any(file_name.endswith(f".{ft}") for ft in ["parquet", "csv", "zip"]):
            cache_paths= [data_dir / f"{file_name}.{ft}" for ft in ["parquet", "csv", "zip"]]
            cached_fp = _file_cached(cache_paths)
        else:
            cache_paths = None
            cached_fp = Path(data_dir, file_name) if Path(data_dir, file_name).exists() else None
            
    if cached_fp:
        print(f"Loading cached data from {cached_fp}")
        return _read_cached_data(cached_fp)

    # Build the SQL query
    sql_query = f"""
        SELECT 
            permno, permco, {date_col}, 
            issuertype, securitytype, securitysubtype, sharetype, 
            usincflg, 
            primaryexch, conditionaltype, tradingstatusflg,
            {tot_ret_col} AS totret, 
            {prc_ret_col} AS retx,
            {prc_col} AS prc, 
            shrout
        FROM crsp.{table}
        WHERE {date_col} >= '{start_date}' 
          AND {date_col} <= '{end_date}'
    """

    # If filtering on permno/permco/ticker
    if filter_by is not None and filter_value is not None:
        filter_value_sql = _format_tuple_for_sql_list(filter_value_tuple)
        sql_query += f" AND {filter_by} IN {filter_value_sql}"

    # Query WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    crsp = db.raw_sql(sql_query, date_cols=[date_col])
    db.close()

    # Clean up
    crsp[["permno", "permco"]] = crsp[["permno", "permco"]].astype(int, errors="ignore")

    # For convenience, align to end-of-month
    crsp["jdate"] = crsp[date_col] + MonthEnd(0)

    # Save to cache
    cache_path = _save_cache_data(crsp, data_dir, cache_paths, file_name, file_type)
    print(f"Saved data to {cache_path}")

    return subset_CRSP_to_common_stock_and_exchanges(crsp)


def subset_CRSP_to_common_stock_and_exchanges(crsp):
    """Subset to common stock universe and stocks traded on NYSE, AMEX and NASDAQ.

    NOTE:
        With the new CIZ format, it is not necessary to apply delisting returns, as they are already applied.
    """
    # In the old SIZ format, this would condition on shrcd = 10 or 11.
    # Now, this is classified under ShareType = ‘NS’

    # Select common stock universe

    # sharetype=='NS': Filters for securities where the 'Share Type' is "Not specified" (no special type). 
    # See here: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/#replicating-common-tasks

    # securitytype=='EQTY': Selects only securities classified as 'EQTY'

    # securitysubtype=='COM': Narrows down to securities with a 'Security Subtype'
    # of 'COM', suggesting these are common stocks.

    # usincflg=='Y': Includes only securities issued by companies incorporated in
    # the U.S., as indicated by the 'U.S. Incorporation Flag'.

    # issuertype.isin(['ACOR', 'CORP']): Further filters securities to those issued
    # by entities classified as either 'ACOR' (Asset-Backed Corporate)
    # or 'CORP' (Corporate), based on the 'Issuer Type' classification.

    mask_filter = (crsp["conditionaltype"] == "RW") & (crsp["tradingstatusflg"] == "A") & \
            (crsp["sharetype"] == "NS") & (crsp["securitytype"] == "EQTY") & \
            (crsp["securitysubtype"] == "COM") & (crsp["usincflg"] == "Y") & \
            (crsp["issuertype"].isin(["ACOR", "CORP"]))

    crsp = crsp[mask_filter]

    ## Select stocks traded on NYSE, AMEX and NASDAQ
    ##
    ## Again, see https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/#replicating-common-tasks
    
    mask_filter_exchanges = crsp["primaryexch"].isin(["N", "A", "Q"])
    crsp = crsp[mask_filter_exchanges]

    return crsp
        


# ==============================================================================================
# INDEX DATA
# ==============================================================================================


def pull_CRSP_index(
    wrds_username: str = WRDS_USERNAME,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    freq: str = "D",
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    file_name: str = None,
    hash_file_name: bool = False,
    file_type: str = None,
) -> pd.DataFrame:
    """
    Pulls the monthly CRSP index files from crsp_a_indexes.msix or  crsp_a_indexes.dsix.
    (Daily)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (dsix).
    https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_m_indexes/dsix/

    (Monthly)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix).
    https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_m_indexes/msix/

    This includes:
        - decile returns: decret1, decret2,...,decret10
        - equal-weighted returns (incl. and excl. dividends): ewretd, ewretx
        - value-weighted returns (incl. and excl. dividends): vwretd, vwretx
        - return on the S&P 500 index: sprtrn
        - level of the S&P 500 index: spindx

    Parameters
    ----------
    wrds_username : str
        WRDS username.
    start_date : str or pd.Timestamp, optional
        Start date to pull data from. Default '1959-01-01' if None.
    end_date : str or pd.Timestamp, optional
        End date to pull data up to. Default is today's date if None.
    data_dir : pathlib.Path or None, optional
        Directory to cache.
    file_name : str, optional
        File name to store the cached data.
    hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name.
    file_type : str, optional
        File type for caching. Default 'parquet'.

    Returns
    -------
    pd.DataFrame
        CRSP Index data from msix.
    """
    # Handle date defaults
    if start_date is None:
        start_date = "1959-01-01"
    elif isinstance(start_date, (pd.Timestamp, datetime)):
        start_date = start_date.strftime("%Y-%m-%d")

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    elif isinstance(end_date, (pd.Timestamp, datetime)):
        end_date = end_date.strftime("%Y-%m-%d")

    if freq.upper() == "M":
        table = "msix"
    elif freq.upper() == "D":
        table = "dsix"
    else:
        raise ValueError("freq must be either 'D' or 'M'")
                         
    # Prepare for caching
    if file_name is None:
        filters = {"start_date": start_date,
                   "end_date": end_date,
                   "freq": freq}
        filter_str = _flatten_dict_to_str(filters)
        if hash_file_name:
            cache_paths = _hash_cache_filename(f"crsp_a_index_{table}", filter_str, data_dir)
        else:
            cache_paths = _cache_filename("crsp_a_index_{table}", filter_str, data_dir)
        
        # Check if file is cached
        cached_fp = _file_cached(cache_paths)
    else:
        if not any(file_name.endswith(f".{ft}") for ft in ["parquet", "csv", "zip"]):
            cache_paths= [data_dir / f"{file_name}.{ft}" for ft in ["parquet", "csv", "zip"]]
            cached_fp = _file_cached(cache_paths)
        else:
            cache_paths = None
            cached_fp = Path(data_dir, file_name) if Path(data_dir, file_name).exists() else None

    if cached_fp:
        print(f"Loading cached data from {cached_fp}")
        return _read_cached_data(cached_fp)

    # Build and run query
    query = f"""
        SELECT * 
        FROM crsp_a_indexes.{table}
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()

    # Save to cache
    cache_path = _save_cache_data(df, data_dir, cache_paths, file_name, file_type)
    print(f"Saved data to {cache_path}")

    return df




def _demo():
    crsp_d = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_stock_d.parquet")
    crsp_m = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_stock_m.parquet")
    crsp_index_d = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_index_d.parquet")

if __name__ == "__main__":

    crsp_d = pull_CRSP_stock(start_date=START_DATE, end_date=END_DATE, freq='D', wrds_username=WRDS_USERNAME, file_name="CRSP_stock_d.parquet")
    crsp_m = pull_CRSP_stock(start_date=START_DATE, end_date=END_DATE, freq='M', wrds_username=WRDS_USERNAME, file_name="CRSP_stock_m.parquet")
    crsp_index_d = pull_CRSP_index(start_date=START_DATE, end_date=END_DATE, freq='D', wrds_username=WRDS_USERNAME, file_name="CRSP_index_d.parquet")
    
