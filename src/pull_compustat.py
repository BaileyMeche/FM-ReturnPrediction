"""
This module pulls and saves data on fundamentals from CRSP and Compustat.
The functions implement caching to avoid repeated downloads, especially during development.

Note: This code uses the new CRSP CIZ format. Information
about the differences between the SIZ and CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

For information about Compustat variables, see:
https://wrds-www.wharton.upenn.edu/documents/1583/Compustat_Data_Guide.pdf

For more information about variables in CRSP, see:
https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
I don't think this is updated for the new CIZ format, though.

Here is some information about the old SIZ CRSP format:
https://wrds-www.wharton.upenn.edu/documents/1095/CRSP_Flat_File_formats_and_notes.pdf


The following is an outdated programmer's guide to CRSP:
https://wrds-www.wharton.upenn.edu/documents/400/CRSP_Programmers_Guide.pdf


"""
from pathlib import Path
from typing import Union, List
from datetime import datetime

import pandas as pd
import wrds

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
    load_cache_data,
    _save_cache_data
)

# ==============================================================================================
# Global Configuration
# ==============================================================================================

RAW_DATA_DIR = config("RAW_DATA_DIR")
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


# ==============================================================================================
# Compustat Data
# ==============================================================================================

description_compustat = {
    "gvkey": "Global Company Key",
    "tic": "Ticker Symbol",
    "datadate": "Data Date",
    "fyear": "Fiscal Year",
    "sale": "Sales/Revenue",
    "cogs": "Cost of Goods Sold",
    "xsga": "Selling, General and Administrative Expense",
    "oibdp": "Operating Income Before Depreciation",
    "ebitda": "Earnings Before Interest, Taxes, Depreciation and Amortization",
    "dp": "Depreciation and Amortization",
    "xint": "Interest Expense, Net",
    "gliv": "Gains/Losses on investments",
    "uniami": "Net Income before Extraordinary Items and after Noncontrolling Interest",
    "ni": "Net Income (Loss)",
    "niadj": "Net Income Adjusted for Common/Ordinary Stock (Capital) Equivalents",
    "epsfi": "Earnings Per Share (Diluted) - Including Extraordinary Items",
    "epsfx": "Earnings Per Share (Diluted) - Excluding Extraordinary Items",
    "epspfi": "Earnings Per Share (Basic) - Including Extraordinary Items",
    "epspx": "Earnings Per Share (Basic) - Excluding Extraordinary Items",
    "dvpd": "Cash Dividends Paid",
    "dvc": "Dividends Common/Ordinary (dvc)",
    "dvt": "Dividends - Total (dvt)", 
    "csho": "Common Shares Outstanding",
    "cshpri": "Common Shares Used to Calculate Earnings Per Share - Basic", 
    "dltt": "Long-Term Debt - Total",
    "lct": "Current Debt - Total",
    "dlc": "Debt in Current Liabilities - Total",
    "at": "Assets - Total",
    "che": "Cash and Short-Term Investments",
    "act": "Current Assets - Total",
    "sich": "SIC Code - Standard Industrial Classification Code - Historical",
    "pstkl": "Preferred Stock - Liquidating Value",
    "txditc": "Deferred Taxes and Investment Tax Credit",
    "pstkrv": "Preferred Stock - Redemption Value",
    # This item represents the total dollar value of the net number of
    # preferred shares outstanding multiplied by the voluntary
    # liquidation or redemption value per share.
    "seq": "Stockholders' Equity - Parent",
    "pstk": "Preferred/Preference Stock (Capital) - Total",
    "indfmt": "Industry Format",
    "datafmt": "Data Format",
    "popsrc": "Population Source",
    "consol": "Consolidation",
}


def pull_Compustat(
        wrds_username: str = WRDS_USERNAME,
        gvkey: Union[str, List] = None,
        vars_str: Union[str,List] = None,
        start_date: str = None,
        end_date: str = None,
        data_dir: Union[None, Path] = RAW_DATA_DIR,
        file_name: str = None,
        hash_file_name: bool = False,
        file_type: str = None,    
    ) -> pd.DataFrame:
    """
    See description_Compustat for a description of the variables.
    Pull Annual Compustat fundamental data with caching

    https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/fundamentals-annual/

    Parameters
    ----------
    wrds_username : str
        WRDS username.
    gvkey : str or list, optional
        Global Company Key. If None, all companies are pulled. The default is None.
    vars_str : str or list, optional
        List of variables to pull from Compustat. If None, all variables are pulled. The default is None.
    start_date : str, optional
        Start date for the data. The default is None.
    end_date : str, optional
        End date for the data. The default is None.
    data_dir : Path, optional
        Directory to save the data. The default is RAW_DATA_DIR.
    file_name : str, optional
        File name to save the data. The default is None.
         hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name. The default is False.
    file_type : str, optional
        File type to save the data. The default is "parquet".
    
    Returns
    -------
    comp : pd.DataFrame
        Compustat data.

    """
    
    # Parse dates:
    if start_date is None:
        start_date = "1959-01-01"
    elif isinstance(start_date, (pd.Timestamp, datetime)):
        start_date = start_date.strftime("%Y-%m-%d")

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    elif isinstance(end_date, (pd.Timestamp, datetime)):
        end_date = end_date.strftime("%Y-%m-%d")

    if vars_str is not None:
        vars_str = ", ".join(vars_str)
    else: 
        vars_str = ("gvkey, datadate, fyear, sale AS sales, ni AS earnings, at AS assets, "
                    "(act - che) - lct - dp AS accruals, "
                    "act - che AS non_cash_current_assets,"
                    "lct,"
                    "dltt + dlc AS total_debt,"
                    "dp AS depreciation, "
                    "dvpd, dvc, dvt, pstk, pstkl, pstkrv, txditc, seq")
    
    # Convert filter_value to a tuple for SQL
    gvkey_tuple = _tickers_to_tuple(gvkey)

    if file_name is None:
        filters = {
            "vars_str": vars_str,
            "start_date": start_date,
            "end_date": end_date}
        if gvkey_tuple is not None:
            filters["gvkey"] = gvkey_tuple
        file_name = _flatten_dict_to_str(filters)
        if hash_file_name:
            cache_paths = _hash_cache_filename("comp_funda", filter_str, data_dir)
        else:
            cache_paths = _cache_filename("comp_funda", filter_str, data_dir)
        
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
        df_cached = _read_cached_data(cached_fp)
        return df_cached

    sql_query = f"""
        SELECT 
            {vars_str}
        FROM 
            comp.funda
        WHERE 
            indfmt='INDL' AND -- industrial reporting format (not financial services format)
            datafmt='STD' AND -- only standardized records
            popsrc='D' AND -- only from domestic sources
            consol='C' AND -- consolidated financial statements
            datadate >= '{start_date}'AND 
            datadate <= '{end_date}'
        """
    # If filtering on gvkey, add to the SQL query
    if gvkey is not None:
        gvkey_filter = _format_tuple_for_sql_list(gvkey_tuple)
        sql_query += f" AND {gvkey} IN {gvkey_filter}"

    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     comp = db.raw_sql(sql_query, date_cols=["datadate"])
    db = wrds.Connection(wrds_username=wrds_username)
    comp = db.raw_sql(sql_query, date_cols=["datadate"])
    db.close()

    # Save to cache
    cache_path = _save_cache_data(comp, data_dir, cache_paths, file_name, file_type)
    print(f"Saved data to {cache_path}")

    return comp


description_CRSP_Comp_link = {
    "gvkey": "Global Company Key - A unique identifier for companies in the Compustat database.",
    "permno": "Permanent Number - A unique stock identifier assigned by CRSP to each security.",
    "linktype": "Link Type - Indicates the type of linkage between CRSP and Compustat records. 'L' types refer to links considered official by CRSP.",
    "linkprim": "Primary Link Indicator - Specifies whether the link is a primary identified by Compustat ('P'), primary assigned by CRSP ('C') connection between the databases, or secondary ('J') for secondary securities for each company (used for total market cap). Primary links are direct matches between CRSP and Compustat entities, while secondary links may represent subsidiary relationships or other less direct connections.",
    "linkdt": "Link Date Start - The starting date for which the linkage between CRSP and Compustat data is considered valid.",
    "linkenddt": "Link Date End - The ending date for which the linkage is considered valid. A blank or high value (e.g., '2099-12-31') indicates that the link is still valid as of the last update.",
}


def pull_CRSP_Comp_link_table(
    wrds_username: str = WRDS_USERNAME,
    gvkey: Union[str, List[str], None] = None,
    data_dir: Union[None, Path] = RAW_DATA_DIR,
    file_name: str = None,
    hash_file_name: bool = False,
    file_type: str = None,    
) -> pd.DataFrame:
    """ 
    Pull the CRSP-Compustat link table.
    https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-crsp-with-compustat/
    
    Parameters
    ----------
    wrds_username : str
        WRDS username.
    gvkey : str or list of str, optional
        Filter by gvkey. If None, no gvkey filter is applied.
    data_dir : pathlib.Path or None, optional
        Directory for caching the data. If None, no caching is performed.
    file_name : str, optional
        If provided, save/read the data under this file name. Otherwise,
        a hashed name is generated from the function filters.
    hash_file_name : bool, optional
        If True, uses a hashed filename for cache. Otherwise uses a verbose name. The default is False.
    file_type : str, optional
        File type to save the data. The default is "parquet".

    Returns
    -------
    pd.DataFrame
        The CRSP-Compustat link table with any specified filters applied.

    """
    # Convert filter_value to a tuple for SQL
    gvkey_tuple = _tickers_to_tuple(gvkey)

    # Build or derive the cache file name
    if file_name is None:
        if gvkey_tuple is not None:
            filters = {"gvkey": gvkey_tuple}
        else:
            filters = {}
        filter_str = _flatten_dict_to_str(filters)
        if hash_file_name:
            cache_paths = _hash_cache_filename("crsp_comp_link_table", filter_str, data_dir)
        else:
            cache_paths = _cache_filename("crsp_comp_link_table", filter_str, data_dir)
        
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
    base_sql = """
        SELECT 
            gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM 
            crsp.ccmxpf_linktable
        WHERE 
            substr(linktype,1,1)='L'
            AND (linkprim ='C' OR linkprim='P')
            AND linktype NOT IN ('LX', 'LD', 'LN')
    """

    if gvkey_tuple is not None:
        gvkey_filter = _format_tuple_for_sql_list(gvkey_tuple)
        base_sql += f" AND gvkey IN {gvkey_filter}"

    # Connect to WRDS, run query
    db = wrds.Connection(wrds_username=wrds_username)
    ccm = db.raw_sql(base_sql, date_cols=["linkdt", "linkenddt"])
    db.close()

    # Save to cache
    cache_path = _save_cache_data(ccm, data_dir, cache_paths, file_name, file_type)
    print(f"Saved data to {cache_path}")

    return ccm


def _demo():
    comp = load_cache_data(data_dir=RAW_DATA_DIR, file_name="Compustat_fund.parquet")
    ccm = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_Comp_Link_Table.parquet")


if __name__ == "__main__":
    comp = pull_Compustat(wrds_username=WRDS_USERNAME, start_date=START_DATE, end_date=END_DATE, file_name="Compustat.parquet",)
    ccm = pull_CRSP_Comp_link_table(wrds_username=WRDS_USERNAME, start_date=START_DATE, end_date=END_DATE, file_name="CRSP_Comp_Link_Table.parquet")