"""
Transformations for Compustat Data
===================================

This module provides transformation utilities for Compustat fundamental data.
It includes functions to convert annual data into a monthly time series, compute
report dates based on fiscal year-end dates, and merge Compustat data with CRSP
data via the CRSP–Compustat linking table to facilitate further analysis, such
as computing the book-to-market ratio.

Functions:
----------
- expand_compustat_annual_to_monthly(comp_annual, date_col='report_date', id_col='permno', value_col='x'):
    Expands an annual fundamental measure to a monthly time series by forward-filling
    each 'report_date' until the next observation, and then pivots the data so that the
    index is month-end dates and columns correspond to firm identifiers.

- add_report_date(comp):
    Computes the report date for each record in the Compustat dataset by adding four
    months to the fiscal year-end date (datadate), reflecting the typical delay in
    accounting data availability.

- merge_CRSP_and_Compustat(crsp, comp, ccm):
    Merges Compustat fundamentals with CRSP market data using the CRSP–Compustat linking
    table. It cleans and restricts the linking data based on valid link dates, aligns the
    Compustat report dates to portfolio formation dates, and merges on common identifiers
    to prepare data for calculating ratios like book-to-market.

Usage:
------
These functions are intended to be used on raw Compustat and CRSP data obtained from
their respective data pulling modules, transforming the data into formats suitable for
empirical asset pricing and financial analysis.
"""


import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, YearEnd

def expand_compustat_annual_to_monthly(
    comp_annual: pd.DataFrame, 
    date_col: str = "report_date", 
    id_col: str = "permno", 
    value_col: str = "x"
) -> pd.DataFrame:
    """
    Expand an annual fundamental item to a monthly timeseries, forward-filling
    from each 'report_date' until the next.  Then pivot so that the rows
    are month-end, columns are permno, and entries are the fundamental.

    Expects comp_annual to have (at least):
       [permno, report_date, x]
    where 'x' is the fundamental measure you want to expand.

    Returns
    -------
    pd.DataFrame
        index = month-end (last calendar day of each month),
        columns = permno,
        values = last known fundamental.
    """
    df = comp_annual[[id_col, date_col, value_col]].dropna(subset=[date_col])
    # Sort
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    out_list = []
    for perm, grp in df.groupby(id_col):
        # set index = report_date
        g = grp.set_index(date_col).sort_index()

        # monthly freq (last day of each month).
        # If you want EXACT month-ends from your CRSP data, you can
        # do something like .asfreq('M') or .resample('M'), but
        # keep in mind that resample('M') typically picks the last calendar day.
        g_monthly = g.resample("M").ffill()  
        # Tag this permno
        g_monthly[id_col] = perm
        out_list.append(g_monthly)

    df_monthly = pd.concat(out_list).reset_index()
    
    # pivot
    pivoted = df_monthly.pivot_table(
        index=date_col, columns=id_col, values=value_col
    )
    pivoted.index.name = None  # rename if desired
    return pivoted


def add_report_date(comp):
    """
    Accounting data are assumed to be known four months after the end of the fiscal year.
    This function computes the report date for each record in the Compustat dataset
    by adding four months to the fiscal year-end date (datadate).
    """
    # Ensure 'datadate' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(comp["datadate"]):
        comp["datadate"] = pd.to_datetime(comp["datadate"])
    
    # Add four months to each 'datadate' to compute the report date
    comp["report_date"] = comp["datadate"] + pd.DateOffset(months=4)
    
    return comp


def merge_CRSP_and_Compustat(crsp, comp, ccm):
    """
    Merge CRSP and Compustat data to compute the book-to-market ratio (beme).

    This function merges Compustat fundamentals with the CRSP data using
    the CRSP-Compustat linking table (ccm). It first cleans the linking table by
    setting missing linkenddt values to today's date, then merges on the company
    identifier (gvkey). The function computes the end-of-year and portfolio
    formation date (jdate) by offsetting the Compustat data date, and restricts
    the links to those where the formation date falls between the link start
    (linkdt) and end (linkenddt) dates. Finally, it merges the restricted
    linking data with the CRSP dataset on permno and jdate, and calculates
    the book-to-market ratio (beme) as the scaled book equity divided by
    December market equity.

    Parameters
    ----------
    crsp (pandas.DataFrame): 
        CRSP data
    comp (pandas.DataFrame):
        Compustat fundamentals data with columns for "gvkey", "datadate", "be" 
        (book equity), and "count" (number of years in Compustat).
    ccm (pandas.DataFrame): 
        CRSP-Compustat linking table containing "gvkey", "permno", "linkdt", 
        and "linkenddt" used to match records across datasets.

    Returns
    ----------
        - crsp_comp_merged (pandas.DataFrame): A merged DataFrame

    """
    # if linkenddt is missing then set to today date
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.to_datetime("today"))

    ccm1 = pd.merge(comp, ccm, how="left", on=["gvkey"])
    ccm1["yearend"] = ccm1["datadate"] + YearEnd(0)
    ccm1["jdate"] = ccm1["yearend"] + MonthEnd(6)
    # set link date bounds
    ccm2 = ccm1[(ccm1["jdate"] >= ccm1["linkdt"]) & (ccm1["jdate"] <= ccm1["linkenddt"])]
    comp_col = pd.unique(["gvkey", "permno", "datadate", "yearend", "jdate"] + list(comp.columns))
    ccm2 = ccm2[comp_col]

    # link comp and crsp
    crsp_comp_merged = pd.merge(crsp, ccm2, how="inner", on=["permno", "jdate"])
    return crsp_comp_merged


def calc_book_equity(comp):
    """Calculate book equity and number of years in Compustat.

    Use pull_CRSP_Compustat.description_crsp for help
    """
    ##
    ## Create a new column 'ps' for preferred stock value in 'comp'.

    # First check if 'pstkrv' (Preferred Stock - Redemption Value) is null.
    # If 'pstkrv' is null, use 'pstkl' (Preferred Stock - Liquidating Value) instead.
    comp = comp.assign(ps = lambda x: x['pstkrv'].fillna(x['pstkl']))
    # Update the 'ps' column. If 'ps' is still null after the previous operation
    # (meaning both 'pstkrv' and 'pstkl' were null), try to use
    # 'pstk' (Preferred/Preference Stock (Capital) - Total) as the preferred
    # stock value. This step ensures that any available preferred stock value is used.
    comp['ps'] = comp['ps'].fillna(comp['pstk'])
    # Another update to the 'ps' column. If 'ps' is still null
    # set the preferred stock value to 0.
    comp['ps'] = comp['ps'].fillna(0)
    # Replace null values in the 'txditc' column (Deferred Taxes and Investment Tax Credit)
    # with 0.
    comp['txditc'] = comp['txditc'].fillna(0)

    ##
    ## Calculate book equity ('be')
    # Book equity is calculated as the sum of 'seq' (Stockholders' Equity - Parent)
    # and 'txditc' (Deferred Taxes and Investment Tax Credit),
    # minus 'ps' (the calculated preferred stock value from previous steps).
    # This formula reflects the accounting equation where book equity is essentially
    # the net worth of the company from a bookkeeping perspective, adjusted for
    # preferred stock and tax considerations.
    comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
    # Update the 'be' (book equity) column to ensure that only positive values
    # are retained. If 'be' is less than or equal to 0, it is replaced with NaN
    comp['be'] = comp['be'].where(comp['be'] > 0, np.nan)

    # Drop NaN values in 'be' (book equity) to ensure that only valid entries are considered.
    comp = comp.dropna(subset=['be'])
    
    return comp