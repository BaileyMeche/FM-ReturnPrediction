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
    comp = comp.drop(columns=['ps', 'pstk', 'pstkrv', 'pstkl'], errors='ignore')  # Clean up by removing intermediate columns
    
    return comp


def expand_compustat_annual_to_monthly(
    comp_annual: pd.DataFrame,
    id_col: str = "gvkey",
    report_date_col: str = "report_date"
) -> pd.DataFrame:
    """
    Transforms an annual Compustat dataset to monthly frequency by:
      1. Dropping 'fyear'.
      2. Creating 'fund_date' = 'report_date'.
      3. For each gvkey, expanding to cover all month-ends from earliest to latest 'report_date'.
      4. Forward-filling all fundamental data.

    Parameters
    ----------
    comp_annual : pd.DataFrame
        Annual Compustat data in "long" format, with columns such as:
          - 'gvkey' (firm identifier)
          - 'datadate' (yearly period end)
          - 'report_date' (4 months after datadate, at month-end)
          - 'fyear' (fiscal year) which we will drop
          - Other columns with fundamental data (e.g. sales, ni, at, etc.).
    id_col : str, default "gvkey"
        The column name identifying the firm.
    report_date_col : str, default "report_date"
        The column name for the monthly period end that is 4 months after datadate.

    Returns
    -------
    pd.DataFrame
        A monthly DataFrame, with columns:
          - <id_col> (e.g., gvkey)
          - fund_date  (the expanded monthly date index)
          - datadate, plus any original Compustat columns (other than fyear)
          - Values are forward-filled from the annual data until the next report_date.
    """

    # 1. Drop fyear (ignore errors in case it's already missing)
    df = comp_annual.drop(columns=["fyear"], errors="ignore").copy()

    # 2. Create new column 'fund_date' (equal to report_date)
    df["fund_date"] = df[report_date_col]

    # 3. Sort and set a MultiIndex [gvkey, fund_date]
    df.set_index([id_col, "fund_date"], inplace=True)
    df.sort_index(inplace=True)

    # 4. For each gvkey, reindex so that we have a row for every month-end date
    #    from the earliest to the latest fund_date, forward-filling values.
    max_date_all = pd.to_datetime(df.index.get_level_values("fund_date")).max()
    def reindex_monthly(group: pd.DataFrame) -> pd.DataFrame:
        # Extract the fund_date level as datetime from the MultiIndex.
        dates = pd.to_datetime(group.index.get_level_values("fund_date"))
        min_date = dates.min()
        max_date = dates.max()
        # Extend the maximum date by 12 months.
        extended_max_date = min(max_date_all, max_date + pd.DateOffset(months=12))
        # Build a range of month-end dates.
        monthly_index = pd.date_range(start=min_date, end=extended_max_date, freq="M")
        
        # Retrieve the firm identifier from the first level of the index.
        gvkey_val = group.index.get_level_values(0)[0]
        # Create a new MultiIndex combining the firm id and the monthly dates.
        new_index = pd.MultiIndex.from_product(
            [[gvkey_val], monthly_index],
            names=[group.index.names[0], "fund_date"]
        )
        
        # Reindex the group using the new MultiIndex and forward-fill.
        return group.reindex(new_index, method="ffill")

    # Apply reindexing group-by-group
    expanded = (
        df.groupby(level=id_col, group_keys=False)
          .apply(reindex_monthly)
    )

    # Rename the second index level to 'fund_date'
    expanded = expanded.rename_axis([id_col, "fund_date"])
    expanded = expanded.reset_index()

    return expanded


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
        Compustat fundamentals data with columns for "gvkey", "fund_date", "be" 
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
    comp = comp.rename(columns={"fund_date": "jdate"})
    ccm1 = pd.merge(comp, ccm, how="left", on=["gvkey"])
    # set link date bounds
    ccm2 = ccm1[(ccm1["jdate"] >= ccm1["linkdt"]) & (ccm1["jdate"] <= ccm1["linkenddt"])]
    comp_col = ["permno"] + list(comp.columns)
    ccm2 = ccm2[comp_col]

    # link comp and crsp
    crsp_comp_merged = pd.merge(crsp, ccm2, how="inner", on=["permno", "jdate"])
    return crsp_comp_merged