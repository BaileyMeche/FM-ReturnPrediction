"""
This module calculates the Lewellen (2014) Table 1, Table 2 and Figure 1.

The Lewellen (2014) paper is available at:
https://faculty.tuck.dartmouth.edu/images/uploads/faculty/jonathan-lewellen/ExpectedStockReturns.pdf

"""
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import polars as pl
from pandas.tseries.offsets import MonthEnd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

from utils import load_cache_data
from pull_compustat import pull_Compustat, pull_CRSP_Comp_link_table
from pull_crsp import pull_CRSP_stock, pull_CRSP_index
from transform_compustat import (expand_compustat_annual_to_monthly,
                                 add_report_date,
                                 merge_CRSP_and_Compustat,
                                 calc_book_equity
                                )
from transform_crsp import calculate_market_equity

# ==============================================================================================
# GLOBAL CONFIGURATION
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def get_subsets(crsp_comp: pd.DataFrame) -> dict:
    """
    Given a monthly CRSP DataFrame with columns at least:
       ['mthcaldt', 'permno', 'me', 'primaryexch'],
    compute the NYSE 20th and 50th percentile of 'me' each month, store them
    in each row as 'me_20' and 'me_50', then build subset DataFrames:

      1) all_stocks          : everyone
      2) all_but_tiny_stocks : rows where me >= me_20
      3) large_stocks        : rows where me >= me_50

    If a particular month has no NYSE stocks (so me_20 or me_50 is NaN),
    then no rows from that month go into the 'all_but_tiny_stocks' or
    'large_stocks' subsets.

    Returns
    -------
    dict
        {
          "all_but_tiny_stocks":  <DataFrame of rows with me >= me_20>,
          "large_stocks":         <DataFrame of rows with me >= me_50>,
          "all_stocks":           crsp_comp   (the entire dataset)
        }
    """
    # 1) Sort for consistent grouping
    crsp_comp = crsp_comp.sort_values(["mthcaldt", "permno"]).copy()

    # 2) Compute month-specific me_20 and me_50 from NYSE
    #    group by mthcaldt, restrict to primaryexch == 'N'
    #    then get quantile(0.2) and quantile(0.5)
    nyse_me_percentiles = (
        crsp_comp
        .loc[crsp_comp["primaryexch"] == "N"]      # keep only NYSE rows
        .groupby("mthcaldt")["me"]
        .quantile([0.2, 0.5])                      # get 20th & 50th
        .unstack(level=1)                          # pivot so columns = [0.2, 0.5]
        .reset_index()
        .rename(columns={0.2: "me_20", 0.5: "me_50"})
    )
    # nyse_stats has columns ['mthcaldt', 'me_20', 'me_50']

    # 3) Merge these percentile columns back to crsp_comp
    crsp_comp = pd.merge(
        crsp_comp,
        nyse_me_percentiles,
        on="mthcaldt",
        how="left"
    )

    # 4) Create boolean columns for "all_but_tiny" and "large"
    #    If me_20 or me_50 is NaN (month has no NYSE?), these will be False
    crsp_comp["is_all_but_tiny"] = crsp_comp["me"] >= crsp_comp["me_20"]
    crsp_comp["is_large"]        = crsp_comp["me"] >= crsp_comp["me_50"]

    # 5) Now build the dictionary of DataFrames
    all_stocks_df = crsp_comp.copy()

    # For "all_but_tiny", we keep only rows with is_all_but_tiny == True
    all_but_tiny_df = crsp_comp.loc[crsp_comp["is_all_but_tiny"] == True].copy()

    # For "large_stocks", keep only rows with is_large == True
    large_stocks_df = crsp_comp.loc[crsp_comp["is_large"] == True].copy()

    subsets_crsp_comp = {
        "All Stocks":          all_stocks_df,
        "All-but-tiny stocks": all_but_tiny_df,
        "Large stocks":        large_stocks_df,
    }
    return subsets_crsp_comp


"""
In the functions below:
Calculate the fundamentals for each firm in the Compustat dataset.
    log_size: Log market value of equity at the end of the prior month
    log_bm: Log book value of equity minus log market value of equity at the end of the prior month
    return_12_2:  Stock return from month -12 to month -2
    accruals: Change in non-cash net working capital minus depreciation in the prior fiscal year
    log_issues_36: Log growth in split-adjusted shares outstanding from month -36 to month -1
    roa: Income before extraordinary items divided by average total assets in the prior fiscal year
    log_assets_growth: Log growth in total assets in the prior fiscal year
    dy: Dividends per share over the prior 12 months divided by price at the end of the prior month
    log_return_13_36: Log stock return from month -13 to month -36
    log_issues_12: Log growth in split-adjusted shares outstanding from month -12 to month -1
    beta_36: Rolling beta (market sensitivity) for each stock, estimated from weekly returns from month -36 to month -1
    std_12: Monthly standard deviation, estimated from daily returns from month -12 to month -1
    debt_price: Short-term plus long-term debt divided by market value at the end of the prior month
    sales_price: Sales in the prior fiscal year divided by market value at the end of the prior month

Accounting data are assumed to be known four months after the end of the fiscal year as calculated in add_report_date(comp) function.
""" 


def calc_log_size(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log market value of equity at the end of the prior month.
    For each (permno, month), we take 'me' from the previous month and log it.
    A new column 'log_size' is created.
    """
    # Shift 'me' by 1 month within each permno
    crsp_comp["me_lag"] = crsp_comp.groupby("permno")["me"].shift(1)
    crsp_comp["log_size"] = np.log(crsp_comp["me_lag"])
    crsp_comp = crsp_comp.drop(columns=["me_lag"])

    return crsp_comp

def calc_log_bm(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log book-to-market ratio = ln(BE_{t-1}) - ln(ME_{t-1}).
    For each (permno, month), we shift 'be' and 'me' by 1 month, then take logs.
    A new column 'log_bm' is created.
    """
    crsp_comp["be_lag"] = crsp_comp.groupby("permno")["be"].shift(1)
    crsp_comp["me_lag"] = crsp_comp.groupby("permno")["me"].shift(1)

    crsp_comp["log_bm"] = np.log(crsp_comp["be_lag"]) - np.log(crsp_comp["me_lag"])
    
    crsp_comp = crsp_comp.drop(columns=["be_lag", "me_lag"])

    return crsp_comp


def calc_return_12_2(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative return from month -12 to month -2 for each t.
    We skip the last month (t-1) and compound returns from t-12 through t-2.
    Creates a new column 'return_12_2'.
    """
    # Shift returns by 2 so that row t sees ret_{t-2} in 'retx_shift2'
    crsp_comp["retx_shift2"] = crsp_comp.groupby("permno")["retx"].shift(2)

    # We'll compute rolling product of 11 monthly returns: t-12..t-2
    # For each permno, we do a rolling(11) on (1 + retx_shift2).
    crsp_comp["1_plus_ret"] = 1 + crsp_comp["retx_shift2"]

    # Rolling product (min_periods=11 ensures we only compute when we have 11 data points)
    crsp_comp["rollprod_11"] = (
        crsp_comp
        .groupby("permno")["1_plus_ret"]
        .rolling(window=11, min_periods=11)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
    )

    crsp_comp["return_12_2"] = crsp_comp["rollprod_11"] - 1

    crsp_comp.drop(["retx_shift2", "1_plus_ret", "rollprod_11"], axis=1, inplace=True)

    return crsp_comp


def calc_accruals(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change in non-cash net working capital minus depreciation
    in the prior fiscal year.
    We assume the monthly row already contains the correct accruals and
    depreciation for that month (e.g., forward-filled from annual data).
    Creates a new column 'accruals_final'.
    """
    crsp_comp["accruals_final"] = crsp_comp["accruals"] - crsp_comp["depreciation"]
    return crsp_comp


def calc_log_issues_36(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in split-adjusted shares outstanding from t-36 to t-1:
        ln(shrout_{t-1}) - ln(shrout_{t-36})
    Creates a new column 'log_issues_36'.
    """
    crsp_comp["shrout_t1"] = crsp_comp.groupby("permno")["shrout"].shift(1)
    crsp_comp["shrout_t36"] = crsp_comp.groupby("permno")["shrout"].shift(36)

    crsp_comp["log_issues_36"] = (
        np.log(crsp_comp["shrout_t1"]) - np.log(crsp_comp["shrout_t36"])
    )
    crsp_comp.drop(["shrout_t1", "shrout_t36"], axis=1, inplace=True)

    return crsp_comp


def calc_log_issues_12(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in shares outstanding from t-12 to t-1:
        ln(shrout_{t-1}) - ln(shrout_{t-12}).
    Creates a new column 'log_issues_12'.
    """
    crsp_comp["shrout_t1"] = crsp_comp.groupby("permno")["shrout"].shift(1)
    crsp_comp["shrout_t12"] = crsp_comp.groupby("permno")["shrout"].shift(12)

    crsp_comp["log_issues_12"] = (
        np.log(crsp_comp["shrout_t1"]) - np.log(crsp_comp["shrout_t12"])
    )
    crsp_comp.drop(["shrout_t1", "shrout_t12"], axis=1, inplace=True)

    return crsp_comp


def calc_roa(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ROA as (income before extraordinary items) / (average total assets) in the prior FY.
    We assume 'roa' or 'earnings' and 'assets' are already properly merged in each monthly row.
    Creates a new column 'roa'.
    """
    # For illustration, if not already done:
    crsp_comp["roa"] = crsp_comp["earnings"] / crsp_comp["assets"]  # or average if you have that
    return crsp_comp


def calc_log_assets_growth(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in total assets from prior year: ln(assets_t / assets_{t-1}).
    We assume monthly data has the correct 'assets' for that month, forward-filled from the annual date.
    Creates a new column 'log_assets_growth'.
    """
    crsp_comp["lag_assets"] = crsp_comp.groupby("permno")["assets"].shift(12)  # or shift(1) if truly each year is only 12 months apart

    crsp_comp["log_assets_growth"] = np.log(crsp_comp["assets"] / crsp_comp["lag_assets"])
    crsp_comp.drop("lag_assets", axis=1, inplace=True)
    return crsp_comp


def calc_dy(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate dividend yield = (sum of dividends over prior 12 months) / price_{t-1}.
    We assume 'dvc' is the monthly dividend in month t, 'prc' is the end-of-month price.
    Creates a new column 'dy'.
    """
    df = crsp_comp.sort_values(["permno", "mthcaldt"]).copy()

    # Rolling sum of dividends over last 12 months
    df["div12"] = (
        df.groupby("permno")["dvc"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Price at t-1
    df["prc_t1"] = df.groupby("permno")["prc"].shift(1)
    df["dy"] = df["div12"] / df["prc_t1"]

    df.drop(["div12", "prc_t1"], axis=1, inplace=True)

    return df


def calc_log_return_13_36(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sum of log monthly returns over [t-36, t-13].
    For each month t, we skip last 12 months and sum the logs from t-36..t-13.
    Creates a new column 'log_return_13_36'.
    """

    # log(1 + retx)
    crsp_comp["log1p_ret"] = np.log(1 + crsp_comp["retx"])

    # shift by 13, then sum 24 rolling
    crsp_comp["log1p_ret_shift13"] = crsp_comp.groupby("permno")["log1p_ret"].shift(13)
    crsp_comp["log_sum_24"] = (
        crsp_comp.groupby("permno")["log1p_ret_shift13"]
        .rolling(window=24, min_periods=24)
        .sum()
        .reset_index(level=0, drop=True)
    )

    crsp_comp["log_return_13_36"] = crsp_comp["log_sum_24"]

    crsp_comp.drop(["log1p_ret", "log1p_ret_shift13", "log_sum_24"], axis=1, inplace=True)

    return crsp_comp


def calc_debt_price(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate (short-term + long-term debt) / market equity at end of prior month.
    Creates a new column 'debt_price'.
    """
    crsp_comp["me_lag"] = crsp_comp.groupby("permno")["me"].shift(1)

    crsp_comp["debt_price"] = crsp_comp["total_debt"] / crsp_comp["me_lag"]

    crsp_comp.drop("me_lag", axis=1, inplace=True)

    return crsp_comp


def calc_sales_price(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate (sales) / market value at the end of prior month.
    Creates a new column 'sales_price'.
    """
    crsp_comp["me_lag"] = crsp_comp.groupby("permno")["me"].shift(1)

    crsp_comp["sales_price"] = crsp_comp["sales"] / crsp_comp["me_lag"]

    crsp_comp.drop("me_lag", axis=1, inplace=True)

    return crsp_comp


def calculate_rolling_beta(crsp_d: pd.DataFrame,
                           crsp_index_d: pd.DataFrame,
                           crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling beta from weekly returns for each stock, estimated
    over the past 36 months (~156 weeks).

    Parameters
    ----------
    crsp_d : pd.DataFrame
        Must have 'dlycaldt' (daily date) and 'retx' (stock daily returns).
    crsp_index_d : pd.DataFrame
        Must have 'caldt' (daily date) and 'vwretx' (daily market returns).
    crsp_comp : pd.DataFrame
        Must have 'permno' (stock identifier) and 'mthcaldt' (month-end date).
    
    Returns
    -------
    pd.DataFrame
        index = last weekly date (which we'll eventually map to a month-end),
        columns = permno,
        values = rolling beta
    """

    df = crsp_d[['permno', 'dlycaldt', 'retx']].copy()
    mkt = crsp_index_d[["caldt","vwretx"]].copy()

    # Rename columns
    df = df.rename(columns={'retx': 'Ri', 'dlycaldt': 'date'})
    mkt = mkt.rename(columns={'vwretx': 'Rm', 'caldt': 'date'})

    # Convert to Polars DataFrame
    df_pl = pl.DataFrame(df)
    mkt_pl = pl.DataFrame(mkt)

    # Join on "date"
    df_joined = df_pl.join(mkt_pl, on="date")

    # Create log-returns (log_Ri, log_Rm) = log(1 + Ri), log(1 + Rm)
    df_joined = df_joined.with_columns([
        (pl.col("Ri") + 1).log().alias("log_Ri"),
        (pl.col("Rm") + 1).log().alias("log_Rm")
    ])

    # Sort by date
    df_joined = df_joined.sort(["permno", "date"])

    # Convert to a LazyFrame to use groupby_rolling
    lazy_df = df_joined.lazy()

    # Use groupby_rolling to aggregate over a 156-week window, grouped by permno.
    # This computes the rolling partial sums needed to approximate beta in log-return space.
    df_beta_lazy = (
        lazy_df
        .group_by_dynamic(
            index_column="date",
            every="1w",
            period="156w", 
            by="permno"
        )
        .agg([
            pl.col("log_Ri").sum().alias("sum_Ri"),
            pl.col("log_Rm").sum().alias("sum_Rm"),
            (pl.col("log_Ri") * pl.col("log_Rm")).sum().alias("sum_RiRm"),
            (pl.col("log_Rm") ** 2).sum().alias("sum_Rm2"),
            pl.count().alias("count_obs"),
        ])
        # Compute beta from the aggregated sums.
        # Using the formula:
        #   beta = [sum_RiRm - (sum_Ri * sum_Rm / N)] / [sum_Rm2 - (sum_Rm^2 / N)]
        .with_columns([
        (
            (pl.col("sum_RiRm") - (pl.col("sum_Ri") * pl.col("sum_Rm") / pl.col("count_obs")))
            /
            (pl.col("sum_Rm2") - (pl.col("sum_Rm")**2 / pl.col("count_obs")))
        ).alias("beta")
        ])
    )

    # Collect the results into an eager DataFrame.
    df_beta_pl = df_beta_lazy.collect()

    # (Optional) Convert back to a pandas DataFrame if needed.
    df_beta = df_beta_pl.to_pandas()

    df_beta['jdate'] = pd.to_datetime(df_beta['date']).dt.to_period('M').dt.to_timestamp('M')
    df_beta.drop_duplicates(subset=['permno', 'jdate'], keep='last', inplace=True)
    
    crsp_comp =  pd.merge(left=crsp_comp, right=df_beta[['permno', 'jdate', 'beta']], on=['permno', 'jdate'], how='left')

    return crsp_comp



def calc_std_12(crsp_d: pd.DataFrame, crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly standard deviation from daily returns over the past ~12 months (252 trading days).
    This is the only function that works on daily data. We then merge the monthly result back.
    Creates a monthly DataFrame of stdevs, then you can merge to your monthly 'crsp_comp'.
    """
    
    df_std_12 = crsp_d.copy()

    # 252-day rolling std
    df_std_12["rolling_std_252"] = (
        df_std_12.groupby("permno")["retx"]
        .rolling(window=252, min_periods=100)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Annualize:
    df_std_12["rolling_std_252"] = df_std_12["rolling_std_252"] * np.sqrt(252)

    # 3) For each month-end, pick the last available daily std in that month
    #    We'll create 'month_end' for daily data, then do a groupby last.
    df_std_12["jdate"] = df_std_12["dlycaldt"].dt.to_period("M").dt.to_timestamp("M")
    df_std_12.drop_duplicates(subset=['permno', 'jdate'], keep='last', inplace=True)
    
    crsp_comp =  pd.merge(left=crsp_comp, right=df_std_12[['permno', 'jdate', 'rolling_std_252']], on=['permno', 'jdate'], how='left')

    return crsp_comp


def filter_companies_table1(crsp_comp: pd.DataFrame, needed_var: list = None) -> set:
    """
    Identify companies that do NOT fit the criteria.

    A company fits the criteria if, for each required variable, it has at least one nonmissing value.
    That is, if for any required variable a company has all missing values, it does NOT fit the criteria:
      - current-month returns (Return, %)
      - beginning-of-month size (Log Size (-1))
      - book-to-market (Log B/M (-1))
      - lagged 12-month returns (Return (-2, -12))

    That is, it removes companyes with all missing values for any of these variables.
    Returns
    -------
    set
        A set of permnos for companies that have all missing values in any one of the required variables.
    """
    if needed_var is None:
        needed_vars = ["retx", "log_size", "log_bm", "return_12_2"]
    else:
        needed_vars = needed_var

    # For each company, check if there is any variable for which all values are missing.
    def has_all_missing(group):
        # group[needed_vars].isna().all() returns a boolean Series per column
        # .any() returns True if any column is entirely missing.
        return group[needed_vars].isna().all().any()

    # Group by company identifier and apply the check.
    flag_series = crsp_comp.groupby("permno").apply(has_all_missing)

    # Companies flagged True have at least one required variable completely missing.
    not_fitting = set(flag_series[flag_series].index)

    return not_fitting


def winsorize(crsp_comp: pd.DataFrame,
                   varlist: list,
                   lower_percentile=1,
                   upper_percentile=99) -> pd.DataFrame:
    """
    Winsorize the columns in `varlist` at [lower_percentile%, upper_percentile%], 
    cross-sectionally *by month* in the long DataFrame.
    Modifies the columns in place.
    """
    df = crsp_comp.sort_values(["mthcaldt", "permno"]).copy()

    for var in varlist:
        # Group by month, compute percentiles for that month
        def _winsorize_subgroup(subdf: pd.DataFrame):
            vals = subdf[var].dropna()
            if len(vals) < 5:
                return subdf  # Not enough data to reliably compute percentiles
            low_val = np.percentile(vals, lower_percentile)
            high_val = np.percentile(vals, upper_percentile)
            subdf[var] = subdf[var].clip(lower=low_val, upper=high_val)
            return subdf

        df = df.groupby("mthcaldt", group_keys=False).apply(_winsorize_subgroup)

    return df

    
def build_table_1(subsets_crsp_comp: dict, 
                  variables_dict: dict) -> pd.DataFrame:
    """
    Build a Table 1 with MultiIndex columns. For each subset in subsets_crsp_comp,
    we calculate time-series average of monthly cross-sectional stats for each variable.

    subsets_crsp_comp : dict
      {
         "All Stocks": <DataFrame>,
         "All-but-tiny stocks": <DataFrame>,
         "Large stocks": <DataFrame>
      }

    variables_dict : dict
      {
        "Return (%)": "retx",
        "Log Size (-1)": "log_size",
        ...
      }

    Returns
    -------
    pd.DataFrame
        A table with one row per variable in `columns_of_interest`. Columns:
          - 'Avg':    The time-series average of the monthly cross-sectional means
          - 'Std':    The time-series average of the monthly cross-sectional stds
          - 'N':      The total number of unique permnos (distinct stocks) that appear for that variable in
    """

    subset_tables = {}  # We'll store a partial table for each subset

    for subset_name, df_subset in subsets_crsp_comp.items():
        rows = []
        
        for var_label, var_col in variables_dict.items():
            # 1) Keep only relevant columns
            if var_col not in df_subset.columns:
                # If for some reason this column doesn't exist, skip or fill with NaN
                rows.append({
                    "Column": var_label,
                    "Avg": np.nan,
                    "Std": np.nan,
                    "N":   np.nan
                })
                continue

            df_clean = df_subset[[var_col, "mthcaldt", "permno"]].copy()
            # 2) Replace inf with NaN, drop rows with NaN in var_col
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_clean.dropna(subset=[var_col], inplace=True)

            if df_clean.empty:
                rows.append({
                    "Column": var_label,
                    "Avg": np.nan,
                    "Std": np.nan,
                    "N":   np.nan
                })
                continue

            # 3) Group by month, compute cross-sectional mean, std
            monthly_stats = df_clean.groupby("mthcaldt")[var_col].agg(["mean", "std"])
            # 4) Time-series average of monthly means, monthly std
            avg_mean = monthly_stats["mean"].mean()
            avg_std  = monthly_stats["std"].mean()

            # 5) N as total distinct permnos in the entire subset (like your friend’s example)
            N = df_clean["permno"].nunique()

            rows.append({"Column": var_label, "Avg": avg_mean, "Std": avg_std, "N": N})

        # Build a partial DataFrame for this subset
        partial_df = pd.DataFrame(rows).set_index("Column")
        # We'll store it
        subset_tables[subset_name] = partial_df

    # Now we merge them side-by-side with MultiIndex columns
    # Example: top-level is subset_name, second-level is [Avg, Std, N]
    partial_dfs = []
    for subset_name, partial_df in subset_tables.items():
        # Rename columns with a MultiIndex
        partial_df.columns = pd.MultiIndex.from_product([
            [subset_name],
            partial_df.columns
        ])
        partial_dfs.append(partial_df)

    # Concatenate along columns (axis=1)
    final_df = pd.concat(partial_dfs, axis=1)

    # Sort columns in a nice order if needed
    # final_df = final_df.reindex(columns=["All Stocks", "All-but-tiny stocks", "Large stocks"], level=0)

    return final_df


if __name__ == "__main__":

    # 1) Load raw data
    comp = load_cache_data(data_dir=RAW_DATA_DIR, file_name="Compustat_fund.parquet")
    ccm  = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_Comp_Link_Table.parquet")
    crsp_d = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_stock_d.parquet")
    crsp_m = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_stock_m.parquet")
    crsp_index_d = load_cache_data(data_dir=RAW_DATA_DIR, file_name="CRSP_index_d.parquet")

    # 2) Calculate market equity
    crsp = calculate_market_equity(crsp_m)

    # 2) Add report date and calculate book equity
    comp = add_report_date(comp)
    comp = calc_book_equity(comp)
    comp = expand_compustat_annual_to_monthly(comp)

    # 3) Merge comp + crsp_m + ccm => crsp_comp
    crsp_comp = merge_CRSP_and_Compustat(crsp, comp, ccm)

    # 4) Calculate all variables
    crsp_comp = crsp_comp.sort_values(["permno", "mthcaldt"])
    crsp_d = crsp_d.sort_values(["permno", "dlycaldt"])
    crsp_index_d = crsp_index_d.sort_values(["caldt"])
    
    crsp_comp = calc_log_size(crsp_comp)
    crsp_comp = calc_log_bm(crsp_comp)
    crsp_comp = calc_return_12_2(crsp_comp)
    crsp_comp = calc_accruals(crsp_comp)  # or calc_accruals(crsp_comp) if you have them merged
    crsp_comp       = calc_roa(crsp_comp)
    crsp_comp = calc_log_assets_growth(crsp_comp)
    crsp_comp = calc_dy(crsp_comp)
    crsp_comp = calc_log_return_13_36(crsp_comp)
    crsp_comp = calc_log_issues_12(crsp_comp)
    crsp_comp = calc_log_issues_36(crsp_comp)
    crsp_comp = calc_debt_price(crsp_comp)
    crsp_comp = calc_sales_price(crsp_comp)
    crsp_comp = calc_std_12(crsp_d, crsp_comp)
    crsp_comp = calculate_rolling_beta(crsp_d, crsp_index_d, crsp_comp)

    # 5) Winsorize the variables to remove outliers
    variables_dict = {
    "Return (%)":                "retx",                # Assuming you are keeping this column name
    "Log Size (-1)":             "log_size",
    "Log B/M (-1)":              "log_bm",
    "Return (-2, -12)":          "return_12_2",
    "Log Issues (-1,-12)":       "log_issues_12",
    "Accruals (-1)":             "accruals_final",
    "ROA (-1)":                  "roa",
    "Log Assets Growth (-1)":    "log_assets_growth",
    "Dividend Yield (-1,-12)":   "dy",
    "Log Return (-13,-36)":      "log_return_13_36",
    "Log Issues (-1,-36)":       "log_issues_36",
    "Beta (-1,-36)":             "rolling_beta",
    "Std Dev (-1,-12)":          "rolling_std_252",
    "Debt/Price (-1)":           "debt_price",
    "Sales/Price (-1)":          "sales_price",
    }
    crsp_comp = winsorize(crsp_comp, variables_dict.values())

    # 6) Create subsets for analysis
    subsets_comp_crsp = get_subsets(crsp_comp) # Dictionary of dataframes corresponding of the data sets

    # 7) Build Table 1
    table_1 = build_table_1(subsets_comp_crsp, variables_dict)


    models = {
        'Model 1: Three Predictors': ['Log Size (-1)', 'Log B/M (-1)', 'Return (-2, -12)'],
        'Model 2: Seven Predictors': [
                            'Log Size (-1)', 'Log B/M (-1)', 'Return (-2, -12)', 'Log Issues (-1,-36)', 
                            'Accruals (-1)', 'ROA (-1)', 'Log Assets Growth (-1)'
                            ],
        'Model 3: Fourteen Predictors': [
                            'Log Size (-1)', 'Log B/M (-1)', 'Return (-2, -12)', 
                            'Log Issues (-1,-12)', 'Accruals (-1)', 'ROA (-1)', 'Log Assets Growth (-1)', 
                            'Dividend Yield (-1,-12)', 'Log Return (-13,-36)', 'Log Issues (-1,-36)', 
                            'Beta (-1,-36)', 'Std Dev (-1,-12)', 'Debt/Price (-1)', 'Sales/Price (-1)'
                            ]
    }
