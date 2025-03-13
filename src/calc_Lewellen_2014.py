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
import wrds
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


def get_subsets(crsp: pd.DataFrame):
    """
    Return a dictionary 'subsets' with keys:
        'all_but_tiny_stocks', 'large_stocks', 'all_stocks'
    and each value is itself a dict whose keys are month-end dates
    (e.g. crsp['mthcaldt']) and values are sets of permno.

    subsets = {
        'all_but_tiny_stocks': { month_end1: set_of_permnos, month_end2: ..., ... },
        'large_stocks':        { month_end1: set_of_permnos, month_end2: ..., ... },
        'all_stocks':          { month_end1: set_of_permnos, month_end2: ..., ... },
    }

    'all_but_tiny_stocks' are those larger than the NYSE 20th percentile
    'large_stocks' are those larger than the NYSE 50th percentile
    'all_stocks' is everyone.

    Parameters
    ----------
    crsp : pd.DataFrame
        Must contain at least 'mthcaldt', 'permno', 'me', and 'primaryexch'.

    Returns
    -------
    dict
        A dictionary of dictionaries, as described above.
    """
    subsets = {
        "all_but_tiny_stocks": {},
        "large_stocks": {},
        "all_stocks": {}
    }

    # Group by the month-end date
    for date, group_df in crsp.groupby("mthcaldt", as_index=False):
        
        # Identify the NYSE subset
        nyse_df = group_df[group_df["primaryexch"] == "N"]
        
        # Compute the 20th and 50th percentile of 'me' in NYSE
        if len(nyse_df) > 0:
            me_20 = nyse_df["me"].quantile(0.2)
            me_50 = nyse_df["me"].quantile(0.5)
        else:
            # If no NYSE stocks (rare or missing data), skip
            me_20 = np.nan
            me_50 = np.nan
        
        # 1. all_stocks = all permnos
        all_stocks_set = set(group_df["permno"].unique())
        subsets["all_stocks"][date] = all_stocks_set

        # 2. all_but_tiny_stocks = me > me_20
        if not np.isnan(me_20):
            abt = group_df.loc[group_df["me"] > me_20, "permno"]
            abt_set = set(abt.unique())
        else:
            abt_set = set()
        subsets["all_but_tiny_stocks"][date] = abt_set

        # 3. large_stocks = me > me_50
        if not np.isnan(me_50):
            large = group_df.loc[group_df["me"] > me_50, "permno"]
            large_set = set(large.unique())
        else:
            large_set = set()
        subsets["large_stocks"][date] = large_set

    return subsets


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
The return for each function is a DataFrame with month-ends as index and permno as columns.
Accounting data are assumed to be known four months after the end of the fiscal year as calculated in add_report_date(comp) function.
""" 


def calc_log_size(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log market value of equity at the end of the prior month.

    Returns
    -------
    pd.DataFrame
        index = month-end date (mthcaldt)
        columns = permno
        values = ln( market_equity_{t-1} )
    """
    # Pivot to wide: row = mthcaldt, col = permno, val = me
    me_pivot = crsp_comp.pivot_table(
        index="mthcaldt", columns="permno", values="me"
    )
    # Shift by 1 row so that row t has prior month's ME
    me_shifted = me_pivot.shift(1)
    # log
    log_size = np.log(me_shifted)
    return log_size

def calc_log_bm(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log book-to-market ratio at the end of the prior month = ln(BE) - ln(ME).

    Assumes crsp_comp has columns 'be' for book equity and 'me' for market equity,
    both already forward-filled so that each permno–month row has the correct fundamental
    valid for that month.

    Returns
    -------
    pd.DataFrame
        index = month-end date
        columns = permno
        values = ln(be_{t-1}) - ln(me_{t-1})
    """
    me_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="me")
    be_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="be")

    me_shifted = me_pivot.shift(1)
    be_shifted = be_pivot.shift(1)

    log_bm = np.log(be_shifted) - np.log(me_shifted)
    return log_bm


def calc_return_12_2(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative return from month -12 to month -2.
    In other words, for each month t, we skip the last month (t-1)
    and compound returns from t-12 through t-2.

    Returns
    -------
    pd.DataFrame
        index = month-end date
        columns = permno
        values = cumulative return (not in logs) over months [t-12, t-2].
    """
    # Pivot monthly returns
    ret_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="retx")
    
    # We want the product of (1+ret) for 11 months: t-12..t-2
    #  - shift(2) so that row t sees ret_{t-2} in the "current" row
    #  - rolling(11) so we pick up t-2..t-12. That’s 11 monthly returns
    ret_shifted = ret_pivot.shift(2)

    def rolling_prod_minus_one(x):
        return np.prod(1 + x) - 1

    # Rolling window of length=11
    # For date t, this sums over t, t-1, ..., t-10 in the SHIFTED data,
    # i.e. ret_{t-2}, ..., ret_{t-12} in original time
    r_12_2 = ret_shifted.rolling(window=11, min_periods=11).apply(
        rolling_prod_minus_one, raw=True
    )
    return r_12_2


def calc_accruals(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change in non-cash net working capital minus depreciation in the prior fiscal year,
    then expand to monthly frequency (index=month-end, columns=permno).

    Assumes `crsp_comp` has columns:
        - 'permno' (or else you must merge compustat with ccm first),
        - 'report_date' = datadate + 4 months,
        - 'accruals',
        - 'depreciation'.

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = accrual measure
    """
    df = crsp_comp.copy()

    # Construct the annual measure
    df["accruals_final"] = df["accruals"] - df["depreciation"]

    # Expand forward from report_date to monthly, pivot
    # (If your 'permno' is not in comp, you need the ccm link or a merged DataFrame.)
    pivoted = expand_compustat_annual_to_monthly(
        comp_annual=df,
        date_col="report_date",
        id_col="permno",
        value_col="accruals_final"
    )
    return pivoted


def calc_log_issues_36(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in split-adjusted shares outstanding from month -36 to month -1.
    For each month t, it is ln(shrout_{t-1}) - ln(shrout_{t-36}).

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = log( ... ) 
    """
    # Pivot shares outstanding
    shrout_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="shrout")

    # shrout_{t-1}
    shrout_t1 = shrout_pivot.shift(1)
    # shrout_{t-36}
    shrout_t36 = shrout_pivot.shift(36)

    log_issues_36 = np.log(shrout_t1) - np.log(shrout_t36)
    return log_issues_36


def calc_roa(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ROA = (income before extraordinary items) / (average total assets) in prior FY,
    then pivot to monthly frequency.

    Here, as a placeholder, we assume comp['roa'] is already computed. If not,
    you'd do something like:
       comp['roa'] = comp['earnings'] / comp['assets']
    with any needed lags or average of assets.

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = ROA
    """
    df = comp.copy()

    df["roa"] = df["earnings"] / df["assets"]  # or average assets

    pivoted = expand_compustat_annual_to_monthly(
        comp_annual=df,
        date_col="report_date",
        id_col="permno",
        value_col="roa"
    )
    return pivoted


def calc_log_assets_growth(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in total assets in the prior fiscal year,
    then pivot to monthly frequency.

    For each firm-year, log_assets_growth = ln(assets_t / assets_{t-1}).

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = log-asset-growth
    """
    df = comp.copy()
    df = df.sort_values(["permno", "datadate"])

    # We'll group by permno and compute one-year lag of 'assets'
    df["lag_assets"] = df.groupby("permno")["assets"].shift(1)
    df["log_assets_growth"] = np.log(df["assets"] / df["lag_assets"])

    pivoted = expand_compustat_annual_to_monthly(
        comp_annual=df,
        date_col="report_date",
        id_col="permno",
        value_col="log_assets_growth"
    )
    return pivoted


def calc_dy(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate dividend yield = (sum of div per share in prior 12 months) / price_{t-1}.
    Using Dividends Common/Ordinary (dvc)
    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = dividend yield
    """
    # Pivot dividends and price
    div_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="dvc")
    prc_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="prc")

    # Rolling sum of dividends over last 12 months
    div_12 = div_pivot.rolling(window=12, min_periods=1).sum()

    # Price at t-1
    prc_shifted = prc_pivot.shift(1)

    dy = div_12 / prc_shifted
    return dy


def calc_log_issues_12(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log growth in shares outstanding from month -12 to month -1:
    ln(shrout_{t-1}) - ln(shrout_{t-12}).

    Returns
    -------
    pd.DataFrame
    """
    shrout_pivot = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="shrout")
    shrout_t1 = shrout_pivot.shift(1)
    shrout_t12 = shrout_pivot.shift(12)
    log_iss_12 = np.log(shrout_t1) - np.log(shrout_t12)
    return log_iss_12


def calc_log_return_13_36(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the sum of log monthly returns over the window [t-36, t-13].

    For each month t, we skip the last 12 months and sum the log returns from
    t-36 up through t-13. The final output is a DataFrame whose row t
    contains the sum of log(1 + ret_m) for months in [t-36, t-13].

    Parameters
    ----------
    crsp_comp : pd.DataFrame
        Must contain monthly returns for each (permno, date).
        We assume:
          - 'mthcaldt' is the month-end date,
          - 'permno' is the security identifier,
          - 'retx' is the monthly return (excluding dividends, if that is your convention).

    Returns
    -------
    pd.DataFrame
        index = month-end (sorted),
        columns = permno,
        values = sum of log(1 + ret) from t-36..t-13.
    """

    # 1) Pivot to wide format:
    #    row index = month-end date, column index = permno, values = monthly returns
    monthly_ret = crsp_comp.pivot_table(index="mthcaldt", columns="permno", values="retx")

    # 2) Convert each monthly return to log(1 + r)
    monthly_log_ret = np.log(1 + monthly_ret)

    # 3) We want, for month t, the sum of log-returns from t-36 through t-13.
    #    That is a 24-month window ending 13 months before t (exclusive of t-12..t-1).
    #    - shift(13) pushes each row down by 13, so row t's log-return is ret_{t-13}.
    #    - rolling(24).sum() aggregates from t-13 back to t-36 inclusive (24 rows).
    shifted = monthly_log_ret.shift(13)
    log_ret_13_36 = shifted.rolling(24).sum()

    return log_ret_13_36



def calculate_rolling_beta(crsp_d: pd.DataFrame,
                           crsp_index_d: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling beta from weekly returns for each stock, estimated
    over the past 36 months (~156 weeks).

    Parameters
    ----------
    crsp_d : pd.DataFrame
        Must have 'dlycaldt' (daily date) and 'retx' (stock daily returns).
    crsp_index_d : pd.DataFrame
        Must have 'caldt' (daily date) and 'vwretx' (daily market returns).
    
    Returns
    -------
    pd.DataFrame
        index = last weekly date (which we'll eventually map to a month-end),
        columns = permno,
        values = rolling beta
    """
    # 1) pivot daily stock returns
    stocks_dret = crsp_d.pivot_table(index="dlycaldt", columns="permno", values="retx")

    # 2) index daily market returns
    market_dret = crsp_index_d.set_index("caldt")["vwretx"].sort_index()

    # 3) convert daily to weekly by e.g. resampling each Friday
    stocks_wret = stocks_dret.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
    market_wret = market_dret.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)

    # 4) rolling covariance and variance (156 weeks ~ 3 years)
    window = 156

    # Cov(i, m)
    rolling_cov = stocks_wret.rolling(window).cov(market_wret)
    # Var(m)
    rolling_var_m = market_wret.rolling(window).var()

    # Beta = cov(i,m) / var(m)
    # rolling_cov is shaped like (date x permno) if it's a "panel" version.
    # The easiest way is to align them carefully:
    betas = rolling_cov.div(rolling_var_m, axis=0)

    # This yields a DataFrame with the same shape as stocks_wret: index=week_ending, columns=permno.
    # If you want to map it to each month-end, you might take the last weekly beta of each month:
    betas_monthly = betas.resample("M").last()

    return betas_monthly



def calc_std_12(crsp_d: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly standard deviation, estimated from daily returns
    over the past ~12 months (252 trading days).

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = stdev of daily returns over last 252 days.
    """
    daily_ret = crsp_d.pivot_table(index="dlycaldt", columns="permno", values="retx")

    # 252-day rolling stdev
    rolling_std = daily_ret.rolling(window=252, min_periods=100).std()

    # Then pick the last stdev in each calendar month
    std_monthly = rolling_std.resample("M").last()

    return std_monthly


def calc_debt_price(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate (short-term + long-term debt) / market equity at end of prior month.
    
    Typically, 'debt' = dlc + dltt from Compustat, forward-filled to each permno–month,
    and 'me' is the CRSP market equity.

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = debt_price ratio
    """
    df = crsp_comp.copy()

    # Pivot both
    debt_pivot = df.pivot_table(index="mthcaldt", columns="permno", values="total_debt")
    me_pivot   = df.pivot_table(index="mthcaldt", columns="permno", values="me")

    # Shift the market equity by 1 to get "prior month"
    me_shifted = me_pivot.shift(1)
    
    debt_price = debt_pivot.div(me_shifted)
    return debt_price


def calc_sales_price(crsp_comp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate (sales in prior fiscal year) / market value at the end of prior month.
    Expects the merged monthly DataFrame to have columns 'sales' and 'me'.

    Returns
    -------
    pd.DataFrame
        index = month-end
        columns = permno
        values = sales / me_{t-1}
    """
    df = crsp_comp.copy()

    sales_pivot = df.pivot_table(index="mthcaldt", columns="permno", values="sales")
    me_pivot    = df.pivot_table(index="mthcaldt", columns="permno", values="me")

    me_shifted = me_pivot.shift(1)

    sales_price = sales_pivot.div(me_shifted)
    return sales_price



def filter_companies_table1(all_variables_dict: dict) -> dict:
    """
    The function filters the companies that have:
      - current-month returns (Return, %)
      - beginning-of-month size (Log Size (-1))
      - book-to-market (Log B/M (-1))
      - lagged 12-month returns (Return (-2, -12))

    Returns
    -------
    dict
        {mth_end_date: set_of_permnos_with_all_required_data}
    """
    # The variable names in the all_variables_dict might be:
    #   'Return (%)'
    #   'Log Size (-1)'
    #   'Log B/M (-1)'
    #   'Return (-2, -12)'
    needed_vars = [
        "Return (%)", 
        "Log Size (-1)", 
        "Log B/M (-1)", 
        "Return (-2, -12)"
    ]

    # We do an intersection-based approach: for each month, find permnos
    # that are non-missing in all the needed variables
    filtered_companies_dict = {}

    # We assume all DFs share roughly the same index of month-ends
    # but let's get the union of all of them:
    all_dates = set()
    for var in needed_vars:
        all_dates = all_dates.union(all_variables_dict[var].index)

    all_dates = sorted(all_dates)  # to iterate in chronological order if needed

    for dt in all_dates:
        # Start with permnos that appear in all data
        # If dt not in the DataFrame, skip
        # Then keep only those with non-missing data
        valid_permnos = None
        for var in needed_vars:
            df = all_variables_dict[var]
            if dt not in df.index:
                # No data at this date
                valid_permnos = set()
                break
            row_data = df.loc[dt]  # this is a Series indexed by permno
            not_na_permnos = row_data[~row_data.isna()].index
            # Intersect with running set
            if valid_permnos is None:
                valid_permnos = set(not_na_permnos)
            else:
                valid_permnos &= set(not_na_permnos)

            if len(valid_permnos) == 0:
                break

        # store it
        filtered_companies_dict[dt] = valid_permnos if valid_permnos else set()

    return filtered_companies_dict


def winsorize(all_variables_dict: dict, lower_percentile=1, upper_percentile=99) -> dict:
    """
    Winsorize all characteristics in all_variables_dict at [1%, 99%], monthly.
    Replaces values below the 1st percentile with that percentile,
    and above the 99th percentile with that percentile.

    Parameters
    ----------
    all_variables_dict : dict
        Each value is a DataFrame with shape (month-end x permno).
    lower_percentile : float
    upper_percentile : float

    Returns
    -------
    dict
        Same structure, but winsorized in place.
    """
    for var, df in all_variables_dict.items():
        # We'll modify df in place
        for dt in df.index:
            row = df.loc[dt]  # row is a Series of shape [permno]
            valid = row.dropna()
            if len(valid) < 5:
                # Not enough data to compute percentiles
                continue

            lower_val = np.percentile(valid, lower_percentile)
            upper_val = np.percentile(valid, upper_percentile)

            clipped = row.clip(lower_val, upper_val)
            df.loc[dt] = clipped
    return all_variables_dict


def build_table_1(all_variables_dict: dict, subsets: dict) -> pd.DataFrame:
    """
    Build Table 1 as described: time-series averages of the monthly cross-sectional
    mean, stdev, and sample size for each variable.  We do it for each of the three
    subsets: all_stocks, all_but_tiny_stocks, large_stocks.

    Returns
    -------
    pd.DataFrame
        A summary table with one row per variable, and columns that contain
        [AllStocks_Avg, AllStocks_Std, AllStocks_N,
         ABT_Avg, ABT_Std, ABT_N,
         Large_Avg, Large_Std, Large_N]
    """
    # 1) filter to required data, then 2) winsorize
    filtered_companies_dict = filter_companies_table1(all_variables_dict)
    all_variables_dict = winsorize(all_variables_dict, lower_percentile=1, upper_percentile=99)

    # We'll do a final DataFrame that has one row per variable
    # and columns for the three subsets * 3 statistics
    subset_names = ["all_stocks", "all_but_tiny_stocks", "large_stocks"]
    stats_cols = []
    for sn in subset_names:
        stats_cols += [f"{sn}_Avg", f"{sn}_Std", f"{sn}_N"]
    table_rows = []

    for var_name, var_df in all_variables_dict.items():
        # We'll collect 9 summary stats for this variable
        row_dict = {"Variable": var_name}

        for sn in subset_names:
            # We gather cross-sectional stats month by month, then average across time
            monthly_means = []
            monthly_stds = []
            monthly_counts = []

            # subset[sn] is a dict: {month_end: set_of_permnos}
            # filtered_companies_dict[dt] is the set that passes the "has data" filter
            # We want the intersection
            for dt in var_df.index:
                if dt not in subsets[sn]:
                    continue
                if dt not in filtered_companies_dict:
                    continue

                final_permnos = subsets[sn][dt].intersection(filtered_companies_dict[dt])
                if len(final_permnos) == 0:
                    continue

                data_slice = var_df.loc[dt, final_permnos].dropna()
                if len(data_slice) == 0:
                    continue

                monthly_means.append(data_slice.mean())
                monthly_stds.append(data_slice.std())
                monthly_counts.append(data_slice.shape[0])

            if len(monthly_means) > 0:
                # Time-series average of cross-sectional means
                mean_of_means = np.mean(monthly_means)
                mean_of_stds  = np.mean(monthly_stds)
                mean_of_counts= np.mean(monthly_counts)
            else:
                mean_of_means = np.nan
                mean_of_stds  = np.nan
                mean_of_counts= np.nan

            row_dict[f"{sn}_Avg"] = mean_of_means
            row_dict[f"{sn}_Std"] = mean_of_stds
            row_dict[f"{sn}_N"]   = mean_of_counts

        table_rows.append(row_dict)

    # Build the final DataFrame
    df_final = pd.DataFrame(table_rows)
    df_final = df_final[["Variable"] + stats_cols]
    return df_final




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

    # 3) Merge comp + crsp_m + ccm => crsp_comp
    crsp_comp = merge_CRSP_and_Compustat(crsp, comp, ccm)

    # 4) Subsets
    subsets = get_subsets(crsp)

    # 5) Calculate all variables
    returns = crsp.pivot_table(index='mthcaldt', columns='permno', values='retx')
    
    log_size          = calc_log_size(crsp_comp)
    log_bm            = calc_log_bm(crsp_comp)
    return_2_12       = calc_return_12_2(crsp_comp)
    accruals          = calc_accruals(crsp_comp)  # or calc_accruals(crsp_comp) if you have them merged
    roa               = calc_roa(crsp_comp)
    log_assets_growth = calc_log_assets_growth(crsp_comp)
    dy                = calc_dy(crsp_comp)
    log_return_13_36  = calc_log_return_13_36(crsp_comp)
    log_issues_12     = calc_log_issues_12(crsp_comp)
    log_issues_36     = calc_log_issues_36(crsp_comp)
    betas             = calculate_rolling_beta(crsp_d, crsp_index_d)
    std_12            = calc_std_12(crsp_d)
    debt_price        = calc_debt_price(crsp_comp)
    sales_price       = calc_sales_price(crsp_comp)

    # 6) Build the dictionary for Table 1
    all_variables_dict = {
        "Return (%)":                returns,                # or your return_2_12 if you literally want that
        "Log Size (-1)":            log_size,
        "Log B/M (-1)":             log_bm,
        "Return (-2, -12)":         return_2_12,
        "Log Issues (-1,-12)":      log_issues_12,
        "Accruals (-1)":            accruals,
        "ROA (-1)":                 roa,
        "Log Assets Growth (-1)":   log_assets_growth,
        "Dividend Yield (-1,-12)":  dy,
        "Log Return (-13,-36)":     log_return_13_36,
        "Log Issues (-1,-36)":      log_issues_36,
        "Beta (-1,-36)":            betas,
        "Std Dev (-1,-12)":         std_12,
        "Debt/Price (-1)":          debt_price,
        "Sales/Price (-1)":         sales_price,
    }

    # 7) Build Table 1
    table_1 = build_table_1(all_variables_dict, subsets)


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
