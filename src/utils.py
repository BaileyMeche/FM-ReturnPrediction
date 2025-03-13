import datetime
from datetime import timedelta
import logging
import re
import zipfile
import gzip
import bz2
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


import holidays
import pandas as pd
import polars as pl
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import inspect

from settings import config

# ==============================================================================================
# Global Configuration
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
OUTPUT_DIR = Path(config("OUTPUT_DIR"))



# =============================================================================
# Helper Functions (Caching, Reading/Writing Files)
# =============================================================================

def _save_figure(
        fig: plt.Figure,
        plot_name_prefix: str,
        output_dir: Union[None, Path] = OUTPUT_DIR,
        dpi: int = 300
) -> None:
    """
    Saves a matplotlib figure to a PNG file if save_plot is True.
    The filename pattern is "<prefix>_YYYYMMDD_HHMMSS.png".

    Parameters:
    fig (plt.Figure): The matplotlib figure to save.
    plot_name_prefix (str): The prefix for the plot filename.
    output_dir (Path): The directory where the plot will be saved.
    """

    filename  = f"{plot_name_prefix}.png"
    # If output_dir is not provided, get the caller's file directory
    if output_dir is None:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])
        if caller_module and caller_module.__file__:
            output_dir = Path(caller_module.__file__).parent
        else:
            output_dir = Path.cwd()  # Fallback to current working directory if the caller module is unknown
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    #logging.info(f"Plot saved to {plot_path}")


def _cache_filename(
    code: str,
    filters_str: str,
    raw_data_dir: Union[None, Path] = RAW_DATA_DIR,
    file_ext_list: List[str] = ["parquet", "csv", "zip"]
) -> List[Path]:
    """
    Generate a cache filename based on the code and filters,
    returning up to three paths: .csv, .parquet, and .zip.
    """
    # Simplify filter string
    if "date" not in filters_str:
        today_str = datetime.datetime.today().strftime('%Y%m%d')
        filters_str += f"_{today_str}"

    safe_filters_str = re.sub(
        r'export=[a-zA-Z]*|[^,]*=',
        '',
        filters_str
    ).replace("/", "_").replace("=", "_").replace(",", "_").replace("-", "").replace(" ", "").replace("'", "")

    filenames = [
        f"{code.replace('/', '_')}__{safe_filters_str}.{file_ext}"
        for file_ext in file_ext_list
    ]
    # Clean up "empty" filters or duplicates
    filenames = [
        filename.replace("__.", ".")
                .replace("_.", ".")
        for filename in filenames
    ]

    # If raw_data_dir is not provided, get the caller's file directory
    if raw_data_dir is None:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])
        if caller_module and caller_module.__file__:
            raw_data_dir = Path(caller_module.__file__).parent
        else:
            raw_data_dir = Path.cwd()  # Fallback to current working directory if the caller module is unknown

    return [raw_data_dir / filename for filename in filenames]


def _hash_cache_filename(
    code: str,
    filters_str: str,
    raw_data_dir: Union[None, Path] = RAW_DATA_DIR,
    file_ext_list: List[str] = ["parquet", "csv", "zip"]
) -> List[Path]:
    """
    Generate a cache filename based on the code and filters,
    returning up to three paths: .csv, .parquet, and .zip.
    """
    # Simplify filter string
    if "date" not in filters_str and "end_date" not in filters_str:
        today_str = datetime.datetime.today().strftime('%Y%m%d')
        filters_str += f"_{today_str}"

    pattern = r'(?P<keep>[^,]*?(date)[^,]*=\[[^\]]*\]|[^,]*?(date)[^,]*=[^,]*)|(?P<remove>[^,]*)'

    # Initialize lists for keeping and removing attributes
    keep_list = []
    hash_list = []
    for match in re.finditer(pattern, filters_str):
        if match.group('keep'):
            keep_list.append(match.group('keep'))
        elif match.group('remove'):
            hash_list.append(match.group('remove'))

        # Construct separated strings
        keep_str = ','.join(keep_list)
        hash_str = ','.join(hash_list)

        safe_keep_str = (code + "_" + re.sub(
                r'export=[a-zA-Z]*|[^,]*=',
                '',
                keep_str
            )).replace("/", "_").replace("=", "_").replace(",", "_").replace("-", "").replace(" ", "").replace("'", "")

        safe_hash_str = re.sub(
            r'export=[a-zA-Z]*|[^,]*=',
            '',
            hash_str
        ).replace("/", "_").replace("=", "_").replace(",", "_").replace("-", "").replace(" ", "").replace("'", "")

    safe_hash_str = safe_hash_str
    
    # Use hashlib.sha to encode safe_hash_str:
    hash_str_hex = hashlib.sha256(safe_hash_str.encode()).hexdigest()[:9]


    filenames = [
            f"{safe_keep_str}_{hash_str_hex}.{file_ext}"
            for file_ext in file_ext_list
        ]
    # Clean up "empty" filters or duplicates
    filenames = [
        filename.replace("__.", ".")
                .replace("_.", ".")
        for filename in filenames
    ]

    # If raw_data_dir is not provided, get the caller's file directory
    if raw_data_dir is None:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])
        if caller_module and caller_module.__file__:
            raw_data_dir = Path(caller_module.__file__).parent
        else:
            raw_data_dir = Path.cwd()  # Fallback to current working directory if the caller module is unknown

    return [raw_data_dir / filename for filename in filenames]


def _file_cached(filepaths: List[Path]) -> Optional[Path]:
    """
    Check if any of the given filepaths exist (parquet, csv, or zip).
    Return the existent first that exists, else None.
    """
    for fp in filepaths:
        if Path(fp).exists():
            return Path(fp)
    return None


def _read_cached_data(filepath: Path) -> pd.DataFrame:
    """
    Read cached data from a file, supporting various formats and compression types.
    """
    fmt = filepath.suffix.lstrip(".")

    if fmt == "csv":
        #logging.info(f"Reading cached data from {filepath}")
        return pd.read_csv(filepath)

    elif fmt == "parquet":
        #logging.info(f"Reading cached data from {filepath}")
        return pd.read_parquet(filepath)

    elif fmt == "zip":
        #logging.info(f"Reading cached data from {filepath}")
        with zipfile.ZipFile(filepath, 'r') as z:
            file_name = z.namelist()[0]  # Assume only one file in the zip
            with z.open(file_name) as f:
                if file_name.endswith('.parquet'):
                    return pd.read_parquet(f)
                else:
                    return pd.read_csv(f)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


def _write_cache_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Write a DataFrame to file for caching.
    """
    # Get file extension from filepath:
    fmt = filepath.suffix.lstrip(".")
    if fmt == "parquet":
        df.to_parquet(filepath, index=False)
    elif fmt =="csv":
        df.to_csv(filepath, index=False)
    elif fmt == "xlsx":
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")
    logging.info(f"Data cached to {filepath}")


def _flatten_dict_to_str(d: Dict[str, Any]) -> str:
    """
    Recursively flatten a dict into a string representation for caching.
    Example:
       {'ticker': ['AAPL','MSFT'], 'date': {'gte': '2020-01-01'}} 
       -> "ticker=['AAPL','MSFT'],date.gte=2020-01-01"
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # recursively flatten sub-dict
            for subk, subv in v.items():
                items.append(f"{k}.{subk}={subv}")
        else:
            items.append(f"{k}={v}")
    return ",".join(items)


def _tickers_to_tuple(tickers: Union[str, List[str]]) ->  tuple:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    elif isinstance(tickers, list):
        tickers_tuple = tuple(tickers)
    else:
        tickers_tuple = tickers
    return tickers_tuple

def _format_tuple_for_sql_list(tickers: tuple) -> str:
    """
    Convert a tuple of tickers into a properly formatted SQL list.
    """
    if len(tickers) == 1:
        return f"('{tickers[0]}')"
    else:
        tickers_list = ", ".join(f"'{ticker}'" for ticker in tickers)
    return f"({tickers_list})"
    
def _save_cache_data(
        df: pd.DataFrame,
        data_dir: str,
        cache_paths: List[Path],
        file_name: str = None,
        file_type: str = None
) -> str:
    """
    Save data to cache.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    data_dir : str
        Directory to save the data.
    cache_paths : list
        List of cache paths.
    file_name : str
        File name to save the data.
    file_type : str
        File type to save the data. If None, searches from file_name, if none, uses "parquet".
    
    Returns
    -------
    str
        Path to the cached data.
    """

    if file_name is None:
        if file_type is None:
            file_type = "parquet"
        cache_path = next((p for p in cache_paths if p.suffix == f".{file_type}"), None)
    else:
        if not any(file_name.endswith(f".{ft}") for ft in ["parquet", "csv", "zip"]):
            if file_type is None:
                file_type = "parquet"
            cache_path = Path(data_dir, f"{file_name}.{file_type}")
        else:
            cache_path = Path(data_dir, file_name)
    _write_cache_data(df, cache_path)

    return cache_path


def load_cache_data(data_dir=RAW_DATA_DIR, file_name=None):
    if file_name is None:
        raise ValueError("file_name must be specified.")
    cached_file_path = Path(data_dir, file_name) if Path(data_dir, file_name).exists() else None
    if cached_file_path:
        return _read_cached_data(cached_file_path)
    else:
        raise FileNotFoundError(f"File {file_name} not found in {data_dir}.")
    

# =============================================================================
# Manipulating DataFrames Utilities
# =============================================================================


def time_series_to_df(returns: Union[pd.DataFrame, pd.Series, List[pd.Series]], name: str = "Returns"):
    """
    Converts returns to a DataFrame if it is a Series or a list of Series.

    Parameters:
        returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.

    Returns:
        pd.DataFrame: DataFrame of returns.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})

        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError(f'{name} must be either a pd.DataFrame or a list of pd.Series')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print(f'Could not convert {name} to float. Check if there are any non-numeric values')
        pass

    return returns


def fix_dates_index(returns: pd.DataFrame):
    """
    Fixes the date index of a DataFrame if it is not in datetime format and convert returns to float.

    Parameters:
        returns (pd.DataFrame): DataFrame of returns.

    Returns:
        pd.DataFrame: DataFrame with datetime index.
    """
    # Check if 'date' is in the columns and set it as the index

    # Set index name to 'date' if appropriate
    
    if returns.index.name is not None:
        if returns.index.name.lower() in ['date', 'dates', 'datetime']:
            returns.index.name = 'date'
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = 'date'
    elif 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    elif 'datetime' in returns.columns.str.lower():
        returns = returns.rename({'Datetime': 'date'}, axis=1)
        returns = returns.rename({'datetime': 'date'}, axis=1)
        returns = returns.set_index('date')

    # Convert dates to datetime if not already in datetime format or if minutes are 0
    try:
        returns.index = pd.to_datetime(returns.index, utc=True)
        if (returns.index.minute == 0).all():
            returns.index = returns.index.date
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns


def _filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str]
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
        df (pd.DataFrame): DataFrame to be filtered.
        keep_columns (list or str): Columns to keep in the DataFrame.
        drop_columns (list or str): Columns to drop from the DataFrame.
        keep_indexes (list or str): Indexes to keep in the DataFrame.
        drop_indexes (list or str): Indexes to drop from the DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    
    df = df.copy()

    # Columns
    if keep_columns is not None:
        keep_columns = [re.escape(col) for col in keep_columns]
        keep_columns = "(?i).*(" + "|".join(keep_columns) + ").*" if isinstance(keep_columns, list) else "(?i).*" + keep_columns + ".*"
        df = df.filter(regex=keep_columns)
        if drop_columns is not None:
            print('Both "keep_columns" and "drop_columns" were specified. "drop_columns" will be ignored.')

    elif drop_columns is not None:
        drop_columns = [re.escape(col) for col in drop_columns]
        drop_columns = "(?i).*(" + "|".join(drop_columns) + ").*" if isinstance(drop_columns, list) else "(?i).*" + drop_columns + ".*"
        df = df.drop(columns=df.filter(regex=drop_columns).columns)

    # Indexes
    if keep_indexes is not None:
        keep_indexes = [re.escape(col) for col in keep_indexes]
        keep_indexes = "(?i).*(" + "|".join(keep_indexes) + ").*" if isinstance(keep_indexes, list) else "(?i).*" + keep_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
        if drop_indexes is not None:
            print('Both "keep_indexes" and "drop_indexes" were specified. "drop_indexes" will be ignored.')

    elif drop_indexes is not None:
        drop_indexes = [re.escape(col) for col in drop_indexes]
        drop_indexes = "(?i).*(" + "|".join(drop_indexes) + ").*" if isinstance(drop_indexes, list) else "(?i).*" + drop_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
    
    return df