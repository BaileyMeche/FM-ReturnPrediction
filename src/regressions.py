import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
from typing import Union, List, Dict
import plotly.graph_objects as go

from utils import time_series_to_df, fix_dates_index, _filter_columns_and_indexes

# ================================================================================================
# Regressions
# ================================================================================================

def calc_regression(
    Y: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    X: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    intercept: bool = True,
    annual_factor: Union[None, int] = None,
    return_model: bool = False,
    return_fitted_values: bool = False,
    p_values: bool = True,
    tracking_error: bool = True,
    r_squared: bool = True,
    rse_mae: bool = True,
    treynor_ratio: bool = False,
    information_ratio: bool = False,
    market_name: str = 'SPY US Equity',
    sortino_ratio: bool = False,
    timeframes: Union[None, dict] = None,
    keep_columns: Union[list, str] = None,
    drop_columns: Union[list, str] = None,
    keep_indexes: Union[list, str] = None,
    drop_indexes: Union[list, str] = None
    ) -> Union[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    
    """
    Performs an OLS regression of a "many-to-many" returns time series with optional intercept, timeframes, statistical ratios, and performance ratios.

    Parameters:
    Y (pd.DataFrame, pd.Series or List or pd.Series): Dependent variable(s) for the regression.
    X (pd.DataFrame, pd.Series or List or pd.Series): Independent variable(s) for the regression.
    intercept (bool, default=True): If True, includes an intercept in the regression.
    annual_factor (int or None, default=None): Factor for annualizing regression statistics.
    return_model (bool, default=False): If True, returns the regression model object.
    return_fitted_values (bool, default=False): If True, returns the fitted values of the regression.
    p_values (bool, default=True): If True, displays p-values for the regression coefficients.
    tracking_error (bool, default=True): If True, calculates the tracking error of the regression.
    r_squared (bool, default=True): If True, calculates the R-squared of the regression.
    rse_mae (bool, default=False): If True, calculates the Mean Absolute Error (MAE) and Relative Squared Error (RSE) of the regression.
    treynor_ratio (bool, default=False): If True, calculates Treynor ratio.
    information_ratio (bool, default=False): If True, calculates Information ratio.
    market_name (str, default='SPY US Equity'): Name of the market index for the Treynor ratio.
    sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.
    timeframes (dict or None, default=None): Dictionary of timeframes to run separate regressions for each period.
    keep_columns (list or str, default=None): Columns to keep in the resulting DataFrame.
    drop_columns (list or str, default=None): Columns to drop from the resulting DataFrame.
    keep_indexes (list or str, default=None): Indexes to keep in the resulting DataFrame.
    drop_indexes (list or str, default=None): Indexes to drop from the resulting DataFrame.
    calc_sortino_ratio (bool, default=False): If True, calculates the Sortino ratio.

    Returns:
    pd.DataFrame or model: Regression summary statistics or the model if `return_model` is True.
    """

    X = time_series_to_df(X) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(X) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float

    Y = time_series_to_df(Y) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(Y) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
    
    y_names = list(Y.columns) if isinstance(Y, pd.DataFrame) else [Y.name]
    X_names = " + ".join(list(X.columns))
    X_names = "Intercept + " + X_names if intercept else X_names

    # Add the intercept
    if intercept:
        X = sm.add_constant(X)
 
    # Check if y and X have the same length
    if len(X.index) != len(Y.index):
        print(f'y has lenght {len(Y.index)} and X has lenght {len(X.index)}. Joining y and X by y.index...')
        df = Y.join(X, how='left')
        df = df.dropna()
        Y = df[y_names]
        X = df.drop(columns=y_names)
        if len(X.index) < len(X.columns) + 1:
            raise Exception('Indexes of y and X do not match and there are less observations than degrees of freedom. Cannot calculate regression')


    if isinstance(timeframes, dict):
        all_timeframes_regressions = pd.DataFrame()
        for name, timeframe in timeframes.items():
            if timeframe[0] and timeframe[1]:
                timeframe_Y = Y.loc[timeframe[0]:timeframe[1]]
                timeframe_X = X.loc[timeframe[0]:timeframe[1]]
            elif timeframe[0]:
                timeframe_Y = Y.loc[timeframe[0]:]
                timeframe_X = X.loc[timeframe[0]:]
            elif timeframe[1]:
                timeframe_Y = Y.loc[:timeframe[1]]
                timeframe_X = X.loc[:timeframe[1]]
            else:
                timeframe_Y = Y.copy()
                timeframe_X = X.copy()
            if len(timeframe_Y.index) == 0 or len(timeframe_X.index) == 0:
                raise Exception(f'No returns data for {name} timeframe')
            
            timeframe_Y = timeframe_Y.rename(columns=lambda col: col + f' ({name})')
            timeframe_regression = calc_regression(
                Y=timeframe_Y,
                X=timeframe_X,
                intercept=intercept,
                annual_factor=annual_factor,
                warnings=False,
                return_model=False,
                return_fitted_values=False,
                p_values=p_values,
                tracking_error=tracking_error,
                r_squared=r_squared,
                rse_mae=rse_mae,
                treynor_ratio=treynor_ratio,
                information_ratio=information_ratio,
                timeframes=None,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
                keep_indexes=keep_indexes,
                drop_indexes=drop_indexes
            )
            timeframe_regression.index = [f"{timeframe_regression.index} ({name})"]
            all_timeframes_regressions = pd.concat(
                [all_timeframes_regressions, timeframe_regression],
                axis=0
            )
        return all_timeframes_regressions
    
    regression_statistics = pd.DataFrame(index=y_names, columns=[])	
    fitted_values_all = pd.DataFrame(index=Y.index, columns=y_names)
    ols_results = {}
    for y_asset in y_names:
        # Fit the regression model: 
        y = Y[y_asset]
        try:
            ols_model = sm.OLS(y, X, missing="drop")
        except ValueError:
            y = y.reset_index(drop=True)
            X = X.reset_index(drop=True)
            ols_model = sm.OLS(y, X, missing="drop")
            print(f'"{y_asset}" Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
        
        ols_result = ols_model.fit()

        if return_model:
            ols_results[y_asset] = ols_result

        elif return_fitted_values:
            fitted_values = ols_results.fittedvalues
            fitted_values = fitted_values.rename(f'{y_asset}^')
            fitted_values_all[y_asset] = fitted_values

        else:
            
            # R-squared
            if r_squared == True:
                regression_statistics.loc[y_asset, 'R-Squared'] = ols_result.rsquared # R-squared
                if intercept == False:
                    print('No intercept in regression. R-Squared might not make statistical sense.')

            # Intercept    
            if intercept == True:
                regression_statistics.loc[y_asset, 'Alpha'] = ols_result.params.iloc[0]
                regression_statistics.loc[y_asset, 'Annualized Alpha'] = ols_result.params.iloc[0] * annual_factor # Annualized Alpha 
                
                if p_values == True: 
                    regression_statistics.loc[y_asset, 'P-Value (Alpha)'] = ols_result.pvalues.iloc[0] # Alpha p-value

            # Betas
            X_names = list(X.columns[1:]) if intercept else list(X.columns)
            betas = ols_result.params[1:] if intercept else ols_result.params
            betas_p_values = ols_result.pvalues[1:] if intercept else ols_result.pvalues
            
            for i in range(len(X_names)):
                regression_statistics.loc[y_asset, f"Beta ({X_names[i]})"] = betas.iloc[i] # Betas
                if p_values == True: 
                    regression_statistics.loc[y_asset, f"P-Value ({X_names[i]})"] = betas_p_values.iloc[i] # Beta p-values

            # Observed Mean and Standard Deviation
            regression_statistics.loc[y_asset, 'Observed Mean'] = y.mean()
            regression_statistics.loc[y_asset, 'Observed Std Dev'] = y.std()

            # Treynor Ratio
            if treynor_ratio == True:
                market_names = ['SPY', 'SPX', 'SP500', 'SPY US Equity']
                if market_name not in market_names:
                    print(f'Neither {market_name} are a factor in the regression. Treynor Ratio cannot be calculated.')
                else:
                    market_name = [m for m in market_names if m in X.columns][0]
                    try:
                        regression_statistics.loc[y_asset, 'Treynor Ratio'] = regression_statistics.loc[y_asset, 'Observed Mean'] / regression_statistics.loc[y_asset, f'Beta ({market_name})'] # Treynor Ratio
                        regression_statistics.loc[y_asset, 'Annualized Treynor Ratio'] = regression_statistics.loc[y_asset, 'Treynor Ratio'] * annual_factor # Annualized Treynor Ratio
                    except:
                        print(f'Treynor Ratio could not be calculated.')
            
            # Residual Standard Error (RSE) and Mean Absolute Error (MAE)
            residuals =  ols_result.resid
            rse = (sum(residuals**2) / (len(residuals) - len(ols_result.params))) ** 0.5 
            
            if rse_mae:
                regression_statistics.loc[y_asset, 'RSE'] = rse
                regression_statistics.loc[y_asset, 'MAE'] = abs(residuals).mean()
            
            # Tracking Error
            if tracking_error == True:
                regression_statistics.loc[y_asset, 'Tracking Error'] = residuals.std() 
                regression_statistics.loc[y_asset, 'Annualized Tracking Error'] = regression_statistics.loc[y_asset, 'Tracking Error'] * (annual_factor ** 0.5) # Annualized Residuals Volatility
            
            # Information Ratio
            if information_ratio == True:
                if intercept:
                    regression_statistics.loc[y_asset, 'Information Ratio'] = regression_statistics.loc[y_asset, 'Alpha'] / residuals.std() # Information Ratio
                    regression_statistics.loc[y_asset, 'Annualized Information Ratio'] = regression_statistics.loc[y_asset, 'Information Ratio'] * (annual_factor ** 0.5) # Annualized Information Ratio
            
            # Fitted Mean and Standard Deviation
            regression_statistics.loc[y_asset, "Fitted Mean"] = ols_result.fittedvalues.mean()
            regression_statistics.loc[y_asset, "Annualized Fitted Mean"] = regression_statistics.loc[y_asset, "Fitted Mean"] * annual_factor
            regression_statistics.loc[y_asset, 'Fitted Std Dev'] = ols_result.fittedvalues.std()
            
            if sortino_ratio:
                try:
                    regression_statistics.loc[y_asset, 'Sortino Ratio'] = regression_statistics.loc[y_asset, 'Fitted Mean'] / Y[Y < 0].std()
                except Exception as e:
                    print(f'Cannot calculate Sortino Ratio: {str(e)}. Set "calc_sortino_ratio" to False or review function')
    
    if return_model:
        return ols_results

    elif return_fitted_values:
        return fitted_values_all
    
    else:
        if regression_statistics.shape[0] == 1:
            regression_statistics = regression_statistics.T
        return _filter_columns_and_indexes(
            regression_statistics,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            keep_indexes=keep_indexes,
            drop_indexes=drop_indexes
        )
    

def calc_regression_rolling(
    returns: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    factors: Union[pd.DataFrame, pd.Series, List[pd.Series]],
    intercept: bool = True,
    moving_window: bool = False,
    exp_decaying_window: bool = False,
    window_size: int = 121,
    decay_alpha: float = 0.94, 
    betas_only: bool = True,
    fitted_values: bool = False,
    residuals: bool = False,
    ) -> Dict[datetime.datetime, pd.DataFrame]:

    """
    Performs a multiple OLS regression of a "one-to-many" returns time series with optional intercept on a rolling window. 
    Allows for different methods of windowing: expanding window (default), moving window, and exponential decay.

    This is the first stage of a Fama-MacBeth model, used to estimate the betas for every asset for every window.

    Parameters:
        returns (pd.DataFrame, pd.Series or List of pd.Series): Dependent variable for the regression.
        factors (pd.DataFrame, pd.Series or List of pd.Series): Independent variable(s) for the regression.
        intercept (bool, default=True): If True, includes an intercept in the regression.
        window_size (int): Number of observations to include in the moving window.
        betas_only (bool): If True, returns only the betas for each asset for each window.
        fitted_values (bool): If True, returns the fitted values.
        residuals (bool): If True, returns the residuals.
        moving_window (bool): If True, uses moving windows of size `window_size`.
        expanding_window (bool): If True, uses expanding windows of minimum size `window_size`.
        exp_decaying_window (bool): If True, uses expanding windows of minimum size `window_size` and exponential decaying weights.
        decay_alpha (float): Decay factor for exponential weighting.
        betas_only (bool, default=True): If True, returns only the betas for each asset for each window.
        fitted_values (bool, default=False): If True, also returns the fitted values of the regression.
        residuals (bool, default=False): If True, also returns the residuals of the regression.

    Returns: a dictionary of dataframes with the regression statistics for each rolling window.
    Returns the intercept (optional) and betas for each asset for each window.
    """
    
    factors = time_series_to_df(factors) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(factors) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
    
    returns = time_series_to_df(returns) # Convert returns to DataFrame if it is a Series or a list of Series
    fix_dates_index(returns) # Fix the date index of the DataFrame if it is not in datetime format and convert returns to float
    
    y_names = list(returns.columns) if isinstance(returns, pd.DataFrame) else [returns.name]
    factor_names = list(factors.columns)
    factor_names = ['Intercept'] + factor_names if (intercept == True and betas_only == False) else factor_names

    # Add the intercept
    if intercept:
        factors = sm.add_constant(factors)
    
    # Check if y and X have the same length
    if len(factors.index) != len(returns.index):
        print(f'y has lenght {len(returns.index)} and X has lenght {len(factors.index)}. Joining y and X by y.index...')
        df = returns.join(factors, how='left')
        df = df.dropna()
        returns = df[y_names]
        factors = df.drop(columns=y_names)
        if len(factors.index) < len(factors.columns) + 1:
            raise Exception('Indexes of y and X do not match and there are less observations than degrees of freedom. Cannot calculate regression')

    regres_columns = ['Beta (' + factor + ')' for factor in factor_names]
    regression_statistics = pd.DataFrame(index=returns.index, columns=regres_columns)

    # Loop through the windows
    for i in range(window_size, len(returns.index), 1):
        if exp_decaying_window:
            y_i = returns.iloc[:i]
            X_i = factors.iloc[:i]
            n_obs = i
            weights = np.array([decay_alpha ** (n_obs - j) for j in range(n_obs)])
            
            try:
                ols_model = sm.WLS(y_i, X_i, missing="drop", weights=weights)
            except ValueError:
                y_i = y_i.reset_index(drop=True)
                X_i = X_i.reset_index(drop=True)
                ols_model = sm.WLS(y_i, X_i, missing="drop", weights=weights)
                print('Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
        else:
            if moving_window:
                y_i = returns.iloc[i-window_size:i]
                X_i = factors.iloc[i-window_size:i]
            else: # Expanding Window
                y_i = returns.iloc[:i]
                X_i = factors.iloc[:i]

            # Fit the regression model: 
            try:
                ols_model = sm.OLS(y_i, X_i, missing="drop")
            except ValueError:
                y_i = y_i.reset_index(drop=True)
                X_i = X_i.reset_index(drop=True)

                ols_model = sm.OLS(y_i, X_i, missing="drop")

                print('Required to reset indexes to make regression work. Try passing "y" and "X" as pd.DataFrame')
            
        ols_results = ols_model.fit() 

        # Process betas for explanatory variables
        coeff = ols_results.params[1:] if (intercept and betas_only) else ols_results.params
        regression_statistics.loc[returns.index[i], regres_columns] = coeff.values # Betas
        
        current_X = factors.loc[returns.index[i], :]
        current_y = returns.iloc[i][0]
        if fitted_values:
            regression_statistics.loc[returns.index[i], 'Fitted Values'] = current_X @ coeff # Fitted Value
        if residuals:
            regression_statistics.loc[returns.index[i], 'Residuals'] = current_y - current_X @ coeff # Residuals
    regression_statistics = regression_statistics.dropna(how='all')  

    return regression_statistics

