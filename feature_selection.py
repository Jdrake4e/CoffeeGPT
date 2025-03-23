import utility_functions as uf
import file_readin_functions as frf
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import statsmodels.api as sm # type: ignore
from statsmodels.tsa.stattools import grangercausalitytests # type: ignore
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Context manager to redirect stdout and stderr to devnull.

    Temporarily suppresses all standard output during execution of the wrapped
    code and restores it afterward.

    Yields
    ------
    file object
        The devnull file object that stdout is redirected to.

    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target

def wrapped_granger_causality_test(target, feature, maxlag):
    """Test for Granger causality between time series variables.

    This function implements a robust Granger causality test to determine whether
    past values of one time series (feature) help predict future values of another
    time series (target) better than the target's past values alone.

    Parameters
    ----------
    target : pandas.Series
        The target time series variable to be predicted.
    feature : pandas.Series
        The feature time series variable to test for causal relationship.
    maxlag : int
        Maximum number of lags to test for causality.

    Returns
    -------
    list of dict
        List of dictionaries containing test results for each lag, with keys:
        
        - 'Lag' : int
            The lag order for this test
        - 'F-Statistic' : float
            The F-statistic value for this lag
        - 'p-value' : float
            The p-value for this lag's test

        Returns an empty list if the test fails.

    Notes
    -----
    Design Decisions:

    1. Statistical Framework:
       - Uses F-test based on sum of squared residuals
       - Tests null hypothesis of no Granger causality
       - Implements multiple lag testing for robustness

    2. Data Handling:
       - Removes missing values before testing
       - Preserves time series ordering
       - Handles non-stationary data gracefully

    3. Error Management:
       - Silent error handling to prevent pipeline breaks
       - Returns empty list instead of raising exceptions
       - Allows for batch processing of multiple tests

    4. Performance:
       - Efficient implementation using statsmodels
       - Minimizes memory usage in results storage
       - Suitable for large-scale feature selection

    Warnings
    --------
    - The test assumes linear relationships between variables
    - Results may be misleading for non-stationary time series
    - Small sample sizes may affect test reliability

    See Also
    --------
    statsmodels.tsa.stattools.grangercausalitytests : Core testing function
    feature_selection_pipeline : Main pipeline using this test

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'target': [100, 102, 105, 108, 110],
    ...     'feature': [7, 6.8, 6.5, 6.3, 6.0]
    ... })
    >>> # Run test
    >>> results = wrapped_granger_causality_test(
    ...     data['target'],
    ...     data['feature'],
    ...     maxlag=2
    ... )
    >>> # Print results
    >>> for r in results:
    ...     print(f"Lag {r['Lag']}: p={r['p-value']:.4f}")
    """

    # 1. INITIALIZATION
    results = []
    
    # 2. EXECUTION WITH ERROR HANDLING
    try:
        # 2.1 Data preparation - concatenate series and handle missing values
        data = pd.concat([target, feature], axis=1).dropna()
        
        # 2.2 Run Granger causality tests for multiple lags
        # Note: verbose=None suppresses stdout from statsmodels
        gc_test = grangercausalitytests(data, maxlag=maxlag, verbose=None)
        
        # 2.3 Extract and format test statistics for each lag
        for lag in range(1, maxlag + 1):
            # Extract F-statistic and p-value from the test results
            # The 'ssr_ftest' uses sum of squared residuals F-test
            f_stat = gc_test[lag][0]['ssr_ftest'][0]  # F-statistic value
            p_val = gc_test[lag][0]['ssr_ftest'][1]   # p-value
            
            # 2.4 Store results in a structured format
            results.append({
                'Lag': lag,              # Current lag being tested
                'F-Statistic': f_stat,   # F-statistic for this lag
                'p-value': p_val         # p-value for this lag
            })
    except:
        # 3. ERROR HANDLING
        # Silently handle any exceptions (insufficient data, non-stationarity, etc.)
        # and return an empty list to indicate the test couldn't be performed
        pass
    
    # 4. RETURN RESULTS
    return results

#TODO Generalize function to an arbitray target variable and date format
#TODO Implement better control over granularity of lag being targeted
#TODO fix this repeating warning: c:\ProgramData\anaconda3\envs\Coffee_cuda\Lib\site-packages\statsmodels\base\model.py:1888: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 2
#   warnings.warn('covariance of constraints does not have full '
# c:\ProgramData\anaconda3\envs\Coffee_cuda\Lib\site-packages\statsmodels\base\model.py:1888: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 2
#   warnings.warn('covariance of constraints does not have full '

def granger_feature_selector(dataframes_dict, alpha=0.05, f_threshold=4.0, min_lag=1, max_lag=7, monthly_data=False):
    """
    Perform Granger causality tests to select predictive features from time series data.
    
    This function identifies features across multiple dataframes that have statistically
    significant predictive power for a target variable (futures prices) using Granger
    causality testing. It handles annual and monthly data aggregation and employs robust
    error handling to ensure feature selection even with inconsistent data.
    
    Parameters
    ----------
    dataframes_dict : dict
        Dictionary of dataframes to analyze. Must include a 'futures' dataframe
        containing the target price variable. Other entries can be either dataframes
        or dictionaries of dataframes.
    alpha : float, optional
        Significance level threshold for p-values, by default 0.05.
        Features with p-values below this threshold are considered significant.
    f_threshold : float, optional
        Minimum F-statistic threshold, by default 4.0.
        Features must exceed this F-statistic value to be selected.
    min_lag : int, optional
        Minimum lag to consider for causality, by default 1.
        Represents the minimum time delay for potential causal effects.
    max_lag : int, optional
        Maximum lag to consider for causality, by default 7.
        Represents the maximum time delay for potential causal effects.
    monthly_data : bool, optional
        Whether the data is monthly (True) or yearly (False), by default False.
        Affects column naming and aggregation approaches.

    Returns
    -------
    list
        List of unique selected feature column names across all dataframes that
        meet the Granger causality criteria for predicting futures prices.
        
    Notes
    -----
    1. Data Preparation and Structure
    ----------------------------------
    1.1. Target Variable Handling
        - For yearly data: Uses 'Price' column as the target variable
        - For monthly data: Uses 'Price_mean' column as the target variable
        - Target is aggregated by year to enable time series causality testing
    
    1.2. Input Data Structure Requirements
        - The 'futures' dataframe must contain the target price variable
        - All dataframes should contain a 'Date' column for time alignment
        - Can accept nested dictionaries of dataframes for hierarchical data
    
    2. Processing and Analysis Flow
    -------------------------------
    2.1. Output Suppression
        - All standard output is redirected to devnull during execution
        - Prevents verbose output from statsmodels during testing
    
    2.2. Target Data Preparation
        - Futures dataframe is extracted and processed
        - Dates are converted to datetime objects
        - Data is aggregated by year for consistent time series analysis
    
    2.3. Feature Dataframe Processing
        - Each dataframe except 'futures' is processed individually
        - Handles both single dataframes and nested dictionaries
        - Maintains 'Date' column for all dataframes by default
    
    2.4. Feature Selection Procedure
        - For each numerical column in each dataframe:
            a. Data is aggregated yearly
            b. Merged with target data on year
            c. Granger causality tests performed for lags min_lag to max_lag
            d. Features selected if they meet both p-value and F-statistic criteria
        - For selected features, corresponding "_was_nan" columns are also included
    
    3. Error Handling and Robustness
    --------------------------------
    3.1. Exception Management
        - Individual dataframe processing errors are caught and handled
        - Ensures the function continues even if some dataframes cause errors
        - Defaults to including only 'Date' column if a dataframe fails processing
    
    3.2. Data Sufficiency Check
        - Skips analysis for merged datasets with insufficient time points
        - Requires at least max_lag + 2 observations for valid causality testing
    
    4. Output Generation
    --------------------
    4.1. Default Inclusions
        - For non-monthly data: 'Date' and 'Price' from futures are always included
        - For monthly data: Price statistics (mean, median, min, max) are included
    
    4.2. Feature Deduplication
        - Returns a list of unique features to avoid redundancy
        - Converts set back to list for consistent output format
    
    Examples
    --------
    >>> # Example with yearly data
    >>> dfs = {
    ...     'futures': df_futures,
    ...     'economic_indicators': df_econ,
    ...     'weather_data': df_weather
    ... }
    >>> selected_features = granger_feature_selector(
    ...     dfs, 
    ...     alpha=0.05, 
    ...     f_threshold=4.0, 
    ...     max_lag=5
    ... )
    >>> print(f"Selected features: {selected_features}")
    """
    # 1. INITIALIZATION AND SETUP
    # 1.1 Determine target column name based on data frequency
    if monthly_data:
        price_column_name = "Price_mean"
    else:
        price_column_name = "Price"
    
    # 1.2 Suppress statsmodels output during processing
    with suppress_output():
        # 1.3 Initialize storage for selected features
        selected_features = {}
        
        # 2. TARGET DATA PREPARATION
        # 2.1 Extract futures dataframe (contains target variable)
        futures = dataframes_dict['futures'].copy()
        
        # 2.2 Convert dates and extract year for aggregation
        futures['Date'] = pd.to_datetime(futures['Date'])
        futures['Year'] = futures['Date'].dt.year
        
        # 2.3 Aggregate futures data by year
        yearly_futures = futures.groupby('Year').agg({
            price_column_name: 'mean'
        }).reset_index()
        
        # 3. PROCESS EACH FEATURE DATAFRAME
        for df_name, df in dataframes_dict.items():
            # 3.1 Skip the target dataframe (futures)
            if df_name == 'futures':
                continue  # Skip futures as it's our target
                
            try:
                # 3.2 Handle both dictionary of dataframes and single dataframes
                if isinstance(df, dict):
                    dfs_to_process = df.values()
                else:
                    dfs_to_process = [df]
                
                # 3.3 Initialize feature list with Date column
                selected_features[df_name] = ['Date']  # Always include Date
                
                # 3.4 Process each dataframe in the collection
                for current_df in dfs_to_process:
                    current_df = current_df.copy()
                    
                    # 3.5 Ensure proper date handling
                    if 'Date' in current_df.columns:
                        current_df['Date'] = pd.to_datetime(current_df['Date'])
                        current_df['Year'] = current_df['Date'].dt.year
                    
                    # 3.6 Identify numerical columns for testing
                    numerical_cols = current_df.select_dtypes(include=[np.number]).columns
                    numerical_cols = [col for col in numerical_cols if col not in ['Date', 'Year']]
                    
                    # 3.7 Aggregate features by year for alignment with target
                    yearly_df = current_df.groupby('Year')[numerical_cols].mean().reset_index()
                    
                    # 3.8 Merge target and feature data on Year
                    merged_df = pd.merge(yearly_futures, yearly_df, on='Year', how='inner')
                    
                    # 3.9 Skip if insufficient data points for the specified lag
                    if len(merged_df) < max_lag + 2:
                        continue
                    
                    # 4. FEATURE TESTING AND SELECTION
                    for feature in numerical_cols:
                        # 4.1 Run Granger causality test for this feature
                        granger_results = wrapped_granger_causality_test(
                            merged_df[price_column_name],
                            merged_df[feature],
                            max_lag
                        )
                        
                        # 4.2 Check if any lag meets selection criteria
                        for result in granger_results:
                            if (result['Lag'] >= min_lag and
                                result['p-value'] < alpha and
                                result['F-Statistic'] > f_threshold):
                                # 4.3 Add the feature and its missing value indicator
                                selected_features[df_name].append(feature)
                                selected_features[df_name].append(f"{feature}_was_nan")
                                break  # Stop checking lags once feature is selected
            except:
                # 5. ERROR HANDLING
                # 5.1 Default to only including Date column if processing fails
                selected_features[df_name] = ['Date']  # At minimum, keep Date
        
        # 6. DEFAULT FEATURE INCLUSION
        # 6.1 Add standard columns based on data frequency
        if not monthly_data:
            # Always include Date and Price in futures features
            selected_features['futures'] = ['Date', 'Price']
        else:
            selected_features['futures'] = ['Price_mean', 'Price_median', 'Price_min', 'Price_max']
        
        # 7. FINAL OUTPUT PREPARATION
        # 7.1 Initialize empty set to store unique column names
        all_features = set()
        
        # 7.2 Collect unique features across all dataframes
        for features_list in selected_features.values():
            # Add each feature to the set (deduplication)
            all_features.update(features_list)

    # 8. RETURN UNIQUE FEATURES LIST
    return list(all_features)