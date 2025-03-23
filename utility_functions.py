import pycountry
import numpy as np
import pandas as pd # type: ignore

# Many identifier had no Official ISO 3 codes, so made many temp ones based off the temp ones found in many un files
iso3_mapping = {
    "Advanced Economies": "AETMP",
    "Advanced Economies excluding US": "EAEUSTMP",
    "Africa": "AFRTMP",
    "Americas": "AMETMP",
    "Asia": "ASIATMP",
    "Emerging and Developing Countries Asia excluding China": "EMDEAECHTMP",
    "Emerging and Developing Countries Europe": "EMDEETMP",
    "Emerging and Developing Economies": "EMDETMP",
    "Europe": "EURTMP",
    "Latin America and the Caribbean": "LACTMP",
    "Middle East and Central Asia": "MECATMP",
    "Oceania": "OCETMP",
    "Sub-Saharan Africa": "SSA",
    "Bolivia (Plurinational State of)":"BOL",
    "Caribbean":"CARTMP",
    "Central America":"CAMTMP",
    "China, mainland":"CHNML",
    "Other non-specified areas":"ONSATMP",
    "Democratic Republic of the Congo":"COD",
    "Eastern Africa":"EAFTMP",
    "Eastern Asia":"EASTMP",
    "Ethiopia PDR":"ETHPDR",
    "European Union (27)":"EUE",
    "Land Locked Developing Countries":"LLDCTMP",
    "Least Developed Countries":"LDCTMP",
    "Low Income Food Deficit Countries":"LIFDC",
    "Melanesia":"MLAS",
    "Middle Africa":"MIDAFR",
    "Net Food Importing Developing Countries":"NFIDCTMP",
    "Northern America":"NAMTMP",
    "Small Island Developing States" : "SIDSTMP",
    "South America":"SAMTMP",
    "South-eastern Asia":"SEATMP",
    "Southern Asia":"SATMP",
    "Southern Europe":"SETMP",
    "Venezuela (Bolivarian Republic of)":"VEN",
    "Western Africa":"WAFTMP",
    "Western Asia":"WASTMP",
    "World": "WLD"
}

# Function to convert country name to ISO 3 code
def country_name_to_iso3(country_name):
    """
    Convert a country or region name to its ISO 3166-1 alpha-3 code or custom temporary code.

    This function handles both standard country names and special regional/economic
    groupings that don't have official ISO codes. It uses a custom mapping for
    non-standard entities and falls back to fuzzy matching for standard country names.

    Parameters
    ----------
    country_name : str
        The name of the country or region to convert.

    Returns
    -------
    str or None
        The three-letter ISO code (for countries) or custom temporary code (for regions).
        Returns None if no matching code is found.

    Notes
    -----
    Design Decisions:
    1. Custom Mapping:
       - Uses temporary codes (XXXXTMP) for non-standard entities
       - Based on common UN file conventions for regional groupings
       - Preserves official ISO codes where they exist (e.g., "BOL" for Bolivia)
    
    2. Fallback Strategy:
       - Primary lookup in custom iso3_mapping dictionary
       - Secondary fuzzy search using pycountry for standard countries
       - Returns None instead of raising exceptions for unmatched names
    
    3. Code Structure:
       - Three-letter codes for consistency with ISO 3166-1
       - TMP suffix indicates non-standard/temporary codes
       - Regional codes follow logical abbreviation patterns

    Examples
    --------
    >>> # Standard country
    >>> country_name_to_iso3("France")
    'FRA'
    >>> # Regional grouping
    >>> country_name_to_iso3("Advanced Economies")
    'AETMP'
    >>> # Non-existent country
    >>> country_name_to_iso3("NonExistentLand")
    None
    """
    # Check if country name exists in ISO3 mapping
    if country_name in iso3_mapping:
        return iso3_mapping[country_name]
    else:
        # Use fuzzy search to find ISO3 code
        try:
            country = pycountry.countries.search_fuzzy(country_name)[0]
            iso3_code = country.alpha_3
            return iso3_code
        except LookupError:
            return None

def transform_futures_data(dataframes_dict):
    """
    Transform futures data to include comprehensive monthly statistics and indicators.

    This function processes raw futures data to create a rich set of monthly
    statistics and technical indicators. It handles data aggregation, volatility
    calculation, and various price-based metrics while ensuring proper date handling.

    Parameters
    ----------
    dataframes_dict : dict
        Dictionary containing dataframes, must include a 'futures' key with
        corresponding DataFrame containing columns: Date, High, Low, Price,
        Open, Vol., Change %.

    Returns
    -------
    dict
        Updated dictionary with transformed futures data, where the 'futures'
        DataFrame now contains monthly statistics including:
        - High_max, Low_min : Highest high and lowest low for the month
        - Price_mean, Price_median, Price_min, Price_max : Price statistics
        - Open_mean, Vol._mean, Change %_mean : Averaged values
        - Price_pct_change : Month-over-month price change percentage
        - Price_volatility : Standard deviation of daily returns
        - Price_range_pct : Monthly price range as percentage of mean

    Notes
    -----
    Design Decisions:
    1. Date Handling:
       - Converts all dates to datetime format for consistency
       - Sets dates to first of month for proper alignment
       - Maintains chronological order in output
    
    2. Statistical Calculations:
       - Uses multiple price points (High, Low, Open) for comprehensive analysis
       - Calculates both simple statistics and derived metrics
       - Preserves original data while adding new insights
    
    3. Performance Considerations:
       - Creates copy of input dictionary to prevent modification
       - Uses efficient pandas operations for calculations
       - Groups operations to minimize data traversal
    
    4. Data Quality:
       - Handles missing values implicitly through aggregation
       - Maintains data types throughout transformation
       - Ensures consistent column naming

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample futures data
    >>> futures_df = pd.DataFrame({
    ...     'Date': ['2023-01-01', '2023-01-02'],
    ...     'High': [105, 106],
    ...     'Low': [98, 99],
    ...     'Price': [100, 101],
    ...     'Open': [99, 100],
    ...     'Vol.': [1000, 1100],
    ...     'Change %': [1.0, 1.1]
    ... })
    >>> data_dict = {'futures': futures_df}
    >>> transformed = transform_futures_data(data_dict)
    >>> print(transformed['futures'].columns)
    """
    # Create a copy of the original dataframes dictionary
    transformed_dict = dataframes_dict.copy()
    
    # Get the futures dataframe
    futures = transformed_dict['futures'].copy()
    
    # Ensure 'Date' is in datetime format
    futures['Date'] = pd.to_datetime(futures['Date'])
    
    # Extract month and year from the 'Date' column
    futures['Month-Year'] = futures['Date'].dt.to_period('M')
    futures['Year'] = futures['Date'].dt.year
    futures['Month'] = futures['Date'].dt.month
    
    # Calculate monthly statistics
    monthly_stats = futures.groupby('Month-Year').agg({
        'High': 'max',
        'Low': 'min',
        'Price': ['mean', 'median', 'min', 'max'],
        'Open': 'mean',
        'Vol.': 'mean',
        'Change %': 'mean'
    })
    
    # Flatten the multi-level columns
    monthly_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
        for col in monthly_stats.columns
    ]
    
    # Calculate percent changes
    monthly_stats['Price_pct_change'] = monthly_stats['Price_mean'].pct_change() * 100
    
    # Convert Month-Year period to datetime for the first day of each month
    monthly_stats = monthly_stats.reset_index()
    monthly_stats['Date'] = monthly_stats['Month-Year'].apply(lambda x: x.to_timestamp(how='start'))
    
    # Drop the Month-Year column as we now have Date
    monthly_stats = monthly_stats.drop('Month-Year', axis=1)
    
    # Add monthly volatility (standard deviation of daily returns within each month)
    volatility = futures.groupby(pd.Grouper(key='Date', freq='MS'))['Price'].std()
    monthly_stats['Price_volatility'] = volatility.values
    
    # Calculate price range as percentage of mean price
    monthly_stats['Price_range_pct'] = (
        (monthly_stats['Price_max'] - monthly_stats['Price_min']) / 
        monthly_stats['Price_mean'] * 100
    )
    
    # Sort by date
    monthly_stats = monthly_stats.sort_values('Date').reset_index(drop=True)
    
    # Update the futures dataframe in the dictionary
    transformed_dict['futures'] = monthly_stats
    
    return transformed_dict

def fix_column_order(data, monthly_data=False, track_nans=True):
    """
    Standardize column order in time series DataFrames with 'Date' first and 'Price' last.

    This function ensures a consistent column order across time series DataFrames,
    placing the 'Date' column first and the price-related column last. It handles
    both daily and monthly data formats, with special handling for monthly price
    column naming and optional NaN tracking columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing time series data with 'Date' and price columns.
    monthly_data : bool, default=False
        If True, handles monthly data where 'Price_mean' might be used instead of 'Price'.
    track_nans : bool, default=True
        If True, preserves _was_nan columns next to their corresponding feature columns.
        If False, removes all _was_nan columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column order: ['Date', ..., 'Price'].
        If track_nans=True, _was_nan columns will be placed immediately after their
        corresponding feature columns.

    Notes
    -----
    Design Decisions:
    1. Column Ordering:
       - 'Date' column always comes first for temporal reference
       - Feature columns in the middle maintain their relative order
       - Price column always comes last for consistency
       - NaN indicator columns are placed immediately after their features
    
    2. Price Column Handling:
       - For daily data: Uses 'Price' column name
       - For monthly data: Converts 'Price_mean' to 'Price' if present
       - Maintains original column if no renaming is needed
    
    3. Data Preservation:
       - Creates a copy to avoid modifying the input DataFrame
       - Preserves all original data, only changes column order
       - Maintains data types of all columns
    
    4. NaN Tracking:
       - When track_nans=True, preserves _was_nan columns next to their features
       - When track_nans=False, removes all _was_nan columns
       - Maintains feature-NaN indicator pairs when tracking is enabled

    Examples
    --------
    >>> # Daily data example with NaN tracking
    >>> df_daily = pd.DataFrame({
    ...     'Feature1': [1, 2],
    ...     'Feature1_was_nan': [0, 1],
    ...     'Date': ['2023-01-01', '2023-01-02'],
    ...     'Price': [100, 101]
    ... })
    >>> result = fix_column_order(df_daily)
    >>> print(result.columns.tolist())
    ['Date', 'Feature1', 'Feature1_was_nan', 'Price']

    >>> # Monthly data example without NaN tracking
    >>> df_monthly = pd.DataFrame({
    ...     'Feature1': [1, 2],
    ...     'Feature1_was_nan': [0, 1],
    ...     'Date': ['2023-01-01', '2023-02-01'],
    ...     'Price_mean': [100, 101]
    ... })
    >>> result = fix_column_order(df_monthly, monthly_data=True, track_nans=False)
    >>> print(result.columns.tolist())
    ['Date', 'Feature1', 'Price']
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # If monthly data, rename Price_mean to Price if it exists
    if monthly_data and 'Price_mean' in df.columns:
        df = df.rename(columns={'Price_mean': 'Price'})
    
    # Get proper price column name
    price_col = 'Price'
    
    # Handle NaN tracking columns
    if not track_nans:
        # Remove all _was_nan columns
        df = df.loc[:, ~df.columns.str.endswith('_was_nan')]
    
    # Get all columns except Date and Price
    middle_columns = [col for col in df.columns if col not in ['Date', price_col]]
    
    # If tracking NaNs, reorder middle columns to keep NaN indicators next to their features
    if track_nans:
        # Group columns by their base name (without _was_nan suffix)
        column_groups = {}
        for col in middle_columns:
            base_name = col.replace('_was_nan', '')
            if base_name not in column_groups:
                column_groups[base_name] = []
            column_groups[base_name].append(col)
        
        # Reorder middle columns to keep NaN indicators next to their features
        reordered_middle = []
        for base_name, cols in column_groups.items():
            # Add the feature column first
            reordered_middle.append(base_name)
            # Add its NaN indicator if it exists
            nan_col = f"{base_name}_was_nan"
            if nan_col in cols:
                reordered_middle.append(nan_col)
        
        middle_columns = reordered_middle
    
    # Create new column order with Date first and Price last
    new_order = ['Date'] + middle_columns + [price_col]
    
    # Reorder the columns
    df = df[new_order]
    
    return df


# TODO fix BUG with monthly
def concat_all_data(data, is_monthly=False, track_nans=True):
    """
    Concatenate and align multiple time series DataFrames with consistent frequency.

    This function combines multiple time series DataFrames while handling frequency
    conversion, missing values, and temporal alignment. It supports both daily and
    monthly data frequencies and includes optional NaN indicators for tracking missing data.

    Parameters
    ----------
    data : dict
        Dictionary of DataFrames to concatenate. Must include a 'futures' key with
        its corresponding DataFrame. Each DataFrame must have a 'Date' column.
    is_monthly : bool, default=False
        If True, treats and resamples data as monthly frequency (MS = Month Start).
        If False, treats and resamples data as daily frequency.
    track_nans : bool, default=True
        If True, creates indicator columns for NaN values in the original data.
        If False, only forward fills missing values without tracking.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing all features from input DataFrames with:
        - Consistent date frequency (daily or monthly)
        - Optional NaN indicators for missing values (if track_nans=True)
        - Forward-filled missing data
        - Deduplicated column names
        - Chronologically sorted rows

    Notes
    -----
    Design Decisions:
    1. Date Handling:
       - Uses futures data date range as reference
       - Converts all dates to datetime format
       - Sets monthly dates to first of month when applicable
       - Creates consistent date index across all data
    
    2. Missing Data Strategy:
       - Optional creation of indicator columns for original NaN values
       - Forward fills missing values for continuity
       - Handles both temporal and feature-wise missing data
       - Preserves information about data availability when track_nans=True
    
    3. Data Alignment:
       - Uses futures data as anchor for date range
       - Resamples all other data to match frequency
       - Handles overlapping and non-overlapping periods
       - Maintains temporal ordering
    
    4. Column Management:
       - Adds suffixes to resolve duplicate column names
       - Preserves original column names where possible
       - Creates intuitive names for NaN indicators when track_nans=True
       - Ensures unique column identifiers

    Warnings
    --------
    - Forward filling may create artificial continuity
    - When track_nans=False, information about original missing values is lost

    See Also
    --------
    transform_futures_data : Function for preprocessing futures data
    fix_column_order : Function for standardizing column order
    """
    # First, handle futures data to establish index
    futures_df = data['futures']
    futures_df['Date'] = pd.to_datetime(futures_df['Date'])
    max_date = futures_df['Date'].max()
    
    # Set appropriate frequency based on is_monthly
    freq = 'MS' if is_monthly else 'D'  # MS = Month Start
    
    # Create index range
    main_index = pd.date_range(
        start=futures_df['Date'].min(),
        end=max_date,
        freq=freq
    )
    
    processed_dfs = {}
    for name, dafr in data.items():
        # Make a copy to avoid modifying original
        df = dafr.copy()
        
        print(f"Processing dataset: {name}")
        
        # Convert Date column to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # If monthly data, ensure all dates are set to first of month
        if is_monthly:
            df['Date'] = df['Date'].apply(lambda x: x.replace(day=1))
        
        # Handle NaN tracking based on parameter
        if track_nans:
            # Create indicator columns (1 if was NaN, 0 if wasn't)
            nan_indicators = df.isna().astype(int)
            # Rename the indicator columns to show they're NaN indicators
            nan_indicators = nan_indicators.add_suffix('_was_nan')
            # Fill the NaN values with 0 in original dataframe
            df = df.fillna(0)
            # Concatenate the original dataframe with the indicators
            df = pd.concat([df, nan_indicators], axis=1)
        else:
            # Simply fill NaN values without tracking
            df = df.fillna(0)
        
        # Remove any duplicate Date entries
        if is_monthly:
            # For monthly data, take the last value of each month
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        else:
            # For daily data, take the first occurrence
            df = df.drop_duplicates(subset=['Date'])
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # If not futures data, resample and fill
        if name != 'futures':
            # Create extended index
            extended_index = pd.date_range(
                start=df.index.min(),
                end=max_date,
                freq=freq
            )
            
            # Resample and forward fill
            if is_monthly:
                # For monthly data, resample to month start and forward fill
                df = df.reindex(extended_index).fillna(method='ffill')
            else:
                # For daily data, forward fill as before
                df = df.reindex(extended_index).fillna(method='ffill')
            
            # Reindex to match the main index
            df = df.reindex(main_index)
        
        processed_dfs[name] = df
    
    # Concatenate all dataframes horizontally
    final_df = pd.concat(processed_dfs.values(), axis=1)
    
    # Reset index to make Date a column
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Clean up column names to remove any duplicates
    duplicates = final_df.columns[final_df.columns.duplicated()]
    for col in duplicates:
        # Find all columns with this name
        dupe_cols = final_df.columns.get_indexer_for(final_df.columns[final_df.columns == col])
        # Rename them with a suffix
        for i, idx in enumerate(dupe_cols[1:], 1):
            final_df.columns.values[idx] = f"{col}_{i}"
    
    # Sort by date
    final_df = final_df.sort_values('Date').reset_index(drop=True)
    
    # For monthly data, ensure all dates are first of month
    if is_monthly:
        final_df['Date'] = final_df['Date'].apply(lambda x: x.replace(day=1))
    
    return final_df

def concat_all_data_yearly_old(data):
    """
    Concatenate multiple time series DataFrames with daily frequency (deprecated).

    .. deprecated:: 1.0.0
        This function is deprecated and will be removed in version 2.0.0.
        Use :func:`concat_all_data` with is_monthly=False instead.

    This function processes multiple time series DataFrames, converting them to
    daily frequency and concatenating them horizontally. It handles missing values,
    date alignment, and column name conflicts.

    Parameters
    ----------
    data : dict
        Dictionary of DataFrames where keys are dataset names and values are
        the corresponding DataFrames. Each DataFrame must have a 'Date' column.
        The dictionary must contain a 'futures' key with its associated DataFrame.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing all features from input DataFrames with:
        - Daily frequency
        - NaN indicators for missing values
        - Forward-filled missing data
        - Deduplicated column names
        - Chronologically sorted rows

    Notes
    -----
    Design Decisions:
    1. Date Processing:
       - Uses futures data date range as reference
       - Converts all dates to datetime format
       - Creates daily frequency index for all data
       - Maintains temporal ordering
    
    2. Missing Data Handling:
       - Creates indicator columns for original NaN values
       - Forward fills missing values for continuity
       - Preserves information about data availability
    
    3. Data Alignment:
       - Uses futures data as anchor for date range
       - Resamples all other data to daily frequency
       - Handles overlapping and non-overlapping periods
    
    4. Column Management:
       - Adds suffixes to resolve duplicate column names
       - Preserves original column names where possible
       - Creates intuitive names for NaN indicators

    Warnings
    --------
    - This function is deprecated and will be removed in version 2.0.0
    - Forward filling may create artificial continuity

    See Also
    --------
    concat_all_data : Newer function that handles both daily and monthly data

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample data
    >>> futures_df = pd.DataFrame({
    ...     'Date': ['2023-01-01', '2023-01-02'],
    ...     'Price': [100, 101]
    ... })
    >>> other_df = pd.DataFrame({
    ...     'Date': ['2023-01-01'],
    ...     'Feature': [1.0]
    ... })
    >>> data_dict = {'futures': futures_df, 'other': other_df}
    >>> result = concat_all_data_yearly_old(data_dict)
    >>> print(result.columns)
    """
    # First, handle futures data to establish daily index
    futures_df = data['futures']
    futures_df['Date'] = pd.to_datetime(futures_df['Date'])
    max_date = futures_df['Date'].max()
    daily_index = pd.date_range(
        start=futures_df['Date'].min(),
        end=max_date,
        freq='D'
    )

    processed_dfs = {}

    for name, dafr in data.items():
        # Make a copy to avoid modifying original
        df = dafr.copy()
        
        print(f"Processing dataset: {name}")
        
        # Convert Date column to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create indicator columns (1 if was NaN, 0 if wasn't)
        nan_indicators = df.isna().astype(int)
        # Rename the indicator columns to show they're NaN indicators
        nan_indicators = nan_indicators.add_suffix('_was_nan')
        # Fill the NaN values with 0 in original dataframe
        df = df.fillna(0)
        # Concatenate the original dataframe with the indicators
        df = pd.concat([df, nan_indicators], axis=1)

        # Remove any duplicate Date entries by taking the first occurrence
        df = df.drop_duplicates(subset=['Date'])

        # Set Date as index
        df.set_index('Date', inplace=True)

        # If not futures data, resample to daily frequency and forward fill to max_date
        if name != 'futures':
            # Create a new index that extends to max_date
            extended_index = pd.date_range(
                start=df.index.min(),
                end=max_date,
                freq='D'
            )
            # Reindex and forward fill
            df = df.reindex(extended_index).fillna(method='ffill')
            # Then reindex to match the futures daily index (in case the data starts later)
            df = df.reindex(daily_index)

        processed_dfs[name] = df

    # Concatenate all dataframes horizontally
    final_df = pd.concat(processed_dfs.values(), axis=1)

    # Reset index to make Date a column and ensure no duplicates
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Date'}, inplace=True)

    # Clean up column names to remove any duplicates
    # Get a list of duplicate columns
    duplicates = final_df.columns[final_df.columns.duplicated()]
    for col in duplicates:
        # Find all columns with this name
        dupe_cols = final_df.columns.get_indexer_for(final_df.columns[final_df.columns == col])
        # Rename them with a suffix
        for i, idx in enumerate(dupe_cols[1:], 1):
            final_df.columns.values[idx] = f"{col}_{i}"

    # Sort by date
    final_df = final_df.sort_values('Date').reset_index(drop=True)

    return final_df

def convert_to_serializable(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    
    This function recursively converts NumPy data types to their Python native 
    equivalents to make them JSON serializable. It handles integers, floats, 
    arrays, and nested structures like dictionaries and lists.
    
    Parameters
    ----------
    obj : any
        The object to convert, which may contain NumPy data types.
    
    Returns
    -------
    any
        The converted object with all NumPy types replaced with Python native types.
    
    Notes
    -----
    Handles the following NumPy types:
    - Integer types (int8, int16, int32, int64, etc.) → Python int
    - Float types (float16, float32, float64) → Python float
    - NumPy arrays → Python lists
    - Nested dictionaries and lists are processed recursively
    
    Examples
    --------
    >>> import numpy as np
    >>> data = {'value': np.int64(42), 'array': np.array([1, 2, 3])}
    >>> serializable_data = convert_to_serializable(data)
    >>> import json
    >>> json.dumps(serializable_data)  # This will succeed
    '{"value": 42, "array": [1, 2, 3]}'
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
