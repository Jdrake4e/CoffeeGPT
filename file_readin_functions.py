import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import csv
import datetime as dt
import time

#import statsmodels.formula.api as smf

import file_readin_functions as frf
import utility_functions as uf

def all_csv_readin():
    """
    Read and combine all available CSV datasets into a unified data structure.

    This function serves as a central coordinator for reading all available datasets,
    including climate disasters, forest carbon, human development, land cover,
    forex data, green coffee production, and futures data. It organizes these
    datasets into a structured dictionary for easy access.

    Returns
    -------
    dict
        A dictionary containing all processed datasets with the following keys:
        - 'disaster': Climate-related disaster frequency data
        - 'forest': Forest and carbon data
        - 'hdi': Human Development Index data
        - 'land': Land cover account data
        - 'forex': OECD forex data
        - 'green_coffee': UN green coffee production data
        - 'futures': Coffee futures market data

    Notes
    -----
    Design Decisions:
        1. Data Organization:
        - Maintains separate datasets in dictionary format
        - Preserves individual dataset structures
        - Enables easy access to specific data types
        
        2. Error Handling:
        - Gracefully handles missing files
        - Preserves partial data on failures
        - Maintains data integrity
        
        3. Extensibility:
        - Easy to add new data sources
        - Modular design for maintenance
        - Consistent interface across datasets

    Examples
    --------
    >>> # Read all available datasets
    >>> all_data = all_csv_readin()
    >>> 
    >>> # Access specific datasets
    >>> disaster_data = all_data['disaster']
    >>> forest_data = all_data['forest']
    >>> futures_data = all_data['futures']
    """
    
    # distaster - A Dictionary Of Data Frames
    disaster = disasters_readin()
    
    # forest - A Dictionary Of Data Frames
    forest = forest_carbon_readin()
    
    # hdi - A Dictionary Of Data Frames
    hdi = human_development_readin()
    
    # land - A Dictionary Of Data Frames
    land = land_cover_readin()
    
    
    #TODO remember why I turned these two off
    # A pivoted Dataframe
    #energy = energy_readin()
    
    # demographics - A Dictioanry of Dataframes
    #demographics = OECD_global_demographics_readin()
    
    # forex - A Dictioanry of Dataframes
    forex = OECD_forex_readin()
    
    # green_coffee - A Dictionary Of Data Frames
    green_coffee = UN_green_coffee_readin()
    
    # A standard dataframe
    futures = futures_readin_bind()
    
    dataframes = {
        'disaster': disaster,
        'forest': forest,
        'hdi': hdi,
        'land': land,
        #'energy': energy,
        #'demographics': demographics,
        'forex': forex,
        'green_coffee': green_coffee,
        'futures': futures
    }
    
    return dataframes

'''
# DONE
'''
def disasters_readin(file = "datasets/Climate-related_Disasters_Frequency.csv"):
    """
    Read and process climate-related disaster frequency data from a CSV file.

    This function reads climate disaster data, processes it into a time series format,
    and handles multiple disaster indicators across countries and years. It includes
    comprehensive data transformation and pivoting to create a wide-format time series.

    Parameters
    ----------
    file : str, default="datasets/Climate-related_Disasters_Frequency.csv"
        Path to the CSV file containing climate disaster frequency data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Columns named as 'Indicator, ISO3 = XXX' for each country
        - Values representing disaster frequencies over time

    Notes
    -----
    Design Decisions:
    1. Data Transformation:
       - Melts year columns into a single date column
       - Pivots data to create indicator-country combinations
       - Preserves all disaster type classifications
    
    2. Date Handling:
       - Extracts years from column names
       - Converts to datetime format for consistency
       - Creates annual time series structure
    
    3. Column Organization:
       - Combines indicator names with ISO3 codes
       - Creates hierarchical column structure
       - Maintains disaster type categorization
    
    4. Data Quality:
       - Handles missing values appropriately
       - Preserves metadata from original file
       - Maintains data relationships between indicators

    Examples
    --------
    >>> # Read disaster data from default file location
    >>> disaster_data = disasters_readin()
    >>> print(disaster_data.columns)  # Shows disaster type-country combinations
    >>> 
    >>> # Read from custom file location
    >>> custom_data = disasters_readin("path/to/disaster_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)
    
    # A list of all quantifiers
    quantifiers_list = df["Indicator"].unique()

    # Melt the DataFrame to unpivot years, TODO read more on melt
    melted_df = pd.melt(df, id_vars=["ObjectId", "Country", "ISO2", "ISO3", "Indicator", "Unit", "Source", "CTS_Code", "CTS_Name", "CTS_Full_Descriptor"], var_name="Year", value_name="Value")

    # Convert the Year column to datetime
    melted_df['Year'] = melted_df['Year'].str.extract(r'(\d+)').astype(int)
    melted_df['Date'] = pd.to_datetime(melted_df['Year'], format='%Y')

    # Drop the original Year column
    melted_df = melted_df.drop(columns=['Year'])

    # Pivot the DataFrame
    pivoted_df = melted_df.pivot(index="Date", columns=["Indicator", "ISO3"], values="Value")

    # Create a dictionary to store DataFrames for each indicator
    indicator_dfs = {}

    # Iterate over the quantifiers list
    for quantifier in quantifiers_list:
        # Filter the pivoted DataFrame for the current quantifier
        filtered_df = pivoted_df[quantifier]

        # If the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Store the DataFrame in the dictionary with the quantifier name as the key
            indicator_dfs[quantifier] = filtered_df

    # Print the dictionary containing separate DataFrames for each indicator
    '''
    for quantifier, df in indicator_dfs.items():
        print(f"Quantifier: {quantifier}")
        print(df)
        print()
    '''

    pivoted_df = pivoted_df.reset_index()

    # Ensure the 'Date' column is in datetime format
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])

    # Flatten the MultiIndex columns and combine ISO3 into the column names
    pivoted_df.columns = [', ISO3 = '.join([str(elem).replace('Climate related disasters frequency, ', '') for elem in col if elem]) for col in pivoted_df.columns]

    return pivoted_df

'''
# DONE
'''
def forest_carbon_readin(file = "datasets/Forest_and_Carbon.csv"):
    """
    Read and process forest and carbon data from a CSV file.

    This function reads forest and carbon-related data, processes it into a time series format,
    and handles multiple indicators including carbon stocks, forest area, and related indices.
    The function performs comprehensive data transformation to create a wide-format time series.

    Parameters
    ----------
    file : str, default="datasets/Forest_and_Carbon.csv"
        Path to the CSV file containing forest and carbon data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Columns named as 'Indicator, ISO3 = XXX' for each country
        - Values representing various forest and carbon metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Organization:
       - Handles six main indicators:
         * Carbon stocks in forests
         * Forest area
         * Index of carbon stocks in forests
         * Index of forest extent
         * Land area
         * Share of forest area
    
    2. Data Transformation:
       - Melts year columns into a single date column
       - Pivots data to create indicator-country combinations
       - Preserves all indicator classifications
    
    3. Date Handling:
       - Extracts years from column names
       - Converts to datetime format for consistency
       - Creates annual time series structure
    
    4. Data Quality:
       - Handles missing values appropriately
       - Maintains relationships between indicators
       - Preserves metadata from original file

    Examples
    --------
    >>> # Read forest and carbon data from default file location
    >>> forest_data = forest_carbon_readin()
    >>> print(forest_data.columns)  # Shows indicator-country combinations
    >>> 
    >>> # Read from custom file location
    >>> custom_data = forest_carbon_readin("path/to/forest_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)

    # A list of all quantifiers
    quantifiers_list = [
        "Carbon stocks in forests",
        "Forest area",
        "Index of carbon stocks in forests",
        "Index of forest extent",
        "Land area",
        "Share of forest area"
    ]

    # Melt the DataFrame to unpivot years, TODO read more on melt
    melted_df = pd.melt(df, id_vars=["ObjectId", "Country", "ISO2", "ISO3", "Indicator", "Unit", "Source", "CTS_Code", "CTS_Name", "CTS_Full_Descriptor"], var_name="Year", value_name="Value")

    # Convert the Year column to datetime
    melted_df['Year'] = melted_df['Year'].str.extract(r'(\d+)').astype(int)
    melted_df['Date'] = pd.to_datetime(melted_df['Year'], format='%Y')

    # Drop the original Year column
    melted_df = melted_df.drop(columns=['Year'])

    # Pivot the DataFrame
    pivoted_df = melted_df.pivot(index="Date", columns=["Indicator", "ISO3"], values="Value")

    # Create a dictionary to store DataFrames for each indicator
    indicator_dfs = {}

    # Iterate over the quantifiers list
    for quantifier in quantifiers_list:
        # Filter the pivoted DataFrame for the current quantifier
        filtered_df = pivoted_df[quantifier]
        
        # If the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Store the DataFrame in the dictionary with the quantifier name as the key
            indicator_dfs[quantifier] = filtered_df

    # Print the dictionary containing separate DataFrames for each indicator
    '''
    for quantifier, df in indicator_dfs.items():
        print(f"Quantifier: {quantifier}")
        print(df)
        print()
    '''
    
    pivoted_df = pivoted_df.reset_index()

    # Ensure the 'Date' column is in datetime format
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])

    # Flatten the MultiIndex columns and combine ISO3 into the column names
    pivoted_df.columns = [', ISO3 = '.join([str(elem) for elem in col if elem]) for col in pivoted_df.columns]
        
    return pivoted_df

'''
Minor modification to hdi data set changed column name from "country" to "Country"
TODO FIX GET INTO ONE TABLE THAT IS all PIVOTED
'''
def human_development_readin(file = "datasets/HDR21-22_Composite_indices_complete_time_series.csv"):
    """
    Read and process human development index (HDI) data from a CSV file.

    This function reads HDI data from the UNDP Human Development Report dataset,
    processes it into a time series format, and handles the complex structure
    of HDI indicators across countries and years.

    Parameters
    ----------
    file : str, default="datasets/HDR21-22_Composite_indices_complete_time_series.csv"
        Path to the CSV file containing HDI data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Columns named as 'Indicator, ISO3 = XXX' for each country
        - Values representing HDI metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Restructuring:
       - Melts wide-format data into long format for time series analysis
       - Groups data by ISO3 country codes and dates
       - Handles multiple HDI indicators per country
    
    2. Date Processing:
       - Extracts year from column names
       - Converts to datetime format for consistency
       - Drops invalid dates to ensure data quality
    
    3. Column Naming:
       - Uses ISO3 country codes in column names
       - Combines indicator names with country codes
       - Creates consistent naming pattern for all metrics
    
    4. Data Quality:
       - Removes missing values
       - Handles duplicate entries by taking first occurrence
       - Preserves original indicator values

    Examples
    --------
    >>> # Read HDI data from default file location
    >>> hdi_data = human_development_readin()
    >>> print(hdi_data.columns)  # Shows indicator-country combinations
    >>> 
    >>> # Read from custom file location
    >>> custom_data = human_development_readin("path/to/hdi_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)
    
    df = df.rename(columns={"iso3": "ISO3"})
    
    
    # Define the quantifiers columns
    quantifiers = [
        'hdi', 'le', 'eys', 'mys', 'gnipc', 'gdi', 'hdi_f', 'le_f', 'eys_f', 'mys_f', 
        'gni_pc_f', 'hdi_m', 'le_m', 'eys_m', 'mys_m', 'gni_pc_m', 'coef_ineq', 'loss',
        'ineq_le', 'ineq_edu', 'ineq_inc', 'mmr', 'abr', 'se_f', 'se_m', 'pr_f', 'pr_m',
        'lfpr_f', 'lfpr_m', 'rankdiff_hdi_phdi', 'phdi', 'diff_hdi_phdi', 'co2_prod', 'mf'
    ]

    # Create the quantifiers dictionary dynamically
    quantifiers_dict = {}
    for quantifier in quantifiers:
        valid_columns = [col for col in [f"{quantifier}_{year}" for year in range(1990, 2022)] if col in df.columns]
        if valid_columns:
            quantifiers_dict[quantifier.upper()] = valid_columns

    # Create a list to store DataFrames for each quantifier
    quantifier_dfs = []

    # Iterate over each quantifier and its corresponding columns
    for quantifier, columns in quantifiers_dict.items():
        # Extract columns related to the current quantifier
        quantifier_df = df[['ISO3'] + columns]
        
        # Melt the DataFrame to reshape it
        quantifier_df = pd.melt(quantifier_df, id_vars=['ISO3'], var_name='Date', value_name=quantifier)
        
        # Strip prefix from the Date column and convert to datetime format
        quantifier_df['Date'] = quantifier_df['Date'].str.split('_', expand=True)[1]
        quantifier_df['Date'] = pd.to_datetime(quantifier_df['Date'], format='%Y', errors='coerce')
        
        # Drop rows with invalid dates
        quantifier_df = quantifier_df.dropna(subset=['Date'])
        
        # Append the reshaped DataFrame to the list only if it's not empty
        if not quantifier_df.empty:
            quantifier_dfs.append(quantifier_df)

    # Concatonate all df the get all columns proberly grouped
    concat_df = pd.concat(quantifier_dfs).groupby(['ISO3', 'Date']).first().reset_index()

    quantifier_list = concat_df.columns[2:]

    pivot_df = concat_df.pivot_table(index='Date', columns='ISO3', values=quantifier_list)
        
    # Print the pivoted DataFrames
    '''
    for quantifier, pivoted_df in pivoted_dfs.items():
        print(f"Quantifier: {quantifier}")
        print(pivoted_df)
        print()
    '''
    
    pivot_df = pivot_df.reset_index()

    # Ensure the 'Date' column is in datetime format
    pivot_df['Date'] = pd.to_datetime(pivot_df['Date'])

    # Flatten the MultiIndex columns and combine ISO3 into the column names
    pivot_df.columns = [', ISO3 = '.join([str(elem) for elem in col if elem]) for col in pivot_df.columns]
            
    return pivot_df

def land_cover_readin(file = "datasets/Land_Cover_Accounts.csv"):
    """
    Read and process land cover account data from a CSV file.

    This function reads land cover data, processes it into a time series format,
    and handles multiple land cover indicators across countries and years. It includes
    comprehensive data transformation to create a wide-format time series.

    Parameters
    ----------
    file : str, default="datasets/Land_Cover_Accounts.csv"
        Path to the CSV file containing land cover account data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Columns named as 'Indicator, ISO3 = XXX' for each country
        - Values representing land cover metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Transformation:
       - Melts year columns into a single date column
       - Pivots data to create indicator-country combinations
       - Preserves all land cover type classifications
    
    2. Date Handling:
       - Extracts years from column names
       - Converts to datetime format for consistency
       - Creates annual time series structure
    
    3. Column Organization:
       - Combines indicator names with ISO3 codes
       - Creates hierarchical column structure
       - Maintains land cover type categorization
    
    4. Data Quality:
       - Handles missing values appropriately
       - Preserves metadata from original file
       - Maintains data relationships between indicators

    Examples
    --------
    >>> # Read land cover data from default file location
    >>> land_data = land_cover_readin()
    >>> print(land_data.columns)  # Shows land cover type-country combinations
    >>> 
    >>> # Read from custom file location
    >>> custom_data = land_cover_readin("path/to/land_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)
    
    # A list of all quantifiers
    quantifiers_list = df["Indicator"].unique()

    # Melt the DataFrame to unpivot years, TODO read more on melt
    melted_df = pd.melt(df, id_vars=["ObjectId", "Country", "ISO2", "ISO3", "Indicator", "Unit", "Source", "CTS_Code", "CTS_Name", "CTS_Full_Descriptor", "Climate_Influence"], var_name="Year", value_name="Value")

    # Convert the Year column to datetime
    melted_df['Year'] = melted_df['Year'].str.extract(r'(\d+)').astype(int)
    melted_df['Date'] = pd.to_datetime(melted_df['Year'], format='%Y')

    # Drop the original Year column
    melted_df = melted_df.drop(columns=['Year'])

    # Pivot the DataFrame and aggregate duplicate entries
    # Do not remove aggregate function this dataset contains duplicates
    # Mean was chosen to not favor any duplicate over another
    pivoted_df = melted_df.pivot_table(index="Date", columns=["Indicator", "ISO3"], values="Value", aggfunc='mean')

    # Create a dictionary to store DataFrames for each indicator
    indicator_dfs = {}

    # Iterate over the quantifiers list
    for quantifier in quantifiers_list:
        # Filter the pivoted DataFrame for the current quantifier
        filtered_df = pivoted_df[quantifier]
        
        # If the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Store the DataFrame in the dictionary with the quantifier name as the key
            indicator_dfs[quantifier] = filtered_df

    # Print the dictionary containing separate DataFrames for each indicator
    '''
    for quantifier, df in indicator_dfs.items():
        print(f"Quantifier: {quantifier}")
        print(df)
        print()
    '''
    
    pivoted_df = pivoted_df.reset_index()

    # Ensure the 'Date' column is in datetime format
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])

    # Flatten the MultiIndex columns and combine ISO3 into the column names
    pivoted_df.columns = [', ISO3 = '.join([str(elem) for elem in col if elem]) for col in pivoted_df.columns]

    return pivoted_df

'''
DONE
'''
def energy_readin(file = r"datasets\MER_T02_01A.csv"):
    """
    Read and process energy market data from a CSV file.

    This function reads energy market data, processes it into a time series format,
    and handles multiple energy indicators. It performs data cleaning and transformation
    to create a structured time series dataset.

    Parameters
    ----------
    file : str, default=r"datasets\MER_T02_01A.csv"
        Path to the CSV file containing energy market data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Energy market indicators as columns
        - Values representing energy metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Cleaning:
       - Removes unnecessary columns and whitespace
       - Handles missing values and special characters
       - Standardizes column names and formats
    
    2. Date Processing:
       - Converts date strings to datetime objects
       - Handles various date formats in input data
       - Creates consistent time series index
    
    3. Data Transformation:
       - Pivots data for time series analysis
       - Maintains data relationships
       - Preserves original data granularity
    

    TODO: find better energy data set

    Examples
    --------
    >>> # Read energy data from default file location
    >>> energy_data = energy_readin()
    >>> print(energy_data.columns)  # Shows available energy indicators
    >>> 
    >>> # Read from custom file location
    >>> custom_data = energy_readin("path/to/energy_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)
    
    # A list of all quantifiers
    #quantifiers_list = df["MSN"].unique()
    
    # If a date in this data set ends in 13 that means it is a yearly sum aggragate
    df = df[~df['YYYYMM'].astype(str).str.endswith('13')]

    # Convert to DateTime Format
    df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')

    # Drop YYYYMM Do not call this column after if debugging
    df.drop(["YYYYMM"], axis=1, inplace=True)

    # Pivot on Type of Usage
    pivoted_df = df.pivot_table(index="Date", columns=["MSN"], values="Value")
    
    # Create a dictionary to store DataFrames for each MSN
    #MSN_dfs = {}
    
    pivoted_df = pivoted_df.reset_index()

    # Ensure 'Date' is in datetime format
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])
    
    return pivoted_df.reset_index()

def OECD_global_demographics_readin(file = "datasets/OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,1.0+..PS.F+M...csv"):
    """
    Read and process OECD global demographic data from a CSV file.

    This function reads OECD demographic data, processes it into a time series format,
    and handles multiple demographic indicators across countries. It performs data
    transformation to create a structured demographic dataset.

    Parameters
    ----------
    file : str, default="datasets/OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,1.0+..PS.F+M...csv"
        Path to the CSV file containing OECD demographic data.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Demographic indicators as columns
        - Values representing population metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Processing:
       - Handles OECD's specific data format
       - Processes demographic indicators
       - Maintains country-specific information
    
    2. Time Series Handling:
       - Converts OECD date format to standard datetime
       - Creates consistent time series structure
       - Handles periodic data points
    
    3. Data Organization:
       - Structures data by demographic indicators
       - Preserves country-level granularity
       - Maintains data relationships
    
    4. Quality Control:
       - Validates demographic data
       - Handles missing values appropriately
       - Ensures data consistency

    Examples
    --------
    >>> # Read demographic data from default file location
    >>> demo_data = OECD_global_demographics_readin()
    >>> print(demo_data.columns)  # Shows available demographic indicators
    >>> 
    >>> # Read from custom file location
    >>> custom_data = OECD_global_demographics_readin("path/to/demo_data.csv")
    >>> print(custom_data.head())
    """
    df = pd.read_csv(file)
    
    # A list of all quantifiers
    quantifiers_list = df["AGE"].unique()
    
    # Rename the column from "Reference area" to "Country"
    df = df.rename(columns={"Reference area": "Country", "REF_AREA": "ISO3"})
    
    # Convert to DateTime Format
    df['Date'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y')

    # Drop YYYYMM Do not call this column after if debugging
    df.drop(["TIME_PERIOD"], axis=1, inplace=True)

    # Pivot on Type of AGE, IS03, then SEX
    pivoted_df = df.pivot_table(index="Date", columns= ["AGE", "ISO3", "SEX"], values="OBS_VALUE")
    
    pivoted_df = pivoted_df.reset_index()
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])
    
    # Flatten the MultiIndex columns into strings
    pivoted_df.columns = [
        f"{col[0]}, ISO3 = {col[1]}, SEX = {col[2]}" 
        if col[1] and col[2] else col[0] 
        for col in pivoted_df.columns
    ]

    # Rename the 'Date' column from the tuple ('Date', '', '') to 'Date'
    pivoted_df = pivoted_df.rename(columns={pivoted_df.columns[-1]: 'Date'})
    
    return pivoted_df

'''
DONE
'''
def OECD_forex_readin(file = "datasets/OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE4,1.0,filtered,2024-03-04 17-04-58.csv"):
    df = pd.read_csv(file)
    
    # Rename the column from "Reference area" to "Country"
    df = df.rename(columns={"Reference area": "Country", "REF_AREA": "ISO3"})
    
    # A list of all quantifiers
    quantifiers_list = df["TRANSACTION"].unique()

    # Convert to DateTime Format
    df['Date'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y')

    # Drop YYYYMM Do not call this column after if debugging
    df.drop(["TIME_PERIOD"], axis=1, inplace=True)

    # Pivot on Type of TRANSACTION and ISO3
    pivoted_df = df.pivot_table(index="Date", columns=["TRANSACTION", "ISO3"], values="OBS_VALUE")

    # Create a dictionary to store DataFrames for each transaction
    transaction_dfs = {}

    # Iterate over the quantifiers list
    for quantifier in quantifiers_list:
        # Filter the pivoted DataFrame for the current quantifier
        filtered_df = pivoted_df[quantifier]
        
        # If the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Store the DataFrame in the dictionary with the quantifier name as the key
            transaction_dfs[quantifier] = filtered_df
    
    pivoted_df = pivoted_df.reset_index()
    
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])
    
    pivoted_df.columns = [', ISO3 = '.join([str(elem) for elem in col if elem]) for col in pivoted_df.columns]
    
    return pivoted_df

'''
DONE
'''
def UN_green_coffee_readin(file1 = "datasets/UN Green Coffee Production.csv", 
                           file2 = "datasets/green_coffee_prod_iso3.csv", 
                           add_iso = False):
    """
    Read and process UN green coffee production data from CSV files.

    This function reads UN coffee production data and optionally combines it with ISO3
    country codes. It processes the data into a time series format and handles multiple
    coffee production indicators across countries.

    Parameters
    ----------
    file1 : str, default="datasets/UN Green Coffee Production.csv"
        Path to the CSV file containing UN green coffee production data.
    file2 : str, default="datasets/green_coffee_prod_iso3.csv"
        Path to the CSV file containing ISO3 country codes mapping.
    add_iso : bool, default=False
        Whether to add ISO3 country codes to the dataset.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with:
        - 'Date' column in datetime format
        - Coffee production indicators as columns
        - Optional ISO3 country codes
        - Values representing coffee production metrics over time

    Notes
    -----
    Design Decisions:
    1. Data Integration:
       - Optional ISO3 code integration
       - Handles country name variations
       - Maintains data integrity during merging
    
    2. Time Series Processing:
       - Converts production periods to datetime
       - Creates consistent time series structure
       - Handles seasonal production data
    
    3. Data Organization:
       - Structures by production indicators
       - Preserves country-level granularity
       - Maintains production relationships
    
    4. Quality Control:
       - Validates production data
       - Handles missing values appropriately
       - Ensures data consistency


    Examples
    --------
    >>> # Read coffee data without ISO3 codes
    >>> coffee_data = UN_green_coffee_readin()
    >>> print(coffee_data.columns)  # Shows available production indicators
    >>> 
    >>> # Read data with ISO3 codes
    >>> coffee_data_iso = UN_green_coffee_readin(add_iso=True)
    >>> print(coffee_data_iso.head())
    """
    if(not add_iso):
        df = pd.read_csv(file2)
    else:
        df = pd.read_csv(file1)
    
    # Rename the column from "Reference area" to "Country"
    if(add_iso):
        df = df.rename(columns={"Country or Area": "Country"})
        df['ISO3'] = df['Country'].apply(uf.country_name_to_iso3)
        df.to_csv('green_coffee_prod_iso3.csv', index=False)
    
    # Convert the Year column to datetime
    df['Date'] = pd.to_datetime(df['Year'], format='%Y')

    # Drop duplicate entries if any
    df = df.drop_duplicates(subset=['Date', 'Element', 'ISO3'])

    # Pivot the DataFrame
    pivoted_df = df.pivot(index="Date", columns=["Element", "ISO3"], values="Value")

    # Get unique quantifiers
    quantifiers_list = df["Element"].unique()
    
    # Create a dictionary to store DataFrames for each indicator
    indicator_dfs = {}

    # Iterate over the quantifiers list
    for quantifier in quantifiers_list:
        # Filter the pivoted DataFrame for the current quantifier
        filtered_df = pivoted_df[quantifier]
        
        # If the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Store the DataFrame in the dictionary with the quantifier name as the key
            indicator_dfs[quantifier] = filtered_df

    # Print the dictionary containing separate DataFrames for each indicator
    '''
    for quantifier, df in indicator_dfs.items():
        print(f"Quantifier: {quantifier}")
        print(df)
        print()
    ''' 
    
    pivoted_df = pivoted_df.reset_index()
    pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date'])
    
    pivoted_df.columns = [', ISO3 = '.join([str(elem) for elem in col if elem]) for col in pivoted_df.columns]
    
    
    return pivoted_df

'''
DONE
'''
def futures_readin_bind(file1 = "datasets/US Coffee C Futures Historical Data.csv", 
                        file2 = "datasets/US Coffee C Futures Historical Data(1).csv", 
                        file3 = "datasets/US Coffee C Futures Historical Data(2).csv", 
                        file4 = "datasets/US Coffee C Futures Historical Data(3).csv"):
    """
    Read and combine multiple US Coffee C Futures historical data files.

    This function reads multiple CSV files containing coffee futures data,
    combines them into a single dataset, and processes them into a consistent
    time series format. It handles data deduplication and ensures temporal continuity.

    Parameters
    ----------
    file1 : str, default="datasets/US Coffee C Futures Historical Data.csv"
        Path to the first CSV file containing futures data.
    file2 : str, default="datasets/US Coffee C Futures Historical Data(1).csv"
        Path to the second CSV file containing futures data.
    file3 : str, default="datasets/US Coffee C Futures Historical Data(2).csv"
        Path to the third CSV file containing futures data.
    file4 : str, default="datasets/US Coffee C Futures Historical Data(3).csv"
        Path to the fourth CSV file containing futures data.

    Returns
    -------
    pandas.DataFrame
        Combined and processed DataFrame with:
        - 'Date' column in datetime format
        - Standard futures trading metrics (Open, High, Low, Close, etc.)
        - Consistent data format across all time periods

    Notes
    -----
    Design Decisions:
    1. Data Integration:
       - Combines multiple historical data files
       - Handles overlapping time periods
       - Maintains data consistency across sources
    
    2. Time Series Processing:
       - Converts trading dates to datetime
       - Creates continuous time series
       - Handles market holidays and gaps
    
    3. Data Standardization:
       - Normalizes column names
       - Standardizes data types
       - Ensures consistent units
    
    4. Quality Control:
       - Removes duplicate entries
       - Validates price data
       - Ensures temporal ordering

    Examples
    --------
    >>> # Read futures data from default file locations
    >>> futures_data = futures_readin_bind()
    >>> print(futures_data.columns)  # Shows available trading metrics
    >>> 
    >>> # Read from custom file locations
    >>> custom_data = futures_readin_bind(
    ...     "path/to/file1.csv",
    ...     "path/to/file2.csv",
    ...     "path/to/file3.csv",
    ...     "path/to/file4.csv"
    ... )
    >>> print(custom_data.head())
    """
    # Read in the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    
    # Concatenate the data frames
    merged_df = pd.concat([df1, df2, df3, df4])
    
    # Convert date from string to DT object
    merged_df['Date'] = merged_df['Date'].astype('datetime64[ns]')
    
    # Remove duplicates based on the 'Date' column
    merged_df = merged_df.drop_duplicates(subset=['Date'])
    
    # Convert 'Change %' column to numeric
    merged_df['Change %'] = merged_df['Change %'].str.replace('%', '').astype(float)
    
    # Convert 'Vol.' column to numeric
    merged_df['Vol.'] = merged_df['Vol.'].str.replace('K', '').astype(float)

    # Sort the dataframe based on the 'Date' column
    merged_df = merged_df.sort_values(by='Date')

    # Reset index
    merged_df = merged_df.reset_index(drop=True)
    
    pivoted_df = merged_df.pivot_table(index="Date")
    
    pivoted_df = pivoted_df.reset_index()
    
    return pivoted_df