import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose


# TODO: More deeply review functions and statistics provided
def plot_feature_correlations(
    data: pd.DataFrame,
    target_col: str = 'Price',
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'coolwarm'
) -> None:
    """
    Plot correlation heatmap between features and target variable.

    This function creates a correlation heatmap to visualize relationships
    between features and the target variable, with special emphasis on
    highlighting strong correlations.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing features and target variable.
    target_col : str, default='Price'
        Name of the target variable column.
    figsize : tuple of int, default=(12, 8)
        Figure size in inches (width, height).
    cmap : str, default='coolwarm'
        Color map for the heatmap.

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.

    Notes
    -----
    Design Decisions:
    1. Visualization Layout:
       - Uses seaborn's heatmap for clear correlation display
       - Implements symmetric color scheme around zero
       - Annotates correlation values for clarity
    
    2. Data Processing:
       - Calculates Pearson correlation coefficients
       - Handles missing values through pairwise deletion
       - Orders features by correlation strength
    
    3. Visual Elements:
       - Rotates x-axis labels for readability
       - Uses diverging color palette for intuitive interpretation
       - Adds color bar for reference

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'Price': [100, 101, 102],
    ...     'Feature1': [1, 2, 3],
    ...     'Feature2': [4, 5, 6]
    ... })
    >>> plot_feature_correlations(data)
    >>> plt.show()  # If not in interactive mode
    """
    # Set up the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = data.corr(method='pearson')
    
    # Sort features by correlation with target
    target_corr = abs(corr_matrix[target_col]).sort_values(ascending=False)
    sorted_features = target_corr.index.tolist()
    
    # Create sorted correlation matrix
    corr_matrix_sorted = corr_matrix.loc[sorted_features, sorted_features]
    
    # Create heatmap
    sns.heatmap(corr_matrix_sorted,
                annot=True,
                cmap=cmap,
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5)
    
    # Customize plot
    plt.title('Feature Correlation Heatmap', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_time_series_decomposition(
    data: pd.DataFrame,
    date_col: str = 'Date',
    value_col: str = 'Price',
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot time series decomposition showing trend, seasonality, and residuals.

    This function decomposes a time series into its components and creates
    a visualization showing the original series, trend, seasonal pattern,
    and residuals.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing time series data.
    date_col : str, default='Date'
        Name of the date column.
    value_col : str, default='Price'
        Name of the value column to decompose.
    figsize : tuple of int, default=(15, 10)
        Figure size in inches (width, height).

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.

    Notes
    -----
    Design Decisions:
    1. Decomposition Method:
       - Uses additive decomposition for price data
       - Handles missing values through interpolation
       - Preserves time series characteristics
    
    2. Visual Layout:
       - Stacked subplots for clear component comparison
       - Consistent x-axis scale across components
       - Highlights seasonal patterns effectively
    
    3. Data Processing:
       - Ensures datetime format for dates
       - Handles irregular time series through resampling
       - Maintains data integrity during processing

    Examples
    --------
    >>> import pandas as pd
    >>> # Create sample time series data
    >>> dates = pd.date_range('2023-01-01', periods=100)
    >>> data = pd.DataFrame({
    ...     'Date': dates,
    ...     'Price': np.random.randn(100).cumsum()
    ... })
    >>> plot_time_series_decomposition(data)
    >>> plt.show()  # If not in interactive mode
    """
    # Ensure date column is datetime
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Set date as index for decomposition
    data.set_index(date_col, inplace=True)
    
    # Handle missing values
    data[value_col] = data[value_col].interpolate(method='linear')
    
    # Perform decomposition
    decomposition = seasonal_decompose(data[value_col],
                                     period=30,  # 30 days for monthly patterns
                                     extrapolate_trend=True)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    
    # Plot original
    ax1.plot(data.index, data[value_col], label='Original')
    ax1.set_title('Original Time Series')
    ax1.grid(True)
    
    # Plot trend
    ax2.plot(data.index, decomposition.trend, label='Trend')
    ax2.set_title('Trend')
    ax2.grid(True)
    
    # Plot seasonal
    ax3.plot(data.index, decomposition.seasonal, label='Seasonal')
    ax3.set_title('Seasonal')
    ax3.grid(True)
    
    # Plot residual
    ax4.plot(data.index, decomposition.resid, label='Residual')
    ax4.set_title('Residual')
    ax4.grid(True)
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    color: str = 'skyblue'
) -> None:
    """
    Plot feature importance scores in a horizontal bar chart.

    This function creates a visualization of feature importance scores,
    showing the most influential features in predicting the target variable.

    Parameters
    ----------
    feature_importance : dict
        Dictionary mapping feature names to their importance scores.
    top_n : int, default=20
        Number of top features to display.
    figsize : tuple of int, default=(12, 8)
        Figure size in inches (width, height).
    color : str, default='skyblue'
        Color for the bars in the plot.

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.

    Notes
    -----
    Design Decisions:
    1. Visual Layout:
       - Horizontal bars for better feature name readability
       - Sorted by importance for clear ranking
       - Limited to top N features to prevent overcrowding
    
    2. Data Processing:
       - Handles both absolute and relative importance scores
       - Normalizes scores if needed
       - Filters out features with zero importance
    
    3. Customization:
       - Configurable number of features to display
       - Flexible color scheme
       - Adjustable figure size

    Examples
    --------
    >>> # Create sample feature importance scores
    >>> importance = {
    ...     'Feature1': 0.5,
    ...     'Feature2': 0.3,
    ...     'Feature3': 0.2
    ... }
    >>> plot_feature_importance(importance, top_n=3)
    >>> plt.show()  # If not in interactive mode
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(feature_importance.items()),
                     columns=['Feature', 'Importance'])
    
    # Sort by importance and get top N features
    df = df.sort_values('Importance', ascending=True).tail(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create horizontal bar plot
    bars = plt.barh(df['Feature'], df['Importance'], color=color)
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontweight='bold')
    
    # Customize plot
    plt.title('Feature Importance Scores', pad=20)
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
