import torch.nn
import numpy as np # type: ignore
import pandas as pd # type: ignore
import sklearn # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
import matplotlib.cm as cm # type: ignore
import datetime as dt
from typing import Any

def prediction_variance(predictions, window=5, base_temp=0.1):
    """
    Calculate rolling variance for recent predictions with safeguards for small windows.
    
    Parameters
    ----------
    predictions : array-like
        Array of recent price predictions.
    window : int, default=5
        Size of the rolling window for variance calculation.
    base_temp : float, default=0.1
        Base temperature/variance to use when insufficient data is available.
        
    Returns
    -------
    float
        Calculated variance scaled by base_temp, or default value if
        insufficient data is available.
        
    Notes
    -----
    This function is designed to provide adaptive variance estimation for
    Monte Carlo simulations. It handles edge cases like empty arrays or
    arrays smaller than the specified window by returning sensible defaults.
    
    The variance is scaled by base_temp to maintain appropriate noise levels
    in the simulation.
    """
    if len(predictions) == 0:
        return base_temp
    elif len(predictions) < window:
        return base_temp * np.var(predictions) if len(predictions) > 1 else base_temp
    return base_temp * np.var(predictions[-window:])

def calculate_dynamic_temperature(recent_predictions, day, base_temp, decay_factor, min_temp, 
                                market_volatility_window=5, volatility_impact=0.2):
    """
    Calculate dynamic temperature for Monte Carlo simulation based on recent market volatility.
    
    Parameters
    ----------
    recent_predictions : list
        List of recent price predictions.
    day : int
        Current forecast day.
    base_temp : float
        Base temperature value.
    decay_factor : float
        Rate at which temperature decays over time (0-1).
    min_temp : float
        Minimum allowed temperature.
    market_volatility_window : int, default=5
        Number of recent predictions to consider for volatility.
    volatility_impact : float, default=0.2
        Scaling factor for how much volatility affects temperature.
    
    Returns
    -------
    float
        Dynamic temperature value used for noise generation in simulations.
        
    Notes
    -----
    This function implements an adaptive approach to noise generation in Monte Carlo
    simulations by considering both time decay and market volatility. The temperature:
    
    1. Decays exponentially over time (forecast horizon)
    2. Increases with market volatility
    3. Has a guaranteed minimum value
    
    This approach helps maintain prediction diversity while adapting to changing
    market conditions and reducing noise as the forecast extends further into the future.
    """
    # Calculate base decayed temperature
    decayed_temp = base_temp * (decay_factor ** day)
    
    # Calculate volatility adjustment if enough history exists
    if len(recent_predictions) >= market_volatility_window:
        # Calculate percentage changes
        pct_changes =   np.diff(recent_predictions[-market_volatility_window:]) / \
                        np.abs(recent_predictions[-market_volatility_window:-1])
        
        # Use rolling volatility as a scaling factor
        volatility = np.std(pct_changes) if len(pct_changes) > 0 else 0
        volatility_adjustment = volatility * volatility_impact
        
        # Combine base temperature with volatility adjustment
        dynamic_temp = decayed_temp * (1 + volatility_adjustment)
    else:
        dynamic_temp = decayed_temp
    
    # Ensure temperature doesn't go below minimum
    return max(min_temp, dynamic_temp)

def create_day_out_dict(accuracy_plot_df, y_test_filtered):
    """
    Create a dictionary of DataFrames where each DataFrame contains data for a specific day_out.
    
    Parameters
    ----------
    accuracy_plot_df : pandas.DataFrame
        DataFrame containing predictions.
    y_test_filtered : pandas.DataFrame
        DataFrame containing actual values.
    
    Returns
    -------
    dict
        Dictionary with day_out as keys and DataFrames as values.
        Each DataFrame contains merged prediction and actual data for a specific forecast horizon.
        
    Notes
    -----
    This function organizes backtesting results by forecast horizon ('day_out'),
    allowing for separate analysis of prediction accuracy at different time distances.
    The resulting dictionary enables detailed evaluation of how model performance
    changes as predictions extend further into the future.
    """
    # Get unique day_out values
    unique_days = sorted(accuracy_plot_df['day_out'].unique())
    
    # Initialize dictionary
    day_out_dict = {}
    
    # Create DataFrame for each day_out
    for day in unique_days:
        # Filter both DataFrames for the current day_out
        predictions_mask = accuracy_plot_df['day_out'] == day
        actuals_mask = y_test_filtered['day_out'] == day
        
        # Create combined DataFrame for this day_out
        day_df = pd.DataFrame({
            'day_out': day,
            'prediction': accuracy_plot_df[predictions_mask]['means'].values,
            'actual': y_test_filtered[actuals_mask]['Price'].values,
            'start_idx': accuracy_plot_df[predictions_mask]['start_idx'].values,
            'valid_idx': accuracy_plot_df[predictions_mask]['valid_idx'].values
        })
        
        # Store in dictionary
        day_out_dict[day] = day_df
    
    return day_out_dict

class MonteCarloForecaster:
    def __init__(
        self,
        model: torch.nn.Module,
        scaler: Any, # A sklearn scaler
        device: torch.device,
        sequence_size: int,
        base_temp: float = 0.01,
        window_size: int = 30
    ):
        """
        Initialize the Monte Carlo forecaster for time series prediction.

        This class implements Monte Carlo simulation methods for forecasting time series
        data, with support for adaptive noise generation, confidence intervals, and
        comprehensive visualization tools.

        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch model for time series prediction.
        scaler : sklearn.preprocessing.StandardScaler
            Fitted scaler for inverse transforming predictions to original scale.
        device : torch.device
            Device to run computations on (CPU or CUDA).
        sequence_size : int
            Length of input sequences for the model.
        base_temp : float, default=0.01
            Base temperature for noise generation in Monte Carlo simulations.
        window_size : int, default=30
            Window size for calculating rolling variance in predictions.

        Attributes
        ----------
        model : torch.nn.Module
            The trained model used for predictions.
        scaler : sklearn.preprocessing.StandardScaler
            Scaler for data transformation.
        device : torch.device
            Computation device.
        sequence_size : int
            Input sequence length.
        base_temp : float
            Base noise temperature.
        window_size : int
            Rolling window size.
        start_indices : numpy.ndarray or None
            Starting indices for backtesting.
        remaining_days_data : pandas.DataFrame or None
            Forecast data for future days.

        Notes
        -----
        Design Decisions:
        1. Model Handling:
           - Accepts any PyTorch module that processes sequences
           - Maintains model on specified device for efficient computation
           - Preserves model state during forecasting
        
        2. Data Processing:
           - Uses scaler for consistent data transformation
           - Maintains sequence size requirements
           - Handles both training and inference data formats
        
        3. Monte Carlo Parameters:
           - Configurable base temperature for noise generation
           - Adaptive window size for variance calculation
           - Flexible initialization for different use cases
        
        4. Memory Management:
           - Stores minimal state information
           - Uses efficient data structures for large simulations
           - Cleans up intermediate results automatically

        Examples
        --------
        >>> import torch
        >>> from sklearn.preprocessing import StandardScaler
        >>> # Initialize with a trained model
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> forecaster = MonteCarloForecaster(
        ...     model=trained_model,
        ...     scaler=fitted_scaler,
        ...     device=device,
        ...     sequence_size=30
        ... )
        """
        self.model = model
        self.scaler = scaler
        self.device = device
        self.sequence_size = sequence_size
        self.base_temp = base_temp
        self.window_size = window_size
        self.start_indices = None
        self.remaining_days_data = None

    def monte_carlo_forecast(   self, 
                                x_test, 
                                num_days_to_predict=100, 
                                num_iterations=1000, 
                                base_temp=0.1, 
                                decay_factor=0.995, 
                                min_temp=0.01, 
                                confidence_levels=[0.68, 0.95]
                            ):
        """
        Enhanced Monte Carlo simulation for price forecasting with adaptive noise and multiple confidence intervals.
        
        Parameters
        ----------
        x_test : torch.Tensor
            Input test data tensor containing feature sequences.
        num_days_to_predict : int, default=100
            Number of days to forecast into the future.
        num_iterations : int, default=1000
            Number of Monte Carlo simulations to run.
        base_temp : float, default=0.1
            Initial temperature for noise generation.
        decay_factor : float, default=0.995
            Factor to decay temperature over time (0-1).
        min_temp : float, default=0.01
            Minimum temperature floor for noise generation.
        confidence_levels : list of float, default=[0.68, 0.95]
            List of confidence levels to calculate (e.g., [0.68, 0.95] for 1σ and 2σ).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing forecast results with columns:
            - Date : Forecast dates
            - Mean_Prediction : Average prediction for each date
            - Std_Prediction : Standard deviation of predictions
            - Confidence_Interval_X.XX_lower : Lower bound for each confidence level
            - Confidence_Interval_X.XX_upper : Upper bound for each confidence level
            
        Notes
        -----
        This method implements a sophisticated Monte Carlo approach to time series
        forecasting by:
        
        1. Running multiple simulations with randomized noise
        2. Using adaptive noise levels based on recent prediction variance
        3. Calculating multiple confidence intervals for uncertainty quantification
        4. Converting scaled predictions back to the original price scale
        
        The method maintains seed values for reproducibility(not that pytorch is allowing reproducability with CUDA for performance reasons if you don't have a b100).
        """
        torch.manual_seed(42)
        np.random.seed(42)
        
        seq_length = x_test.shape[1]
        num_features = x_test.shape[2]
        
        predicted_prices_all = np.zeros((num_iterations, num_days_to_predict))
        
        # Calculate percentiles for each confidence level
        percentiles = []
        for conf_level in confidence_levels:
            lower = (1 - conf_level) / 2 * 100
            upper = (1 + conf_level) / 2 * 100
            percentiles.extend([lower, upper])
        
        for seed in range(num_iterations):
            rng = np.random.RandomState(seed)
            predicted_prices = np.zeros(num_days_to_predict)
            
            self.model.eval()
            with torch.no_grad():
                last_sequence = x_test[-1].unsqueeze(0).to(self.device)
                last_known_features = last_sequence[0, -1, :-1].cpu().numpy()
                
                # Track recent predictions for adaptive noise
                recent_predictions = []
                
                for day in range(num_days_to_predict):
                    # Adaptive temperature based on prediction history
                    dynamic_temp = base_temp + prediction_variance(predicted_prices[:day])
                    
                    # Generate prediction with noise
                    next_day_prediction = self.model(last_sequence).squeeze().item()
                    noise = rng.normal(0, np.sqrt(dynamic_temp))
                    next_day_prediction += noise
                    
                    predicted_prices[day] = next_day_prediction
                    recent_predictions.append(next_day_prediction)
                    
                    # Update sequence for next prediction
                    next_day_features = np.zeros(num_features)
                    next_day_features[:-1] = last_known_features
                    next_day_features[-1] = next_day_prediction
                    
                    next_day_tensor = torch.tensor(next_day_features,
                                                dtype=torch.float32,
                                                device=self.device).view(1, 1, -1)
                    
                    last_sequence = torch.cat([
                        last_sequence[:, 1:, :],
                        next_day_tensor
                    ], dim=1)
                
                # Transform predictions back to original scale
                full_features = np.zeros((len(predicted_prices), num_features))
                full_features[:, -1] = predicted_prices
                predicted_prices_all[seed] = self.scaler.inverse_transform(full_features)[:, -1]
                
                print(f'Processed start index {seed}/{num_iterations}')
        
        # Calculate statistics
        mean_predictions = np.mean(predicted_prices_all, axis=0)
        std_predictions = np.std(predicted_prices_all, axis=0)
        confidence_intervals = np.percentile(predicted_prices_all, percentiles, axis=0)
        
        # Generate dates for forecast period
        start_date = pd.Timestamp('2024-10-23')
        remaining_days = pd.date_range(start=start_date, periods=num_days_to_predict, freq='D')
        
        # Create results DataFrame
        results = {'Date': remaining_days,
                'Mean_Prediction': mean_predictions,
                'Std_Prediction': std_predictions}
        
        # Add confidence intervals to results
        for i, conf_level in enumerate(confidence_levels):
            lower_idx = i * 2
            upper_idx = i * 2 + 1
            results[f'Confidence_Interval_{conf_level:.2f}_lower'] = confidence_intervals[lower_idx]
            results[f'Confidence_Interval_{conf_level:.2f}_upper'] = confidence_intervals[upper_idx]
        
        remaining_days_data = pd.DataFrame(results)
        self.remaining_days_data = remaining_days_data
        
        return remaining_days_data


    def monte_carlo_backtesting(self, 
                              test_data, 
                              x_test, 
                              forecast_horizon: int = 20, 
                              num_simulations: int = 1000, 
                              num_start_points: int = 100,
                              base_temperature: float = 0.1, 
                              temperature_decay: float = 0.995, 
                              min_temperature: float = 0.01,
                              confidence_levels: list = [0.68, 0.95]
                              ):
        """
        Perform enhanced Monte Carlo backtesting with adaptive noise and confidence intervals.

        Parameters
        ----------
        test_data : pandas.DataFrame
            DataFrame containing test data with 'Price' and feature columns.
        x_test : torch.Tensor
            Tensor of preprocessed test data sequences.
        forecast_horizon : int, default=20
            Number of days ahead to forecast for each simulation.
        num_simulations : int, default=1000
            Number of Monte Carlo simulations to run per starting point.
            Minimum value is 30 for statistical significance.
        num_start_points : int, default=100
            Number of different starting points in the test set.
            Minimum value is 10 for statistical significance.
        base_temperature : float, default=0.1
            Initial temperature for noise generation.
        temperature_decay : float, default=0.995
            Factor to decay temperature over prediction horizon.
        min_temperature : float, default=0.01
            Minimum temperature floor for noise generation.
        confidence_levels : list of float, default=[0.68, 0.95]
            Confidence levels for interval calculation.

        Returns
        -------
        dict
            Dictionary containing backtesting results for each start point:
            - predictions : numpy.ndarray
                Array of shape (num_iterations, num_days_to_predict)
            - mean : numpy.ndarray
                Mean prediction for each day
            - std : numpy.ndarray
                Standard deviation of predictions
            - start_date : datetime
                Starting date for this forecast
            - confidence_intervals : dict
                Confidence intervals at specified levels

        Notes
        -----
        Statistical Requirements:
        1. Minimum Sample Size:
           - At least 30 simulations per start point for reliable statistics
           - At least 10 start points for robust backtesting coverage
           - These minima ensure reliable confidence interval estimation
        
        2. Temperature Control:
           - Base temperature controls initial prediction uncertainty
           - Temperature decay reduces uncertainty over forecast horizon
           - Minimum temperature prevents unrealistic stability
        
        3. Implementation Details:
           - Uses adaptive noise based on prediction variance
           - Maintains temporal alignment with test data
           - Preserves random state for reproducibility
        

        Warnings
        --------
        - Long prediction horizons may accumulate errors
        - High num_iterations may impact performance
        - Confidence intervals assume normal distribution

        See Also
        --------
        monte_carlo_backtesting_plot : Function to visualize these results
        monte_carlo_backtesting_diagnostics_plot : Function for diagnostic analysis

        Examples
        --------
        >>> forecaster = MonteCarloForecaster(model, device)
        >>> results = forecaster.monte_carlo_backtesting(
        ...     test_data=test_data,
        ...     x_test=x_test,
        ...     num_days_to_predict=10,
        ...     num_iterations=500
        ... )
        >>> print(f"Mean prediction horizon: {len(results[0]['mean'])} days")
        """
        # Input validation
        if num_simulations < 30:
            raise ValueError("num_simulations must be at least 30 for statistical significance")
        if num_start_points < 10:
            raise ValueError("num_start_points must be at least 10 for statistical significance")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1")
        if not all(0 < level < 1 for level in confidence_levels):
            raise ValueError("confidence_levels must be between 0 and 1")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Prepare data
        feature_data = test_data.drop(columns=['Price'])
        sequence_length = x_test.shape[1]
        num_features = x_test.shape[2]
        
        # Calculate evenly distributed start indices in the test set
        feature_length = len(x_test)
        start_indices = np.linspace(0, feature_length - forecast_horizon - 1, 
                                num_start_points, dtype=int)
        self.start_indices = start_indices
        
        # Calculate percentiles for each confidence level
        percentiles = []
        for conf_level in confidence_levels:
            lower = (1 - conf_level) / 2 * 100
            upper = (1 + conf_level) / 2 * 100
            percentiles.extend([lower, upper])
        
        # Initialize dictionary to store results for each start point
        backtesting_results = {
            start_idx: {
                'predictions': np.zeros((num_simulations, forecast_horizon)),
                'mean': None,
                'std': None,
                'start_date': feature_data['Date'].iloc[start_idx + self.sequence_size],
                'confidence_intervals': None
            }
            for start_idx in start_indices
        }
        
        # Generate predictions for each start point
        for start_idx in start_indices:
            for simulation_idx in range(num_simulations):
                rng = np.random.RandomState(simulation_idx)
                predicted_prices = np.zeros(forecast_horizon)
                
                self.model.eval()
                with torch.no_grad():
                    # Use the sequence at the start index as initial input
                    initial_sequence = x_test[start_idx].unsqueeze(0).to(self.device)
                    current_sequence = initial_sequence
                    last_known_features = current_sequence[0, -1, :-1].cpu().numpy()
                    
                    # Track recent predictions for adaptive noise
                    recent_predictions = []
                    
                    for day in range(forecast_horizon):
                        # Adaptive temperature based on prediction history
                        dynamic_temp = base_temperature + prediction_variance(predicted_prices[:day])
                        
                        # Generate prediction with noise
                        next_day_prediction = self.model(current_sequence).squeeze().item()
                        noise = rng.normal(0, np.sqrt(dynamic_temp))
                        next_day_prediction += noise
                        
                        predicted_prices[day] = next_day_prediction
                        recent_predictions.append(next_day_prediction)
                        
                        # Update sequence for next prediction
                        next_day_features = np.zeros(num_features)
                        next_day_features[:-1] = last_known_features
                        next_day_features[-1] = next_day_prediction
                        
                        next_day_tensor = torch.tensor(next_day_features,
                                                    dtype=torch.float32,
                                                    device=self.device).view(1, 1, -1)
                        
                        current_sequence = torch.cat([
                            current_sequence[:, 1:, :],
                            next_day_tensor
                        ], dim=1)
                    
                    # Transform predictions back to original scale
                    full_features = np.zeros((len(predicted_prices), num_features))
                    full_features[:, -1] = predicted_prices
                    backtesting_results[start_idx]['predictions'][simulation_idx] = (
                        self.scaler.inverse_transform(full_features)[:, -1]
                    )
            
            # Calculate statistics for this start point
            predictions_all = backtesting_results[start_idx]['predictions']
            backtesting_results[start_idx]['mean'] = np.mean(predictions_all, axis=0)
            backtesting_results[start_idx]['std'] = np.std(predictions_all, axis=0)
            
            # Calculate confidence intervals
            confidence_intervals = np.percentile(predictions_all, percentiles, axis=0)
            backtesting_results[start_idx]['confidence_intervals'] = {
                conf_level: {
                    'lower': confidence_intervals[i * 2],
                    'upper': confidence_intervals[i * 2 + 1]
                }
                for i, conf_level in enumerate(confidence_levels)
            }
            
            print(f'Processed start index {start_idx}/{start_indices[-1]}')
        
        return backtesting_results

    def monte_carlo_forecast_plot(self,
                                remaining_days_data: pd.DataFrame,
                                predictions: pd.DataFrame,
                                test_data: pd.DataFrame,
                                DateMax: int = 100,
                                confidence_colors: list = None,
                                price_column: str = 'Price',
                                title: str = 'Price Forecast with Confidence Intervals'):
        """
        Enhanced plotting function for forecasting results with flexible confidence intervals.
        
        Parameters
        ----------
        remaining_days_data : pandas.DataFrame
            DataFrame containing future predictions and confidence intervals.
        predictions : pandas.DataFrame
            DataFrame containing model predictions for historical data.
        test_data : pandas.DataFrame
            Original test dataset containing actual price values.
        DateMax : int, default=100
            Maximum number of future dates to plot.
        confidence_colors : list of str, optional
            List of colors for confidence intervals. Defaults to red-based palette.
        price_column : str, default='Price'
            Name of the price column in test_data.
        title : str, default='Price Forecast with Confidence Intervals'
            Plot title.
            
        Returns
        -------
        None
            Displays the plot using matplotlib's pyplot.
            
        Notes
        -----
        This function creates a comprehensive visualization that includes:
        
        1. Historical actual prices
        2. Historical model predictions
        3. Future forecasted prices with multiple confidence intervals
        4. Historical mean price reference line
        
        The visualization employs a customized color scheme for clarity and uses
        transparent confidence intervals to show prediction uncertainty. The function
        handles both date-indexed and date-columned DataFrames appropriately.
        """
        # Set style for better visualization
        # plt.style.use('whitegrid')
        
        # Default color scheme for confidence intervals
        if confidence_colors is None:
            confidence_colors = ['#ffcccc', '#ff9999']  # Light to darker red
        
        # Get dates - handle both cases where Date might be in columns or index
        if 'Date' in test_data.columns:
            dates = test_data['Date']
        else:
            test_data = test_data.reset_index()
            dates = test_data['Date']
        
        # Extract price data
        y_test_df = test_data[price_column][self.sequence_size:]
        dates = dates[self.sequence_size:]
        
        # Create figure with specified size
        plt.figure(figsize=(15, 8))
        
        # Find confidence interval columns
        ci_columns = [col for col in remaining_days_data.columns 
                    if 'Confidence_Interval' in col]
        confidence_levels = sorted(list(set([
            float(col.split('_')[2]) for col in ci_columns 
            if col.split('_')[2].replace('.', '').isdigit()
        ])), reverse=True)
        
        # Plot confidence intervals from widest to narrowest
        for i, conf_level in enumerate(confidence_levels):
            lower_col = f'Confidence_Interval_{conf_level:.2f}_lower'
            upper_col = f'Confidence_Interval_{conf_level:.2f}_upper'
            
            if lower_col in remaining_days_data.columns and upper_col in remaining_days_data.columns:
                plt.fill_between(
                    remaining_days_data['Date'].iloc[:DateMax],
                    remaining_days_data[lower_col].iloc[:DateMax],
                    remaining_days_data[upper_col].iloc[:DateMax],
                    color=confidence_colors[i % len(confidence_colors)],
                    alpha=0.3,
                    label=f'{conf_level*100:.0f}% Confidence Interval'
                )
        
        # Plot historical and predicted data
        plt.plot(dates, y_test_df, 
                color='#2E86C1', 
                label='Actual Price', 
                linewidth=2)
        
        plt.plot(dates, predictions, 
                color='#28B463', 
                label='Predicted Price', 
                linewidth=2, 
                linestyle='--')
        
        # Plot mean future prediction
        plt.plot(remaining_days_data['Date'].iloc[:DateMax],
                remaining_days_data['Mean_Prediction'].iloc[:DateMax],
                color='#E74C3C',
                label='Mean Future Prediction',
                linewidth=2)
        
        # Plot historical mean
        plt.axhline(y=np.nanmean(y_test_df), 
                    color='#34495E', 
                    linestyle=':', 
                    label='Historical Mean',
                    alpha=0.7)
        
        # Enhance plot appearance
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Price', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, pad=20, fontweight='bold')
        
        # Format axis
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend with custom styling
        legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left',
                        borderaxespad=0.,
                        frameon=True,
                        fancybox=True,
                        shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()

    def monte_carlo_backtesting_plot(self,
                                  x_test: pd.DataFrame,
                                  backtesting_results: dict,
                                  test_data: pd.DataFrame,
                                  forecast_horizon: int = 20,
                                  num_start_points: int = 100,
                                  confidence_colors: list = None,
                                  price_column: str = 'Price',
                                  title: str = 'Price Backtesting with Confidence Intervals'):
        """
        Create comprehensive visualization of Monte Carlo backtesting results.

        Parameters
        ----------
        x_test : pandas.DataFrame
            Preprocessed test data used for predictions.
        backtesting_results : dict
            Dictionary containing Monte Carlo simulation results from backtesting.
        test_data : pandas.DataFrame
            Original test dataset with actual prices.
        forecast_horizon : int, default=20
            Number of days predicted in each forecast.
        num_start_points : int, default=100
            Number of forecast starting points used.
        confidence_colors : list of str, optional
            Colors for confidence interval bands. Defaults to red-based palette.
        price_column : str, default='Price'
            Name of the price column in test_data.
        title : str, default='Price Backtesting with Confidence Intervals'
            Plot title.

        Returns
        -------
        None
            Displays the plot using matplotlib's pyplot.

        Notes
        -----
        Design Decisions:
        1. Visual Layout:
           - Large figure size (15, 8) for detail visibility
           - Clear separation of actual vs predicted values
           - Multiple confidence intervals with transparency
        
        2. Visual Elements:
           - Rainbow color scheme for different forecast periods
           - Transparent confidence intervals (alpha=0.1)
           - Historical mean reference line
           - Clear date formatting and grid lines
        
        3. Plot Components:
           - Actual price line (blue)
           - Individual forecast lines (rainbow colors)
           - Confidence intervals (semi-transparent)
           - Historical mean (dotted gray)
        
        4. Layout Considerations:
           - Large figure size (15x8) for detail visibility
           - Rotated x-axis labels for readability
           - Legend positioned outside plot
           - Grid lines for value reference
        """
        # Input validation
        if not backtesting_results:
            raise ValueError("backtesting_results cannot be empty")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1")
        if num_start_points < 10:
            raise ValueError("num_start_points must be at least 10 for statistical significance")
        
        # Default color scheme for confidence intervals
        if confidence_colors is None:
            confidence_colors = ['#ffcccc', '#ff9999']  # Light to darker red
        
        # Prepare data
        feature_data = test_data.drop(columns=[price_column])
        actual_prices = test_data[price_column][self.sequence_size:]
        
        # Get dates - handle both cases where Date might be in columns or index
        if 'Date' in test_data.columns:
            dates = test_data['Date']
        else:
            test_data = test_data.reset_index()
            dates = test_data['Date']
        
        dates = dates[self.sequence_size:]
        
        # Create figure with specified size
        plt.figure(figsize=(15, 8))
        
        # Plot actual prices
        plt.plot(dates, actual_prices,
                color='#2E86C1',
                label='Actual Price',
                linewidth=2)
        
        # Get start indices from backtesting_results
        start_indices = sorted(backtesting_results.keys())
        
        # Plot each forecast with confidence intervals
        colors = plt.cm.rainbow(np.linspace(0, 1, len(start_indices)))
        for start_idx, color in zip(start_indices, colors):
            start_date = backtesting_results[start_idx]['start_date']
            forecast_dates = pd.date_range(start=start_date,
                                        periods=forecast_horizon,
                                        freq='D')
            
            # Plot mean prediction
            #TODO delete if changes worked
            '''
            for conf_level, interval_data in forecast_results[start_idx]['confidence_intervals'].items():
                plt.fill_between(dates_forecast,
            '''
            plt.plot(forecast_dates,
                    backtesting_results[start_idx]['mean'],
                    color=color,
                    label=f'Forecast from {start_date.strftime("%Y-%m-%d")}',
                    linewidth=1.5,
                    linestyle='--')
            
            # Plot confidence intervals
            for conf_level, interval_data in backtesting_results[start_idx]['confidence_intervals'].items():
                plt.fill_between(forecast_dates,
                                interval_data['lower'],
                                interval_data['upper'],
                                color=color,
                                alpha=0.1,
                                label=f'{conf_level*100:.0f}% CI ({start_date.strftime("%Y-%m-%d")})')
        
        # Plot historical mean
        plt.axhline(y=np.nanmean(actual_prices),
                    color='#34495E',
                    linestyle=':',
                    label='Historical Mean',
                    alpha=0.7)
        
        # Enhance plot appearance
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Price', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, pad=20, fontweight='bold')
        
        # Format axis
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        #TODO add a color bar for ledgend normal legends dont work here too many items
        
        # Adjust layout
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()


#TODO Generalize to an arbitray target variable and date format
    def monte_carlo_backtesting_diagnotsics_plot(self, 
                                          forecast_results: dict, 
                                          test_data: pd.DataFrame):
        """
        Generate diagnostic plots for Monte Carlo backtesting analysis.

        This function creates a two-panel diagnostic visualization showing prediction
        accuracy and error distribution patterns across the backtesting period.
        Designed to help identify systematic biases and error patterns.

        Parameters
        ----------
        forecast_results : dict
            Dictionary containing Monte Carlo simulation results from backtesting.
            Must include mean predictions and confidence intervals.
        test_data : pandas.DataFrame
            Original test dataset with actual prices in 'Price' column.

        Returns
        -------
        None
            Displays the plot using matplotlib's pyplot.

        Notes
        -----
        Design Decisions:
        1. Plot Layout:
           - Two-panel vertical arrangement (12, 10)
           - Top panel: Temporal error patterns
           - Bottom panel: Error distribution analysis
        
        2. Diagnostic Metrics:
           - Day-wise prediction accuracy
           - Error distribution characteristics
           - Systematic bias indicators
        
        3. Visual Elements:
           - Clear color coding for error types
           - Reference lines for perfect prediction
           - Confidence bands for uncertainty
        
        4. Analysis Features:
           - Temporal clustering of errors
           - Bias detection in predictions
           - Outlier identification

        See Also
        --------
        monte_carlo_backtesting : Function that generates the analyzed results
        monte_carlo_backtesting_plot : Main visualization of predictions

        Examples
        --------
        >>> forecaster = MonteCarloForecaster(model, device)
        >>> results = forecaster.monte_carlo_backtesting(test_data, x_test)
        >>> forecaster.monte_carlo_backtesting_diagnotsics_plot(
        ...     forecast_results=results,
        ...     test_data=test_data
        ... )
        >>> plt.show()  # If not in interactive mode
        """
        y_test_df = test_data['Price'][self.sequence_size:]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        accuracy_plot_data = []

        for start_index in forecast_results:
            for i, mean in enumerate(forecast_results[start_index]["mean"]):
                accuracy_plot_data.append({
                    "day_out": i+1,
                    "start_idx": start_index,
                    "valid_idx": start_index + i,
                    "end_idx": start_index + len(forecast_results[start_index]["mean"]) - 1,
                    "means": mean
                })

        accuracy_plot_df = pd.DataFrame(accuracy_plot_data)

        # Initialize an empty list to collect the rows
        y_test_filtered_list = []

        # "day_out" Using day_out from accuracy_plot_df instead of i+1
        for i, val_idx in enumerate(accuracy_plot_df["valid_idx"]):
            y_test_filtered_list.append({
                "Price": y_test_df.iloc[val_idx],
                "day_out": accuracy_plot_df.iloc[i]["day_out"] 
            })

        # Convert the list to a DataFrame
        y_test_filtered = pd.DataFrame(y_test_filtered_list)

        # Calculate RMSE for each day_out
        day_out_rmse = []
        for day in sorted(accuracy_plot_df['day_out'].unique()):
            day_mask = accuracy_plot_df['day_out'] == day
            rmse = np.sqrt(mean_squared_error(
                y_test_filtered[day_mask]['Price'],
                accuracy_plot_df[day_mask]['means']
            ))
            day_out_rmse.append({
                'day_out': day,
                'rmse': rmse
            })

        # Convert to DataFrame for easy plotting
        rmse_df = pd.DataFrame(day_out_rmse)

        # Calculate percentage increase from day 1
        first_day_rmse = rmse_df.iloc[0]['rmse']
        rmse_df['pct_increase'] = ((rmse_df['rmse'] - first_day_rmse) / first_day_rmse).round(2)

        # Plot 1: RMSE over time
        ax1.plot(rmse_df['day_out'], rmse_df['rmse'], 
                marker='o', linewidth=2, markersize=8, 
                color='#2E86C1', label='RMSE')

        # Customize first plot
        ax1.set_xlabel('Forecast Horizon (Days)')
        ax1.set_ylabel('RMSE')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('RMSE vs Forecast Horizon')

        # Add value labels on the points
        for x, y in zip(rmse_df['day_out'], rmse_df['rmse']):
            ax1.annotate(f'{y:.2f}', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')

        # Plot 2: Percentage increase
        ax2.plot(rmse_df['day_out'], rmse_df['pct_increase'], 
                marker='s', linewidth=2, markersize=8, 
                color='#E74C3C', label='% Increase')
        ax2.fill_between(rmse_df['day_out'], rmse_df['pct_increase'], 
                        alpha=0.2, color='#E74C3C')

        # Customize second plot
        ax2.set_xlabel('Forecast Horizon (Days)')
        ax2.set_ylabel('Percentage Increase from Day 1 (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_title('Percentage Increase in RMSE from Day 1')

        # Add value labels on the points
        for x, y in zip(rmse_df['day_out'], rmse_df['pct_increase']):
            ax2.annotate(f'{y*100:.1f}%', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
    def monte_carlo_backtesting_pred_true_plot(self, 
                                            forecast_results : dict, 
                                            test_data : pd.DataFrame):
        """
        Generate scatter plot comparing actual vs predicted prices from Monte Carlo backtesting.
        
        Parameters
        ----------
        forecast_results : dict
            Dictionary containing Monte Carlo simulation results from backtesting.
            Must contain mean predictions for each forecast start point.
        test_data : pandas.DataFrame
            Original test dataset with actual prices in 'Price' column.
            
        Returns
        -------
        None
            Displays the scatter plot using matplotlib's pyplot.
            
        Notes
        -----
        This function creates a scatter plot visualization that:
        
        1. Compares actual prices against predicted prices
        2. Color-codes points by forecast horizon (days out)
        3. Includes a perfect prediction reference line
        4. Uses a color gradient to show how prediction accuracy changes with forecast distance
        
        The plot helps identify systematic biases in the model predictions and visualize
        how prediction accuracy varies across different price levels and forecast horizons.
        """
        y_test_df = test_data['Price'][self.sequence_size:]
        
        accuracy_plot_data = []

        for start_index in forecast_results:
            for i, mean in enumerate(forecast_results[start_index]["mean"]):
                accuracy_plot_data.append({
                    "day_out": i+1,
                    "start_idx": start_index,
                    "valid_idx": start_index + i,
                    "end_idx": start_index + len(forecast_results[start_index]["mean"]) - 1,
                    "means": mean
                })

        accuracy_plot_df = pd.DataFrame(accuracy_plot_data)
        
        # Initialize an empty list to collect the rows
        y_test_filtered_list = []

        # "day_out" Using day_out from accuracy_plot_df instead of i+1
        for i, val_idx in enumerate(accuracy_plot_df["valid_idx"]):
            y_test_filtered_list.append({
                "Price": y_test_df.iloc[val_idx],
                "day_out": accuracy_plot_df.iloc[i]["day_out"] 
            })

        # Convert the list to a DataFrame
        y_test_filtered = pd.DataFrame(y_test_filtered_list)
        
        # Normalize 'day_out' for colormap
        norm = plt.Normalize(accuracy_plot_df['day_out'].min(), accuracy_plot_df['day_out'].max())
        cmap = cm.viridis  

        plt.figure(figsize=(10, 6))

        # Scatter plot with color based on 'day_out'
        scatter = plt.scatter(
            y_test_filtered["Price"], 
            accuracy_plot_df['means'], 
            c=accuracy_plot_df['day_out'], 
            cmap=cmap, 
            label='Actual vs. Predicted'
        )

        # Plot perfect prediction line
        plt.plot(
            [y_test_filtered["Price"].min(), y_test_filtered["Price"].max()], 
            [y_test_filtered["Price"].min(), y_test_filtered["Price"].max()], 
            'k--', lw=2, color='red', label='Perfect Prediction'
        )

        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs. Predicted Prices')

        # Add colorbar to indicate days out
        cbar = plt.colorbar(scatter)
        cbar.set_label('Days Out')

        plt.legend()
        plt.show()
    
    