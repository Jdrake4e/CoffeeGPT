import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

def evaluate_test_set(model, test_loader, scaler, test_data, sequence_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate model performance on test set and generate predictions.

    This function evaluates a trained model on test data, handling the conversion between
    scaled and original values, and computing key performance metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader containing the test dataset.
    scaler : sklearn.preprocessing._data.StandardScaler
        Fitted scaler object used to inverse transform predictions.
    test_data : pandas.DataFrame
        Original test data containing all features. Used to determine feature dimensionality
        and for date indexing.
    sequence_size : int
        The size of input sequences used during training.
    device : str, optional
        Device to run evaluation on, by default 'cuda' if available else 'cpu'.

    Returns
    -------
    numpy.ndarray
        predictions_original : Model predictions in original scale.
    numpy.ndarray
        true_values_original : True values in original scale.
    float
        rmse : Root Mean Square Error between predictions and true values.
    pandas.DatetimeIndex
        dates : DatetimeIndex corresponding to the predictions.

    Notes
    -----
    Design Decisions:
    1. The function assumes the target variable is the last column in the dataset
       (excluding the index). This design choice simplifies the interface but requires
       consistent data preprocessing.
    2. Predictions are generated in batch mode for memory efficiency, especially
       important for large test sets.
    3. The scaler is applied to a zero-filled array with the same dimensionality
       as the training data to ensure correct inverse transformation.

    See Also
    --------
    plot_predictions : Function to visualize these predictions against true values.

    Examples
    --------
    >>> model = TransformerModel(input_dim=5)
    >>> predictions, true_values, rmse, dates = evaluate_test_set(
    ...     model, test_loader, scaler, test_data, sequence_size=10
    ... )
    >>> print(f"Test RMSE: {rmse:.4f}")
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            predictions = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    predictions = np.array(all_predictions)
    true_values = np.array(all_targets)
    
    # Get the correct number of features from test_data
    num_features = test_data.shape[1]-1
    
    # Create dummy arrays with correct number of features
    pred_full = np.zeros((len(predictions), num_features))
    true_full = np.zeros((len(true_values), num_features))
    
    # Assuming the target is the last column
    target_idx = num_features - 1
    
    # Put predictions and true values in the correct column
    pred_full[:, target_idx] = predictions.flatten()
    true_full[:, target_idx] = true_values.flatten()
    
    # Inverse transform
    predictions_original = scaler.inverse_transform(pred_full)[:, target_idx]
    true_values_original = scaler.inverse_transform(true_full)[:, target_idx]
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_values_original, predictions_original))
    
    # Get dates starting from the beginning of the test set
    dates = test_data.index[:len(predictions_original)]
    
    return predictions_original, true_values_original, rmse, dates

def plot_predictions(predictions, true_values, test_data, sequence_size):
    """
    Create comprehensive visualization of model predictions versus true values.

    This function generates a two-panel plot showing both time series and regression
    analysis of model predictions against true values. The visualization includes
    statistical metrics and regression analysis to evaluate model performance.

    Parameters
    ----------
    predictions : array-like
        Model predictions in the original scale.
    true_values : array-like
        Actual values in the original scale.
    test_data : pandas.DataFrame
        Test dataset containing the date index. Must have either a 'Date' column
        or a DatetimeIndex.
    sequence_size : int
        The size of input sequences used during training, used for proper date alignment.

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.

    Notes
    -----
    Design Decisions:
    1. Two-Panel Layout:
       - Top panel: Time series plot showing temporal patterns and divergences
       - Bottom panel: Regression plot with perfect prediction line for bias analysis
    
    2. Visual Elements:
       - Uses alpha=0.5 for scatter plots to show point density
       - Includes both regression line and perfect prediction line for comparison
       - Grid enabled for better readability
    
    3. Metrics Display:
       - RMSE (Root Mean Square Error) for absolute error measurement
       - MAE (Mean Absolute Error) for average deviation
       - R² score for goodness of fit
    
    4. Date Handling:
       - Flexibly handles both 'Date' column and DatetimeIndex formats
       - Adjusts for sequence_size to ensure proper temporal alignment

    See Also
    --------
    evaluate_test_set : Function that generates the predictions used by this visualization.
    plot_training_history : Complementary function for visualizing training metrics.

    Examples
    --------
    >>> predictions = model.predict(test_data)
    >>> plot_predictions(predictions, true_values, test_data, sequence_size=10)
    >>> plt.show()  # If not in interactive mode
    """
    
    if 'Date' in test_data.columns:
        dates = test_data['Date']
    else:
        # If Date is in the index, reset index to get it as a column
        test_data = test_data.reset_index()
        dates = test_data['Date']
    
    # Extract Dates
    dates = dates[sequence_size:]
    
    # Ensure arrays are 1-dimensional
    predictions = np.asarray(predictions).flatten()
    true_values = np.asarray(true_values).flatten()
    
    plt.figure(figsize=(12, 8))
    
    # Time series plot (line plot)
    plt.subplot(2, 1, 1)
    plt.plot(dates, predictions, 
            label='Predictions', linewidth=2, alpha=0.8)
    plt.plot(dates, true_values, 
            label='True Values', linewidth=2, alpha=0.8)
    plt.title('Predictions vs True Values Over Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Regression plot (scatter plot)
    plt.subplot(2, 1, 2)
    plt.scatter(true_values, predictions, s=20, alpha=0.5)  # Set point size to 20 and alpha to 0.5

    # Add regression line
    z = np.polyfit(true_values, predictions, 1)
    p = np.poly1d(z)
    plt.plot(true_values, p(true_values), 
            "r--", linewidth=2, alpha=0.8, 
            label=f'Regression Line (R² = {np.corrcoef(true_values, predictions)[0,1]**2:.4f})')

    # Add perfect prediction line
    plt.plot([min(true_values), max(true_values)],
            [min(true_values), max(true_values)],
            'g--', linewidth=2, alpha=0.8, label='Perfect Prediction')

    plt.title('Prediction vs True Value Correlation')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    rmse = np.sqrt(np.mean((predictions - true_values)**2))
    mae = np.mean(np.abs(predictions - true_values))
    r2 = np.corrcoef(predictions, true_values)[0,1]**2
    
    print("\nMetrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
def plot_training_history(trainer, figsize=(10, 6)):
    """
    Visualize training and validation loss history.

    Creates a detailed visualization of model training progress, showing both training
    and validation loss curves. Includes special handling for infinite values and
    implements sophisticated styling for clear visualization.

    Parameters
    ----------
    trainer : ModelTrainer
        Trainer object containing training history. Must have the following attributes:
        - validation_loss_history : list of float
        - epoch_history : list of int
        - training_loss_history : list of float
    figsize : tuple of int, optional
        Figure size in inches (width, height), by default (10, 6)

    Returns
    -------
    None
        Displays the plot using matplotlib's pyplot.

    Notes
    -----
    Design Decisions:
    1. Data Preprocessing:
       - Converts infinite values to NaN to prevent plotting issues
       - Maintains data integrity while handling numerical anomalies
    
    2. Visual Design:
       - Implements markers for better epoch differentiation
       - Customized line width and opacity for optimal readability
    
    3. Plot Elements:
       - Dual line plot showing both training and validation metrics
       - Legend with appropriately sized fonts
       - Grid lines for easier metric comparison
       - Padding added to title for better spacing
    
    4. Error Handling:
       - Gracefully handles infinite values in loss histories
       - Preserves plot functionality even with partial data corruption

    See Also
    --------
    plot_predictions : Complementary function for visualizing model predictions.
    ModelTrainer : Class that generates the training history used here.

    Examples
    --------
    >>> trainer = ModelTrainer(model, device)
    >>> # After training
    >>> plot_training_history(trainer, figsize=(12, 8))
    >>> plt.show()  # If not in interactive mode
    """
    # Extract histories from trainer and convert inf to nan
    val_loss = np.array(trainer.validation_loss_history)
    epochs = np.array(trainer.epoch_history)
    train_loss = np.array(trainer.training_loss_history)
    
    # Convert inf to nan for all three arrays
    val_loss = np.where(np.isinf(val_loss), np.nan, val_loss)
    epochs = np.where(np.isinf(epochs), np.nan, epochs)
    train_loss = np.where(np.isinf(train_loss), np.nan, train_loss)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create the line plots
    sns.lineplot(x=epochs, y=train_loss, label='Training Loss', 
                marker='o', linewidth=2)
    sns.lineplot(x=epochs, y=val_loss, label='Validation Loss', 
                marker='o', linewidth=2)
    
    # Customize the plot
    plt.title('Training and Validation Loss Over Time', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def calculate_directional_accuracy(test_data, forecast_results, sequence_size):
    """
    Calculate directional accuracy of predictions by forecast day and overall.
    
    Parameters
    ----------
    test_data : pandas.DataFrame
        DataFrame containing actual price data.
    forecast_results : dict
        Dictionary containing forecast results.
    sequence_size : int
        Integer representing the sequence size used in the model.
    
    Returns
    -------
    dict
        Dictionary containing accuracy metrics and detailed results:
        
        - accuracy_by_day : pandas.DataFrame
            DataFrame with daily accuracy metrics.
        - overall_accuracy : float
            Overall accuracy percentage.
        - detailed_metrics : dict
            Dictionary with additional performance metrics.
        - detailed_results : pandas.DataFrame
            DataFrame with prediction details.
            
    Notes
    -----
    This function calculates the accuracy of predicted price directions (up or down)
    compared to actual price movements. It performs analysis for each forecast day
    and calculates aggregate metrics across all predictions.
    
    The function creates a comprehensive set of metrics including:
    - Direction prediction accuracy (overall and by day)
    - Separate metrics for upward and downward movement predictions
    - Error measurements (RMSE, MAPE)
    - Direction bias analysis
    """
    y_test_df = test_data['Price'][sequence_size:]
    
    # Create accuracy plot data more efficiently
    accuracy_plot_data = []
    for start_index, forecast in forecast_results.items():
        for i, mean in enumerate(forecast["mean"]):
            accuracy_plot_data.append({
                "day_out": i + 1,
                "start_idx": start_index,
                "valid_idx": start_index + i,
                "end_idx": start_index + len(forecast["mean"]) - 1,
                "predicted_price": mean
            })
    
    accuracy_plot_df = pd.DataFrame(accuracy_plot_data)
    
    # Create filtered test data with vectorized operations
    valid_indices = accuracy_plot_df["valid_idx"].values
    y_test_filtered = pd.DataFrame({
        "actual_price": y_test_df.iloc[valid_indices].values,
        "day_out": accuracy_plot_df["day_out"].values,
        "valid_idx": valid_indices
    })
    
    # Add previous day's actual price
    y_test_filtered["prev_price"] = y_test_df.iloc[y_test_filtered["valid_idx"] - 1].values
    
    # Calculate slopes
    y_test_filtered["slope_actual"] = np.sign(y_test_filtered["actual_price"] - y_test_filtered["prev_price"])
    accuracy_plot_df["slope_predicted"] = np.sign(accuracy_plot_df["predicted_price"] - y_test_filtered["prev_price"])
    
    # Merge dataframes
    merged_df = pd.merge(
        y_test_filtered,
        accuracy_plot_df[["day_out", "slope_predicted", "predicted_price"]],
        on="day_out"
    )
    
    # Calculate detailed metrics by day
    accuracy_by_day = merged_df.groupby("day_out").apply(
        lambda x: pd.Series({
            'correct_predictions': (x["slope_actual"] == x["slope_predicted"]).sum(),
            'total_predictions': len(x),
            'accuracy_percentage': (x["slope_actual"] == x["slope_predicted"]).mean() * 100,
            'upward_accuracy': (
                (x["slope_actual"] == 1) & (x["slope_predicted"] == 1)
            ).sum() / (x["slope_actual"] == 1).sum() * 100 if (x["slope_actual"] == 1).sum() > 0 else 0,
            'downward_accuracy': (
                (x["slope_actual"] == -1) & (x["slope_predicted"] == -1)
            ).sum() / (x["slope_actual"] == -1).sum() * 100 if (x["slope_actual"] == -1).sum() > 0 else 0,
            'avg_prediction_error': np.abs(x["actual_price"] - x["predicted_price"]).mean()
        })
    ).round(2)
    
    # Calculate overall metrics
    correct_predictions = (merged_df["slope_actual"] == merged_df["slope_predicted"]).sum()
    total_predictions = len(merged_df)
    overall_accuracy = (correct_predictions / total_predictions) * 100
    
    # Additional overall metrics
    detailed_metrics = {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'mape': np.mean(np.abs((merged_df["actual_price"] - merged_df["predicted_price"]) / merged_df["actual_price"])) * 100,
        'rmse': np.sqrt(np.mean((merged_df["actual_price"] - merged_df["predicted_price"])**2)),
        'direction_bias': (merged_df["slope_predicted"].mean() - merged_df["slope_actual"].mean())
    }
    
    return {
        'accuracy_by_day': accuracy_by_day,
        'overall_accuracy': overall_accuracy,
        'detailed_metrics': detailed_metrics,
        'detailed_results': merged_df
    }

def plot_accuracy_metrics(accuracy_by_day, detailed_metrics):
    """
    Create comprehensive visualization of accuracy metrics.
    
    Parameters
    ----------
    accuracy_by_day : pandas.DataFrame
        DataFrame containing accuracy metrics by day.
    detailed_metrics : dict
        Dictionary containing overall performance metrics.
        
    Returns
    -------
    None
        Displays visualization plots using matplotlib.
        
    Notes
    -----
    This function creates a three-panel visualization showing:
    1. Accuracy percentage by forecast day
    2. Directional accuracy for upward vs downward movements
    3. Average prediction error by day
    
    The function also prints additional metrics including MAPE, RMSE,
    and direction bias for comprehensive model evaluation.
    """
    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Accuracy by Day
    plt.subplot(3, 1, 1)
    sns.barplot(
        data=accuracy_by_day.reset_index(),
        x='day_out',
        y='accuracy_percentage',
        color='skyblue'
    )
    plt.title('Prediction Accuracy by Day', pad=20)
    plt.xlabel('Days Out')
    plt.ylabel('Accuracy (%)')
    plt.axhline(y=detailed_metrics['overall_accuracy'], color='red', linestyle='--', 
                label=f'Overall Accuracy: {detailed_metrics["overall_accuracy"]:.1f}%')
    plt.legend()

    # Plot 2: Directional Accuracy (Upward vs Downward)
    plt.subplot(3, 1, 2)
    direction_data = pd.melt(
        accuracy_by_day.reset_index(),
        id_vars=['day_out'],
        value_vars=['upward_accuracy', 'downward_accuracy'],
        var_name='direction',
        value_name='accuracy'
    )
    sns.lineplot(
        data=direction_data,
        x='day_out',
        y='accuracy',
        hue='direction',
        marker='o'
    )
    plt.title('Upward vs Downward Movement Prediction Accuracy', pad=20)
    plt.xlabel('Days Out')
    plt.ylabel('Accuracy (%)')

    # Plot 3: Prediction Error
    plt.subplot(3, 1, 3)
    sns.lineplot(
        data=accuracy_by_day.reset_index(),
        x='day_out',
        y='avg_prediction_error',
        color='green',
        marker='o'
    )
    plt.title('Average Prediction Error by Day', pad=20)
    plt.xlabel('Days Out')
    plt.ylabel('Average Absolute Error')
    
    plt.tight_layout()
    plt.show()

    # Print additional metrics
    print("\nDetailed Performance Metrics:")
    print(f"MAPE: {detailed_metrics['mape']:.2f}%")
    print(f"RMSE: {detailed_metrics['rmse']:.2f}")
    print(f"Direction Bias: {detailed_metrics['direction_bias']:.3f}")

def bland_altman_plot(sequence_size, forecast_results, test_data):
    """
    Creates a Bland-Altman plot comparing actual vs predicted prices,
    with points colored by prediction horizon (days out).
    
    Parameters
    ----------
    sequence_size : int
        Integer representing the sequence size used in the model.
    forecast_results : dict
        Dictionary containing forecast results.
    test_data : pandas.DataFrame
        DataFrame containing actual price data.
        
    Returns
    -------
    None
        Displays Bland-Altman plot using matplotlib.
        
    Notes
    -----
    The Bland-Altman plot (or difference plot) is used to visualize the agreement
    between two measurement methods. In this context, it shows the difference between
    predicted and actual prices against their mean, providing insights into systematic
    bias and limits of agreement.
    
    Points are colored by forecast horizon (days out) to show how prediction
    accuracy changes with increasing forecast distance. The function also prints
    statistical information about mean difference and limits of agreement.
    """
    y_test_df = test_data['Price'][sequence_size:]
    
    # Prepare data
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
    
    # Get actual values
    y_test_filtered_list = []
    for i, val_idx in enumerate(accuracy_plot_df["valid_idx"]):
        y_test_filtered_list.append({
            "Price": y_test_df.iloc[val_idx],
            "day_out": accuracy_plot_df.iloc[i]["day_out"]
        })
    
    y_test_filtered = pd.DataFrame(y_test_filtered_list)
    
    # Calculate Bland-Altman metrics
    differences = accuracy_plot_df['means'] - y_test_filtered['Price']
    means = (accuracy_plot_df['means'] + y_test_filtered['Price']) / 2
    
    # Calculate limits of agreement
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Normalize 'day_out' for colormap
    norm = plt.Normalize(accuracy_plot_df['day_out'].min(), accuracy_plot_df['day_out'].max())
    cmap = cm.viridis
    
    # Scatter plot with color based on 'day_out'
    scatter = plt.scatter(means, 
                         differences,
                         c=accuracy_plot_df['day_out'],
                         cmap=cmap,
                         alpha=0.9,
                         label='Differences')
    
    # Add mean and limits of agreement lines
    plt.axhline(y=0, color='black', linestyle='-', label='No difference')
    plt.axhline(y=mean_diff, color='red', linestyle='--', label='Mean Difference')
    plt.axhline(y=upper_limit, color='gray', linestyle=':', label='+1.96 SD')
    plt.axhline(y=lower_limit, color='gray', linestyle=':', label='-1.96 SD')
    
    plt.xlabel('Mean of Actual and Predicted Prices')
    plt.ylabel('Difference (Predicted - Actual)')
    plt.title('Bland-Altman Plot of Price Predictions')
    
    # Add colorbar to indicate days out
    cbar = plt.colorbar(scatter)
    cbar.set_label('Days Out')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print statistical summary
    print(f"Mean difference: {mean_diff:.2f}")
    print(f"Upper limit of agreement: {upper_limit:.2f}")
    print(f"Lower limit of agreement: {lower_limit:.2f}")

def analyze_accuracy_by_blocks(merged_df, n=10):
    """
    Analyze prediction accuracy across different blocks of data.
    
    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing prediction results and actual values.
    n : int, default=10
        Number of blocks to divide the data into.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing block analysis results, with metrics for each time block
        including directional accuracy percentages and average prediction errors.
        
    Notes
    -----
    This function segments the dataset into time-based blocks to analyze how
    prediction accuracy varies across different periods. It helps identify
    whether the model performs consistently over time or if there are specific
    periods where performance deteriorates.
    
    The function creates a visualization showing directional accuracy across
    the time blocks and returns detailed metrics for further analysis.
    """
    # Assign a block number to each row based on its position in the dataset
    merged_df["block"] = pd.qcut(merged_df.index, n, labels=False)

    # Calculate slope match
    slope_match = (merged_df["slope_actual"] == merged_df["slope_predicted"])

    # Group by the blocks and calculate slope match statistics
    block_analysis = merged_df.groupby("block").apply(
        lambda df: pd.Series({
            "slope_match_count": (df["slope_actual"] == df["slope_predicted"]).sum(),
            "total_comparisons": len(df),
            "slope_match_percentage": (df["slope_actual"] == df["slope_predicted"]).mean() * 100,
            "avg_prediction_error": np.abs(df["actual_price"] - df["predicted_price"]).mean()
        })
    )

    # Reset index for a cleaner DataFrame
    block_analysis = block_analysis.reset_index()
    
    # Print or visualize the results
    print("\nAccuracy Analysis by Time Blocks:")
    print(block_analysis)

    plt.figure(figsize=(10, 6))
    plt.bar(block_analysis["block"], block_analysis["slope_match_percentage"], color="skyblue")
    plt.xlabel("Block Number")
    plt.ylabel("Directional Accuracy (%)")
    plt.title(f"Directional Accuracy Across {n} Time Blocks")
    plt.xticks(block_analysis["block"])
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return block_analysis

def analyze_accuracy_by_price_range(merged_df, n=10):
    """
    Analyze prediction accuracy across different price ranges.
    
    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing prediction results and actual values.
    n : int, default=10
        Number of price ranges to analyze.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing price range analysis results, with metrics for each
        price range including directional accuracy percentages, price boundaries,
        and average prediction errors.
        
    Notes
    -----
    This function segments the dataset into price-based ranges to analyze how
    prediction accuracy varies across different price levels. It helps identify
    whether the model performs consistently across all price ranges or if there
    are specific price levels where performance is better or worse.
    
    The function creates a visualization showing directional accuracy across
    the price ranges and returns detailed metrics for further analysis. This
    information can be valuable for understanding model limitations and potential
    price-dependent biases.
    """
    # Create blocks based on the range of actual prices
    merged_df["price_block"] = pd.qcut(merged_df["actual_price"], n, labels=False)

    # Group by the price blocks and calculate slope match statistics
    price_analysis = merged_df.groupby("price_block").apply(
        lambda df: pd.Series({
            "slope_match_count": (df["slope_actual"] == df["slope_predicted"]).sum(),
            "total_comparisons": len(df),
            "slope_match_percentage": (df["slope_actual"] == df["slope_predicted"]).mean() * 100,
            "price_min": df["actual_price"].min(),
            "price_max": df["actual_price"].max(),
            "avg_prediction_error": np.abs(df["actual_price"] - df["predicted_price"]).mean()
        })
    )

    # Reset index for a cleaner DataFrame
    price_analysis = price_analysis.reset_index()
    
    # Print the results
    print("\nAccuracy Analysis by Price Range:")
    print(price_analysis)

    plt.figure(figsize=(10, 6))
    plt.bar(
        price_analysis.index, 
        price_analysis["slope_match_percentage"], 
        color="lightgreen", 
        edgecolor="black"
    )
    plt.xlabel("Price Range")
    plt.ylabel("Directional Accuracy (%)")
    plt.title(f"Directional Accuracy Across {n} Price Ranges")
    plt.xticks(price_analysis.index, labels=[
        f"[{row['price_min']:.2f}, {row['price_max']:.2f}]" for _, row in price_analysis.iterrows()], rotation=45
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return price_analysis