import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_synthetic_data(num_points=1000, seed=None, noise_level=0.1, jump_probability=0.05):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a date range
    start_date = datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(num_points)]
    
    # Format dates as d/m/y
    formatted_dates = [date.strftime('%d/%m/%Y') for date in dates]
    
    # Generate more complex periodic data with random frequency and amplitude
    base_frequencies = [4, 8, 16, 32, 64]
    base_amplitudes = [1, 0.5, 0.25, 0.1, 0.05]
    periodic_data = np.zeros(num_points)
    
    for base_freq, base_amp in zip(base_frequencies, base_amplitudes):
        # Add random noise to frequency and amplitude
        freq_noise = np.random.uniform(0.8, 1.4)
        amp_noise = np.random.uniform(0.8, 1.4)
        
        # Generate the sine wave with noise
        periodic_data += (base_amp * amp_noise) * np.sin(np.linspace(0, base_freq * freq_noise * np.pi, num_points))
    
    # Add controlled noise (Gaussian)
    noise = np.random.normal(0, noise_level, num_points)
    data_with_noise = periodic_data + noise
    
    # Apply cumulative effect to the price data
    for i in range(1, num_points):
        data_with_noise[i] += 0.1 * data_with_noise[i-1]  # Add 10% of the previous price to the current price
    
    # Introduce extreme events
    extreme_event_probability = 0.01
    extreme_event_dates = np.random.choice(dates, size=int(num_points * extreme_event_probability), replace=False)
    for date in extreme_event_dates:
        index = dates.index(date)
        data_with_noise[index] += np.random.choice([-1, 1])  # Significant increase or decrease
    
    # Introduce event-based jumps or decreases
    event_dates = np.random.choice(dates, size=int(num_points * jump_probability), replace=False)
    for date in event_dates:
        index = dates.index(date)
        data_with_noise[index] += np.random.choice([-1, 1]) * np.random.rand()

    # Add indicator for weekdays (1) vs weekends (0)
    weekday_indicator = [1 if date.weekday() < 5 else 0 for date in dates]

    # Apply exponential transformation to ensure positive prices
    data_with_noise = np.exp(data_with_noise)

    # Add a small constant to prevent prices from reaching zero
    data_with_noise += 0.01

    # Recalculate median price after transformation
    median_price = np.median(data_with_noise)
    high_low_indicator = [1 if price > median_price else 0 for price in data_with_noise]

    # Add climatic indicator (e.g., sunny, rainy, cloudy)
    climatic_conditions = ['Sunny', 'Rainy', 'Cloudy']
    climatic_indicator = np.random.choice(climatic_conditions, num_points)

    # Add political indicator (0 for stable, 1 for unstable)
    political_indicator = np.random.choice([0, 1], num_points, p=[0.8, 0.2])

    # Adjust price based on climatic and political indicators
    for i in range(num_points):
        if climatic_indicator[i] == 'Rainy':
            data_with_noise[i] -= 0.1  # Decrease price slightly on rainy days
        if political_indicator[i] == 1:
            data_with_noise[i] -= 0.2  # Decrease price on politically unstable days

    # Introduce permanent market changes
    market_change_date = np.random.choice(dates)
    market_change_index = dates.index(market_change_date)
    permanent_change = np.random.choice([-0.5, 0.5])  # Permanent decrease or increase
    data_with_noise[market_change_index:] += permanent_change

    # Map climatic conditions to numeric values
    climatic_mapping = {'Sunny': 0, 'Rainy': 1, 'Cloudy': 2}
    climatic_indicator_numeric = [climatic_mapping[condition] for condition in climatic_indicator]

    # Create a DataFrame with the correct column names
    df = pd.DataFrame({
        'Date': formatted_dates,
        'Price': data_with_noise,
        'Weekday': weekday_indicator,
        'High_Low': high_low_indicator,
        'Climatic': climatic_indicator_numeric,
        'Political': political_indicator
    }) 
    
    return df

# Example usage
df = generate_synthetic_data(num_points=3*365, seed=42)
print(df.head())

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Apply a rolling average to smooth the Price data
window_size = 5  # You can adjust this window size
df['Price'] = df['Price'].rolling(window=window_size, min_periods=1).median()

# Set the style of the plot
sns.set_style("whitegrid")

# Create a line plot for Smoothed Price over Date
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Price', label='Price')

# Set the x-axis to show only the year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Add title and labels
plt.title('Synthetic Financial Data: Smoothed Price Over Time')
plt.xlabel('Date')
plt.ylabel('Smoothed Price')

# Show the plot
plt.tight_layout()
plt.show()