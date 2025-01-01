import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates

# Step 1: Load the data
def load_data(file_path):
    try:
        # Load the data into a pandas DataFrame
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Explore the data (basic information)
def explore_data(data):
    print("Data Information:")
    print(data.info())
    print("\nFirst few rows of the data:")
    print(data.head())
    print("\nDescriptive statistics:")
    print(data.describe())

# Step 3: Clean the data (focus on `TransactionNo` for modeling)
def clean_data(data):
    # Since we don't have `discounted_price`, let's focus on `TransactionNo` or a count of sales
    # Convert 'DateTime' to datetime format
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    # Ensure the 'TransactionNo' is numeric (it should already be)
    data['TransactionNo'] = pd.to_numeric(data['TransactionNo'], errors='coerce')
    print("\nData cleaned.")
    return data

# Step 5: Time-series forecasting using ARIMA
def time_series_forecast(data, time_column, target_column):
    """
    Forecasts future values using ARIMA for time-series data.
    Args:
        data (pd.DataFrame): The cleaned time-series data
        time_column (str): The column representing time
        target_column (str): The column to predict (target)
    """
    # Convert time column to datetime format (in case it hasn't been done yet)
    data[time_column] = pd.to_datetime(data[time_column])

    # Set time column as index
    data.set_index(time_column, inplace=True)

    # Aggregate data by day (sum of transactions per day)
    daily_data = data.groupby(data.index.date)[target_column].sum()

    # Visualize the original data
    plt.figure(figsize=(10, 6))
    plt.plot(daily_data.index, daily_data, label="Actual Data", color='blue')
    plt.title("Time Series - Actual Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Fit ARIMA model (Auto-Regressive Integrated Moving Average)
    model = ARIMA(daily_data, order=(5, 1, 0))  # (p, d, q) for ARIMA model
    model_fit = model.fit()

    # Print model summary
    print(model_fit.summary())

    # Forecast future values
    forecast_steps = 30  # Forecast the next 30 days (you can adjust this number)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a forecast index based on the last date in the data
    forecast_index = pd.date_range(daily_data.index[-1], periods=forecast_steps+1, freq='D')[1:]

    # Plot the actual data and forecasted data
    plt.figure(figsize=(10, 6))
    plt.plot(daily_data.index, daily_data, label="Actual Data", color='blue')
    plt.plot(forecast_index, forecast, label="Forecasted Data", color='red', linestyle='--')
    plt.title("Time Series Forecast - ARIMA")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # Format x-axis for better date display
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.show()

    # Return forecasted values for further analysis if needed
    return forecast, forecast_index

# Main function to run the steps
def main():
    file_path = "/Users/seerat/Downloads/archive/Bakery.csv"  # Replace with your actual file path

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Explore the data
    explore_data(data)

    # Clean the data
    cleaned_data = clean_data(data)

    # Time-series forecast
    time_column = "DateTime"  # Use the 'DateTime' column from your CSV
    target_column = "TransactionNo"  # Example target column for forecasting
    forecast, forecast_index = time_series_forecast(cleaned_data, time_column, target_column)

    # You can use forecast and forecast_index for further analysis, comparison, or visualization
    print(f"\nForecasted values for the next {len(forecast)} time periods:")
    for date, value in zip(forecast_index, forecast):
        print(f"{date}: {value}")

if __name__ == "__main__":
    main()
