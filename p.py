import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- PART 1: DATA PREPARATION AND NORMALIZATION ---
# This section creates structured CSV files for Power BI.

# Load the dataset from the uploaded archive.
# Note: In Google Colab, you would need to adjust the path to your file.
# Example for Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# file_path = '/content/drive/My Drive/Sample - Superstore.csv'
file_path = 'Sample - Superstore.csv'  # Assumes the file is in the same directory.

try:
    # Read the file assuming no header, and use the 'latin1' encoding to fix the previous error.
    df_raw = pd.read_csv(file_path, encoding='latin1', header=None)

    # Manually define the correct column names based on the original dataset structure.
    # This step ensures all dataframes created from df_raw will have correct headers.
    column_names = [
        'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID',
        'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code',
        'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
        'Sales', 'Quantity', 'Discount', 'Profit'
    ]

    # Assign the correct column names to the DataFrame.
    df_raw.columns = column_names

    # Drop the first row since it contains the original headers.
    df_raw = df_raw.iloc[1:].copy()

    print("Raw data loaded successfully with corrected column names and header row dropped.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check your file path.")
    exit()

# 1. Create the 'Orders' fact table
# This table contains the core transactional data.
orders_df = df_raw[['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID', 'Product ID', 'Sales', 'Quantity', 'Discount', 'Profit']].copy()
orders_df.drop_duplicates(inplace=True)

# Convert the relevant columns to the correct data types.
# This fixes the new ValueError by converting potential non-numeric values to NaN.
orders_df['Sales'] = pd.to_numeric(orders_df['Sales'], errors='coerce')
orders_df['Profit'] = pd.to_numeric(orders_df['Profit'], errors='coerce')
orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'])
orders_df['Ship Date'] = pd.to_datetime(orders_df['Ship Date'])

# Remove any rows where 'Sales' or 'Profit' could not be converted (i.e., contain NaN)
orders_df.dropna(subset=['Sales', 'Profit'], inplace=True)

# Save the 'orders' table to a new CSV file.
orders_df.to_csv('superstore_orders.csv', index=False)
print("Created 'superstore_orders.csv' with transactional data.")

# 2. Create the 'Products' dimension table
# This table contains unique product information.
products_df = df_raw[['Product ID', 'Category', 'Sub-Category', 'Product Name']].drop_duplicates()
products_df.to_csv('superstore_products.csv', index=False)
print("Created 'superstore_products.csv' with product details.")

# 3. Create the 'Customers' dimension table
# This table contains unique customer and location information.
customers_df = df_raw[['Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region']].drop_duplicates()
customers_df.to_csv('superstore_customers.csv', index=False)
print("Created 'superstore_customers.csv' with customer and region data.")

# 4. Create the 'Dates' dimension table
# This table is crucial for time intelligence in Power BI.
dates_df = pd.DataFrame(orders_df['Order Date'].unique(), columns=['Date'])
dates_df['Date'] = pd.to_datetime(dates_df['Date'])
dates_df['Year'] = dates_df['Date'].dt.year
dates_df['MonthNumber'] = dates_df['Date'].dt.month
dates_df['MonthName'] = dates_df['Date'].dt.strftime('%B')
dates_df['Quarter'] = dates_df['Date'].dt.quarter
dates_df.to_csv('superstore_dates.csv', index=False)
print("Created 'superstore_dates.csv' for time-based analysis.")

print("\nAll data preparation CSV files have been created.")

# --- PART 2: SALES FORECASTING WITH PROPHET ---
# This section prepares data for forecasting and generates a new CSV with predictions.

# Prepare data for Prophet: Prophet requires 'ds' (date) and 'y' (value) columns.
df_prophet = orders_df.copy()
df_prophet = df_prophet.groupby('Order Date')['Sales'].sum().reset_index()
df_prophet.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)

# Initialize and fit the Prophet model.
# Note: You may need to install Prophet first: pip install prophet
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(df_prophet)

# Create a future dataframe for 12 months.
future = model.make_future_dataframe(periods=12, freq='MS')

# Make the forecast.
forecast = model.predict(future)

# Save the forecast to a new CSV file.
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('superstore_forecast.csv', index=False)
print("\nCreated 'superstore_forecast.csv' with sales predictions.")

# Optional: Plot the forecast to visualize the results.
fig1 = model.plot(forecast)
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

print("\nProcess completed successfully.")
