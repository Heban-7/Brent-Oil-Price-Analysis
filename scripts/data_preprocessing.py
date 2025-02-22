import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

# Load dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
    return df

# Summary statistics
def summary_statistics(df):
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

    print("\n\nData information")
    print(df.info())

    print('\n\nCheck for missing value')
    print(df.isnull().sum())

# Check skewness and kurtosis
def sckewness_kurtosis(df):
    print("\nSkewness:", skew(df['Price']))
    print("Kurtosis:", kurtosis(df['Price']))

# Stationarity test
def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print("\nADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])



# Visualizing price trends
def price_trend(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label="Brent Oil Price", color='blue')
    plt.title("Brent Oil Price Over Time")
    plt.xlabel("Year")
    plt.ylabel("Price (USD per barrel)")
    plt.legend()
    plt.show()

# Rolling mean and standard deviation
def rolling_mean_std(df):
    df['Rolling_Mean'] = df['Price'].rolling(window=10).mean()
    df['Rolling_Std'] = df['Price'].rolling(window=10).std()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Price'], label='Original', color='blue')
    plt.plot(df['Rolling_Mean'], label='Rolling Mean', color='red')
    plt.plot(df['Rolling_Std'], label='Rolling Std', color='green')
    plt.title("Rolling Mean & Standard Deviation")
    plt.legend()
    plt.show()

# Distribution plot
def distribution_plot(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price'].dropna(), bins=20, kde=True, color='purple')
    plt.title("Price Distribution")
    plt.show()

# Box plot to detect outliers
def box_plot(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df['Price'], color='cyan')
    plt.title("Boxplot of Brent Oil Prices")
    plt.show()
