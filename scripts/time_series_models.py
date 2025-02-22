import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
import pymc3 as pm
import arviz as az
import ruptures as rpt
from pmdarima import auto_arima
import numpy as np
import joblib 
import os, sys

sys.path.append(os.path.abspath('..'))
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    # Linear interpolation for missing values
    df['Price'] = df['Price'].interpolate(method='linear') 

    return df

def visualize_price(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Brent Oil Prices Over Time')
    plt.legend()
    plt.show()

def cusum(df):
    mean_price = df['Price'].mean()
    cusum = np.cumsum(df['Price'] - mean_price)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, cusum, label='CUSUM')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('CUSUM Value')
    plt.title('CUSUM Analysis')
    plt.legend()
    plt.show()

# Bayesian Change Point Detection 
def bayesian_changepoint_detection(series):
    mean_price = series['Price'].mean()
    with pm.Model() as model:
        # Priors
        mean_prior = pm.Normal('mean_prior', mu=mean_price, sigma=10)
        change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(series)-1)

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mean_prior, sigma=10, observed=series['Price'])

        # Inference
        trace = pm.sample(1000, tune=1000, cores=2)

    joblib.dump(trace, '../data/bayesian_changepoint_trace.pkl')  # Save the Bayesian model trace
    return trace


# ARIMA Model with Auto-Selection of Parameters
def train_arima(series):
    optimal_order = auto_arima(series, seasonal=False, stepwise=True, trace=True).order
    model = sm.tsa.ARIMA(series, order=optimal_order)
    results = model.fit()
    joblib.dump(results, '../data/arima_model.pkl')  # Save the ARIMA model
    return results.summary()

# GARCH Model with Optimized Parameters
def train_garch(series):
    model = arch_model(series, vol='Garch', p=1, q=1, mean='Zero', dist='StudentsT')
    results = model.fit(update_freq=5)
    joblib.dump(results, '../data/garch_model.pkl')  # Save the GARCH model
    return results.summary()


