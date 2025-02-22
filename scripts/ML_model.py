import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os, sys
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath('..'))
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    # Linear interpolation for missing values
    df['Price'] = df['Price'].interpolate(method='linear') 

    return df

# Preparing data for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

def input_output(df, seq_length):
    X, y = create_sequences(df['Price'].values.reshape(-1,1), seq_length)
    # Convert data to float32 for TensorFlow
    X, y = X.astype(np.float32), y.astype(np.float32)
    return X, y


# Train the model on CUDA if available
def train_model(seq_length, X,y):
    # Define LSTM Model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(100),
        Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    with tf.device("/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"):
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Save model
    model.save("../data/lstm_model.h5")
    print("Model Trained and Saved Succussfuly")
    return model

# Predictions and Evaluation
def model_evaluation(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")

    # Save evaluation metrics
    metrics = {"RMSE": rmse, "MAE": mae, "R-squared": r2}
    joblib.dump(metrics, "../data/stm_model_metrics.pkl")
