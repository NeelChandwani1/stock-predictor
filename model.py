import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Prepare data for training
def prepare_data(stock_data):
    stock_data['Prediction'] = stock_data['Close'].shift(-30)  # Predict 30 days into the future
    X = np.array(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']][:-30])
    y = np.array(stock_data['Prediction'][:-30])
    return train_test_split(X, y, test_size=0.2)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Predict future prices
def predict(model, X_test):
    return model.predict(X_test)

if __name__ == "__main__":
    # Example: Predict Apple stock prices
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_model(X_train, y_train)
    predictions = predict(model, X_test)

    print("Predictions:", predictions)
