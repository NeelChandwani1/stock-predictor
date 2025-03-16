import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch stock data (e.g., Apple)
ticker = "AAPL"  # You can change this to any stock symbol
stock_data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

# Add a new column for the 'next day's close price'
stock_data['Next Close'] = stock_data['Close'].shift(-1)

# Drop the last row since the next close value is missing
stock_data.dropna(inplace=True)

# Features and target variable
X = stock_data[['Close']]  # You can add more features like moving averages, volume, etc.
y = stock_data['Next Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Model Accuracy: ", model.score(X_test, y_test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
