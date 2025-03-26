import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ticker = "AAPL"
stock_data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

stock_data['Next Close'] = stock_data['Close'].shift(-1)

stock_data.dropna(inplace=True)

X = stock_data[['Close']] 
y = stock_data['Next Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy: ", model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
