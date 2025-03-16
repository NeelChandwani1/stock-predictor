from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Function to get stock data and make predictions
def get_stock_predictions(ticker):
    # Fetch stock data
    stock_data = yf.download(ticker, start="2010-01-01", end="2025-01-01")

    # Add a new column for the 'next day's close price'
    stock_data['Next Close'] = stock_data['Close'].shift(-1)

    # Drop the last row since the next close value is missing
    stock_data.dropna(inplace=True)

    # Features and target variable
    X = stock_data[['Close']]
    y = stock_data['Next Close']

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, y, label='Actual', color='blue')
    plt.plot(stock_data.index, predictions, label='Predicted', color='red')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Save the plot to a BytesIO object and encode it in base64 for embedding in the HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if ticker:
            plot_url = get_stock_predictions(ticker)
            return render_template('index.html', plot_url=plot_url, ticker=ticker)
    return render_template('index.html', plot_url=None)

if __name__ == "__main__":
    app.run(debug=True)
