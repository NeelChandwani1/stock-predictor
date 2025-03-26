from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import io
import base64
import talib  # Technical Analysis Library

app = Flask(__name__)

# Enhanced data preprocessing
def preprocess_data(ticker):
    # Get extended historical data
    stock_data = yf.download(ticker, start="2000-01-01")
    
    # Add technical indicators
    stock_data['50_MA'] = stock_data['Close'].rolling(50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(200).mean()
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
    macd, signal, _ = talib.MACD(stock_data['Close'])
    stock_data['MACD'] = macd - signal
    
    # Add fundamental data
    try:
        fundamentals = yf.Ticker(ticker).info
        stock_data['PE_Ratio'] = fundamentals.get('trailingPE', np.nan)
        stock_data['Profit_Margin'] = fundamentals.get('profitMargins', np.nan)
    except:
        pass
    
    # Clean data
    stock_data.dropna(inplace=True)
    return stock_data

# LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_stock_predictions(ticker):
    # Get enhanced data
    stock_data = preprocess_data(ticker)
    
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(stock_data[['Close','50_MA','RSI','MACD']])
    
    # Create sequences
    X, y = [], []
    sequence_length = 60
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Close price is first column
        
    X, y = np.array(X), np.array(y)
    
    # Split data (80% train, 20% test)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    test_predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((len(predictions), scaled_data.shape[1]-1))), axis=1)
    )[:,0]
    
    # Prepare plot
    plt.figure(figsize=(14,7))
    actual_prices = stock_data['Close'][-len(y_test):]
    plt.plot(actual_prices.index, actual_prices, label='Actual Price', color='blue')
    plt.plot(actual_prices.index, test_predictions, label='Predicted Price', color='red', linestyle='--')
    
    # Add confidence interval
    confidence = np.std(test_predictions - actual_prices.values)
    plt.fill_between(actual_prices.index, 
                    test_predictions - confidence, 
                    test_predictions + confidence, 
                    color='orange', alpha=0.3)
    
    plt.title(f'{ticker} Price Prediction (LSTM Model)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Calculate model metrics
    mse = np.mean((test_predictions - actual_prices.values)**2)
    accuracy = max(0, 100 * (1 - mse/np.var(actual_prices.values)))
    
    # Save plot
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return plot_url, accuracy, mse, test_predictions[-1]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()
        try:
            plot_url, accuracy, mse, last_pred = get_stock_predictions(ticker)
            return render_template('index.html', 
                                 plot_url=plot_url,
                                 ticker=ticker,
                                 accuracy=f"{accuracy:.1f}",
                                 mse=f"{mse:.2f}",
                                 prediction=f"{last_pred:.2f}")
        except Exception as e:
            error = f"Error processing {ticker}: {str(e)}"
            return render_template('index.html', error=error)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)