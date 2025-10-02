from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from io import BytesIO
import base64
from datetime import date
import os

app = Flask(__name__)

# ----------------------
# Global storage per ticker
# ----------------------
trained_model = None
trained_scaler = None
trained_ticker = None

# ----------------------
# Helper: matplotlib -> base64
# ----------------------
def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return "data:image/png;base64," + img_base64

# ----------------------
# Build LSTM
# ----------------------
def build_lstm_model(input_shape, num_layers=2, units_per_layer=None):
    if units_per_layer is None:
        units_per_layer = [50] * num_layers
    model = Sequential()
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        if i == 0:
            model.add(LSTM(units_per_layer[i], activation='sigmoid',
                           return_sequences=return_sequences,
                           input_shape=input_shape))
        else:
            model.add(LSTM(units_per_layer[i], activation='sigmoid',
                           return_sequences=return_sequences))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------------------
# Routes
# ----------------------
@app.route('/')
def forecast_page():
    today = date.today().isoformat()
    return render_template("forecast.html", today=today)

# ----------------------
# Run Analysis (train model)
# ----------------------
@app.route('/api/train')
def api_train():
    global trained_model, trained_scaler, trained_ticker

    start_date = request.args.get('start_date', '2008-01-01')
    end_date = request.args.get('end_date', date.today().isoformat())
    ticker = request.args.get('ticker', 'TATASTEEL.NS').upper()
    num_layers = int(request.args.get('num_layers', 2))
    units = [int(u) for u in request.args.get('units', '50,50').split(',')]
    epochs = int(request.args.get('epochs', 50))

    # Download data
    nifty_data = yf.download("^NSEI", start=start_date, end=end_date)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    common_dates = nifty_data.index.intersection(stock_data.index)
    stock_data = stock_data.loc[common_dates]
    stock_data['nifty_pct_change'] = nifty_data['Close'].pct_change()

    data = stock_data[["Open","High","Low","Close","nifty_pct_change"]].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = scaled_data[:, [0, 4]].reshape((scaled_data.shape[0],1,2))
    y = scaled_data[:,1:4]

    # Train model (no callbacks)
    model = build_lstm_model(input_shape=(1,2), num_layers=num_layers, units_per_layer=units)
    model.fit(X, y, epochs=epochs, batch_size=10, verbose=0)

    # Save globally
    trained_model = model
    trained_scaler = scaler
    trained_ticker = ticker

    # Predictions for charts & RMSE
    y_pred = model.predict(X)
    combined_pred = np.concatenate([X[:,0,:1], y_pred, X[:,0,1:]], axis=1)
    combined_actual = np.concatenate([X[:,0,:1], y, X[:,0,1:]], axis=1)
    y_pred_actual = scaler.inverse_transform(combined_pred)[:,1:4]
    y_actual = scaler.inverse_transform(combined_actual)[:,1:4]

    rmse_high = np.sqrt(mean_squared_error(y_actual[:,0], y_pred_actual[:,0]))
    rmse_low = np.sqrt(mean_squared_error(y_actual[:,1], y_pred_actual[:,1]))
    rmse_close = np.sqrt(mean_squared_error(y_actual[:,2], y_pred_actual[:,2]))

    # Forecast plot
    plt.figure(figsize=(10,5))
    plt.plot(y_actual[:,2], linestyle='dashed', color='blue', label='Actual Close')
    plt.plot(y_pred_actual[:,2], color='black', label='Predicted Close')
    plt.title(f"{ticker} Close Price Prediction")  # <-- added ticker name
    plt.legend()
    forecast_plot = plot_to_base64()

    # Residual plot
    resid = y_actual - y_pred_actual
    plt.figure(figsize=(8,4))
    plt.hist(resid[:,2], bins=50, alpha=0.6, color='blue')
    plt.title(f"{ticker} Residuals (Close)")  # <-- added ticker name
    residual_plot = plot_to_base64()

    return jsonify({
        "rmse_high": round(rmse_high,4),
        "rmse_low": round(rmse_low,4),
        "rmse_close": round(rmse_close,4),
        "forecast_plot": forecast_plot,
        "residual_plot": residual_plot
    })

# ----------------------
# Forecast (after model trained)
# ----------------------
@app.route('/api/forecast')
def api_forecast():
    global trained_model, trained_scaler, trained_ticker
    if trained_model is None:
        return jsonify({"error": "Please run analysis first"}), 400

    ticker = request.args.get('ticker', 'TATASTEEL.NS').upper()
    nifty_ch = float(request.args.get('nifty_ch', 0.33))/100

    if ticker != trained_ticker:
        return jsonify({"error": f"Model not trained for {ticker}. Please run analysis first."}), 400

    data_today = yf.Ticker(ticker).history(period="1d")
    if data_today.empty:
        return jsonify({"error": "No data available for today"}), 400

    open_price1 = data_today['Open'].iloc[0]
    input_array = np.array([[open_price1, 0,0,0, nifty_ch]])
    scaled_input = trained_scaler.transform(input_array)
    X_today = np.array([[scaled_input[0][0], scaled_input[0][4]]]).reshape(1,1,2)
    y_today_pred = trained_model.predict(X_today)
    combined_today = np.array([[scaled_input[0][0], y_today_pred[0][0], y_today_pred[0][1], y_today_pred[0][2], scaled_input[0][4]]])
    actual_pred = trained_scaler.inverse_transform(combined_today)[0][1:4]

    return jsonify({
        "today_open_text": f"Today's Open: {round(open_price1,2)}",
        "open": round(open_price1,2),
        "high": round(actual_pred[0],2),
        "low": round(actual_pred[1],2),
        "close": round(actual_pred[2],2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
    app.run(host="0.0.0.0", port=port)
