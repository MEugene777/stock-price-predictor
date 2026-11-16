import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load API credentials from your .env file
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca API keys! Check your .env file.")

# 1. Fetch Historical Data
SYMBOL = "NVDA"
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 11, 1)

print(f"Fetching historical data for {SYMBOL}...")
client = StockHistoricalDataClient(API_KEY, API_SECRET)
request_params = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE
)

bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()

# 2. Feature Engineering
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(5).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

# Target: Next trading day's closing price
df["target"] = df["close"].shift(-1)
df = df.dropna()

features = ["close", "volume", "volatility", "ma_5", "ma_10", "ma_20"]
X = df[features]
y = df["target"]

# 3. Chronological Train/Test Split (80/20)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Train & Evaluate Models
print("Training models...\n")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest MAE:      ${mean_absolute_error(y_test, rf_preds):.2f}")

gb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
print(f"Gradient Boosting MAE:  ${mean_absolute_error(y_test, gb_preds):.2f}")

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print(f"Linear Regression MAE:  ${mean_absolute_error(y_test, lr_preds):.2f}")

# 5. Plot Comparison
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label="Actual Price", linewidth=2, color="black")
plt.plot(rf_preds, label="Random Forest", alpha=0.8)
plt.plot(gb_preds, label="Gradient Boosting", alpha=0.8)
plt.plot(lr_preds, label="Linear Regression", alpha=0.8)

plt.title(f"{SYMBOL} Next-Day Price Prediction Model Comparison")
plt.xlabel("Trading Days (Test Set)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
print("\nChart saved as model_comparison.png!")
plt.show()