 # ==============================
# GREEN FINANCE EMPIRICAL ANALYSIS (FINAL VERSION)
# ==============================
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. PARAMETERS (EASY TO MODIFY)
# ------------------------------

TICKERS = {
    "ESG": "ESGU",           # ESG ETF
    "Traditional": "SPY",    # S&P 500
    "Green": "ICLN"          # Clean Energy
}

START_DATE = "2015-01-01"

# ------------------------------
# 2. DOWNLOAD DATA (SOURCE)
# ------------------------------

# Data source: Yahoo Finance market data accessed via the yfinance API.
print("Downloading data from Yahoo Finance...")
data = yf.download(list(TICKERS.values()), start=START_DATE, auto_adjust=False)["Adj Close"]

# Rename columns using ticker symbols (robust to Yahoo column order)
ticker_to_name = {ticker: name for name, ticker in TICKERS.items()}
data = data.rename(columns=ticker_to_name)
data = data[list(TICKERS.keys())]

# Ticker coverage audit to validate data quality
coverage_rows = []
for asset in data.columns:
    series = data[asset]
    coverage_rows.append({
        "Asset": asset,
        "Ticker": TICKERS[asset],
        "First Valid Date": series.first_valid_index().date() if series.first_valid_index() is not None else None,
        "Last Valid Date": series.last_valid_index().date() if series.last_valid_index() is not None else None,
        "Missing Observations": int(series.isna().sum()),
        "Total Observations": int(series.shape[0])
    })

coverage_df = pd.DataFrame(coverage_rows)
print("\n=== TICKER DATA COVERAGE CHECK ===")
print(coverage_df)

# Save raw data (PROOF FOR TEACHER)
data.to_csv("raw_price_data.csv")

# ------------------------------
# 3. RETURNS
# ------------------------------

daily_returns = data.pct_change(fill_method=None).dropna()
monthly_returns = data.resample('ME').last().pct_change(fill_method=None).dropna()

# Save returns
daily_returns.to_csv("daily_returns.csv")
monthly_returns.to_csv("monthly_returns.csv")

# ------------------------------
# 4. ADVANCED METRICS
# ------------------------------

def performance_metrics(returns, freq=252):
    ann_return = returns.mean() * freq
    volatility = returns.std() * np.sqrt(freq)
    sharpe = ann_return / volatility
    
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Max Drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = drawdown.min()
    
    return pd.DataFrame({
        "Return": ann_return,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Skewness": skewness,
        "Kurtosis": kurtosis
    })

daily_perf = performance_metrics(daily_returns, 252)
monthly_perf = performance_metrics(monthly_returns, 12)

# Save tables
daily_perf.to_csv("daily_performance.csv")
monthly_perf.to_csv("monthly_performance.csv")

# ------------------------------
# 5. ALPHA & BETA (VS MARKET)
# ------------------------------

market = daily_returns["Traditional"]

def alpha_beta(asset, market):
    cov = np.cov(asset, market)[0][1]
    beta = cov / np.var(market)
    alpha = asset.mean()*252 - beta * (market.mean()*252)
    return alpha, beta

alpha_beta_results = {}

for col in daily_returns.columns:
    if col != "Traditional":
        alpha, beta = alpha_beta(daily_returns[col], market)
        alpha_beta_results[col] = {"Alpha": alpha, "Beta": beta}

alpha_beta_df = pd.DataFrame(alpha_beta_results).T
alpha_beta_df.to_csv("alpha_beta.csv")

# ------------------------------
# 6. SUB-PERIOD ANALYSIS
# ------------------------------

periods = {
    "Pre-COVID": daily_returns["2015-01-01":"2019-12-31"],
    "COVID": daily_returns["2020-01-01":"2020-12-31"],
    "Post-COVID": daily_returns["2021-01-01":]
}

for name, dataset in periods.items():
    perf = performance_metrics(dataset)
    perf.to_csv(f"{name}_performance.csv")

# ------------------------------
# 7. CUMULATIVE RETURNS (GRAPH)
# ------------------------------

cum_returns = (1 + daily_returns).cumprod()

plt.figure()
cum_returns.plot(title="Cumulative Returns")
plt.savefig("cumulative_returns.png")
plt.close()

# ------------------------------
# 8. DRAWDOWN (GRAPH)
# ------------------------------

rolling_max = cum_returns.cummax()
drawdown = cum_returns / rolling_max - 1

plt.figure()
drawdown.plot(title="Drawdown")
plt.savefig("drawdown.png")
plt.close()

# ------------------------------
# 9. SHARPE RATIO (GRAPH)
# ------------------------------

plt.figure()
daily_perf["Sharpe"].plot(kind="bar", title="Sharpe Ratios")
plt.savefig("sharpe.png")
plt.close()

# ------------------------------
# 10. SUMMARY OUTPUT
# ------------------------------

print("\n=== DAILY PERFORMANCE ===")
print(daily_perf)

print("\n=== MONTHLY PERFORMANCE ===")
print(monthly_perf)

print("\n=== ALPHA & BETA ===")
print(alpha_beta_df)

print("\nAll outputs saved to project folder.")

# --- NEW SECTION 11: OUTPUT DIRECTORY SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NEW SECTION 12: DATA TRANSPARENCY & METADATA ---
print("\n=== DATA TRANSPARENCY ===")
print("Source: Yahoo Finance (via yfinance)")
print(f"Tickers: {list(TICKERS.values())}")
print(f"Start date: {START_DATE}")
print(f"Observations (price rows): {len(data)}")
print(f"Sample period: {data.index.min().date()} to {data.index.max().date()}")

with open(os.path.join(OUTPUT_DIR, "data_summary.txt"), "w", encoding="utf-8") as f:
    f.write("Data source: Yahoo Finance via yfinance\n")
    f.write(f"Tickers used: {', '.join(list(TICKERS.values()))}\n")
    f.write(f"Named assets: {', '.join(list(TICKERS.keys()))}\n")
    f.write(f"Sample period: {data.index.min().date()} to {data.index.max().date()}\n")
    f.write("Frequency: Daily prices; Monthly returns from month-end resampling (ME)\n")
    f.write(f"Number of daily observations: {len(data)}\n")

# Save existing core outputs into outputs folder (in addition to your original saves).
data.to_csv(os.path.join(OUTPUT_DIR, "raw_price_data.csv"))
daily_returns.to_csv(os.path.join(OUTPUT_DIR, "daily_returns.csv"))
monthly_returns.to_csv(os.path.join(OUTPUT_DIR, "monthly_returns.csv"))
daily_perf.to_csv(os.path.join(OUTPUT_DIR, "daily_performance.csv"))
monthly_perf.to_csv(os.path.join(OUTPUT_DIR, "monthly_performance.csv"))
alpha_beta_df.to_csv(os.path.join(OUTPUT_DIR, "alpha_beta.csv"))
coverage_df.to_csv(os.path.join(OUTPUT_DIR, "ticker_coverage.csv"), index=False)

for name, dataset in periods.items():
    perf = performance_metrics(dataset)
    perf.to_csv(os.path.join(OUTPUT_DIR, f"{name}_performance.csv"))

# --- NEW SECTION 13: DESCRIPTIVE STATISTICS (DAILY RETURNS) ---
descriptive_stats = pd.DataFrame({
    "mean": daily_returns.mean(),
    "std": daily_returns.std(),
    "min": daily_returns.min(),
    "max": daily_returns.max(),
    "skewness": daily_returns.skew(),
    "kurtosis": daily_returns.kurtosis()
})
descriptive_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

# --- NEW SECTION 14: CORRELATION MATRIX + HEATMAP ---
correlation_matrix = daily_returns.corr()
correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(correlation_matrix.values, cmap="YlGnBu", vmin=-1, vmax=1)
ax.set_title("Correlation Matrix of Daily Returns")
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.index)))
ax.set_xticklabels(correlation_matrix.columns)
ax.set_yticklabels(correlation_matrix.index)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

fig.colorbar(heatmap, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close(fig)

# --- NEW SECTION 15: ROLLING 30-DAY ANALYSIS ---
rolling_window = 30
rolling_volatility = daily_returns.rolling(rolling_window).std() * np.sqrt(252)
rolling_sharpe = (daily_returns.rolling(rolling_window).mean() / daily_returns.rolling(rolling_window).std()) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(10, 6))
rolling_sharpe.plot(ax=ax)
ax.set_title("Rolling 30-Day Sharpe Ratio")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe Ratio (Annualized)")
ax.legend(title="Asset")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rolling_sharpe.png"), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
rolling_volatility.plot(ax=ax)
ax.set_title("Rolling 30-Day Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (Annualized)")
ax.legend(title="Asset")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rolling_volatility.png"), dpi=300)
plt.close(fig)

# --- NEW SECTION 16: SUMMARY TABLE FOR REPORT ---
summary_table = daily_perf[["Return", "Volatility", "Sharpe", "Max Drawdown"]].copy()
summary_table = summary_table.round(4)
summary_table.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"))

# --- NEW SECTION 17: DAILY VS MONTHLY COMPARISON ---
frequency_comparison = pd.DataFrame({
    "Daily Sharpe": daily_perf["Sharpe"],
    "Monthly Sharpe": monthly_perf["Sharpe"],
    "Daily Volatility": daily_perf["Volatility"],
    "Monthly Volatility": monthly_perf["Volatility"]
}).round(4)
frequency_comparison.to_csv(os.path.join(OUTPUT_DIR, "frequency_comparison.csv"))

# --- NEW SECTION 18: PROFESSIONAL GRAPH EXPORTS ---
fig, ax = plt.subplots(figsize=(10, 6))
cum_returns.plot(ax=ax)
ax.set_title("Cumulative Returns: ESG vs Traditional vs Green")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend(title="Asset")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns.png"), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
drawdown.plot(ax=ax)
ax.set_title("Drawdown Comparison")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown")
ax.legend(title="Asset")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "drawdown.png"), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
daily_perf["Sharpe"].plot(kind="bar", ax=ax, color=["#2E8B57", "#1F4E79", "#66A61E"])
ax.set_title("Sharpe Ratios (Daily)")
ax.set_xlabel("Asset")
ax.set_ylabel("Sharpe Ratio")
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sharpe.png"), dpi=300)
plt.close(fig)

# --- NEW SECTION 19: SHORT DATA-DRIVEN INTERPRETATION ---
best_asset = summary_table["Sharpe"].idxmax()
best_sharpe = summary_table.loc[best_asset, "Sharpe"]
worst_asset = summary_table["Sharpe"].idxmin()
worst_sharpe = summary_table.loc[worst_asset, "Sharpe"]

print("\n=== INTERPRETATION ===")
print(
    f"{best_asset} shows the strongest risk-adjusted performance (Sharpe={best_sharpe:.2f}), "
    f"while {worst_asset} is lower (Sharpe={worst_sharpe:.2f})."
)
print(f"\nProfessional outputs exported to: {OUTPUT_DIR}")
