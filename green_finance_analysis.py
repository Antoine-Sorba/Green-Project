# ==============================
# GREEN FINANCE EMPIRICAL ANALYSIS (FINAL VERSION)
# ==============================
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


plt.style.use("seaborn-v0_8-whitegrid")


# ------------------------------
# 1. PARAMETERS (EASY TO MODIFY)
# ------------------------------

TICKERS = {
    "ESG": "ESGU",           # ESG ETF
    "Traditional": "SPY",    # S&P 500
    "Green": "ICLN"          # Clean Energy
}

START_DATE = "2015-01-01"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ROLLING_WINDOW = 30


def save_root_and_output_csv(dataframe, filename, index=True):
    dataframe.to_csv(BASE_DIR / filename, index=index)
    dataframe.to_csv(OUTPUT_DIR / filename, index=index)


def performance_metrics(returns, freq=252):
    ann_return = returns.mean() * freq
    volatility = returns.std() * np.sqrt(freq)
    sharpe = ann_return / volatility

    skewness = returns.skew()
    kurtosis = returns.kurtosis()

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


def alpha_beta(asset, market):
    cov = np.cov(asset, market)[0][1]
    beta = cov / np.var(market)
    alpha = asset.mean() * 252 - beta * (market.mean() * 252)
    return alpha, beta


def download_price_data():
    # Data source: Yahoo Finance market data accessed via the yfinance API.
    print("Downloading data from Yahoo Finance...")
    raw_data = yf.download(list(TICKERS.values()), start=START_DATE, auto_adjust=False)["Adj Close"]

    ticker_to_name = {ticker: name for name, ticker in TICKERS.items()}
    raw_data = raw_data.rename(columns=ticker_to_name)
    raw_data = raw_data[list(TICKERS.keys())]
    return raw_data


def build_coverage_table(price_data):
    coverage_rows = []

    for asset in price_data.columns:
        series = price_data[asset]
        coverage_rows.append({
            "Asset": asset,
            "Ticker": TICKERS[asset],
            "First Valid Date": series.first_valid_index().date() if series.first_valid_index() is not None else None,
            "Last Valid Date": series.last_valid_index().date() if series.last_valid_index() is not None else None,
            "Missing Observations": int(series.isna().sum()),
            "Total Observations": int(series.shape[0])
        })

    coverage_table = pd.DataFrame(coverage_rows)
    print("\n=== TICKER DATA COVERAGE CHECK ===")
    print(coverage_table)
    return coverage_table


def compute_returns(price_data):
    daily_returns = price_data.pct_change(fill_method=None).dropna()
    monthly_returns = price_data.resample("ME").last().pct_change(fill_method=None).dropna()
    return daily_returns, monthly_returns


def export_data_summary(price_data):
    print("\n=== DATA TRANSPARENCY ===")
    print("Source: Yahoo Finance (via yfinance)")
    print(f"Tickers: {list(TICKERS.values())}")
    print(f"Start date: {START_DATE}")
    print(f"Observations (price rows): {len(price_data)}")
    print(f"Sample period: {price_data.index.min().date()} to {price_data.index.max().date()}")

    with open(OUTPUT_DIR / "data_summary.txt", "w", encoding="utf-8") as file:
        file.write("Data source: Yahoo Finance via yfinance\n")
        file.write(f"Tickers used: {', '.join(list(TICKERS.values()))}\n")
        file.write(f"Named assets: {', '.join(list(TICKERS.keys()))}\n")
        file.write(f"Sample period: {price_data.index.min().date()} to {price_data.index.max().date()}\n")
        file.write("Frequency: Daily prices; Monthly returns from month-end resampling (ME)\n")
        file.write(f"Number of daily observations: {len(price_data)}\n")


def build_alpha_beta_table(daily_returns):
    market = daily_returns["Traditional"]
    alpha_beta_results = {}

    for col in daily_returns.columns:
        if col != "Traditional":
            alpha, beta = alpha_beta(daily_returns[col], market)
            alpha_beta_results[col] = {"Alpha": alpha, "Beta": beta}

    return pd.DataFrame(alpha_beta_results).T


def build_periods(daily_returns):
    return {
        "Pre-COVID": daily_returns["2015-01-01":"2019-12-31"],
        "COVID": daily_returns["2020-01-01":"2020-12-31"],
        "Post-COVID": daily_returns["2021-01-01":]
    }


def export_period_tables(periods):
    for name, dataset in periods.items():
        period_performance = performance_metrics(dataset)
        period_performance.to_csv(BASE_DIR / f"{name}_performance.csv")
        period_performance.to_csv(OUTPUT_DIR / f"{name}_performance.csv")


def build_descriptive_statistics(daily_returns):
    return pd.DataFrame({
        "mean": daily_returns.mean(),
        "std": daily_returns.std(),
        "min": daily_returns.min(),
        "max": daily_returns.max(),
        "skewness": daily_returns.skew(),
        "kurtosis": daily_returns.kurtosis()
    })


def save_line_plot(dataframe, title, ylabel, filename, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    dataframe.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(title="Asset")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / filename), dpi=300)
    plt.close(fig)


def save_bar_plot(series, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax, color=["#2E8B57", "#1F4E79", "#66A61E"])
    ax.set_title(title)
    ax.set_xlabel("Asset")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / filename), dpi=300)
    plt.close(fig)


def save_correlation_heatmap(correlation_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(correlation_matrix.values, cmap="YlGnBu", vmin=-1, vmax=1)

    ax.set_title("Correlation Matrix of Daily Returns")
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for row_index in range(correlation_matrix.shape[0]):
        for column_index in range(correlation_matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{correlation_matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(heatmap, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "correlation_heatmap.png"), dpi=300)
    plt.close(fig)


def print_summary(daily_perf, monthly_perf, alpha_beta_df, summary_table):
    print("\n=== DAILY PERFORMANCE ===")
    print(daily_perf)

    print("\n=== MONTHLY PERFORMANCE ===")
    print(monthly_perf)

    print("\n=== ALPHA & BETA ===")
    print(alpha_beta_df)

    print("\nAll outputs saved to project folder.")

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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    price_data = download_price_data()
    coverage_df = build_coverage_table(price_data)

    # Save raw data (PROOF FOR TEACHER)
    save_root_and_output_csv(price_data, "raw_price_data.csv")

    daily_returns, monthly_returns = compute_returns(price_data)
    save_root_and_output_csv(daily_returns, "daily_returns.csv")
    save_root_and_output_csv(monthly_returns, "monthly_returns.csv")

    daily_perf = performance_metrics(daily_returns, 252)
    monthly_perf = performance_metrics(monthly_returns, 12)
    save_root_and_output_csv(daily_perf, "daily_performance.csv")
    save_root_and_output_csv(monthly_perf, "monthly_performance.csv")

    alpha_beta_df = build_alpha_beta_table(daily_returns)
    save_root_and_output_csv(alpha_beta_df, "alpha_beta.csv")
    coverage_df.to_csv(OUTPUT_DIR / "ticker_coverage.csv", index=False)

    periods = build_periods(daily_returns)
    export_period_tables(periods)

    descriptive_stats = build_descriptive_statistics(daily_returns)
    descriptive_stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    correlation_matrix = daily_returns.corr()
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    rolling_volatility = daily_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    rolling_sharpe = (
        daily_returns.rolling(ROLLING_WINDOW).mean()
        / daily_returns.rolling(ROLLING_WINDOW).std()
    ) * np.sqrt(252)

    summary_table = daily_perf[["Return", "Volatility", "Sharpe", "Max Drawdown"]].copy().round(4)
    summary_table.to_csv(OUTPUT_DIR / "summary_table.csv")

    frequency_comparison = pd.DataFrame({
        "Daily Sharpe": daily_perf["Sharpe"],
        "Monthly Sharpe": monthly_perf["Sharpe"],
        "Daily Volatility": daily_perf["Volatility"],
        "Monthly Volatility": monthly_perf["Volatility"]
    }).round(4)
    frequency_comparison.to_csv(OUTPUT_DIR / "frequency_comparison.csv")

    cum_returns = (1 + daily_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = cum_returns / rolling_max - 1

    # Preserve the original project-level exports while also producing polished output copies.
    plt.figure()
    cum_returns.plot(title="Cumulative Returns")
    plt.savefig(str(BASE_DIR / "cumulative_returns.png"))
    plt.close()

    plt.figure()
    drawdown.plot(title="Drawdown")
    plt.savefig(str(BASE_DIR / "drawdown.png"))
    plt.close()

    plt.figure()
    daily_perf["Sharpe"].plot(kind="bar", title="Sharpe Ratios")
    plt.savefig(str(BASE_DIR / "sharpe.png"))
    plt.close()

    export_data_summary(price_data)
    save_correlation_heatmap(correlation_matrix)
    save_line_plot(rolling_sharpe, "Rolling 30-Day Sharpe Ratio", "Sharpe Ratio (Annualized)", "rolling_sharpe.png")
    save_line_plot(rolling_volatility, "Rolling 30-Day Volatility", "Volatility (Annualized)", "rolling_volatility.png")
    save_line_plot(cum_returns, "Cumulative Returns: ESG vs Traditional vs Green", "Growth of $1", "cumulative_returns.png")
    save_line_plot(drawdown, "Drawdown Comparison", "Drawdown", "drawdown.png")
    save_bar_plot(daily_perf["Sharpe"], "Sharpe Ratios (Daily)", "Sharpe Ratio", "sharpe.png")

    print_summary(daily_perf, monthly_perf, alpha_beta_df, summary_table)


if __name__ == "__main__":
    main()
