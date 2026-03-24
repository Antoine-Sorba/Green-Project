# Green Finance Empirical Analysis

Empirical finance analysis comparing an ESG ETF, a traditional market benchmark, and a clean energy ETF using Yahoo Finance data accessed through `yfinance`.

## Overview

This project analyzes three ETFs:

- `ESGU` as the ESG proxy
- `SPY` as the traditional market benchmark
- `ICLN` as the green energy proxy

The script computes and exports:

- raw adjusted price data
- daily and monthly returns
- performance metrics
- alpha and beta versus the market benchmark
- descriptive statistics
- correlation matrix and heatmap
- rolling 30-day Sharpe ratio and volatility
- summary tables and graphs for report use

## Data Source

All market data is downloaded from Yahoo Finance via the `yfinance` Python package.

## Important Data Note

`ESGU` begins later than the other ETFs in the sample. The script includes a ticker coverage audit to document first valid dates and missing observations.

## Repository Structure

- `green_finance_analysis.py`: main analysis script
- `requirements.txt`: Python dependencies
- `outputs/`: generated CSV files and figures created when the script runs

## Installation

Use Python 3.9 or newer.

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 green_finance_analysis.py
```

## Generated Outputs

The script creates an `outputs/` directory and exports analysis artifacts such as:

- `raw_price_data.csv`
- `daily_returns.csv`
- `monthly_returns.csv`
- `summary_table.csv`
- `ticker_coverage.csv`
- `correlation_heatmap.png`
- `rolling_sharpe.png`
- `rolling_volatility.png`

## Methodological Notes

- Prices are based on Yahoo Finance adjusted close data.
- Daily returns are computed with `pct_change(fill_method=None)`.
- Monthly returns are computed from month-end resampled prices using `resample('ME').last()`.
- Performance measures include annualized return, annualized volatility, Sharpe ratio, and maximum drawdown.

## GitHub Publishing Recommendation

The `outputs/` folder is ignored by default because it contains generated artifacts. Re-run the script locally to regenerate the analysis outputs.