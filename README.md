# Green Finance Empirical Analysis

Empirical asset management-style analysis comparing an ESG ETF, a traditional benchmark ETF, and a clean energy ETF using Python and Yahoo Finance data.

## Project Snapshot

This repository studies three investment universes:

- `ESGU` as the ESG proxy
- `SPY` as the traditional market benchmark
- `ICLN` as the clean energy proxy

The code produces publication-ready tables and figures for a university empirical finance report, including:

- raw adjusted price series
- daily and monthly returns
- annualized performance metrics
- alpha and beta relative to the benchmark
- descriptive statistics and correlation analysis
- rolling 30-day Sharpe ratio and rolling volatility
- export-ready CSV tables and chart files

## Why This Repository Is Useful

- The workflow is transparent: data source, sample period, and ticker coverage are explicitly documented.
- The script exports clean files that can be inserted directly into a report or presentation.
- The analysis is reproducible with a single Python script and a short dependency list.

## Data Source

All price data is downloaded from Yahoo Finance through `yfinance`.

## Important Data Caveat

`ESGU` starts later than `SPY` and `ICLN`, so the early sample contains missing values for the ESG series. This is expected and documented in the exported ticker coverage audit.

## Repository Layout

```text
green_finance_project/
├── green_finance_analysis.py
├── README.md
├── requirements.txt
└── outputs/
```

## Installation

Use Python 3.9 or newer.

```bash
python3 -m pip install -r requirements.txt
```

## Run The Analysis

```bash
python3 green_finance_analysis.py
```

## Main Outputs

Running the script creates an `outputs/` folder containing files such as:

- `raw_price_data.csv`
- `daily_returns.csv`
- `monthly_returns.csv`
- `ticker_coverage.csv`
- `descriptive_statistics.csv`
- `correlation_matrix.csv`
- `summary_table.csv`
- `frequency_comparison.csv`
- `correlation_heatmap.png`
- `rolling_sharpe.png`
- `rolling_volatility.png`

## Methodology Summary

- Adjusted close prices are used as the core price series.
- Daily returns are computed with `pct_change(fill_method=None)`.
- Monthly returns are computed from month-end prices with `resample("ME").last()`.
- Performance metrics include annualized return, annualized volatility, Sharpe ratio, maximum drawdown, skewness, and kurtosis.
- Additional diagnostics include a benchmark-relative alpha/beta table, descriptive statistics, and rolling risk indicators.

## Research Positioning

This repository is structured as a compact empirical finance project rather than a software package. The emphasis is on readable code, transparent methodology, and exportable outputs suitable for academic or professional reporting.

## Notes For GitHub Visitors

The `outputs/` directory is ignored in version control because it contains generated artifacts. Re-run the script locally to regenerate all tables and figures.