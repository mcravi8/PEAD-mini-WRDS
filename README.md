# PEAD Mini (WRDS)  

Minimal post-earnings announcement drift (PEAD) backtest using WRDS (IBES + CRSP).  
Pulls IBES estimates & actuals, constructs SUE, forms SUE deciles, builds long/short D10−D1 portfolios, applies simple transaction & borrow costs, and saves metrics + plots.

## Quick Summary

This repository contains a lightweight research pipeline that:

- Loads IBES EPS estimates & actuals and CRSP daily returns via WRDS.
- Computes SUE per announcement: `SUE = (actual − mean_est) / std_est`.
- Ranks events into deciles (grouped by week × market-cap bucket).
- Builds equal-weight long (top decile) / short (bottom decile) portfolios starting T+1 for `hold_days`.
- Applies transaction costs (commission + slippage) and borrow costs for shorts.
- Computes IC (Spearman), CAPM stats (alpha, beta, t-stats, R² via statsmodels), Sharpe, and turnover.
- Writes `outputs/pead_summary.json`, `outputs/turnover.json`, and PNG plots.

## Files of Interest

- `pead_mini_wrds_v2/pead/cli/run.py` — Main CLI entry point (run the backtest)
- `pead_mini_wrds_v2/pead/data/wrds_io.py` — WRDS SQL extraction & cleaning
- `outputs/` — Generated JSON + PNG results (if present)
- `.gitignore` — Suggested to prevent leaking secrets / env files

## Requirements

- Python 3.8+ (tested with 3.10)
- WRDS account (username + password); `wrds` Python package
- Recommended packages (install in virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have `requirements.txt`, at minimum install:

```bash
pip install pandas numpy matplotlib wrds statsmodels psycopg2-binary
```

## Setup & WRDS Access

1. Set WRDS username (export or environment):

```bash
export WRDS_USERNAME="your_wrds_username"
```

2. The first run will prompt to create a `~/.pgpass` file to avoid entering credentials repeatedly. Alternatively, create `~/.pgpass` yourself (DO NOT commit it).
3. Confirm WRDS connectivity (network / VPN if required by your institution).

## Run the Pipeline

From project root (example):

```bash
source .venv/bin/activate
python -m pead_mini_wrds_v2.pead.cli.run \
  --start 1990-01-01 --end 2020-12-31 \
  --hold_days 30 \
  --n_deciles 10 \
  --universe_source nasdaq_by_date \
  --commission_bps 10 \
  --slippage_bps 10 \
  --borrow_bps 50 \
  --clean_outputs \
  --plot_log
```

### Key Flags

- `--start` / `--end` — Date range (inclusive)
- `--hold_days` — Holding horizon in trading days (portfolio held from T+1)
- `--n_deciles` — Number of SUE buckets (default 10)
- `--universe_source` — Currently `nasdaq_by_date` supported (date-valid NASDAQ membership)
- `--commission_bps`, `--slippage_bps`, `--borrow_bps` — In basis points
- `--clean_outputs` — Remove prior files in `outputs/`
- `--plot_log` — Plot log cumulative P&L

## Outputs

After a successful run, you will find (in `outputs/`):

- `pead_summary.json` — JSON with main metrics:
  - `ic_spearman_mean` (monthly average Spearman IC)
  - `n_obs` (number of daily return observations)
  - `ls_alpha`, `ls_beta`, `ls_tstat_alpha`, `ls_tstat_beta`, `ls_r2`
  - `sharpe_annualized`
- `turnover.json` — Turnover & gross exposure summary:
  - `avg_daily_turnover`, `avg_monthly_turnover`, `max_daily_turnover`
  - `avg_long_gross`, `avg_short_gross`
- Plots:
  - `ls_cum.png` / `ls_cum_log.png` — Overall LS cumulative P&L (level or log)
  - `ls_cum_by_cap.png` / `ls_cum_by_cap_log.png` — LS P&L by cap bucket (SMALL/MID/LARGE)
  - `decile_perf.png` — Average forward return by decile

### How to Read JSON from Python

```python
import json
with open("outputs/pead_summary.json", "r") as f:
    summary = json.load(f)
with open("outputs/turnover.json", "r") as f:
    turnover = json.load(f)
print(summary)
print(turnover)
```

## Notes About Correctness & Realism

- **Look-ahead**: The backtest begins on T+1 trading day after the announcement (no same-day look-ahead).
- **Costs**: Costs are computed daily from portfolio turnover: `(commission + slippage) * turnover`. Borrow cost applied to short exposure divided by 252 (annual → daily).
- **Regression**: OLS and t-stats are computed via `statsmodels` (robust and correct). Newey-West/HAC option is available if needed.
- **Survivorship**: If a static universe is used (e.g., a single-date NASDAQ snapshot), survivorship bias may occur. The pipeline supports `nasdaq_by_date` to mitigate this (date-valid membership).

### Limitations

- Ticker mapping relies on CRSP name history (`dsenames`) — suitable for demo; production would benefit from IBES↔CRSP link tables.
- Delisting returns (`dlret`) may not be fully incorporated, which could affect exit handling.
- Announcement times are day-level (no intraday timestamps).

## Findings & Comments

### Summary of Numeric Results

**Metrics** (`pead_summary.json`):
```json
{
  "ic_spearman_mean": 0.01831909711214039,
  "n_obs": 7812,
  "ls_alpha": -0.00047724273093889763,
  "ls_beta": 0.037001754017027794,
  "ls_tstat_alpha": -1.919981598050007,
  "ls_tstat_beta": 1.6815179342204407,
  "ls_r2": 0.00036190515689416003,
  "sharpe_annualized": -0.3331873656061828
}
```

**Turnover** (`turnover.json`):
```json
{
  "avg_daily_turnover": 0.3233022777174975,
  "avg_monthly_turnover": 0.3236148943790858,
  "max_daily_turnover": 3.0277777777777777,
  "avg_long_gross": 1.1353440147338274,
  "avg_short_gross": 1.1492362445083324
}
```

### Plain-English Interpretation

- **IC (Spearman) = 0.018**  
  The average cross-sectional signal is small and positive, showing a weak rank correlation between SUE and 30-day forward returns across months. The signal’s predictive strength appears limited.

- **Observations (n_obs) = 7812**  
  This represents daily return observations in the pipeline. Separating the count of earnings announcements used for IC calculation would add clarity, as these may differ.

- **CAPM Regression**  
  - `ls_alpha ≈ -0.00048` (daily): The portfolio has a slightly negative raw alpha (~−0.048% per day, unannualized).  
  - `ls_beta ≈ 0.037`: The portfolio is nearly market-neutral with low market exposure.  
  - `ls_tstat_alpha ≈ −1.92`: The alpha is marginally significant, close to the 5% threshold but not robust.  
  - `ls_tstat_beta ≈ 1.68`: The beta lacks strong significance.  
  - `ls_r2 ≈ 0.00036`: The market explains almost no variation, with returns mostly idiosyncratic.  

- **Sharpe ≈ −0.33 (annualized)**  
  The negative Sharpe ratio indicates the cost-adjusted long-short portfolio produced negative risk-adjusted returns, consistent with the negative alpha.

- **Turnover ≈ 0.323 (daily)**  
  Average daily turnover (~32% of the portfolio) is high, suggesting significant trading activity. The `max_daily_turnover ≈ 3.03` shows extreme rebalancing days (>300% gross traded), likely due to clustered earnings announcements. High turnover increases trading costs and operational complexity.

- **Gross Exposures ≈ 1.13 (long) / 1.15 (short)**  
  These reflect mild leverage/gross exposures. Specifying whether weights are normalized per side or represent gross notional would enhance clarity.

### Visuals — What They Show and How to Read Them

- **Overall LS Cumulative (Log)**  
  <img src="https://raw.githubusercontent.com/mcravi8/PEAD-mini-WRDS/main/outputs/ls_cum_log.png" alt="Overall LS Cumulative Log" width="75%">  
  The log cumulative plot shows a downward trend, indicating the cost-adjusted long-short portfolio lost value over time. The log scale highlights multiplicative drawdowns and long-term trends, consistent with the negative Sharpe and alpha.

- **LS Cumulative by Cap Bucket (Log)**  
  <img src="https://raw.githubusercontent.com/mcravi8/PEAD-mini-WRDS/main/outputs/ls_cum_by_cap_log.png" alt="LS Cumulative by Cap Bucket Log" width="75%">  
  Performance differs across market caps: SMALL and LARGE buckets decline more steadily, while MID shows a flatter or less negative path. This suggests varying SUE signal effectiveness by cap bucket. The charts represent per-cap portfolios (long top-decile, short bottom-decile within each cap), not simple mean returns of tickers in each group.

### Potential Enhancements

- **High Turnover**: The ~32% daily turnover is significant. Exploring weekly rebalancing, minimum holding windows, or grouping announcements within a week might reduce spikes from clustered earnings.

- **Cost Sensitivity**: The cost model uses `(commission + slippage) * turnover` + daily borrow cost. Testing higher cost assumptions (e.g., >10 bps commission + slippage) could clarify P&L sensitivity.

- **Weak IC**: The IC of 0.018 is near zero. Plotting monthly IC time series or histograms, or using bootstrap methods, might reveal whether the signal is consistently positive or driven by outliers.

- **Regression Robustness**: The `statsmodels` OLS implementation is reliable. Applying Newey-West (HAC) standard errors with `maxlags=10` or `int(4*(n_obs/100)**(2/9))` could account for autocorrelation in daily returns.

- **Delisting/Survivorship**: Ensuring CRSP delisting returns (`dlret`) are fully incorporated would prevent potential bias from treating exits as zero-return survivors.

- **Event Alignment**: The T+1 weight builder avoids look-ahead bias. Verifying the handling of multiple events for the same ticker within the holding window could prevent double-allocating or overwriting positions.

- **Cap Bucket Assignment**: Confirming that `mcap_prev` uses the previous trading day’s market cap and that SMALL/MID/LARGE thresholds align with the design would ensure consistency.
