from __future__ import annotations
import numpy as np
import pandas as pd


def run_pead_backtest(
    deciles: pd.DataFrame,
    returns: pd.DataFrame,
    hold_days: int = 20,
    n_deciles: int = 10,
):
    """
    Long the top decile (== n_deciles), short the bottom (== 1), equal-weight.
    Build a rolling portfolio over 'hold_days' trading days from the NEXT day after the event.
    Returns:
      perf: dict with counts/meta
      ls_daily: pd.Series of daily long-short returns (not cumulative)
      dec_curves: {} (placeholder)
    """
    idx = returns.index
    n = len(idx)

    long_sum = np.zeros(n, dtype=float)
    long_cnt = np.zeros(n, dtype=float)
    short_sum = np.zeros(n, dtype=float)
    short_cnt = np.zeros(n, dtype=float)

    # Ensure clean types
    dec = deciles.dropna(subset=['decile']).copy()
    if dec.empty:
        ls_daily = pd.Series(np.nan, index=idx)
        perf = {"n_events": 0, "n_long_events": 0, "n_short_events": 0, "hold_days": int(hold_days)}
        return perf, ls_daily, {}

    for _, row in dec.iterrows():
        d = pd.to_datetime(row['date'])
        t = row['ticker']
        dc = row['decile']

        if pd.isna(dc) or t not in returns.columns:
            continue

        # start from NEXT trading day after event
        pos = idx.searchsorted(d, side='left')
        if pos < len(idx) and idx[pos] <= d:
            pos += 1
        start = pos
        end = min(start + hold_days, n)
        if start >= n:
            continue

        r = returns.iloc[start:end][t].astype(float).values
        if len(r) == 0:
            continue

        if dc == n_deciles:  # long
            long_sum[start:end] += np.nan_to_num(r, nan=0.0)
            long_cnt[start:end] += ~np.isnan(r)
        elif dc == 1:        # short (subtract)
            short_sum[start:end] += np.nan_to_num(r, nan=0.0)
            short_cnt[start:end] += ~np.isnan(r)

    # Mean by side
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_long = np.divide(long_sum, np.where(long_cnt == 0, np.nan, long_cnt))
        mean_short = np.divide(short_sum, np.where(short_cnt == 0, np.nan, short_cnt))

    ls_daily = pd.Series(mean_long - mean_short, index=idx)
    ls_daily = ls_daily.replace([np.inf, -np.inf], np.nan)

    perf = {
        "n_events": int(dec['decile'].notna().sum()),
        "n_long_events": int((dec['decile'] == n_deciles).sum()),
        "n_short_events": int((dec['decile'] == 1).sum()),
        "hold_days": int(hold_days),
    }
    return perf, ls_daily, {}
