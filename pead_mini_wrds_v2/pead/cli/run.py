# pead_mini_wrds_v2/pead/cli/run.py
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pead_mini_wrds_v2.pead.data.wrds_io import load_wrds_events_returns_mcap


# ----------------------------
# Helpers
# ----------------------------

def _week_monday(d):
    """Return the Monday of the week for each date (naive dates)."""
    d = pd.to_datetime(d, errors="coerce")
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def _cap_bucket(mcap):
    """Assign market cap bucket (nan -> UNK)."""
    if pd.isna(mcap):
        return "UNK"
    if mcap >= 10_000_000_000:
        return "LARGE"
    if mcap >= 2_000_000_000:
        return "MID"
    if mcap > 0:
        return "SMALL"
    return "UNK"


def _assign_deciles_within_group(g: pd.DataFrame, n_deciles: int) -> pd.DataFrame:
    """Assign deciles robustly within group g using rank -> qcut, fallback to rank bins."""
    s = g['sue'].astype(float)
    valid = s.dropna()
    if len(valid) < 2:
        g['decile'] = np.nan
        return g
    n_unique = min(n_deciles, valid.nunique())
    try:
        q = pd.qcut(valid.rank(method='first'), q=n_unique, labels=range(1, n_unique + 1), duplicates='drop')
        # If qcut produced fewer bins than expected, fallback
        if pd.Series(q).nunique() < n_unique:
            raise ValueError("insufficient unique bins")
        g.loc[valid.index, 'decile'] = q.astype(int)
    except Exception:
        rk = valid.rank(pct=True, method='first')
        dec = np.ceil(rk * n_deciles).clip(1, n_deciles).astype(int)
        g['decile'] = np.nan
        g.loc[valid.index, 'decile'] = dec
    g['decile'] = g['decile'].astype('float')
    return g


def _build_weight_matrices(deciles: pd.DataFrame,
                           returns: pd.DataFrame,
                           hold_days: int,
                           top_decile: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long & short weight matrices. Start on next trading day to avoid same-day look-ahead."""
    dates = returns.index
    tickers = returns.columns
    w_long = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_short = pd.DataFrame(0.0, index=dates, columns=tickers)

    dec = deciles.dropna(subset=['decile']).copy()
    dec['decile'] = dec['decile'].astype(int)

    for _, row in dec.iterrows():
        d, t, k = row['date'], row['ticker'], int(row['decile'])
        try:
            d0 = pd.to_datetime(d)
        except Exception:
            continue
        if (t not in tickers) or (d0 not in dates):
            continue
        i0 = dates.get_loc(d0) + 1  # start next trading day
        i1 = min(i0 + hold_days, len(dates))
        if i0 >= i1:
            continue
        idx = dates[i0:i1]
        if k == top_decile:
            w_long.loc[idx, t] += 1.0
        elif k == 1:
            w_short.loc[idx, t] += 1.0

    # normalize by number of active names each day
    long_gross = (w_long > 0).sum(axis=1).replace(0, np.nan)
    short_gross = (w_short > 0).sum(axis=1).replace(0, np.nan)
    w_long = w_long.div(long_gross, axis=0).fillna(0.0)
    w_short = w_short.div(short_gross, axis=0).fillna(0.0)
    return w_long, w_short


def _apply_costs_and_pnl(w_long: pd.DataFrame,
                         w_short: pd.DataFrame,
                         returns: pd.DataFrame,
                         commission_bps: float,
                         slippage_bps: float,
                         borrow_bps: float) -> tuple[pd.Series, pd.Series]:
    """
    Compute P&L and turnover:
      - pnl_long = (w_long * returns).sum(axis=1)
      - pnl_short = -(w_short * returns).sum(axis=1)
      - turnover = sum(abs(diff(weights))) per day (both sides)
      - trade_cost = turnover * (commission+slippage)/1e4
      - borrow_cost = short_exposure * borrow_bps/1e4 / 252
      - ls_daily = pnl_long + pnl_short - trade_cost - borrow_cost
    """
    pnl_long = (w_long * returns).sum(axis=1)
    pnl_short = -(w_short * returns).sum(axis=1)

    dw_long = w_long.diff().abs().sum(axis=1).fillna(0.0)
    dw_short = w_short.diff().abs().sum(axis=1).fillna(0.0)
    turnover = dw_long + dw_short

    trade_bps = commission_bps + slippage_bps
    trade_cost = turnover * (trade_bps / 1e4)
    borrow_cost = w_short.sum(axis=1) * (borrow_bps / 1e4) / 252.0

    ls_daily = pnl_long + pnl_short - trade_cost - borrow_cost
    return ls_daily.fillna(0.0), turnover.fillna(0.0)


# ----------------------------
# Plotting
# ----------------------------

def plot_cumulative(ls_path_series: pd.Series, ls_by_cap: dict[str, pd.Series], plot_log: bool, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    if plot_log:
        y = np.log(ls_path_series.replace(0, np.nan))
        ylabel = "Log Cumulative"
    else:
        y = ls_path_series - 1.0
        ylabel = "Cumulative Return"
    plt.plot(y.index, y.values)
    plt.title("Long–Short Cumulative Return" + (" (log)" if plot_log else ""))
    plt.xlabel("Date"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outdir / ("ls_cum_log.png" if plot_log else "ls_cum.png"), dpi=150); plt.close()

    plt.figure()
    for cap, series in ls_by_cap.items():
        if plot_log:
            s = np.log(series.replace(0, np.nan))
        else:
            s = series - 1.0
        plt.plot(s.index, s.values, label=cap)
    plt.legend()
    plt.title("Long–Short Cumulative by Cap Bucket" + (" (log)" if plot_log else ""))
    plt.xlabel("Date"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outdir / ("ls_cum_by_cap_log.png" if plot_log else "ls_cum_by_cap.png"), dpi=150); plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="PEAD v2 — grouped by (week × cap) with costs/turnover")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hold_days", type=int, default=30)
    ap.add_argument("--n_deciles", type=int, default=10)
    # universe_source argument retained but only 'nasdaq_by_date' is supported now
    ap.add_argument("--universe_source", type=str, default="nasdaq_by_date",
                    help="Only 'nasdaq_by_date' is supported (nasdaq_2020 has been removed).")
    ap.add_argument("--universe", type=str, default=None)
    ap.add_argument("--commission_bps", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--borrow_bps", type=float, default=0.0)
    ap.add_argument("--clean_outputs", action="store_true")
    ap.add_argument("--plot_log", action="store_true")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parents[3] / "outputs"
    if args.clean_outputs:
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(exist_ok=True)

    # Enforce supported universe_source
    if args.universe_source != "nasdaq_by_date":
        raise ValueError("Only 'nasdaq_by_date' is supported for --universe_source. The 'nasdaq_2020' option has been removed.")

    # Load data
    events, returns, mcap, mkt = load_wrds_events_returns_mcap(
        args.start, args.end,
        universe_source=args.universe_source,
        universe=args.universe,
    )
    print(f"Loaded {len(events)} events across {events['ticker'].nunique()} tickers; returns shape {returns.shape}")

    # prev-day mcap joined to events
    mcap_prev = mcap.shift(1)

    def _lookup_mcap(r):
        d, t = r['date'], r['ticker']
        try:
            d0 = pd.to_datetime(d)
        except Exception:
            return np.nan
        if t in mcap_prev.columns:
            if d0 in mcap_prev.index:
                val = mcap_prev.loc[d0, t]
                return np.nan if pd.isna(val) else float(val)
            pos = mcap_prev.index.searchsorted(d0)
            if pos <= 0:
                return np.nan
            val = mcap_prev.iloc[pos - 1].get(t, np.nan)
            return np.nan if pd.isna(val) else float(val)
        return np.nan

    ev = events.copy()
    ev['mcap_prev'] = ev.apply(_lookup_mcap, axis=1)
    ev['cap_bucket'] = ev['mcap_prev'].apply(_cap_bucket)
    ev['week_monday'] = _week_monday(ev['date'])

    # Deciles within (week_monday × cap_bucket). Keep group columns.
    ev = (
        ev.sort_values(['week_monday', 'cap_bucket', 'ticker'])
          .groupby(['week_monday', 'cap_bucket'], group_keys=False)
          .apply(lambda g: _assign_deciles_within_group(g, args.n_deciles))
          .reset_index(drop=True)
    )

    deciles = ev[['date', 'ticker', 'sue', 'decile', 'cap_bucket', 'week_monday']].copy()

    # Build portfolios
    w_long, w_short = _build_weight_matrices(deciles, returns, args.hold_days, top_decile=args.n_deciles)
    ls_daily, turnover = _apply_costs_and_pnl(
        w_long, w_short, returns,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        borrow_bps=args.borrow_bps,
    )

    # stats: convert to numpy floats
    mkt_aligned = mkt.reindex(returns.index).fillna(0.0).astype(float)
    X = np.asarray(mkt_aligned.values, dtype='float64')
    Y = np.asarray(ls_daily.fillna(0.0).values, dtype='float64')

    X_ = np.vstack([np.ones_like(X), X]).T  # [1, MKT]
    beta_hat, *_ = np.linalg.lstsq(X_, Y, rcond=None)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])

    resid = Y - X_ @ beta_hat
    t_alpha = alpha / (np.std(resid, ddof=1) / np.sqrt(len(resid))) if len(resid) > 2 else np.nan
    t_beta  = beta  / (np.std(X, ddof=1) / np.sqrt(len(X))) if len(X) > 2 else np.nan
    denom = np.var(Y, ddof=0) * len(Y)
    r2 = float(1.0 - (resid @ resid) / (denom + 1e-12))
    sharpe = float(np.sqrt(252.0) * (np.mean(Y) / (np.std(Y, ddof=1) + 1e-12)))

    # IC on 30D forward returns (compute forward total return using daily compounding)
    def _fwd(d, t, k=30):
        if (t not in returns.columns) or (pd.to_datetime(d) not in returns.index):
            return np.nan
        i = returns.index.get_loc(pd.to_datetime(d)) + 1
        j = min(i + k, len(returns.index))
        if i >= j:
            return np.nan
        path = returns.iloc[i:j][t].astype(float)
        if path.isna().any():
            return np.nan
        return float(np.prod(1.0 + path.values) - 1.0)

    deciles['fwd30'] = [ _fwd(d, t, 30) for d, t in zip(deciles['date'], deciles['ticker']) ]
    deciles['month'] = pd.to_datetime(deciles['date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
    ic_vals = []
    for _, dfm in deciles.dropna(subset=['sue', 'fwd30']).groupby('month'):
        if dfm['sue'].nunique() >= 5 and dfm['fwd30'].nunique() >= 5:
            ic_vals.append(pd.Series(dfm['sue']).corr(pd.Series(dfm['fwd30']), method='spearman'))
    ic_mean = float(np.nanmean(ic_vals)) if ic_vals else np.nan

    summary = {
        "ic_spearman_mean": ic_mean,
        "n_obs": int(len(returns.index)),
        "ls_alpha": float(alpha),
        "ls_beta": float(beta),
        "ls_tstat_alpha": float(t_alpha),
        "ls_tstat_beta": float(t_beta),
        "ls_r2": float(r2),
        "sharpe_annualized": float(sharpe),
        "filters": {}
    }
    (outdir / "pead_summary.json").write_text(json.dumps(summary, indent=2))

    long_gross = w_long.sum(axis=1)
    short_gross = w_short.sum(axis=1)
    turnover_report = {
        "avg_daily_turnover": float(turnover.mean()),
        "avg_monthly_turnover": float(turnover.resample("ME").mean().mean()),
        "max_daily_turnover": float(turnover.max()),
        "avg_long_gross": float(long_gross.mean()),
        "avg_short_gross": float(short_gross.mean()),
    }
    (outdir / "turnover.json").write_text(json.dumps(turnover_report, indent=2))

    # Save plots (use compounding for cumulative P&L)
    ls_path = (1.0 + pd.Series(Y, index=returns.index)).cumprod()
    # Also compute per-cap L/S by building per-cap portfolios (like original, not simple mean of returns)
    ls_by_cap = {}
    for cap in ["SMALL", "MID", "LARGE"]:
        sel = deciles[(deciles['cap_bucket'] == cap) & deciles['decile'].notna()]
        if sel.empty:
            continue
        wl, ws = _build_weight_matrices(sel, returns, args.hold_days, top_decile=args.n_deciles)
        pnl_cap, _ = _apply_costs_and_pnl(wl, ws, returns, args.commission_bps, args.slippage_bps, args.borrow_bps)
        path_cap = (1.0 + pnl_cap.fillna(0.0)).cumprod()
        ls_by_cap[cap] = path_cap

    plot_cumulative(ls_path, ls_by_cap, args.plot_log, outdir)

    print(f"Saved outputs to {outdir}/: pead_summary.json, ls_cum.png, ls_cum_by_cap.png, turnover.json")
    print("\n==== PEAD SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print("\n==== TURNOVER REPORT ====")
    print(json.dumps(turnover_report, indent=2))


if __name__ == "__main__":
    main()
