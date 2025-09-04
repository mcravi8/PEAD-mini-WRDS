from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _safe_float_series(x) -> pd.Series:
    s = pd.Series(x, copy=False)
    # Try hard to get to float64
    try:
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        s = s.astype("float64", errors="ignore")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.astype("float64")

def summarize_metrics(deciles: pd.DataFrame,
                      ls_daily: pd.Series,
                      returns: pd.DataFrame) -> dict:
    """
    - deciles: events with columns ['date','ticker','sue','fwd_ret','decile'] (date as Timestamp)
    - ls_daily: Series of daily long-short returns (not cumulative)
    - returns: wide DataFrame with market column 'MKT' (same date index as ls_daily or superset)
    """
    # ---------- Information Coefficient (cross-sectional Spearman) ----------
    ic_vals = []
    cols = [c for c in ["date", "sue", "fwd_ret"] if c in deciles.columns]
    if set(cols) == {"date", "sue", "fwd_ret"}:
        # work per-day; need variability on both sides
        for d, g in deciles[cols].dropna().groupby("date"):
            if g["sue"].nunique() < 2 or g["fwd_ret"].nunique() < 2 or len(g) < 3:
                continue
            # Spearman via ranking then Pearson
            ic = g["sue"].rank().corr(g["fwd_ret"].rank(), method="pearson")
            ic_vals.append(ic)
    ic_mean = float(np.nanmean(ic_vals)) if len(ic_vals) else np.nan

    # ---------- Long/Short alpha/beta vs market ----------
    y = _safe_float_series(ls_daily).dropna()
    n_obs = int(y.shape[0])

    if "MKT" in returns.columns:
        m = _safe_float_series(returns["MKT"]).reindex(y.index)
    else:
        # If market isn't present, regress on a zero series (beta=0, alpha=mean)
        m = pd.Series(0.0, index=y.index, dtype="float64")

    df_reg = pd.DataFrame({"y": y, "m": m}).dropna()
    if len(df_reg) >= 5:
        X = sm.add_constant(df_reg["m"].astype("float64"))
        model = sm.OLS(df_reg["y"].astype("float64"), X).fit()
        alpha = float(model.params.get("const", np.nan))
        beta  = float(model.params.get("m", np.nan))
        t_a   = float(model.tvalues.get("const", np.nan))
        t_b   = float(model.tvalues.get("m", np.nan))
        r2    = float(model.rsquared)
    else:
        alpha = beta = t_a = t_b = r2 = np.nan

    # ---------- Sharpe (annualized) ----------
    # crude frequency guess
    ann_scale = np.sqrt(252 if n_obs > 200 else (52 if n_obs > 40 else 12))
    mu = float(y.mean())
    sd = float(y.std(ddof=1))
    sharpe = float(mu / sd * ann_scale) if sd > 0 and np.isfinite(sd) else np.nan

    return {
        "ic_spearman_mean": ic_mean,
        "n_obs": n_obs,
        "ls_alpha": alpha,
        "ls_beta": beta,
        "ls_tstat_alpha": t_a,
        "ls_tstat_beta": t_b,
        "ls_r2": r2,
        "sharpe_annualized": sharpe,
    }
