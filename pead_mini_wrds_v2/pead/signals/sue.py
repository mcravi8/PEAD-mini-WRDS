import pandas as pd


def assign_deciles_daily(events: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    For each DATE independently, rank SUE and assign deciles 1..n_deciles.
    If too few names on a day, decile is left as NA.
    """
    ev = events.copy()

    def by_day_qcut(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        # need at least n_deciles distinct slots; guard small groups
        if s.dropna().shape[0] < n_deciles:
            return pd.Series([None] * len(s), index=s.index)
        try:
            r = s.rank(method='first')
            d = pd.qcut(r, q=n_deciles, labels=range(1, n_deciles + 1), duplicates='drop')
            return d.astype('Int64')
        except Exception:
            return pd.Series([None] * len(s), index=s.index)

    ev['decile'] = ev.groupby('date', group_keys=False)['sue'].apply(by_day_qcut)
    return ev
