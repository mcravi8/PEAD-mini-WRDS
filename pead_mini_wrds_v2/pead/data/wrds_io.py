# pead_mini_wrds_v2/pead/data/wrds_io.py
from __future__ import annotations
from typing import Tuple, Optional, List
import os
import pandas as pd


def _connect():
    import wrds
    return wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))


def _load_ibes_events(db, start: str, end: str) -> pd.DataFrame:
    sql_act = f"""
        select ticker, anndats::date as anndate, value::float8 as actual
        from ibes.actu_epsus
        where measure='EPS' and anndats between '{start}' and '{end}'
    """
    act = db.raw_sql(sql_act, date_cols=['anndate'])

    sql_det = f"""
        select ticker, anndats::date as anndate, value::float8 as est
        from ibes.det_epsus
        where measure='EPS' and anndats between '{start}' and '{end}'
    """
    det = db.raw_sql(sql_det, date_cols=['anndate'])

    cons = (
        det.groupby(['ticker', 'anndate'])
           .agg(meanest=('est', 'mean'), stdev=('est', 'std'))
           .reset_index()
    )
    cons.loc[cons['stdev'] <= 0, 'stdev'] = pd.NA

    ibes = cons.merge(act, on=['ticker', 'anndate'], how='inner')
    ibes['sue'] = (ibes['actual'] - ibes['meanest']) / ibes['stdev']

    events = (
        ibes[['anndate', 'ticker', 'sue']]
        .rename(columns={'anndate': 'date'})
        .dropna(subset=['date', 'ticker', 'sue'])
        .copy()
    )
    dts = pd.to_datetime(events['date'], errors='coerce', utc=True)
    events['date'] = dts.dt.tz_convert(None).dt.normalize()
    events['ticker'] = events['ticker'].astype(str).str.upper()
    return events


def _listing_windows(db, start: str, end: str) -> pd.DataFrame:
    sql = f"""
        select permno, ticker, exchcd,
               namedt::date as namedt,
               coalesce(nameendt, '9999-12-31'::date) as nameendt
        from crsp.dsenames
        where ticker is not null
          and namedt <= '{end}'
          and coalesce(nameendt, '9999-12-31'::date) >= '{start}'
    """
    df = db.raw_sql(sql, date_cols=['namedt', 'nameendt'])
    df['ticker'] = df['ticker'].astype(str).str.upper()
    return df


def _load_returns_daily(db, start: str, end: str, tickers_needed: Optional[List[str]]) -> pd.DataFrame:
    sql = f"""
        select d.date::date as date, d.permno, d.ret::float8 as ret
        from crsp.dsf d
        where d.date between '{start}' and '{end}'
    """
    dsf = db.raw_sql(sql, date_cols=['date'])
    names = _listing_windows(db, start, end)[['permno', 'ticker', 'namedt', 'nameendt']]
    crsp = dsf.merge(names, on='permno', how='left')
    crsp = crsp[(crsp['date'] >= crsp['namedt']) & (crsp['date'] <= crsp['nameendt'])]
    crsp = crsp.dropna(subset=['ticker'])
    crsp['ticker'] = crsp['ticker'].astype(str).str.upper()
    if tickers_needed:
        crsp = crsp[crsp['ticker'].isin(tickers_needed)]
    ret = pd.pivot_table(crsp, index='date', columns='ticker', values='ret', aggfunc='mean').sort_index()
    ret.index = pd.DatetimeIndex(pd.to_datetime(ret.index, utc=True).tz_convert(None).normalize())
    return ret


def _load_mkt_daily(db, start: str, end: str) -> pd.Series:
    sql = f"""
        select date::date as date, vwretd::float8 as mkt
        from crsp.dsi
        where date between '{start}' and '{end}'
    """
    mkt = db.raw_sql(sql, date_cols=['date']).set_index('date').sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(mkt.index, utc=True).tz_convert(None).normalize())
    return pd.Series(mkt['mkt'].values, index=idx, name='MKT')


def _load_mcap_daily(db, start: str, end: str, tickers_needed: Optional[List[str]]) -> pd.DataFrame:
    sql = f"""
        select d.date::date as date, d.permno,
               abs(d.prc)::float8 as prc, d.shrout::float8 as shrout
        from crsp.dsf d
        where d.date between '{start}' and '{end}'
    """
    dsf = db.raw_sql(sql, date_cols=['date'])
    dsf['mcap'] = dsf['prc'] * dsf['shrout'] * 1000.0
    names = _listing_windows(db, start, end)[['permno', 'ticker', 'namedt', 'nameendt']]
    crsp = dsf.merge(names, on='permno', how='left')
    crsp = crsp[(crsp['date'] >= crsp['namedt']) & (crsp['date'] <= crsp['nameendt'])]
    crsp = crsp.dropna(subset=['ticker'])
    crsp['ticker'] = crsp['ticker'].astype(str).str.upper()
    if tickers_needed:
        crsp = crsp[crsp['ticker'].isin(tickers_needed)]
    mc = pd.pivot_table(crsp, index='date', columns='ticker', values='mcap', aggfunc='mean').sort_index()
    mc.index = pd.DatetimeIndex(pd.to_datetime(mc.index, utc=True).tz_convert(None).normalize())
    return mc


def load_wrds_events_returns_mcap(
    start: str,
    end: str,
    universe_source: str = "nasdaq_by_date",
    universe: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    db = _connect()

    events = _load_ibes_events(db, start, end)

    if universe:
        keep = [t.strip().upper() for t in universe.split(",") if t.strip()]
        events = events[events['ticker'].isin(keep)]

    # Only 'nasdaq_by_date' supported now — remove 'nasdaq_2020' legacy branch.
    if universe_source == "nasdaq_by_date":
        names = _listing_windows(db, start, end)
        nas = names[names['exchcd'] == 3].copy()
        nas['namedt'] = pd.to_datetime(nas['namedt'], utc=True).dt.tz_convert(None).dt.normalize()
        nas['nameendt'] = pd.to_datetime(nas['nameendt'], utc=True).dt.tz_convert(None).dt.normalize()
        events = (
            events.merge(nas[['ticker', 'namedt', 'nameendt']], on='ticker', how='left')
                  .query("date >= namedt and date <= nameendt")
                  .drop(columns=['namedt', 'nameendt'])
        )
    else:
        # explicit error for unsupported source (avoids silent fallback)
        raise ValueError("universe_source must be 'nasdaq_by_date' (the 'nasdaq_2020' option has been removed).")

    tickers_needed = sorted(events['ticker'].unique().tolist())
    returns = _load_returns_daily(db, start, end, tickers_needed)
    mcap = _load_mcap_daily(db, start, end, tickers_needed)
    mkt = _load_mkt_daily(db, start, end)

    common = sorted(set(returns.columns).intersection(set(mcap.columns)))
    returns = returns[common]
    mcap = mcap[common]

    return events, returns, mcap, mkt
