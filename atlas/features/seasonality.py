"""Mavsumiylik featurelari va tarixiy mavsumiy statistika (UI uchun)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ..data.calendar import calendar_features


def season_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    f = pd.DataFrame(index=d)
    two_pi = 2 * np.pi
    f["m_sin"] = np.sin(two_pi * (d.month - 1) / 12)
    f["m_cos"] = np.cos(two_pi * (d.month - 1) / 12)
    f["dow_sin"] = np.sin(two_pi * d.dayofweek / 7)
    f["dow_cos"] = np.cos(two_pi * d.dayofweek / 7)
    f["doy_sin"] = np.sin(two_pi * d.dayofyear / 365.25)
    f["doy_cos"] = np.cos(two_pi * d.dayofyear / 365.25)
    f["dom"] = (d.day - 1) / 30.0
    f["month_end"] = ((d + pd.Timedelta(days=3)).month != d.month).astype(float)
    f["quarter_end"] = ((d.month % 3 == 0) & (d.day >= 24)).astype(float)
    # Hisobot mavsumi: chorak boshlangandan 2–6 hafta keyin
    wq = ((d.month - 1) % 3) * 4.345 + d.day / 7
    f["earnings_season"] = ((wq >= 2) & (wq <= 6.5)).astype(float)
    # Bayramlar (AQSh): keyingi bayramgacha kunlar
    hol = USFederalHolidayCalendar().holidays(d.min() - pd.Timedelta(days=5), d.max() + pd.Timedelta(days=400))
    idx = hol.searchsorted(d)
    nxt = hol[np.minimum(idx, len(hol) - 1)]
    days = np.clip((nxt - d).days, 0, 30)
    f["holiday_days"] = days / 30.0
    f["holiday_next3"] = (days <= 3).astype(float)
    # Qishloq xo'jaligi davrlari (Shimoliy yarimshar): ekish (apr–may), o'sish (iyun–avg), hosil (sen–noy)
    f["plant"] = d.month.isin([4, 5]).astype(float)
    f["grow"] = d.month.isin([6, 7, 8]).astype(float)
    f["harvest"] = d.month.isin([9, 10, 11]).astype(float)
    f = f.join(calendar_features(d))
    return f.astype("float32")


SEASON_DIM = len(season_features(pd.date_range("2020-01-01", periods=3)).columns)


def seasonal_profile(df: pd.DataFrame, by: str = "month") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aktivning tarixiy mavsumiy profili: har oy/hafta uchun o'rtacha qaytim, o'sish ehtimoli, yillar soni.

    by: 'month' | 'week' | 'dow'
    """
    s = df.set_index("date")["close"].astype(float)
    if by == "month":
        r = np.log(s.resample("ME").last()).diff().dropna()
        key = r.index.month
    elif by == "week":
        r = np.log(s.resample("W-FRI").last()).diff().dropna()
        key = r.index.isocalendar().week.values
    else:
        r = np.log(s).diff().dropna()
        key = r.index.dayofweek
    g = pd.DataFrame({"k": key, "r": r.values, "year": r.index.year})
    prof = g.groupby("k").agg(mean=("r", "mean"), median=("r", "median"), std=("r", "std"),
                              up_prob=("r", lambda x: (x > 0).mean()), n=("r", "size"))
    # t-statistika: kichik namunada "mavsumiylik" tasodif bo'lishi mumkin — UI'da ko'rsatamiz
    prof["t_stat"] = prof["mean"] / (prof["std"] / np.sqrt(prof["n"])).replace(0, np.nan)
    prof[["mean", "median", "std"]] *= 100  # foizda
    by_year = g.pivot_table(index="year", columns="k", values="r", aggfunc="sum") * 100
    return prof.reset_index().rename(columns={"k": by}), by_year
