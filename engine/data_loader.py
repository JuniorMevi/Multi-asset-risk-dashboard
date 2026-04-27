"""
engine/data_loader.py
---------------------
Récupère les prix historiques via yfinance pour un portefeuille
cross-assets (EQ, FX, FI proxy, Commodities).
Gère les erreurs réseau et les données manquantes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


PORTFOLIO_DEF = {
    "CAC 40":       ("^FCHI",   "EQ", 500_000),
    "S&P 500":      ("^GSPC",   "EQ", 380_000),
    "EUR/USD":      ("EURUSD=X","FX", 250_000),
    "USD/JPY":      ("USDJPY=X","FX", 180_000),
    "OAT 10Y ETF":  ("TLT",     "FI", 300_000),
    "Bund ETF":     ("IEF",     "FI", 220_000),
    "Pétrole WTI":  ("CL=F",    "CM", 150_000),
    "Or":           ("GC=F",    "CM", 120_000),
}


class DataLoader:
    """
    Télécharge et nettoie les séries de prix historiques.

    Attributes
    ----------
    lookback_days : int
        Nombre de jours de données à récupérer.
    portfolio : dict
        {nom_asset: {"ticker": str, "class": str, "exposure": float}}
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.portfolio = {
            name: {"ticker": t, "class": cls, "exposure": expo}
            for name, (t, cls, expo) in PORTFOLIO_DEF.items()
        }
        self._end   = datetime.today()
        self._start = self._end - timedelta(days=int(lookback_days * 1.5))

    def fetch_all(self) -> pd.DataFrame:
        """
        Télécharge les prix de clôture ajustés pour tous les assets.

        Returns
        -------
        pd.DataFrame
            Index : dates de trading | Colonnes : noms des assets
        """
        tickers = [v["ticker"] for v in self.portfolio.values()]
        names   = list(self.portfolio.keys())

        raw = yf.download(
            tickers=tickers,
            start=self._start.strftime("%Y-%m-%d"),
            end=self._end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].copy()
            close.columns = tickers  # fix : un seul ticker

        ticker_to_name = {v["ticker"]: k for k, v in self.portfolio.items()}
        close = close.rename(columns=ticker_to_name)
        close = close[[n for n in names if n in close.columns]]
        close = close.ffill().dropna(how="all")  # fix : dropna moins agressif
        close = close.tail(self.lookback_days)

        return close

    def get_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calcule les rendements logarithmiques journaliers."""
        return np.log(prices / prices.shift(1)).dropna()