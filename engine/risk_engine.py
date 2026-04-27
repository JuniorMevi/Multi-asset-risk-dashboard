"""
engine/risk_engine.py
---------------------
Moteur de calcul des métriques de risque financier :
  - VaR historique & paramétrique (95%, 99%)
  - Expected Shortfall (CVaR)
  - Volatilité rolling
  - Corrélations cross-asset
  - PnL journalier simulé
  - Monte Carlo (500 scénarios)
  - Sharpe ratio, Max Drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Constantes ─────────────────────────────────────────────────────────────────
Z_99 = norm.ppf(0.99)   # 2.3263
Z_95 = norm.ppf(0.95)   # 1.6449
MC_SCENARIOS = 500
ROLLING_WINDOW = 30     # jours pour vol rolling et PnL


class RiskEngine:
    """
    Calcule l'ensemble des métriques de risque pour un portefeuille cross-assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Prix de clôture ajustés (index=dates, colonnes=noms assets).
    horizon : int
        Horizon VaR en jours ouvrés.
    confidence : float
        Niveau de confiance (0.95 ou 0.99).
    portfolio : dict
        Métadonnées du portefeuille (ticker, classe, exposition).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        horizon: int,
        confidence: float,
        portfolio: dict,
    ):
        self.prices      = prices
        self.horizon     = horizon
        self.confidence  = confidence
        self.portfolio   = portfolio
        self.returns     = np.log(prices / prices.shift(1)).dropna()

    # ── API publique ────────────────────────────────────────────────────────────

    def compute_all(self) -> dict:
        """
        Lance tous les calculs et retourne un dictionnaire de métriques.

        Returns
        -------
        dict avec clés :
          per_asset  : DataFrame, métriques par asset
          portfolio  : dict, métriques agrégées
          pnl_series : DataFrame, PnL cumulé 30J par asset
          corr       : DataFrame, matrice de corrélations
          mc_pnl     : np.ndarray, distribution Monte Carlo du portefeuille
          var_by_class: dict, VaR par classe d'actifs
        """
        per_asset   = self._compute_per_asset()
        pnl_series  = self._compute_pnl_series()
        corr        = self._compute_correlations()
        mc_pnl      = self._monte_carlo_portfolio()
        var_by_class = self._var_by_class(per_asset)
        portfolio   = self._compute_portfolio_metrics(per_asset, mc_pnl)

        return {
            "per_asset":    per_asset,
            "portfolio":    portfolio,
            "pnl_series":   pnl_series,
            "corr":         corr,
            "mc_pnl":       mc_pnl,
            "var_by_class": var_by_class,
        }

    # ── Calculs par asset ───────────────────────────────────────────────────────

    def _compute_per_asset(self) -> pd.DataFrame:
        rows = []
        for name, meta in self.portfolio.items():
            if name not in self.returns.columns:
                continue
            ret  = self.returns[name].dropna()
            expo = meta["exposure"]

            vol_ann   = ret.std() * np.sqrt(252)
            vol_30d   = ret.tail(ROLLING_WINDOW).std() * np.sqrt(252)
            var_param = self._var_parametric(ret, expo)
            var_hist  = self._var_historical(ret, expo)
            cvar      = self._cvar(ret, expo)
            pnl_d     = float(ret.iloc[-1] * expo) if len(ret) else 0.0
            pnl_mtd   = float(ret.tail(20).sum() * expo)
            sharpe    = self._sharpe(ret)
            max_dd    = self._max_drawdown(self.prices[name])

            rows.append({
                "asset":      name,
                "class":      meta["class"],
                "exposure":   expo,
                "vol_ann":    round(vol_ann * 100, 2),     # %
                "vol_30d":    round(vol_30d * 100, 2),     # %
                "var_param":  round(var_param, 0),          # €
                "var_hist":   round(var_hist, 0),           # €
                "cvar":       round(cvar, 0),               # €
                "pnl_d":      round(pnl_d, 0),              # €
                "pnl_mtd":    round(pnl_mtd, 0),            # €
                "sharpe":     round(sharpe, 2),
                "max_dd":     round(max_dd * 100, 2),       # %
            })

        return pd.DataFrame(rows).set_index("asset")

    # ── VaR paramétrique ────────────────────────────────────────────────────────

    def _var_parametric(self, returns: pd.Series, exposure: float) -> float:
        """
        VaR paramétrique (normale).
        VaR = z * σ * √h * Exposition
        """
        z   = Z_99 if self.confidence >= 0.99 else Z_95
        sig = returns.std()
        return abs(exposure * sig * z * np.sqrt(self.horizon))

    # ── VaR historique ──────────────────────────────────────────────────────────

    def _var_historical(self, returns: pd.Series, exposure: float) -> float:
        """
        VaR historique : quantile empirique des P&L simulés.
        P&L_i = r_i * exposition
        """
        pnl = returns * exposure
        return abs(float(np.percentile(pnl, (1 - self.confidence) * 100)))

    # ── Expected Shortfall (CVaR) ───────────────────────────────────────────────

    def _cvar(self, returns: pd.Series, exposure: float) -> float:
        """
        CVaR = moyenne des pertes au-delà de la VaR (tail risk).
        """
        pnl       = returns * exposure
        threshold = np.percentile(pnl, (1 - self.confidence) * 100)
        tail      = pnl[pnl <= threshold]
        return abs(float(tail.mean())) if len(tail) > 0 else 0.0

    # ── Ratios & statistiques ───────────────────────────────────────────────────

    def _sharpe(self, returns: pd.Series, rf: float = 0.03) -> float:
        """Sharpe ratio annualisé (rf = taux sans risque annuel)."""
        excess = returns.mean() * 252 - rf
        vol    = returns.std() * np.sqrt(252)
        return float(excess / vol) if vol > 0 else 0.0

    def _max_drawdown(self, prices: pd.Series) -> float:
        """Max Drawdown sur la fenêtre d'analyse."""
        roll_max = prices.cummax()
        dd       = (prices - roll_max) / roll_max
        return float(dd.min())

    # ── PnL cumulé sur 30J ─────────────────────────────────────────────────────

    def _compute_pnl_series(self) -> pd.DataFrame:
        """
        Retourne le PnL cumulé (en €) sur les 30 derniers jours pour chaque asset.
        """
        ret_30 = self.returns.tail(ROLLING_WINDOW)
        pnl    = pd.DataFrame(index=ret_30.index)

        for name, meta in self.portfolio.items():
            if name in ret_30.columns:
                pnl[name] = (ret_30[name] * meta["exposure"]).cumsum()

        return pnl.round(0)

    # ── Corrélations ───────────────────────────────────────────────────────────

    def _compute_correlations(self) -> pd.DataFrame:
        """Matrice de corrélation de Pearson sur les rendements (fenêtre 30J)."""
        ret_30 = self.returns.tail(ROLLING_WINDOW)
        assets = [n for n in self.portfolio if n in ret_30.columns]
        return ret_30[assets].corr().round(2)

    # ── Monte Carlo portefeuille ────────────────────────────────────────────────

    def _monte_carlo_portfolio(self) -> np.ndarray:
        """
        Simule MC_SCENARIOS scénarios de P&L du portefeuille à horizon h jours.
        Utilise la décomposition de Cholesky pour respecter les corrélations.

        Returns
        -------
        np.ndarray shape (MC_SCENARIOS,) — distribution des P&L en €
        """
        assets = [n for n in self.portfolio if n in self.returns.columns]
        ret    = self.returns[assets]
        expos  = np.array([self.portfolio[n]["exposure"] for n in assets])

        mu    = ret.mean().values
        cov   = ret.cov().values

        # Décomposition de Cholesky : Σ = L @ L.T
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(assets)))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov)))

        # Simulation : rendements corrélés sur h jours
        z        = np.random.standard_normal((MC_SCENARIOS, len(assets)))
        sim_ret  = mu * self.horizon + (L @ z.T).T * np.sqrt(self.horizon)
        pnl_sim  = (sim_ret * expos).sum(axis=1)

        return np.round(pnl_sim, 0)

    # ── VaR agrégée par classe ─────────────────────────────────────────────────

    def _var_by_class(self, per_asset: pd.DataFrame) -> dict:
        """Agrège la VaR paramétrique par classe d'actifs."""
        result = {}
        for cls in per_asset["class"].unique():
            mask = per_asset["class"] == cls
            result[cls] = round(float(per_asset.loc[mask, "var_param"].sum()), 0)
        return result

    # ── Métriques portefeuille agrégées ────────────────────────────────────────

    def _compute_portfolio_metrics(self, per_asset: pd.DataFrame, mc_pnl: np.ndarray) -> dict:
        conf_pct = int(self.confidence * 100)
        return {
            "total_exposure": int(per_asset["exposure"].sum()),
            "var_param":      int(per_asset["var_param"].sum()),
            "var_hist":       int(per_asset["var_hist"].sum()),
            "cvar":           int(per_asset["cvar"].sum()),
            "pnl_d":          int(per_asset["pnl_d"].sum()),
            "pnl_mtd":        int(per_asset["pnl_mtd"].sum()),
            "mc_var":         int(abs(np.percentile(mc_pnl, (1 - self.confidence) * 100))),
            "mc_cvar":        int(abs(mc_pnl[mc_pnl <= np.percentile(mc_pnl, (1-self.confidence)*100)].mean())),
            "horizon":        self.horizon,
            "confidence":     self.confidence,
            "conf_pct":       conf_pct,
            "n_assets":       len(per_asset),
        }

    # ── Affichage console ──────────────────────────────────────────────────────

    def print_summary(self, metrics: dict) -> None:
        pa  = metrics["per_asset"]
        ptf = metrics["portfolio"]

        print(f"\n  {'ASSET':<20} {'CLASSE':>6} {'EXPO (€)':>10} {'VaR{conf_pct}% (€)':>12} {'Vol 30J':>8} {'PnL J-1':>10}".format(
            conf_pct=ptf["conf_pct"]))
        print("  " + "-" * 72)

        for asset, row in pa.iterrows():
            sign = "+" if row["pnl_d"] >= 0 else ""
            print(
                f"  {asset:<20} {row['class']:>6} {row['exposure']:>10,.0f}"
                f" {row['var_param']:>12,.0f} {row['vol_30d']:>7.1f}%"
                f" {sign}{row['pnl_d']:>9,.0f}"
            )

        print("  " + "-" * 72)
        sign = "+" if ptf["pnl_d"] >= 0 else ""
        print(
            f"  {'TOTAL':<20} {'':>6} {ptf['total_exposure']:>10,.0f}"
            f" {ptf['var_param']:>12,.0f} {'':>8}"
            f" {sign}{ptf['pnl_d']:>9,.0f}"
        )
        print(f"\n  VaR MC {ptf['conf_pct']}% : {ptf['mc_var']:>10,.0f} € | CVaR MC : {ptf['mc_cvar']:>10,.0f} €")
