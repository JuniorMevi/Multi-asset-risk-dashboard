# Multi-Asset Risk Dashboard

Outil d'analyse et de reporting de risque sur un portefeuille **cross-assets** (Equity, FX, Fixed Income, Commodities) développé en Python.

---

## Fonctionnalités

### Métriques de risque calculées
- **VaR paramétrique** (sous hypothèse normale) — niveaux 95% et 99%, horizon 1J / 5J / 10J
- **VaR historique** — quantile empirique des P&L simulés
- **Expected Shortfall (CVaR)** — moyenne des pertes au-delà de la VaR (tail risk)
- **VaR Monte Carlo** — simulation de 500 scénarios corrélés via décomposition de Cholesky
- **Volatilité rolling** — fenêtres 30J et annualisée
- **Sharpe Ratio** annualisé par asset
- **Max Drawdown** sur la fenêtre d'analyse
- **P&L journalier et MTD** (month-to-date) par position

### Couverture cross-assets
| Classe     | Assets couverts                        |
|------------|----------------------------------------|
| Equity     | CAC 40, S&P 500                        |
| FX         | EUR/USD, USD/JPY                       |
| Fixed Income | OAT 10Y (proxy TLT), Bund (proxy IEF)|
| Commodities| Pétrole WTI, Or                        |

### Rapports générés
- Rapport **HTML autonome** (single-file, sans serveur) avec :
  - KPIs agrégés portefeuille
  - Tableau détaillé par asset (VaR, CVaR, Vol, PnL, Sharpe, MaxDD)
  - Courbe PnL cumulée 30J (Chart.js)
  - Histogramme Monte Carlo des P&L
  - Matrice de corrélations cross-asset colorée
  - VaR agrégée par classe d'actifs (barres)

---

## Architecture

```
multi_asset_risk_dashboard/
├── main.py                  # Point d'entrée CLI
├── requirements.txt
├── engine/
│   ├── data_loader.py       # Téléchargement & nettoyage (yfinance)
│   └── risk_engine.py       # Moteur de calcul (VaR, CVaR, MC, corrélations)
└── reporting/
    └── html_report.py       # Génération du rapport HTML (Chart.js)
```

---

## Installation

```bash
git clone https://github.com/JuniorMevi/multi-asset-risk-dashboard.git
cd multi-asset-risk-dashboard
pip install -r requirements.txt
```

---

## Utilisation

```bash
# Rapport standard (VaR 99%, horizon 1J)
python main.py

# Horizon 5 jours
python main.py --horizon 5

# VaR 95%, horizon 10J
python main.py --conf 0.95 --horizon 10

# Sans génération du rapport HTML
python main.py --no-report
```

Le rapport HTML est sauvegardé dans `reports/risk_report_YYYYMMDD_HHMMSS.html`.

---

## Formules utilisées

### VaR paramétrique
```
VaR = z_α × σ × √h × Exposition
```
avec `z_α = 2.326` (99%) ou `1.645` (95%), `σ` = volatilité journalière, `h` = horizon.

### Expected Shortfall (CVaR)
```
CVaR = E[P&L | P&L ≤ -VaR]
```
Moyenne des pertes dépassant la VaR — mesure plus conservative que la VaR seule.

### Monte Carlo avec corrélations
```
R_sim = μ·h + L·Z·√h
```
où `L` est la matrice de Cholesky de la covariance (`Σ = L·Lᵀ`) et `Z ~ N(0,I)`.

---

## Stack technique

| Composant          | Technologie             |
|--------------------|-------------------------|
| Données marché     | yfinance (Yahoo Finance)|
| Calcul numérique   | numpy, scipy            |
| Manipulation data  | pandas                  |
| Visualisation      | Chart.js (CDN, rapport HTML) |
| Reporting          | HTML/CSS vanilla        |
| Runtime            | Python 3.10+            |

---

## Exemple de sortie console

```
============================================================
  MULTI-ASSET RISK DASHBOARD
============================================================

[1/3] Récupération des données de marché...
      8 assets chargés en 2.3s

[2/3] Calcul des métriques de risque...

  ASSET                CLASSE   EXPO (€)   VaR99% (€)  Vol 30J    PnL J-1
  ------------------------------------------------------------------------
  CAC 40               EQ       500 000      7 314      11.8%     +1 243
  S&P 500              EQ       380 000      6 813      14.2%       -892
  EUR/USD              FX       250 000      3 789       6.1%       +234
  ...
  ------------------------------------------------------------------------
  TOTAL                         2 100 000   34 217                 +1 987

  VaR MC 99% :     31 450 €  |  CVaR MC :     38 920 €

[3/3] Génération du rapport HTML...
      Rapport sauvegardé : reports/risk_report_20260424_143022.html
============================================================
```

---

## Auteur

**Junior MEVI** — Étudiant Cycle Ingénieur IT for Finance, EFREI Paris  
LinkedIn : [@juniormevi](https://linkedin.com/in/junior-mevi)
