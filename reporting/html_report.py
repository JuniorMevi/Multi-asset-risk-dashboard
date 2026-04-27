"""
reporting/html_report.py
------------------------
Génère un rapport HTML autonome (single-file, sans dépendance externe)
avec Chart.js embarqué via CDN, tableaux Bootstrap-like et métriques de risque.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ── Palette de couleurs par classe d'actifs ────────────────────────────────────
CLASS_COLORS = {
    "EQ": "#378ADD",
    "FX": "#639922",
    "FI": "#BA7517",
    "CM": "#D85A30",
}
CLASS_LABELS = {
    "EQ": "Equity",
    "FX": "FX",
    "FI": "Fixed Income",
    "CM": "Commodities",
}


class HtmlReport:
    """
    Génère le rapport HTML complet du Risk Dashboard.

    Parameters
    ----------
    metrics : dict  — sortie de RiskEngine.compute_all()
    engine  : RiskEngine
    args    : argparse.Namespace
    """

    def __init__(self, metrics: dict, engine, args):
        self.metrics   = metrics
        self.engine    = engine
        self.args      = args
        self.generated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def generate(self, output_dir: str = "reports") -> str:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(
            output_dir,
            f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        html = self._build_html()
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        return fname

    # ── Construction HTML ──────────────────────────────────────────────────────

    def _build_html(self) -> str:
        ptf = self.metrics["portfolio"]
        pa  = self.metrics["per_asset"]
        corr = self.metrics["corr"]
        mc  = self.metrics["mc_pnl"]
        var_cls = self.metrics["var_by_class"]

        return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Risk Dashboard — {self.generated}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
{self._css()}
</head>
<body>
<div class="container">
  {self._header(ptf)}
  {self._kpis(ptf)}
  {self._asset_table(pa)}
  <div class="charts-row">
    {self._pnl_chart_card()}
    {self._var_bars_card(var_cls)}
  </div>
  {self._corr_table(corr)}
  {self._mc_chart_card(mc, ptf)}
  {self._footer()}
</div>
{self._scripts(pa, var_cls, mc, ptf)}
</body>
</html>"""

    # ── Sections HTML ──────────────────────────────────────────────────────────

    def _header(self, ptf: dict) -> str:
        sign = "+" if ptf["pnl_d"] >= 0 else ""
        color = "#3a8a3a" if ptf["pnl_d"] >= 0 else "#c0392b"
        return f"""
<div class="header">
  <div>
    <h1>Multi-Asset Risk Dashboard</h1>
    <span class="subtitle">Horizon {ptf['horizon']}J &nbsp;|&nbsp; Confiance {ptf['conf_pct']}% &nbsp;|&nbsp; {self.generated}</span>
  </div>
  <div class="header-pnl" style="color:{color}">
    <span class="pnl-label">PnL J-1</span>
    <span class="pnl-val">{sign}{ptf['pnl_d']:,.0f} €</span>
  </div>
</div>"""

    def _kpis(self, ptf: dict) -> str:
        kpis = [
            ("Exposition totale",       f"{ptf['total_exposure']:,.0f} €",    f"{ptf['n_assets']} assets",          "neu"),
            (f"VaR paramétrique {ptf['conf_pct']}%", f"{ptf['var_param']:,.0f} €",    f"Horizon {ptf['horizon']}J",         "neg"),
            (f"VaR historique {ptf['conf_pct']}%",   f"{ptf['var_hist']:,.0f} €",     "Quantile empirique",                 "neg"),
            ("CVaR (Expected Shortfall)", f"{ptf['cvar']:,.0f} €",            "Tail risk moyen",                    "neg"),
            ("VaR Monte Carlo",          f"{ptf['mc_var']:,.0f} €",           f"{500} scénarios MC",                "neg"),
            ("PnL MTD",                  f"{ptf['pnl_mtd']:+,.0f} €",         "Mois en cours (~20J)",               "pos" if ptf['pnl_mtd'] >= 0 else "neg"),
        ]
        cards = "\n".join(
            f'<div class="kpi"><div class="kpi-label">{l}</div>'
            f'<div class="kpi-val {cls}">{v}</div>'
            f'<div class="kpi-sub">{sub}</div></div>'
            for l, v, sub, cls in kpis
        )
        return f'<div class="kpis">{cards}</div>'

    def _asset_table(self, pa: pd.DataFrame) -> str:
        rows = ""
        for asset, row in pa.iterrows():
            cls   = row["class"]
            badge = f'<span class="badge badge-{cls.lower()}">{cls}</span>'
            pnl_d_cls = "pos" if row["pnl_d"] >= 0 else "neg"
            pnl_m_cls = "pos" if row["pnl_mtd"] >= 0 else "neg"
            rows += f"""
<tr>
  <td class="asset-name">{asset}</td>
  <td>{badge}</td>
  <td class="num">{row['exposure']:,.0f} €</td>
  <td class="num neg">-{row['var_param']:,.0f} €</td>
  <td class="num neg">-{row['var_hist']:,.0f} €</td>
  <td class="num neg">-{row['cvar']:,.0f} €</td>
  <td class="num">{row['vol_30d']:.1f}%</td>
  <td class="num {pnl_d_cls}">{row['pnl_d']:+,.0f} €</td>
  <td class="num {pnl_m_cls}">{row['pnl_mtd']:+,.0f} €</td>
  <td class="num">{row['sharpe']:.2f}</td>
  <td class="num neg">{row['max_dd']:.1f}%</td>
</tr>"""
        return f"""
<div class="section">
  <div class="section-title">Positions &amp; métriques de risque par asset</div>
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th>Asset</th><th>Classe</th><th>Exposition</th>
      <th>VaR param.</th><th>VaR hist.</th><th>CVaR</th>
      <th>Vol 30J</th><th>PnL J-1</th><th>PnL MTD</th>
      <th>Sharpe</th><th>Max DD</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</div>"""

    def _pnl_chart_card(self) -> str:
        return """
<div class="chart-card">
  <div class="section-title">PnL cumulé — 30 derniers jours</div>
  <div class="legend" id="pnl-legend"></div>
  <div style="position:relative;height:220px;">
    <canvas id="pnlChart" role="img" aria-label="Courbe PnL cumulée 30 jours par asset class"></canvas>
  </div>
</div>"""

    def _var_bars_card(self, var_cls: dict) -> str:
        bars = ""
        max_v = max(var_cls.values()) if var_cls else 1
        for cls, val in sorted(var_cls.items(), key=lambda x: -x[1]):
            pct  = val / max_v * 100
            color = CLASS_COLORS.get(cls, "#888")
            bars += f"""
<div class="var-row">
  <span class="var-label">{CLASS_LABELS.get(cls, cls)}</span>
  <div class="var-track"><div class="var-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
  <span class="var-val">-{val:,.0f} €</span>
</div>"""
        return f"""
<div class="chart-card">
  <div class="section-title">VaR paramétrique par classe d'actifs</div>
  {bars}
</div>"""

    def _corr_table(self, corr: pd.DataFrame) -> str:
        assets = list(corr.columns)
        header = "".join(f"<th>{a.split()[0]}</th>" for a in assets)
        rows   = ""
        for asset in assets:
            cells = ""
            for a2 in assets:
                v = corr.loc[asset, a2]
                if asset == a2:
                    bg = "#f0f0f0"; tc = "#555"
                elif v > 0.5:
                    bg = "#B5D4F4"; tc = "#042C53"
                elif v > 0.2:
                    bg = "#E6F1FB"; tc = "#185FA5"
                elif v < -0.1:
                    bg = "#FAECE7"; tc = "#712B13"
                else:
                    bg = "#fff"; tc = "#555"
                cells += f'<td style="background:{bg};color:{tc};">{v:.2f}</td>'
            rows += f"<tr><td class='asset-name'>{asset.split()[0]}</td>{cells}</tr>"

        return f"""
<div class="section">
  <div class="section-title">Matrice de corrélations cross-asset (30J)</div>
  <div class="table-wrap">
  <table class="corr-table">
    <thead><tr><th></th>{header}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <div class="legend" style="margin-top:8px;">
    <span><span class="ldot" style="background:#B5D4F4;"></span>Forte corrél. positive (&gt;0.5)</span>
    <span><span class="ldot" style="background:#E6F1FB;"></span>Corrél. positive (&gt;0.2)</span>
    <span><span class="ldot" style="background:#FAECE7;"></span>Corrél. négative</span>
  </div>
</div>"""

    def _mc_chart_card(self, mc: np.ndarray, ptf: dict) -> str:
        return f"""
<div class="section">
  <div class="section-title">Distribution des P&amp;L — Monte Carlo ({len(mc):,} scénarios, horizon {ptf['horizon']}J)</div>
  <div style="position:relative;height:160px;">
    <canvas id="histChart" role="img" aria-label="Histogramme distribution Monte Carlo des PnL"></canvas>
  </div>
  <div class="legend" style="margin-top:8px;">
    <span><span class="ldot" style="background:rgba(226,75,74,0.6);"></span>Scénarios de perte</span>
    <span><span class="ldot" style="background:rgba(99,153,34,0.6);"></span>Scénarios de gain</span>
    <span style="margin-left:12px;">VaR MC {ptf['conf_pct']}% : <strong>-{ptf['mc_var']:,.0f} €</strong></span>
    <span>CVaR MC : <strong>-{ptf['mc_cvar']:,.0f} €</strong></span>
  </div>
</div>"""

    def _footer(self) -> str:
        return f"""
<div class="footer">
  Généré le {self.generated} &nbsp;·&nbsp;
  Multi-Asset Risk Dashboard &nbsp;·&nbsp;
  Données : Yahoo Finance (yfinance) &nbsp;·&nbsp;
  VaR paramétrique sous hypothèse de normalité
</div>"""

    # ── Scripts Chart.js ───────────────────────────────────────────────────────

    def _scripts(self, pa: pd.DataFrame, var_cls: dict, mc: np.ndarray, ptf: dict) -> str:
        pnl_series = self.metrics["pnl_series"]

        # PnL chart data
        labels_pnl = [str(d.date()) for d in pnl_series.index]
        datasets   = []
        legend_html = ""
        for name, meta in self.engine.portfolio.items():
            if name not in pnl_series.columns:
                continue
            cls   = meta["class"]
            color = CLASS_COLORS.get(cls, "#888")
            data  = [int(v) for v in pnl_series[name].tolist()]
            datasets.append({
                "label": name, "data": data,
                "borderColor": color, "borderWidth": 1.5,
                "pointRadius": 0, "tension": 0.35, "fill": False
            })
            legend_html += f'<span><span class="ldot" style="background:{color};"></span>{name}</span>'

        # Monte Carlo histogram
        arr  = mc.tolist()
        mn, mx = min(arr), max(arr)
        n_bins = 25
        step = (mx - mn) / n_bins
        counts = [0] * n_bins
        for v in arr:
            b = min(int((v - mn) / step), n_bins - 1)
            counts[b] += 1
        hist_labels = [f"{int(mn + i * step):,}€" for i in range(n_bins)]
        hist_colors = [
            "rgba(226,75,74,0.65)" if (mn + (i + 0.5) * step) < 0 else "rgba(99,153,34,0.65)"
            for i in range(n_bins)
        ]

        return f"""
<script>
// PnL Chart
(function() {{
  document.getElementById('pnl-legend').innerHTML = `{legend_html}`;
  new Chart(document.getElementById('pnlChart'), {{
    type: 'line',
    data: {{ labels: {json.dumps(labels_pnl)}, datasets: {json.dumps(datasets)} }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 6, autoSkip: true, font: {{ size: 10 }} }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ callback: v => (v/1000).toFixed(0)+'K€', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.06)' }} }}
      }}
    }}
  }});
}})();

// Monte Carlo Histogram
(function() {{
  new Chart(document.getElementById('histChart'), {{
    type: 'bar',
    data: {{
      labels: {json.dumps(hist_labels)},
      datasets: [{{ data: {json.dumps(counts)}, backgroundColor: {json.dumps(hist_colors)}, borderWidth: 0 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 8, autoSkip: true, font: {{ size: 9 }} }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,0.06)' }} }}
      }}
    }}
  }});
}})();
</script>"""

    # ── CSS ────────────────────────────────────────────────────────────────────

    def _css(self) -> str:
        return """<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 13px; color: #1a1a1a; background: #f4f5f7; line-height: 1.5; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
.header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid #dde; }
.header h1 { font-size: 20px; font-weight: 600; color: #111; }
.subtitle { font-size: 12px; color: #888; }
.header-pnl { text-align: right; }
.pnl-label { display: block; font-size: 11px; color: #888; }
.pnl-val { font-size: 22px; font-weight: 600; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px; }
.kpi { background: #fff; border: 1px solid #e8e8ec; border-radius: 8px; padding: 12px 14px; }
.kpi-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.kpi-val { font-size: 20px; font-weight: 600; }
.kpi-sub { font-size: 11px; color: #aaa; margin-top: 2px; }
.pos { color: #2e7d32; }
.neg { color: #c0392b; }
.neu { color: #555; }
.section { margin-bottom: 20px; background: #fff; border: 1px solid #e8e8ec; border-radius: 10px; padding: 16px; }
.section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: #888; margin-bottom: 12px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: right; padding: 6px 8px; font-weight: 500; color: #888; border-bottom: 1px solid #eee; font-size: 10px; text-transform: uppercase; white-space: nowrap; }
th:first-child, th:nth-child(2) { text-align: left; }
td { padding: 7px 8px; text-align: right; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }
td:first-child, td:nth-child(2) { text-align: left; }
tr:last-child td { border-bottom: none; }
.asset-name { font-weight: 500; color: #1a1a1a; }
.num { font-variant-numeric: tabular-nums; }
.badge { display: inline-block; font-size: 10px; padding: 2px 7px; border-radius: 4px; font-weight: 600; }
.badge-eq { background: #E6F1FB; color: #0C447C; }
.badge-fx { background: #EAF3DE; color: #27500A; }
.badge-fi { background: #FAEEDA; color: #633806; }
.badge-cm { background: #FAECE7; color: #712B13; }
.charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
.chart-card { background: #fff; border: 1px solid #e8e8ec; border-radius: 10px; padding: 16px; }
.var-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 12px; }
.var-label { width: 90px; color: #555; flex-shrink: 0; }
.var-track { flex: 1; height: 6px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }
.var-fill { height: 100%; border-radius: 3px; }
.var-val { width: 90px; text-align: right; color: #c0392b; font-weight: 500; flex-shrink: 0; }
.corr-table td, .corr-table th { text-align: center; min-width: 54px; font-size: 11px; }
.corr-table td:first-child, .corr-table th:first-child { text-align: left; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px; color: #666; }
.ldot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
.footer { text-align: center; font-size: 11px; color: #bbb; margin-top: 24px; padding-top: 12px; border-top: 1px solid #eee; }
@media (max-width: 700px) { .kpis { grid-template-columns: 1fr 1fr; } .charts-row { grid-template-columns: 1fr; } }
</style>"""
