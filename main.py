"""
Multi-Asset Risk Dashboard
--------------------------
Point d'entrée principal. Lance la collecte de données,
le calcul des métriques de risque et génère le rapport HTML.

Usage:
    python main.py
    python main.py --horizon 5 --conf 0.99
    python main.py --no-report
"""

import argparse
import time
from engine.data_loader import DataLoader
from engine.risk_engine import RiskEngine
from reporting.html_report import HtmlReport


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Asset Risk Dashboard")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Horizon VaR en jours (défaut: 1)")
    parser.add_argument("--conf", type=float, default=0.99,
                        help="Niveau de confiance VaR (défaut: 0.99)")
    parser.add_argument("--lookback", type=int, default=252,
                        help="Fenêtre historique en jours (défaut: 252)")
    parser.add_argument("--no-report", action="store_true",
                        help="Désactive la génération du rapport HTML")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  MULTI-ASSET RISK DASHBOARD")
    print("=" * 60)

    # 1. Chargement des données
    print("\n[1/3] Récupération des données de marché...")
    t0 = time.time()
    loader = DataLoader(lookback_days=args.lookback)
    prices = loader.fetch_all()
    print(f"      {len(prices.columns)} assets chargés en {time.time()-t0:.1f}s")

    # 2. Calcul des métriques de risque
    print("\n[2/3] Calcul des métriques de risque...")
    engine = RiskEngine(
        prices=prices,
        horizon=args.horizon,
        confidence=args.conf,
        portfolio=loader.portfolio,
    )
    metrics = engine.compute_all()
    engine.print_summary(metrics)

    # 3. Génération du rapport
    if not args.no_report:
        print("\n[3/3] Génération du rapport HTML...")
        report = HtmlReport(metrics=metrics, engine=engine, args=args)
        path = report.generate()
        print(f"      Rapport sauvegardé : {path}")
        print(f"\n      Ouvrez le fichier dans votre navigateur pour visualiser.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
