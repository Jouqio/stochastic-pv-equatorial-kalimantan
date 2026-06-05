#!/usr/bin/env python3
"""
scripts/run_full_pipeline.pyy
============================
End-to-end entry point for the stochastic PV modeling pipeline.

Usage:
    python scripts/run_full_pipeline.py

Outputs:
    results/model_metrics_summary.csv
    figures/figure1_pr_distribution.png
    figures/figure2_actual_vs_predicted.png
    figures/figure3_residual_diagnostics.png
    figures/figure4_model_benchmarking.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_generation         import generate_bontang_met_data
from src.stochastic_loss_framework import compute_stochastic_pr
from src.models                  import train_all_models
from src.diagnostics             import breusch_godfrey, white_test, shapiro_wilk

SEED    = 42
N_TR    = 106   # 80% training (Jan 2015 – Oct 2023)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED)


def main():
    print("=" * 60)
    print(" STOCHASTIC PV MODELING — BONTANG, EAST KALIMANTAN")
    print("=" * 60)

    # ── Step 1: Meteorological data ──────────────────────────────────────
    print("\n[1/4] Generating synthetic meteorological data …")
    df = generate_bontang_met_data(n_months=132, random_seed=SEED)
    print(f"      {len(df)} months generated ({df['date'].iloc[0].strftime('%b %Y')}"
          f" – {df['date'].iloc[-1].strftime('%b %Y')})")

    # ── Step 2: Stochastic loss framework ────────────────────────────────
    print("\n[2/4] Computing stochastic PV power (7-component loss model) …")
    pv = compute_stochastic_pr(
        ghi    = df["GHI"].values,
        t2m    = df["T2M"].values,
        rh2m   = df["RH2M"].values,
        ws10m  = df["WS10M"].values,
        cloud  = df["CLOUD_AMT"].values,
        precip = df["PRECTOTCORR"].values,
        random_seed=SEED,
    )
    df["PR"]       = pv["PR"]
    df["PV_Power"] = pv["Y"]
    print(f"      PR: μ = {pv['PR'].mean():.4f}  σ = {pv['PR'].std():.4f}")

    # ── Step 3: Feature engineering & split ──────────────────────────────
    print("\n[3/4] Feature engineering and model training …")
    feat_cols = ["GHI","T2M","CLOUD_AMT","RH2M","WS10M","PRECTOTCORR","GHI_T2M","GHI_CLOUD"]
    df["GHI_T2M"]   = df["GHI"] * df["T2M"]
    df["GHI_CLOUD"] = df["GHI"] * df["CLOUD_AMT"]

    X = df[feat_cols].values
    y = df["PV_Power"].values
    X_tr, y_tr = X[:N_TR], y[:N_TR]
    X_te, y_te = X[N_TR:], y[N_TR:]

    sc = StandardScaler(); Xtr_sc=sc.fit_transform(X_tr); Xte_sc=sc.transform(X_te)

    results, _ = train_all_models(X_tr, y_tr, X_te, y_te)

    # ── Step 4: Statistical diagnostics ──────────────────────────────────
    print("\n[4/4] Running statistical diagnostics …")
    res_tr = y_tr - results["Corrected OLS"]["model"].predict(Xtr_sc)
    bg  = breusch_godfrey(res_tr, nlags=2)
    wt  = white_test(res_tr, Xtr_sc)
    sw  = shapiro_wilk(res_tr)

    # ── Print benchmarking table ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  MODEL PERFORMANCE — TEST SET (n=26 months)")
    print("─" * 60)
    order = ["XGBoost","Random Forest","Corrected OLS","SVR (RBF)"]
    rows  = []
    for name in order:
        m = results[name]["metrics"]
        rows.append({"Model":name,"R²":m["r2"],"RMSE":m["rmse"],"MAE":m["mae"],"MAPE%":m["mape"]})
        print(f"  {name:<20} R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%")

    print("\n  DIAGNOSTICS (OLS residuals, training set):")
    print(f"  Breusch–Godfrey  LM = {bg['lm_stat']:.2f}   p = {bg['p_value']:.4f}")
    print(f"  White            LM = {wt['lm_stat']:.2f}  p = {wt['p_value']:.4f}")
    print(f"  Shapiro–Wilk      W = {sw['w_stat']:.4f}  p = {sw['p_value']:.4f}")
    print("─" * 60)

    # ── Save results CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "model_metrics_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")
    print("\n  Run  python scripts/generate_figures.py  to create figures.")


if __name__ == "__main__":
    main()
