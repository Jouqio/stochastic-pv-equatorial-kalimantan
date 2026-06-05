"""
src/models.py
=============
Model training and evaluation for stochastic PV power prediction.
Four benchmarked approaches (Table 1, manuscript):
    1. Corrected OLS with interaction terms  (R² = 0.847)
    2. Random Forest                          (R² = 0.891)
    3. Support Vector Regression (RBF)        (R² = 0.824)
    4. XGBoost  ← champion                   (R² = 0.903)
"""

import numpy as np
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

__all__ = ["train_all_models", "compute_metrics"]

RANDOM_STATE = 42


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return R², RMSE, MAE, MAPE for a prediction array."""
    return {
        "r2":   r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> dict:
    """
    Train and evaluate all four benchmark models.

    Parameters
    ----------
    X_train, y_train : training features and targets
    X_test,  y_test  : test features and targets

    Returns
    -------
    dict keyed by model name; each value is:
        {'model': fitted model, 'y_pred': predictions, 'metrics': dict}
    """
    # Standardise features (important for OLS and SVR)
    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_train)
    Xte_sc   = scaler.transform(X_test)

    results = {}

    # ── 1. Corrected OLS ─────────────────────────────────────────────────
    ols = LinearRegression()
    ols.fit(Xtr_sc, y_train)
    p_ols = ols.predict(Xte_sc)
    results["Corrected OLS"] = {
        "model":  ols,
        "y_pred": p_ols,
        "metrics": compute_metrics(y_test, p_ols),
        "residuals_train": y_train - ols.predict(Xtr_sc),
    }

    # ── 2. Random Forest ─────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(Xtr_sc, y_train)
    p_rf = rf.predict(Xte_sc)
    results["Random Forest"] = {
        "model":  rf,
        "y_pred": p_rf,
        "metrics": compute_metrics(y_test, p_rf),
    }

    # ── 3. SVR (RBF) ─────────────────────────────────────────────────────
    svr = SVR(kernel="rbf", C=50, gamma="scale", epsilon=0.01)
    svr.fit(Xtr_sc, y_train)
    p_svr = svr.predict(Xte_sc)
    results["SVR (RBF)"] = {
        "model":  svr,
        "y_pred": p_svr,
        "metrics": compute_metrics(y_test, p_svr),
    }

    # ── 4. XGBoost ───────────────────────────────────────────────────────
    xgb_m = xgb.XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.75, colsample_bytree=0.75,
        reg_lambda=2, min_child_weight=5,
        random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_m.fit(Xtr_sc, y_train,
              eval_set=[(Xte_sc, y_test)], verbose=False)
    p_xgb = xgb_m.predict(Xte_sc)
    results["XGBoost"] = {
        "model":  xgb_m,
        "y_pred": p_xgb,
        "metrics": compute_metrics(y_test, p_xgb),
        "feature_importance": xgb_m.feature_importances_,
    }

    return results, scaler
