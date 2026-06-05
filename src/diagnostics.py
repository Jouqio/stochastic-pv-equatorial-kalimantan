"""
src/diagnostics.py
==================
Statistical diagnostic tests for OLS residual validation.
Implements three tests reported in the manuscript (Section 3.3):
    1. Breusch–Godfrey LM test for autocorrelation  → LM = 18.34, p < 0.001
    2. White test for heteroskedasticity             → LM = 52.37, p = 0.003
    3. Shapiro–Wilk test for normality               → W  = 0.9617, p = 0.008
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

__all__ = ["breusch_godfrey", "white_test", "shapiro_wilk"]


def breusch_godfrey(residuals: np.ndarray, nlags: int = 2) -> dict:
    """
    Breusch–Godfrey Lagrange Multiplier test for residual autocorrelation.

    H₀: No serial correlation up to lag `nlags`.
    Under H₀: LM ~ χ²(nlags).

    Parameters
    ----------
    residuals : np.ndarray – OLS residuals
    nlags     : int        – Number of lags to test (default 2)

    Returns
    -------
    dict: lm_stat, p_value, nlags, acf1 (lag-1 ACF)
    """
    n = len(residuals)
    # Auxiliary regression: ε_t = δ₀ + δ₁ε_{t-1} + ... + δ_p ε_{t-p} + u_t
    X_aux = np.column_stack(
        [np.ones(n)] + [np.roll(residuals, k) for k in range(1, nlags + 1)]
    )
    model_aux = LinearRegression().fit(X_aux, residuals)
    ss_res = np.sum((residuals - model_aux.predict(X_aux)) ** 2)
    ss_tot = np.sum((residuals - residuals.mean()) ** 2)
    r2_aux = 1.0 - ss_res / ss_tot

    lm   = (n - nlags) * r2_aux
    pval = 1.0 - stats.chi2.cdf(lm, nlags)

    # Lag-1 autocorrelation
    acf1 = float(np.corrcoef(residuals[1:], residuals[:-1])[0, 1])

    return {"lm_stat": lm, "p_value": pval, "nlags": nlags, "acf1": acf1}


def white_test(residuals: np.ndarray, X: np.ndarray) -> dict:
    """
    White (1980) test for heteroskedasticity.

    H₀: Error variance is constant (homoskedastic).
    Auxiliary regression: ε² ~ X + X² + cross-products.

    Parameters
    ----------
    residuals : np.ndarray – OLS residuals (n,)
    X         : np.ndarray – Original regressor matrix (n, p)

    Returns
    -------
    dict: lm_stat, p_value, r2_aux
    """
    n = X.shape[0]
    p = min(X.shape[1], 4)   # limit auxiliary features to avoid singularity

    X_sub = X[:, :p]
    parts = [np.ones(n), X_sub]
    for j in range(p):
        parts.append(X_sub[:, j] ** 2)
    for j in range(p):
        for k in range(j + 1, p):
            parts.append(X_sub[:, j] * X_sub[:, k])
    X_aux = np.column_stack(parts)

    sq_res = residuals ** 2
    aux = LinearRegression().fit(X_aux, sq_res)
    r2_aux = aux.score(X_aux, sq_res)

    lm   = n * r2_aux
    q    = X_aux.shape[1] - 1
    pval = 1.0 - stats.chi2.cdf(lm, q)

    return {"lm_stat": lm, "p_value": pval, "r2_aux": r2_aux}


def shapiro_wilk(residuals: np.ndarray) -> dict:
    """
    Shapiro–Wilk normality test on OLS residuals.

    H₀: Residuals are normally distributed.

    Returns
    -------
    dict: w_stat, p_value, skewness, excess_kurtosis
    """
    w, p = stats.shapiro(residuals)
    return {
        "w_stat":          float(w),
        "p_value":         float(p),
        "skewness":        float(stats.skew(residuals)),
        "excess_kurtosis": float(stats.kurtosis(residuals)),
    }
