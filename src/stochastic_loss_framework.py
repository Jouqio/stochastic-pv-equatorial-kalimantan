"""
src/stochastic_loss_framework.py
=================================
Seven-component physics-informed stochastic loss framework for
photovoltaic performance ratio (PR) generation.

Master equation:
    Y(t) = GHI(t) × η_STC
          × [1 − β_eff(T_cell − 25)]     # temperature derating
          × [1 − L_soil(t)]               # soiling
          × [1 − L_spectral(t)]           # spectral mismatch
          × η_inv(t)                      # inverter (Sandia)
          × R_deg(t)                      # degradation
          × [1 − L_wire]                  # wiring
          × ξ_intermit(t)                 # intermittency

Reference:
    Abdi, S.N. (2025). Applied Energy (under review).
"""

import numpy as np

__all__ = ["compute_stochastic_pr"]


def compute_stochastic_pr(
    ghi:    np.ndarray,
    t2m:    np.ndarray,
    rh2m:   np.ndarray,
    ws10m:  np.ndarray,
    cloud:  np.ndarray,
    precip: np.ndarray,
    random_seed: int = 42,
) -> dict:
    """
    Compute stochastic Performance Ratio (PR) and PV power output
    using the seven-component physics-informed loss framework.

    Parameters
    ----------
    ghi    : np.ndarray – Global Horizontal Irradiance (kWh/m²/day)
    t2m    : np.ndarray – 2-m air temperature (°C)
    rh2m   : np.ndarray – 2-m relative humidity (%)
    ws10m  : np.ndarray – 10-m wind speed (m/s)
    cloud  : np.ndarray – Cloud amount (%)
    precip : np.ndarray – Precipitation (mm/day)
    random_seed : int   – Fixed seed for reproducibility

    Returns
    -------
    dict with keys:
        'PR'      : np.ndarray – Monthly Performance Ratio (calibrated μ=0.76, σ=0.04)
        'Y'       : np.ndarray – PV power output (kWh/kWp/day)
        'losses'  : dict       – Individual loss components for diagnostics
    """
    rng = np.random.default_rng(random_seed)
    N = len(ghi)
    idx = np.arange(N)

    # ── Component 1: Temperature derating ────────────────────────────────
    # β_eff ~ N(0.0045, 0.0003²)  [positive magnitude]
    beta = np.abs(rng.normal(0.0045, 0.0003, N))
    # NOCT cell temperature model
    t_cell = t2m + 25.0 * (ghi / 0.8) * 9.5 / (5.7 + ws10m)
    temp_fac = np.clip(1.0 - beta * (t_cell - 25.0), 0.78, 0.96)

    # ── Component 2: Seasonal soiling with rain cleaning ─────────────────
    # dL_soil/dt = r_accum − η_clean × 𝟙{PRECIP>5mm} × L_soil
    l_soil = np.zeros(N)
    for i in range(N):
        mo = (i % 12) + 1
        if mo in [6, 7, 8, 9]:     # dry season
            mu_acc = 0.003
        elif mo in [11, 12, 1, 2, 3, 4]:   # wet season
            mu_acc = 0.001
        else:                        # transitional
            mu_acc = 0.002
        mu_stoch = max(0.0, float(rng.normal(mu_acc, mu_acc * 0.23)))
        eta_clean = float(rng.beta(8, 2)) if precip[i] > 5.0 else 0.0
        prev = l_soil[i - 1] if i > 0 else 0.0
        l_soil[i] = float(np.clip(prev + mu_stoch * 30 - eta_clean, 0.0, 0.15))

    # ── Component 3: Inverter efficiency (Sandia model) ──────────────────
    # η_inv(p) = η_max × p / (p + k),  p normalised [0,1]
    p_norm  = np.clip(ghi / 6.0, 0.01, 1.0)
    eta_inv = np.clip(0.98 * p_norm / (p_norm + 0.06), 0.85, 0.98)
    eta_inv *= rng.beta(95, 5, N)       # MPPT variation

    # ── Component 4: Spectral mismatch ───────────────────────────────────
    l_spec = (0.050 * (cloud / 100.0)
              + 0.030 * (rh2m / 100.0 - 0.60)
              + rng.normal(0, 0.015, N))
    l_spec = np.clip(l_spec, 0.0, 0.15)

    # ── Component 5: Module degradation ──────────────────────────────────
    # r_d ~ N(0.0095, 0.0025²) %/year
    r_d   = float(rng.normal(0.0095, 0.0025))
    r_deg = (1.0 - r_d) ** (idx / 12.0)

    # ── Component 6: Wiring losses ───────────────────────────────────────
    l_wire = 0.015 + 0.0002 * (t2m - 25.0) + rng.normal(0, 0.003, N)
    l_wire = np.clip(l_wire, 0.005, 0.025)

    # ── Component 7: Atmospheric intermittency ───────────────────────────
    # σ_regime: clear=0.05, broken-cloud=0.12, overcast=0.08
    sigma_xi = np.where(cloud < 40, 0.05,
                np.where(cloud < 70, 0.12, 0.08))
    xi = np.clip(
        rng.normal(1.0, sigma_xi)
        * (1.0 + rng.normal(0.005, 0.025, N)),   # satellite bias
        0.60, 1.20
    )

    # ── Assemble PR ──────────────────────────────────────────────────────
    PR_raw = (temp_fac
              * (1.0 - l_soil)
              * (1.0 - l_spec)
              * eta_inv
              * r_deg
              * (1.0 - l_wire)
              * xi)
    PR_raw = np.clip(PR_raw, 0.40, 1.00)

    # Calibrate to target distribution: μ = 0.76, σ = 0.04
    PR = (PR_raw - PR_raw.mean()) / PR_raw.std() * 0.04 + 0.76
    PR = np.clip(PR, 0.55, 0.92)

    Y = PR * ghi

    return {
        "PR":   PR,
        "Y":    Y,
        "losses": {
            "temp_fac":  temp_fac,
            "l_soil":    l_soil,
            "l_spec":    l_spec,
            "eta_inv":   eta_inv,
            "r_deg":     r_deg,
            "l_wire":    l_wire,
            "xi":        xi,
            "t_cell":    t_cell,
        },
    }
