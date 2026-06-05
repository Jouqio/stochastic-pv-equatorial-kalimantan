"""
src/data_generation.py
======================
Synthetic monthly meteorological data generator for
Bontang, East Kalimantan (0.1333°N, 117.50°E).

Reproduces the 132-month (Jan 2015 – Dec 2025) dataset used in:
  Abdi, S.N. (2025). Stochastic Performance Modeling of PV Systems
  in Equatorial Maritime Climate. Applied Energy (under review).
"""

import numpy as np
import pandas as pd

__all__ = ["generate_bontang_met_data"]


def generate_bontang_met_data(
    n_months: int = 132,
    start_date: str = "2015-01-01",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly-averaged meteorological data for Bontang,
    East Kalimantan, representative of the Köppen Af equatorial climate.

    Parameters
    ----------
    n_months    : int   – Number of months (default 132 = Jan 2015–Dec 2025)
    start_date  : str   – First month as 'YYYY-MM-DD'
    random_seed : int   – NumPy random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        date, GHI, T2M, RH2M, WS10M, CLOUD_AMT, PRECTOTCORR, ONI, DMI, regime

    Notes
    -----
    ENSO regimes modelled:
        El Niño 2015–16  (ONI peak +2.6°C)
        Neutral  2017–19
        La Niña  2020–23  (34-month triple-dip, ONI ≈ −1.0°C)
        Recovery 2024–25
    """
    rng = np.random.default_rng(random_seed)
    idx = np.arange(n_months)
    dates = pd.date_range(start=start_date, periods=n_months, freq="MS")

    # ── Synthetic ONI (Oceanic Niño Index) ──────────────────────────────
    enso = np.zeros(n_months)
    enso[0:12]  =  2.0 + 0.6 * np.cos(2 * np.pi * np.arange(12) / 12)
    enso[12:24] =  0.2
    enso[24:36] = -0.3
    enso[36:48] =  0.4
    enso[48:60] = -0.6
    enso[60:84] = -1.0 - 0.2 * np.cos(2 * np.pi * np.arange(24) / 24)
    enso[84:]   = -0.3
    enso_factor = 1.0 + 0.05 * enso / 2.6

    # ── GHI (kWh/m²/day) ────────────────────────────────────────────────
    GHI = np.clip(
        (4.82 + 0.35 * np.sin(2 * np.pi * idx / 12 + 1.57)
         + rng.normal(0, 0.15, n_months)) * enso_factor,
        3.5, 6.0
    )

    # ── T2M (°C) ────────────────────────────────────────────────────────
    T2M = np.clip(
        26.5 + 1.5 * np.sin(2 * np.pi * idx / 12 + 0.5)
        + 0.3 * enso / 2.6 + rng.normal(0, 0.5, n_months),
        24.0, 28.5
    )

    # ── WS10M (m/s) ─────────────────────────────────────────────────────
    WS10M = np.clip(
        4.5 + 0.8 * np.sin(2 * np.pi * idx / 12 + 3.0)
        + rng.normal(0, 0.8, n_months),
        2.0, 7.0
    )

    # ── CLOUD_AMT (%) ────────────────────────────────────────────────────
    CLOUD = np.clip(
        71.4 + (GHI - 4.82) / (-0.06)
        - 10.0 * np.cos(2 * np.pi * idx / 12)
        + 8.0 * enso / 2.6
        + rng.normal(0, 4.0, n_months),
        30.0, 95.0
    )

    # ── RH2M (%) ─────────────────────────────────────────────────────────
    RH2M = np.clip(
        80.0 - 10.0 * (GHI - 4.82) / 1.5
        + 5.0 * np.sin(2 * np.pi * idx / 12 - 1.57)
        + 3.0 * enso / 2.6
        + rng.normal(0, 2.0, n_months),
        65.0, 95.0
    )

    # ── PRECTOTCORR (mm/day) ─────────────────────────────────────────────
    PREC = np.clip(
        8.3 + 6.0 + 4.0 * np.sin(2 * np.pi * idx / 12 - 1.57)
        + 2.0 * enso / 2.6
        + rng.gamma(3.0, 1.5, n_months),
        0.5, 25.0
    )

    # ── DMI (Dipole Mode Index) ──────────────────────────────────────────
    DMI = np.clip(
        np.roll(enso, 3) * 0.6 + rng.normal(0, 0.3, n_months),
        -2.0, 2.0
    )

    # ── ENSO regime labels ───────────────────────────────────────────────
    regime = np.where(enso > 0.5, "El Niño",
              np.where(enso < -0.5, "La Niña", "Neutral"))

    return pd.DataFrame({
        "date":        dates,
        "GHI":         GHI,
        "T2M":         T2M,
        "RH2M":        RH2M,
        "WS10M":       WS10M,
        "CLOUD_AMT":   CLOUD,
        "PRECTOTCORR": PREC,
        "ONI":         enso,
        "DMI":         DMI,
        "regime":      regime,
    })
