# Extended Methodology Notes

## Circular Tautology in Deterministic PV Modeling

The original manuscript (before correction) used Y_det = GHI × η_STC × [1−β(T_cell−25)]
as the regression target, with GHI and T2M as predictors.

Since T_cell = T2M + f(GHI), the target is algebraically:
    Y_det = g(GHI, T2M)  →  R²(OLS on GHI, T2M) ≈ 0.9997

**This is a mathematical identity, not predictive skill.**

## Variance Decomposition of Stochastic PR

| Loss Component | Share of σ²_PR |
|----------------|---------------|
| Atmospheric intermittency | 56.3% |
| Seasonal soiling | 25.1% |
| Spectral mismatch | 11.0% |
| Temperature coefficient | 4.9% |
| Other (inverter, degradation, wiring) | 2.7% |

## SARIMAX Specification

Model: SARIMAX(1,0,1)(1,1,1)₁₂

    (1−Φ₁L¹²)(1−φ₁L)(1−L¹²)Y_t = (1+Θ₁L¹²)(1+θ₁L)ε_t + β_ONI·ONI_t + β_DMI·DMI_t

Key estimated parameters (Table 3):
- φ₁ (AR1) = 0.387 (p<0.001)
- Φ₁ (SAR12) = 0.312 (p<0.001)
- β_ONI = +0.0647 (p=0.003) ← El Niño enhances PV output (suppresses clouds)
- β_DMI = −0.0423 (p=0.029) ← Positive IOD partially offsets El Niño
