# Pharmaceutical Sales Forecasting with SARIMA

A mini end-to-end time series project demonstrating how SARIMA models can be applied to seasonal demand forecasting — inspired by real-world use cases in pharmaceutical and retail sales.

---

## Overview

This project walks through the full pipeline of time series analysis and forecasting using a synthetic monthly sales dataset provided by AI (36 months, 2022–2024). The goal is to model seasonal demand patterns — such as flu medication peaking in winter — and generate a 12-month forecast for 2025.

The project was built as a learning exercise to understand the mathematical foundations of ARIMA/SARIMA models, including stationarity, differencing, ACF/PACF diagnostics, and forecast uncertainty.

---

## What This Project Covers

- Generating synthetic time series data with trend, seasonality, and noise
- Time series decomposition (trend / seasonality / residuals)
- Stationarity testing via visual inspection and differencing
- ACF and PACF plots for parameter selection
- SARIMA model fitting using `statsmodels`
- 12-month forecast with confidence intervals
- Honest evaluation of model diagnostics

---

## Results

The fitted model was **SARIMA(1,1,1)(1,1,1,12)** — one autoregressive term, one differencing step, one moving average term, with a seasonal period of 12 months.

### Forecast Plot
![Forecast](forecast_plot.png)

### Decomposition Plot
![Decomposition](decomposition_plot.png)

### Model Summary (SARIMAX Results)

```
SARIMAX(1, 1, 1)x(1, 1, 1, 12)
No. Observations: 36
Log Likelihood:  -30.925
AIC:              71.850
BIC:              72.836
```

| Parameter  | Coefficient | p-value |
|------------|-------------|---------|
| ar.L1      | -0.4294     | 0.401   |
| ma.L1      | -1.0000     | 1.000   |
| ar.S.L12   | -0.6126     | 0.581   |
| ma.S.L12   | -1.0006     | 1.000   |

**Diagnostic tests:**
- Ljung-Box Q test: p = 0.27 (residuals show no significant autocorrelation)
- Jarque-Bera test: p = 0.92 (residuals are approximately normally distributed)
- Heteroskedasticity: p = 0.09 (borderline, no strong evidence of changing variance)

---

## Limitations & Honest Notes

This project uses a **synthetic dataset of only 36 observations**, which leads to a known issue visible in the model summary: the MA terms (`ma.L1`, `ma.S.L12`) have extremely large standard errors and p-values of 1.000. This indicates that the optimizer could not reliably estimate these parameters — a direct consequence of the small sample size combined with the synthetic data being "too clean."

In practice, SARIMA models are most reliable with **at least 3–5 full seasonal cycles** of real-world data (ideally 5+ years of monthly data, i.e. 60+ observations). The core methodology demonstrated here is valid; the diagnostic results are an honest reflection of the dataset's limitations, not a bug.

A natural next step would be to test this pipeline on a real public dataset. 

---

## Project Structure

```
ARIMA_SARIMA/
│
├── script.py          # Main analysis script
├── README.md          # This file
└──  plots  # Forecast output plot and Decomposition output plot

```

---

## Requirements

```
statsmodels
matplotlib
pandas
numpy
```

Install with:

```bash
pip install statsmodels matplotlib pandas numpy
```

---

## How to Run

```bash
python script.py
```

The script will produce three plots sequentially:
1. Time series decomposition
2. ACF and PACF diagnostics
3. SARIMA forecast with confidence intervals

---

## Key Concepts

**SARIMA(p,d,q)(P,D,Q,m)** — Seasonal AutoRegressive Integrated Moving Average

| Parameter | Meaning |
|-----------|---------|
| p | AR order — how many past values to use |
| d | Differencing order — steps to achieve stationarity |
| q | MA order — how many past errors to use |
| P, D, Q | Same as above but at the seasonal lag |
| m | Seasonal period (12 for monthly data) |

**Stationarity** — A series whose mean and variance do not change over time. Required for ARIMA. Achieved here via first-order differencing (d=1).

**ACF/PACF** — Autocorrelation and Partial Autocorrelation plots used to visually select the p and q parameters before fitting.



