import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── 1. Φόρτωση δεδομένων ─────────────────────────────────────────────────────
# Μηνιαίες πωλήσεις φαρμάκου
dates = pd.date_range(start='2022-01', periods=36, freq='MS')
np.random.seed(42)
trend     = np.linspace(100, 150, 36)
seasonal  = np.array([35,25,5,-12,-22,-28,-30,-25,-12,5,22,38] * 3)
noise     = np.random.normal(0, 8, 36)
sales     = trend + seasonal + noise

df = pd.Series(sales, index=dates, name='Πωλήσεις')

# ── 2. Αποσύνθεση ────────────────────────────────────────────────────────────
decomp = seasonal_decompose(df, model='additive', period=12)
decomp.plot()
plt.suptitle('Αποσύνθεση Χρονοσειράς', fontsize=13)
plt.tight_layout()
plt.show()

# ── 3. ACF & PACF (για επιλογή p, q) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df.diff().dropna(),  lags=20, ax=axes[0], title='ACF')
plot_pacf(df.diff().dropna(), lags=16, ax=axes[1], title='PACF')
plt.tight_layout()
plt.show()

# ── 4. Εκπαίδευση SARIMA ─────────────────────────────────────────────────────
# SARIMA(p,d,q)(P,D,Q,m)
# p=1, d=1, q=1  ← από ACF/PACF
# P=1, D=1, Q=1  ← εποχιακό μέρος
# m=12           ← 12 μήνες ανά εποχή
model = SARIMAX(df,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)
print(results.summary())

# ── 5. Πρόβλεψη 12 μηνών (2025) ─────────────────────────────────────────────
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
conf_int      = forecast.conf_int()  # διάστημα εμπιστοσύνης

# ── 6. Γράφημα ───────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(df, label='Ιστορικά δεδομένα', color='steelblue')
plt.plot(forecast_mean, label='Πρόβλεψη 2025', color='tomato', linewidth=2)
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 alpha=0.2, color='tomato',
                 label='Διάστημα εμπιστοσύνης')
plt.title('Πρόβλεψη Πωλήσεων — SARIMA(1,1,1)(1,1,1,12)')
plt.xlabel('Ημερομηνία')
plt.ylabel('Πωλήσεις (χιλ.)')
plt.legend()
plt.tight_layout()
plt.show()