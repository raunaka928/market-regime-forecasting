# Market Regime Forecasting: Predicting Volatility Regimes with ML

Binary classification on 25 years of SPY and VIX data, forecasting next-day market volatility regime. Four models of increasing complexity — linear, ensemble, kernel, and sequential — are compared to isolate what each additional capability contributes to minority-class recall.

**Course:** DADS7275 — Machine Learning and Data Analytics, Northeastern University
**Author:** Raunak Amanna

---

## Problem

Directional price forecasting on a broad equity index is close to a coin flip at daily horizons. This project reframes the target from *price* to *market state*: given today's price-derived technical indicators, will tomorrow be a high-volatility regime or a calm one?

A day is labeled **High Risk (1)** when VIX closes above 20, and **Stable (0)** otherwise. The label is shifted forward one day so that features observable at close on day *t* predict the regime of day *t+1*.

Because a missed volatility spike leaves a portfolio fully exposed to a drawdown while a false alarm costs roughly one day of foregone return, the governing metric is **recall on the High Risk class**, not accuracy. Target: recall > 0.60.

## Data

| | |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Window | 2000-01-01 to 2025-01-01 |
| Instruments | `SPY` (features), `^VIX` (labels) |
| Rows after cleaning | **6,089** trading days |

`auto_adjust=True` on the SPY download so prices account for dividends and splits — without it, split events create artificial discontinuities that propagate into every rolling feature. VIX is used **only** to construct labels and is never provided as a feature (see Design Notes).

## Features

Raw close price is non-stationary — SPY traded near $100 in 2000 and near $500 in 2024 — so every feature is constructed to be stationary and scale-independent.

| Feature | Definition |
|---|---|
| `Returns` | `Price.pct_change()` |
| `RSI` | 14-period, `100 − 100/(1 + RS)`, `RS = mean(gains)/mean(losses)` |
| `SMA_50_Dist` | `(Price − SMA₅₀) / SMA₅₀` |
| `SMA_200_Dist` | `(Price − SMA₂₀₀) / SMA₂₀₀` |

Expressing the moving averages as *fractional deviation from trend* means −0.10 reads as "10% below trend" identically whether the SMA is 120 or 480, making the feature comparable across the full 25-year span.

## Leakage controls

1. **One-day forward label shift.** Without it, the target would be same-day regime — derivable from same-day VIX and therefore not a forecasting problem.
2. **Strictly chronological split** at 2020-01-01. Train ≈ 4,832 days (2000–2019), test = 1,257 days (2020–2025). A random split would let 2023 data inform predictions about 2015.
3. **Scaler fit on training data only** — `fit_transform` on train, `transform` on test, no refit.

The Keras `validation_split=0.2` takes the final 20% of the training array before shuffling, so the LSTM's validation fold is chronologically 2016–2019 — correct for time series.

## Models

All sklearn models use `class_weight='balanced'` and `random_state=42`. A `DummyClassifier(strategy='most_frequent')` provides a floor reference.

**1. Logistic Regression** — linear baseline. Included so the inadequacy of a linear boundary is demonstrated rather than assumed.

**2. Random Forest** — 100 trees, `max_depth=5`. Represents conjunctions like "low RSI **and** negative returns **and** negative SMA divergence," a three-way interaction no hyperplane can express.

**3. SVM, RBF kernel** — `SVC(kernel='rbf', probability=True)`. The kernel implicitly maps the 4-D feature space into an infinite-dimensional space where a linear separator corresponds to a flexible non-linear boundary. Mechanistically distinct from the Random Forest, so it independently confirms the non-linearity finding.

**4. LSTM** — the only model with temporal context. Input is a 10-trading-day rolling window of all four features, shape `(n, 10, 4)`.

```python
Sequential([
    LSTM(32, input_shape=(10, 4)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
# Adam, binary_crossentropy, 30 epochs max, batch_size=32
# EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

The 10-day window makes the first 10 test rows unusable, so the LSTM is evaluated on **1,247** rows rather than 1,257.

## Results — test set 2020–2025

| Model | Recall (Class 1) | Precision | F1 | Accuracy | ROC-AUC | TP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.54 | 0.91 | 0.68 | 0.75 | 0.769 | 326 | 278 |
| Random Forest | 0.59 | 0.89 | 0.71 | 0.77 | 0.868 | 356 | 248 |
| SVM (RBF) | 0.66 | 0.88 | 0.75 | 0.79 | 0.870 | 399 | 205 |
| **LSTM** | **0.72** | 0.85 | 0.78 | 0.80 | **0.884** | ~435 | ~169 |

*Class-1 support: 604.*

**Recall improves monotonically with complexity: 0.54 → 0.59 → 0.66 → 0.72.** Each increment is attributable to a specific added capability:

- **LR → RF** (+0.05, 30 more true positives): non-linear multi-feature interactions
- **RF → SVM** (+0.07, 43 fewer false negatives): margin optimization in a projected space
- **SVM → LSTM** (+0.06, ~36 fewer false negatives): temporal memory

End to end, the LSTM **reduces false negatives by 39%** relative to the linear baseline (278 → ~169). Precision degrades gradually (0.91 → 0.85), which is the correct trade under the cost asymmetry above: 77 false positives is strictly preferable to 278 missed spikes.

## Diagnostics

- **ROC curves** — Logistic Regression separates visibly from the three non-linear models across the full curve, confirming its weakness is structural rather than a threshold artifact. RF and SVM nearly overlap; LSTM dominates.
- **Precision-Recall curves** — in the operationally relevant 0.6–0.8 recall band, the LSTM sustains higher precision than every other model, evidence that it is more *discriminating* rather than merely more aggressive.
- **Threshold sweep** on Random Forest at 0.30 / 0.40 / 0.50 / 0.60, mapping the precision-recall frontier explicitly instead of accepting the sklearn default.
- **Confusion matrices** for all models, with the LSTM plotted separately due to the row-count difference.

## Feature importance

`SMA_200_Dist` ranks first and `RSI` second in **both** the Logistic Regression coefficients and the Random Forest Gini importances:

| Feature | LR coefficient |
|---|---|
| `SMA_200_Dist` | −1.6452 |
| `RSI` | −0.4902 |
| `SMA_50_Dist` | +0.3002 |
| `Returns` | −0.0256 |

Agreement across two mechanistically unrelated fitting procedures is stronger evidence that long-term structural trend breakdown is a genuine precursor than either result would be alone. All coefficient signs are economically coherent: price below trend, weakening momentum, and negative returns all associate with elevated forward volatility.

## EDA findings

- **Class balance** — 3,788 Stable (62.2%) / 2,301 High Risk (37.8%), ratio ≈ 1.65:1. Motivates balanced class weighting and rules out accuracy as a metric (a constant-Stable predictor scores 62%).
- **VIX distribution** — strongly right-skewed, bulk below 20 with a tail to ~80. This is what makes 20 a defensible cut point rather than an arbitrary one.
- **Regime clustering** — High Risk labels land precisely on known macro shocks (dot-com 2000–02, GFC 2008–09, Euro debt 2011, China selloff 2015–16, COVID 2020) without being told about them. Critically, regimes arrive in *persistent blocks* rather than i.i.d. draws — the direct justification for a sequence model.
- **Class-conditional distributions** — separate in their means (RSI averages 48.6 on High Risk days vs 59.8 on Stable) but overlap heavily through their bodies. No single feature admits a clean threshold, so discriminating power must come from joint interaction.
- **Target correlations** — `SMA_200_Dist` −0.526, `SMA_50_Dist` −0.361, `RSI` −0.336, `Returns` −0.079.

## Design notes

**Distribution shift is present and intentional.** The train period is 35.1% High Risk; the test period is 48.0%. The 2020–2025 window contains the COVID crash, the 2022 bear market, and the rate-hike cycle, making it substantially more volatile than the training era. This mirrors deployment — the future is always drawn from an unknown distribution — but means test metrics are not in-distribution and should be read accordingly.

**VIX is deliberately excluded from the feature set.** VIX is strongly autocorrelated, so a persistence rule ("tomorrow's regime = today's regime") would score high recall while learning nothing about precursors. The research question here is narrower and more interesting: *can price-derived technical indicators alone anticipate a regime shift?* Excluding VIX and current regime from the features is what makes that question answerable. See `additional_analysis.py` for a persistence benchmark quantifying the gap.

## Limitations

- **Four features only.** A production system would add rates and the yield curve, macro releases, cross-asset signals (credit spreads, DXY), and sentiment/positioning data.
- **Binary threshold discards magnitude.** VIX 21 and VIX 80 receive the same label. A three-class scheme (Stable / Elevated / Crisis) would support graduated position sizing.
- **No hyperparameter tuning.** No grid search, no `TimeSeriesSplit` cross-validation. RF depth, SVM defaults, and the LSTM architecture were fixed a priori, so all results are single-configuration point estimates with no variance estimate.
- **Classification metrics are not economic metrics.** "Catches 72% of high-risk days" does not establish profitability. There is no backtest, no transaction-cost model, and no buy-and-hold comparison, so no claim about returns or drawdown reduction is supported by this work.
- **RSI uses simple rolling means** for average gain/loss rather than Wilder's exponential smoothing — a common simplification, but technically a different indicator from the canonical formulation.

## Contents

| File | Description |
|---|---|
| `market_regime_forecasting.ipynb` | End-to-end notebook: ingestion, feature engineering, four EDA figures, five models, threshold sweep, ROC/PR curves, confusion matrices |
| `report.pdf` | Full written report with methodology, figures, per-model analysis, and discussion |
| `additional_analysis.py` | Persistence baseline and SHAP interpretability cells to append to the notebook |
| `requirements.txt` | Dependencies |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook market_regime_forecasting.ipynb
```

Data is fetched live from Yahoo Finance at runtime, so figures may shift marginally if the underlying series is revised. Seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`, `random_state=42`).

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Lundberg, S., & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Wilder, J. W. (1978). *New concepts in technical trading systems.* Trend Research.
