"""
PHASE 1 — CONFIGURATION AND SHARED ESTIMATION PRIMITIVES

This module defines the parameters of the discretized Generalized
Langevin Equation (GLE) governing the currency pair's log-return
process, together with the estimation primitives common to every
phase that performs a regression.

Mathematical role
------------------
The continuous-time GLE

    dr/dt = Omega*r(t) - integral_0^t K(t-tau) r(tau) dtau + eta(t)

is discretized on a daily grid as a linear autoregressive model with
exogenous inputs (ARX):

    r(t+1) = sum_{k=1}^{p} K_k * r(t-k)      [memory kernel]
           + sum_j Omega_j * x_j(t)          [network coupling]
           + eta(t)                          [noise]

choose_lag_order() selects the memory-kernel order p by Akaike
Information Criterion (AIC) applied to r(t) alone -- a scalar
approximation to full VARX order selection, justified because at daily
granularity the own-process autocorrelation structure dominates
lag-order identifiability.

build_design_matrix() assembles the regression matrix X (own lags plus
every exogenous column supplied downstream) and target y(t)=r(t+1),
i.e. the exact discretization of the GLE above.

make_scaled_elasticnet_pipeline() implements the estimator that
performs projection-operator model reduction in practice: an L1/L2
(elastic-net) penalty applied to a standardized design matrix.
Standardization (zero mean, unit variance per column) is mathematically
required before L1 penalization, because the L1 norm ||beta||_1 =
sum|beta_j| penalizes raw coefficient magnitude, which is
unit-dependent; without rescaling, a feature measured in large natural
units would be penalized more heavily per unit of true predictive
contribution than a feature measured in small units, biasing which
coefficients survive shrinkage independent of their actual explanatory
power. After standardization, the estimator's active set (nonzero
coefficients) is the estimated topology of the reduced network,
J* = {j : Omega_j != 0}: the concrete numerical realization of
Mori-Zwanzig projection, in which every variable outside J* has been
projected onto the residual eta(t), exactly as the formalism
prescribes.
"""

import numpy as np
import pandas as pd

# ── the currency pair under study ────────────────────────────────────
CURRENCY_PAIR = "REPLACE_WITH_YOUR_PAIR"   # any ticker your data source understands

# ── candidate coupled instruments ("network neighbors") ──────────────
# you decide what belongs here -- other pairs, commodities, indices,
# rate proxies, anything you hypothesize is coupled. Selection of what
# actually matters happens statistically in Phase 4, not here.
NEIGHBORS = [
    # {"name": "REPLACE_ME", "ticker": "REPLACE_ME"},
]

# ── news topic categories ─────────────────────────────────────────────
NEWS_CATEGORIES = [
    # "REPLACE_ME",
]

# ── demo-mode fallback sizes (used ONLY if the lists above are empty) ──
NUM_SYNTHETIC_NEIGHBORS_DEFAULT = 6
NUM_SYNTHETIC_NEWS_DEFAULT = 4
NUM_TRULY_COUPLED_DEFAULT = 2

# ── data window ────────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

# ── state space ─────────────────────────────────────────────────────────
VOLATILITY_WINDOW = 5

# ── model hyperparameters (shared by Phase 4 and Phase 7 -- one grid,
#    fit once, no inconsistency between the candidate fit and the
#    backtest refits) ──────────────────────────────────────────────────
MAX_LAG = 5
ELASTICNET_ALPHAS = np.logspace(-6, -1, 30)
ELASTICNET_L1_RATIOS = [0.2, 0.5, 0.8, 1.0]
ELASTICNET_CV_SPLITS = 5

N_SIMS = 1000
FORECAST_HORIZON = 30
REFIT_EVERY = 20
TEST_WINDOW = 200
RANDOM_SEED = 42

NEWS_API_KEY_ENV_VAR = "NEWS_API_KEY"
NEWS_API_LOOKBACK_WARNING = (
    "NewsAPI's free tier only serves the last ~30 days on /everything -- "
    "it cannot backfill a multi-year window. For historical training data, "
    "use an archival news source instead and reserve NewsAPI for live/"
    "ongoing updates once the model is deployed."
)


# ── shared feature-matrix builders (single source of truth for phases
#    4, 6, and 7 -- previously duplicated three times with drift risk) ──

def choose_lag_order(r: pd.Series, max_lag: int = MAX_LAG) -> int:
    """AIC-based lag order for the memory kernel, from r's own history.
    Approximate: ignores exogenous features, which keeps this a cheap
    O(max_lag) scan instead of a full VARX order search."""
    from statsmodels.tsa.ar_model import ar_select_order
    sel = ar_select_order(r.values, maxlag=max_lag, ic="aic")
    p = len(sel.ar_lags) if sel.ar_lags else 1
    return max(1, min(p, max_lag))


def build_design_matrix(df: pd.DataFrame, p: int):
    """Target: next-day return. Features: own lags 1..p (memory kernel)
    + every other column (network neighbors + news, whatever Phase 3
    produced -- nothing hardcoded)."""
    other_cols = [c for c in df.columns if c not in ("r", "sigma")]
    X = pd.DataFrame(index=df.index)
    for k in range(1, p + 1):
        X[f"r_lag{k}"] = df["r"].shift(k)
    for c in other_cols:
        X[c] = df[c]
    y = df["r"].shift(-1).rename("r_next")
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data["r_next"], list(X.columns)


def build_no_memory_matrix(df: pd.DataFrame):
    """Same feature set minus the own-lag (memory-kernel) columns --
    the baseline that isolates whether the memory kernel earns its
    keep (used by Phase 7)."""
    other_cols = [c for c in df.columns if c not in ("r", "sigma")]
    X = df[other_cols].copy()
    y = df["r"].shift(-1).rename("r_next")
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data["r_next"]


def make_elasticnet_cv():
    """Elastic-net regression with the regularization path (alpha) and
    the L1/L2 mixing ratio (l1_ratio) selected by k-fold cross-
    validation on a TimeSeriesSplit -- folds respect temporal order, so
    no fold is validated on data preceding its own training window.
    Used identically wherever the discretized GLE is fit: the candidate
    model estimation in Phase 4 and the walk-forward backtest in
    Phase 7."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    return ElasticNetCV(
        l1_ratio=ELASTICNET_L1_RATIOS,
        alphas=ELASTICNET_ALPHAS,
        cv=TimeSeriesSplit(n_splits=ELASTICNET_CV_SPLITS),
        max_iter=20000,
        random_state=RANDOM_SEED,
    )


def make_scaled_elasticnet_pipeline():
    """Composes a per-column standardization (subtract mean, divide by
    standard deviation) with the elastic-net regression. Every
    candidate feature is z-scored before the L1/L2 penalty is applied,
    so the resulting sparse selection reflects genuine explanatory
    contribution to r(t+1) rather than the arbitrary measurement units
    of each feature."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("model", make_elasticnet_cv())])