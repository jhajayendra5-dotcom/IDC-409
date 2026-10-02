"""
PHASE 1 — CONFIGURATION AND SHARED ESTIMATION PRIMITIVES

Framework
---------
The object of study is the log-return process r(t) of a currency pair,
treated as an open, driven, dissipative complex system. r(t) is one
degree of freedom of a larger system whose remaining degrees of freedom
(other instruments, information flow, order flow) are unobserved or only
partially observed. The framework approximates the reduced dynamics of
r(t) by a Generalized Langevin Equation (GLE) with memory and temporally
dependent noise, estimated entirely from data:

    dr/dt = Omega*r(t) - int_0^t K(t-tau) r(tau) dtau
            + sum_j Omega_j x_j(t) + eta(t)

K is the memory kernel, x_j(t) are observed exogenous drives (network
neighbors and news channels), and eta(t) is the noise process.

Assumptions made and not made
-----------------------------
  - No equilibrium. The Mori-Zwanzig projection that produces the GLE
    structure is an exact identity for arbitrary dynamics, in or out of
    equilibrium. Nothing here imposes a fluctuation-dissipation relation
    between K and the noise autocorrelation, a Boltzmann ensemble or
    temperature for the projected-out variables, or detailed balance. K
    and eta are estimated from data with no relation imposed between
    them, and departure from time-reversal symmetry is measured
    (Phase 5), not assumed away.
  - No Markov property. The memory integral makes r(t) non-Markovian on
    its own. The memory depth p is selected from data (AIC); it
    truncates a kernel whose support may be unbounded, and its adequacy
    is tested (Phase 7), not presumed.
  - Stationarity is used; equilibrium is not. Fixed-coefficient
    estimation requires the statistics of r(t) to be invariant under time
    translation (verified in Phase 2). A driven non-equilibrium steady
    state satisfies this. Parameter drift is tested in Phase 7.

Discretization
--------------
On a daily grid the GLE becomes a linear autoregression with exogenous
inputs:

    r(t+1) = sum_{k=1}^{p} K_k * r(t-k+1)     [memory kernel, r_lag1 = r(t)]
           + sum_j Omega_j * x_j(t)           [drive coupling]
           + eta(t)                           [noise]

Functions
---------
choose_lag_order() selects the memory depth p by Akaike Information
Criterion applied to r(t) alone -- a scalar approximation to full VARX
order selection, justified because at daily granularity the own-process
autocorrelation structure dominates lag-order identifiability.

build_design_matrix() assembles X (own lags, indexed so that r_lag1 is
r(t), the most recent value known when r(t+1) is predicted, plus every
drive column and the observed variable sigma(t)) and y(t) = r(t+1).

make_scaled_elasticnet_pipeline() is the sparse estimator: an L1/L2
(elastic-net) penalty on a standardized design matrix. Standardization
is required because the L1 norm penalizes raw coefficient magnitude,
which depends on measurement units; without it, a feature in large
natural units is penalized more per unit of true contribution than one
in small units. After standardization the active set (nonzero
coefficients) is the estimated set of couplings J* = {j : Omega_j != 0};
every variable outside J* is treated as absorbed into eta(t).

check_stationarity(), check_cointegration(), check_structural_break()
implement three econometric tests used downstream (Phases 2, 3, 7):
unit-root stationarity of r(t), Engle-Granger cointegration between
price levels, and CUSUM parameter stability.
"""

import numpy as np
import pandas as pd

# ── the currency pair under study ────────────────────────────────────
# USD/INR. India's exchange rate regime is a managed float (RBI
# intervenes at its discretion; it is not a hard peg), which is the
# relevant context for interpreting both the network-coupling
# estimates in Phase 4 and the structural-break diagnostic in Phase 7:
# discretionary intervention is a plausible source of regime shifts
# that a fixed-coefficient linear model cannot represent internally.
CURRENCY_PAIR = "USDINR=X"

# ── candidate coupled instruments ("network neighbors") ──────────────
# Each entry's ticker is a Yahoo Finance/yfinance quote symbol.
# Selection of which of these actually carry predictive weight is not
# decided here -- it is estimated statistically by the sparse
# regression in Phase 4.
#
# The set is chosen to minimize pairwise redundancy among candidate
# regressors: DXY alone represents broad dollar strength, since EUR/USD
# and USD/JPY are its two largest constituent weights (57.6% and
# 13.6%) and so carry little information beyond it; Brent alone
# represents crude-oil exposure, since WTI and Brent move together at
# above 0.9 correlation and Brent is the more relevant reference for
# Asian import pricing; gold alone represents precious-metals exposure,
# since gold and silver are themselves highly correlated and gold is
# the better-documented driver of India's import bill. A candidate set
# with high pairwise correlation would let the elastic net's L2
# component spread a single true effect's weight across several
# collinear columns rather than attributing it to one, which weakens
# the network-topology interpretation of Phase 4's active-set output.
NEIGHBORS = [
    {"name": "dxy",      "ticker": "DX-Y.NYB"},   # US Dollar Index
    {"name": "brent_oil","ticker": "BZ=F"},       # crude oil, Brent -- India is a net oil
                                                   # importer, so the rupee is structurally
                                                   # exposed to crude oil price shocks
    {"name": "gold",     "ticker": "GC=F"},       # gold; India is among the largest
                                                   # physical-demand markets for gold
    {"name": "vix",      "ticker": "^VIX"},       # equity implied-volatility index,
                                                   # a standard risk-on/risk-off proxy
    {"name": "sp500",    "ticker": "^GSPC"},      # US equity index
    {"name": "nikkei",   "ticker": "^N225"},      # Japan equity index -- retained
                                                   # alongside sp500 despite some
                                                   # correlation, since it is the
                                                   # only Asian-session risk proxy
                                                   # in the set
    {"name": "us10y",    "ticker": "^TNX"},       # US 10-year Treasury yield; the US leg
                                                   # of the interest-rate differential
                                                   # driving carry-trade flows into/out of INR.
                                                   # NOTE: no equivalent free, machine-readable
                                                   # yfinance series exists for the India 10-year
                                                   # government bond yield at the time of writing;
                                                   # the India leg of the differential is therefore
                                                   # not represented here. A connector to an
                                                   # official RBI/CCIL data feed or FRED's India
                                                   # series would close this gap.
    {"name": "usdcny",   "ticker": "USDCNY=X"},   # renminbi; a standard proxy for broad
                                                   # emerging/Asian-currency sentiment not
                                                   # already captured by DXY (CNY is not a
                                                   # DXY basket component)
    {"name": "btc_usd",  "ticker": "BTC-USD"},    # cryptocurrency, tested here purely as a
                                                   # candidate risk-sentiment correlate
]

# ── news topic categories ─────────────────────────────────────────────
# Free-text query strings, each traceable to a specific macro-driver or
# news-and-sentiment concept. No topic is assumed relevant in advance;
# as with NEIGHBORS, relevance is determined statistically in Phase 4,
# not by this list.
#
# The set is chosen for minimal semantic overlap: "Federal Reserve
# interest rate" alone represents the Fed-policy-expectations channel,
# since topics such as inflation prints, employment data, and
# quantitative easing are themselves largely proxies for the same
# underlying channel; "geopolitical risk" and "trade war tariffs"
# jointly span adversarial-policy news without a separate "sanctions"
# category, which would overlap both. Each topic issues at least one
# request against the news source (see NEWS_API_LOOKBACK_WARNING
# below), so the size of this list is a direct driver of API-quota
# consumption.
NEWS_CATEGORIES = [
    "Federal Reserve interest rate",
    "European Central Bank policy",
    "Bank of Japan monetary policy",
    "India inflation CPI",
    "India GDP growth",
    "India trade deficit current account",
    "geopolitical risk",
    "trade war tariffs",
    "risk appetite safe haven flows",
]

# ── demo-mode fallback sizes (used ONLY if the lists above are empty) ──
NUM_SYNTHETIC_NEIGHBORS_DEFAULT = 6
NUM_SYNTHETIC_NEWS_DEFAULT = 4
NUM_TRULY_COUPLED_DEFAULT = 2
NUM_TRUE_MEMORY_LAGS_DEFAULT = 3

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
# Concentrated toward the L1 end of the mixing range: the active set
# is interpreted as the estimated network topology (Phase 4), and a
# mixing ratio weighted toward the L2 term tends to spread a single
# true effect's weight across collinear columns (a "grouping effect")
# rather than concentrating it on one, which works against the sparse,
# individually-attributable coupling estimate the interpretation
# requires.
ELASTICNET_L1_RATIOS = [0.5, 0.7, 0.85, 1.0]
ELASTICNET_CV_SPLITS = 5

# ── irreversibility diagnostic (Phase 5) ──────────────────────────────
IRREVERSIBILITY_WINDOW = 10        # L: path length, in days, of the window whose forward
                                   # and time-reversed path measures are compared
N_REVERSIBLE_SURROGATES = 200      # size of the reversible-null ensemble
SPECTRAL_SMOOTHING_HALF_WIDTH = 5  # frequency bins each side, smoothing the cross-spectral
                                   # matrix from which the reversible surrogates are drawn

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


# ── shared feature-matrix builders and estimators (the single source
#    of truth used identically by phases 4, 6, and 7) ──────────────────

def choose_lag_order(r: pd.Series, max_lag: int = MAX_LAG) -> int:
    """AIC-based lag order for the memory kernel, from r's own history.
    Approximate: ignores exogenous features, which keeps this a cheap
    O(max_lag) scan instead of a full VARX order search."""
    from statsmodels.tsa.ar_model import ar_select_order
    sel = ar_select_order(r.values, maxlag=max_lag, ic="aic")
    p = len(sel.ar_lags) if sel.ar_lags else 1
    return max(1, min(p, max_lag))


def build_design_matrix(df: pd.DataFrame, p: int):
    """Target: next-day return, y(t) = r(t+1). Features: own lags
    r_lag1,...,r_lag_p (the memory kernel) plus every other column
    (network neighbors, news, and sigma -- whatever Phase 3 produced,
    nothing hardcoded).

    r_lag_k(t) = r(t-k+1), so r_lag1 = r(t): the most recent return
    value known at the moment r(t+1) is predicted, and r_lag_p =
    r(t-p+1), the oldest value within the memory-kernel window. This
    indexing convention is shared exactly by Phase 6's forward
    simulation, which seeds its own lag buffer with the most recently
    observed or simulated return as r_lag1, so a fitted coefficient K_k
    always multiplies the same relative position in a return's own
    history at both estimation and simulation time.

    sigma(t), the realized-volatility component of the state vector
    defined in Phase 2, is included as an ordinary regressor alongside
    the network and news columns: it is a nonlinear function of recent
    returns (a rolling standard deviation), not a linear combination of
    the r_lag columns, so it can carry genuine information -- e.g. an
    asymmetric volatility-level effect on the conditional mean -- that
    the linear lag terms alone cannot represent."""
    other_cols = [c for c in df.columns if c != "r"]
    X = pd.DataFrame(index=df.index)
    for k in range(1, p + 1):
        X[f"r_lag{k}"] = df["r"].shift(k - 1)
    for c in other_cols:
        X[c] = df[c]
    y = df["r"].shift(-1).rename("r_next")
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data["r_next"], list(X.columns)


def build_no_memory_matrix(df: pd.DataFrame):
    """Same feature set as build_design_matrix minus the own-lag
    (memory-kernel) columns -- the baseline that isolates whether the
    memory kernel earns its keep (used by Phase 7). sigma(t) and every
    network/news column are retained, since neither is an own-lag
    term."""
    other_cols = [c for c in df.columns if c != "r"]
    X = df[other_cols].copy()
    y = df["r"].shift(-1).rename("r_next")
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data["r_next"]


def make_elasticnet_cv():
    """Elastic-net regression with the regularization path (alpha) and
    the L1/L2 mixing ratio (l1_ratio) selected by k-fold cross-
    validation on a TimeSeriesSplit -- folds respect temporal order, so
    no fold is validated on data preceding its own training window.
    Used for Phase 4's candidate-model fit, where alpha and l1_ratio are
    not yet known and must themselves be selected from data."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import TimeSeriesSplit
    return ElasticNetCV(
        l1_ratio=ELASTICNET_L1_RATIOS,
        alphas=ELASTICNET_ALPHAS,
        cv=TimeSeriesSplit(n_splits=ELASTICNET_CV_SPLITS),
        max_iter=20000,
        random_state=RANDOM_SEED,
    )


def check_stationarity(series: pd.Series):
    """Augmented Dickey-Fuller test of the null hypothesis that `series`
    contains a unit root (is non-stationary). Used on r(t) in Phase 2 to
    verify, rather than merely assert, that the log-difference transform
    has produced a series whose statistics do not depend on t -- the
    property fixed-coefficient estimation requires, and one that a
    stationary non-equilibrium steady state can satisfy as readily as an
    equilibrium state. Returns (adf_statistic, p_value); p < 0.05
    rejects the unit-root null, i.e. supports stationarity."""
    from statsmodels.tsa.stattools import adfuller
    stat, pvalue = adfuller(series.dropna().values, autolag="AIC",
                             result_object=False)[:2]
    return stat, pvalue


def check_cointegration(level_series_a: pd.Series, level_series_b: pd.Series):
    """Engle-Granger two-step test of the null hypothesis of no
    cointegrating relationship between two I(1) (unit-root, level)
    series. Applied in Phase 3 to price LEVELS (not returns) of the
    currency pair against each network neighbor: a significant result
    indicates the two series share a common stochastic trend, a
    distinct notion of long-run linkage from the short-run coupling
    Phase 4 estimates on returns. Returns (test_statistic, p_value)."""
    from statsmodels.tsa.stattools import coint
    aligned = pd.concat([level_series_a, level_series_b], axis=1).dropna()
    stat, pvalue, _ = coint(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
    return stat, pvalue


def check_structural_break(resid: np.ndarray):
    """CUSUM test (Ploberger-Kramer / Brown-Durbin-Evans family) of the
    null hypothesis of parameter stability, applied in Phase 7 to the
    candidate model's walk-forward residuals. Under the null, the
    cumulative sum of scaled OLS-type residuals stays within a
    Brownian-bridge critical envelope; a significant result indicates a
    regime shift the fixed-coefficient linear specification has not
    captured. The null distribution assumes approximately homoskedastic
    input: residuals with known, separately modeled heteroskedasticity
    (here, GARCH-driven volatility clustering) should be divided by
    their conditional standard deviation before this test, so that
    already-characterized second-order dependence is not mistaken for
    parameter instability. Returns (test_statistic, p_value)."""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    stat, pvalue, _ = breaks_cusumolsresid(resid)
    return stat, pvalue


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


def make_scaled_elasticnet_fixed(alpha: float, l1_ratio: float):
    """Fixed-hyperparameter counterpart to make_scaled_elasticnet_pipeline:
    re-estimates coefficients under a given (alpha, l1_ratio) directly,
    with no internal cross-validated search over the ~30x4x5 candidate
    grid. Used by Phase 7's walk-forward backtest with alpha and
    l1_ratio taken from Phase 4's cross-validated fit, treating the
    regularization strength as fixed over the evaluation horizon while
    the coefficients themselves are re-estimated on each expanding
    window -- standard walk-forward practice for a quantity, the
    optimal regularization strength, that is not expected to change
    materially every REFIT_EVERY days."""
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()),
                      ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                            max_iter=20000, random_state=RANDOM_SEED))])
