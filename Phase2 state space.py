"""
PHASE 2 — OBSERVED STATE CONSTRUCTION AND STATIONARITY CHECK

Role
----
Mori-Zwanzig projection presupposes a choice of observed variables;
every other degree of freedom then enters the reduced dynamics only
through the memory kernel and the noise. This phase defines the observed
variables of the currency pair:

    r(t)     = ln( P(t) / P(t-1) )
    sigma(t) = std( r(t-w+1 : t) ),  w = VOLATILITY_WINDOW

The pair [r(t), sigma(t)] is not presumed Markovian. The dynamics of
r(t) are non-Markovian, and the memory is carried explicitly by the
lagged terms of Phase 4. sigma(t) is a summary of the recent
second-moment history of r(t): a nonlinear function of recent returns,
supplied as an additional observed variable.

Log-returns are used because price is an integrated (unit-root)
process, whereas r(t) can have statistics that do not depend on t
(stationarity), which fixed-coefficient estimation requires.
Stationarity is a property of a steady state, not of equilibrium: a
driven, dissipative system can sustain a stationary non-equilibrium
steady state with nonzero probability currents and entropy production.
The ADF test below checks stationarity only.

Output: state_space.csv (date, price, r, sigma)

Synthetic fallback
------------------
When no live price feed is available, synthetic_universe() generates a
test process whose ground truth is known, so that downstream estimation
can be checked against it. It is a test fixture, not a model of any
market:

    r(t)   = sum_{k=1}^{q} K_k r(t-k) + sum_i Omega_i f_i(t-1) + eta(t)
    K_k    = K_1 * 0.5^(k-1),   q = NUM_TRUE_MEMORY_LAGS_DEFAULT
    eta(t) = sqrt(h(t)) z(t),   z(t) ~ N(0,1)
    h(t)   = omega + alpha r(t-1)^2 + beta h(t-1)       [GARCH(1,1)]

The f_i(t) are independent latent drives, a subset with nonzero
coupling Omega_i and the rest pure noise. The drives act on r(t) but r
does not act back on them; this directed coupling breaks the
time-reversal symmetry of the joint (r, f) process, which is the
property Phase 5 measures.
"""

import numpy as np
import pandas as pd
import phase1_config as cfg


def fetch_real_price(ticker: str, start: str, end: str) -> pd.Series:
    """Swap for whatever data source you use; only the return shape matters."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("empty download")
    return df["Close"].rename("price")


def _neighbor_names():
    if cfg.NEIGHBORS:
        return [n["name"] for n in cfg.NEIGHBORS]
    return [f"neighbor_{i+1}" for i in range(cfg.NUM_SYNTHETIC_NEIGHBORS_DEFAULT)]


def synthetic_universe(start: str, end: str, seed: int = None):
    seed = cfg.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    names = _neighbor_names()
    factors = {name: rng.standard_normal(n) * 0.005 for name in names}

    n_true = min(cfg.NUM_TRULY_COUPLED_DEFAULT, len(names))
    true_coupled = names[:n_true]
    coupling = {name: rng.choice([-1, 1]) * rng.uniform(0.3, 0.9) for name in true_coupled}
    n_lags = cfg.NUM_TRUE_MEMORY_LAGS_DEFAULT
    k1_true = rng.uniform(0.15, 0.30)
    kernel_true = k1_true * 0.5 ** np.arange(n_lags)   # decaying multi-lag memory kernel

    omega_g, alpha_g, beta_g = 1e-6, 0.08, 0.88
    var = np.zeros(n)
    var[0] = omega_g / (1 - alpha_g - beta_g)
    eps = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        var[t] = omega_g + alpha_g * r[t - 1] ** 2 + beta_g * var[t - 1]
        coupling_term = sum(coupling[name] * factors[name][t - 1] for name in true_coupled)
        memory_term = sum(kernel_true[k] * r[t - 1 - k] for k in range(n_lags) if t - 1 - k >= 0)
        r[t] = memory_term + coupling_term + np.sqrt(var[t]) * eps[t]

    price = 1.0 * np.exp(np.cumsum(r))
    price_series = pd.Series(price, index=dates, name="price")
    factors_df = pd.DataFrame(factors, index=dates)

    print(f"[phase2] synthetic ground truth (demo mode only): "
          f"memory kernel K={np.round(kernel_true, 4).tolist()}, "
          f"coupling={ {k: round(float(v), 4) for k, v in coupling.items()} }")
    return price_series, factors_df


def build_state_space(price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": price})
    df["r"] = np.log(df["price"] / df["price"].shift(1))
    df["sigma"] = df["r"].rolling(cfg.VOLATILITY_WINDOW).std()
    return df.dropna().reset_index().rename(columns={"index": "date"})


def main():
    try:
        if cfg.CURRENCY_PAIR.startswith("REPLACE"):
            raise ValueError("phase1_config.CURRENCY_PAIR not set")
        price = fetch_real_price(cfg.CURRENCY_PAIR, cfg.START_DATE, cfg.END_DATE)
        source = "real data source"
    except Exception as e:
        print(f"[phase2] live fetch unavailable ({e}); using generic synthetic fallback")
        price, factors_df = synthetic_universe(cfg.START_DATE, cfg.END_DATE)
        factors_df.to_csv("synthetic_factors.csv", index_label="date")
        source = "synthetic"

    state = build_state_space(price)
    state.to_csv("state_space.csv", index=False)

    print(f"[phase2] source = {source}  |  rows = {len(state)}")
    print(state.head())
    print(state[["r", "sigma"]].describe())

    adf_stat, adf_p = cfg.check_stationarity(state["r"])
    print(f"\n[phase2] ADF stationarity test on r(t): statistic={adf_stat:.4f}, p={adf_p:.4g}")
    if adf_p < 0.05:
        print("[phase2] unit-root null rejected -- r(t) is consistent with the "
              "time-translation invariance required for fixed-coefficient estimation.")
    else:
        print("[phase2] unit-root null NOT rejected -- reassess the return "
              "transform before proceeding to Phase 3.")


if __name__ == "__main__":
    main()
