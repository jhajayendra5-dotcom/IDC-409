"""
PHASE 2 — STATE-SPACE CONSTRUCTION

Physical and mathematical role
--------------------------------
The Mori-Zwanzig projection formalism requires an a priori choice of
"relevant" (slow) variables onto which the full microscopic dynamics
are projected; every other degree of freedom is subsequently
integrated out into the memory kernel and noise term of the GLE. This
phase defines that relevant subspace for the currency pair as the
two-component state vector

    s(t) = [ r(t), sigma(t) ]

    r(t)     = ln( P(t) / P(t-1) )
    sigma(t) = std( r(t-w+1 : t) ),  w = VOLATILITY_WINDOW

r(t), the log-return, is the additive, approximately stationary
observable whose dynamics the GLE governs; log-differencing rather
than raw price is used because price is a non-stationary (integrated)
process, which is incompatible with the implicit stationarity
assumption underlying the GLE's fluctuation-dissipation structure.

sigma(t), the trailing realized volatility, is the minimal auxiliary
variable required for s(t) to be approximately Markovian: r(t) alone
is not, since its conditional variance depends on recent history
(volatility clustering), whereas the pair [r(t), sigma(t)] jointly
captures first- and second-moment state.

Output: state_space.csv (date, price, r, sigma)

Synthetic fallback
--------------------
In the absence of a live price feed, synthetic_universe() generates a
return series from a closed-form, fully known data-generating process,
so downstream estimation can be validated against ground truth:

    r(t) = K1*r(t-1) + sum_i Omega_i*f_i(t-1) + eta(t)
    eta(t) = sqrt(h(t))*z(t),         z(t) ~ N(0,1)
    h(t)   = omega + alpha*r(t-1)^2 + beta*h(t-1)     [GARCH(1,1)]

where f_i(t) are independent latent factor processes, a subset of
which are assigned nonzero true coupling Omega_i while the remainder
are pure noise. This reproduces, by construction, the network-plus-
memory-plus-noise structure the later phases are built to recover, and
the GARCH(1,1) recursion reproduces the volatility clustering that is
empirically near-universal in currency return series.
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
    k1_true = rng.uniform(0.05, 0.2)

    omega_g, alpha_g, beta_g = 1e-6, 0.08, 0.88
    var = np.zeros(n)
    var[0] = omega_g / (1 - alpha_g - beta_g)
    eps = rng.standard_normal(n)
    r = np.zeros(n)
    for t in range(1, n):
        var[t] = omega_g + alpha_g * r[t - 1] ** 2 + beta_g * var[t - 1]
        coupling_term = sum(coupling[name] * factors[name][t - 1] for name in true_coupled)
        r[t] = k1_true * r[t - 1] + coupling_term + np.sqrt(var[t]) * eps[t]

    price = 1.0 * np.exp(np.cumsum(r))
    price_series = pd.Series(price, index=dates, name="price")
    factors_df = pd.DataFrame(factors, index=dates)

    print(f"[phase2] synthetic ground truth (demo mode only): "
          f"k1={k1_true:.4f}, coupling={coupling}")
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


if __name__ == "__main__":
    main()