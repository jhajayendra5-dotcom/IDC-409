"""
PHASE 2 — OBSERVED STATE CONSTRUCTION AND STATIONARITY CHECK

Role
----
Mori-Zwanzig projection presupposes a choice of observed variables;
every other degree of freedom then enters the reduced dynamics only
through the memory kernel and the noise.

This phase defines the observed variables of the currency pair:

    r(t)     = ln(P(t) / P(t-1))
    sigma(t) = std(r(t-w+1 : t)), w = VOLATILITY_WINDOW

The pair [r(t), sigma(t)] is not presumed Markovian.

Output:
    state_space.csv
        date, price, r, sigma

Data modes
----------
1. REAL DATA MODE (default)
   Loads historical USD/INR prices from:

       usd_inr.csv

   Expected CSV format:

       date,price
       2020-01-01,71.35
       2020-01-02,71.42
       ...

2. SYNTHETIC VALIDATION MODE

       py phase2_state_space.py --demo

   Generates a controlled synthetic process with known memory-kernel
   and coupling ground truth.

3. YAHOO MODE (optional)

       py phase2_state_space.py --yahoo

   Attempts to download data using yfinance.

IMPORTANT
---------
Synthetic data is NOT automatically used when real data fails.
This prevents a failed live-data download from being mistaken for
an empirical USD/INR result.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import phase1_config as cfg


# ---------------------------------------------------------------------
# OPTIONAL YAHOO FINANCE DATA SOURCE
# ---------------------------------------------------------------------

def fetch_real_price(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download historical price data from Yahoo Finance.

    This is optional. The recommended reproducible experiment uses
    a local CSV file instead.
    """

    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError("Yahoo Finance returned an empty dataset.")

    close = df["Close"]

    # yfinance may return a DataFrame instead of a Series.
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]

    close = pd.Series(
        close,
        name="price"
    )

    close.index = pd.to_datetime(close.index)

    close = close.dropna()

    if close.empty:
        raise ValueError("Yahoo Finance returned no valid closing prices.")

    return close


# ---------------------------------------------------------------------
# LOCAL CSV DATA SOURCE
# ---------------------------------------------------------------------

def load_price_csv(
    path: str,
    start: str,
    end: str,
    date_column: str = "date",
    price_column: str = "price",
) -> pd.Series:
    """
    Load reproducible historical price data from a local CSV.

    Expected format:

        date,price
        2020-01-01,71.35
        2020-01-02,71.42
        ...

    Additional CSV columns are allowed.
    """

    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price CSV not found: {csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)

    if date_column not in df.columns:
        raise ValueError(
            f"CSV is missing date column '{date_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    if price_column not in df.columns:
        raise ValueError(
            f"CSV is missing price column '{price_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df[price_column] = pd.to_numeric(
        df[price_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[date_column, price_column]
    )

    df = df.sort_values(date_column)

    # Keep only the configured experiment period.
    df = df[
        (df[date_column] >= pd.Timestamp(start))
        & (df[date_column] <= pd.Timestamp(end))
    ]

    if df.empty:
        raise ValueError(
            f"No price observations found between "
            f"{start} and {end}."
        )

    if (df[price_column] <= 0).any():
        raise ValueError(
            "Price data contains zero or negative values. "
            "Log returns require strictly positive prices."
        )

    price = pd.Series(
        df[price_column].to_numpy(),
        index=df[date_column],
        name="price",
    )

    # Remove duplicate dates.
    price = price[
        ~price.index.duplicated(keep="last")
    ]

    return price


# ---------------------------------------------------------------------
# SYNTHETIC VALIDATION UNIVERSE
# ---------------------------------------------------------------------

def _neighbor_names():
    if cfg.NEIGHBORS:
        return [
            n["name"]
            for n in cfg.NEIGHBORS
        ]

    return [
        f"neighbor_{i+1}"
        for i in range(
            cfg.NUM_SYNTHETIC_NEIGHBORS_DEFAULT
        )
    ]


def synthetic_universe(
    start: str,
    end: str,
    seed: int = None
):
    """
    Generate a synthetic process with known ground truth.

    This is a validation fixture, NOT a model of the financial market.
    """

    seed = (
        cfg.RANDOM_SEED
        if seed is None
        else seed
    )

    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        start,
        end
    )

    n = len(dates)

    names = _neighbor_names()

    factors = {
        name: rng.standard_normal(n) * 0.005
        for name in names
    }

    n_true = min(
        cfg.NUM_TRULY_COUPLED_DEFAULT,
        len(names)
    )

    true_coupled = names[:n_true]

    coupling = {
        name: rng.choice([-1, 1])
        * rng.uniform(0.3, 0.9)
        for name in true_coupled
    }

    n_lags = (
        cfg.NUM_TRUE_MEMORY_LAGS_DEFAULT
    )

    k1_true = rng.uniform(
        0.15,
        0.30
    )

    kernel_true = (
        k1_true
        * 0.5 ** np.arange(n_lags)
    )

    # GARCH(1,1) parameters.
    omega_g = 1e-6
    alpha_g = 0.08
    beta_g = 0.88

    var = np.zeros(n)

    var[0] = (
        omega_g
        / (1 - alpha_g - beta_g)
    )

    eps = rng.standard_normal(n)

    r = np.zeros(n)

    for t in range(1, n):

        var[t] = (
            omega_g
            + alpha_g * r[t - 1] ** 2
            + beta_g * var[t - 1]
        )

        coupling_term = sum(
            coupling[name]
            * factors[name][t - 1]
            for name in true_coupled
        )

        memory_term = sum(
            kernel_true[k]
            * r[t - 1 - k]
            for k in range(n_lags)
            if t - 1 - k >= 0
        )

        r[t] = (
            memory_term
            + coupling_term
            + np.sqrt(var[t]) * eps[t]
        )

    price = np.exp(
        np.cumsum(r)
    )

    price_series = pd.Series(
        price,
        index=dates,
        name="price"
    )

    factors_df = pd.DataFrame(
        factors,
        index=dates
    )

    print(
        "[phase2] synthetic ground truth "
        "(demo mode only): "
        f"memory kernel K="
        f"{np.round(kernel_true, 4).tolist()}, "
        f"coupling="
        f"{ {k: round(float(v), 4) for k, v in coupling.items()} }"
    )

    return (
        price_series,
        factors_df
    )


# ---------------------------------------------------------------------
# OBSERVED STATE CONSTRUCTION
# ---------------------------------------------------------------------

def build_state_space(
    price: pd.Series
) -> pd.DataFrame:

    price = price.sort_index()

    price = price[
        ~price.index.duplicated(
            keep="last"
        )
    ]

    if (price <= 0).any():
        raise ValueError(
            "Price series contains zero or negative values."
        )

    df = pd.DataFrame(
        {"price": price}
    )

    # Log return:
    # r(t) = ln(P(t) / P(t-1))
    df["r"] = np.log(
        df["price"]
        / df["price"].shift(1)
    )

    # Rolling volatility.
    df["sigma"] = (
        df["r"]
        .rolling(
            cfg.VOLATILITY_WINDOW
        )
        .std()
    )

    return (
        df
        .dropna()
        .reset_index()
        .rename(
            columns={"index": "date"}
        )
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 observed state "
            "construction and stationarity check."
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use synthetic validation data "
            "with known ground truth."
        ),
    )

    parser.add_argument(
        "--csv",
        default="usd_inr.csv",
        help=(
            "Historical USD/INR CSV file. "
            "Default: usd_inr.csv"
        ),
    )

    parser.add_argument(
        "--yahoo",
        action="store_true",
        help=(
            "Attempt to download data "
            "from Yahoo Finance."
        ),
    )

    args = parser.parse_args()

    # ================================================================
    # MODE 1: SYNTHETIC DEMO
    # ================================================================

    if args.demo:

        print(
            "[phase2] mode = synthetic demo"
        )

        price, factors_df = (
            synthetic_universe(
                cfg.START_DATE,
                cfg.END_DATE
            )
        )

        factors_df.to_csv(
            "synthetic_factors.csv",
            index_label="date"
        )

        source = "synthetic_demo"

    # ================================================================
    # MODE 2: LOCAL REAL DATA
    # ================================================================

    elif not args.yahoo:

        print(
            "[phase2] mode = real data "
            "(local CSV)"
        )

        print(
            f"[phase2] loading: {args.csv}"
        )

        price = load_price_csv(
            args.csv,
            cfg.START_DATE,
            cfg.END_DATE
        )

        source = "local_csv"

    # ================================================================
    # MODE 3: OPTIONAL YAHOO
    # ================================================================

    else:

        print(
            "[phase2] mode = Yahoo Finance"
        )

        print(
            f"[phase2] ticker = "
            f"{cfg.CURRENCY_PAIR}"
        )

        try:

            if cfg.CURRENCY_PAIR.startswith(
                "REPLACE"
            ):
                raise ValueError(
                    "phase1_config.CURRENCY_PAIR "
                    "is not configured."
                )

            price = fetch_real_price(
                cfg.CURRENCY_PAIR,
                cfg.START_DATE,
                cfg.END_DATE
            )

            source = "yahoo"

        except Exception as e:

            raise RuntimeError(
                "\n"
                "[phase2] Yahoo Finance "
                "download failed.\n"
                f"Reason: {e}\n"
                "\n"
                "The program has STOPPED instead "
                "of silently switching to "
                "synthetic data.\n"
                "\n"
                "For the real experiment, place "
                "historical prices in:\n"
                "    usd_inr.csv\n"
                "\n"
                "Then run:\n"
                "    py phase2_state_space.py\n"
                "\n"
                "For the synthetic validation test, run:\n"
                "    py phase2_state_space.py --demo\n"
            ) from e

    # ================================================================
    # BUILD OBSERVED STATE
    # ================================================================

    state = build_state_space(
        price
    )

    state.to_csv(
        "state_space.csv",
        index=False
    )

    print(
        f"\n[phase2] source = {source}"
        f" | rows = {len(state)}"
    )

    print(
        f"[phase2] date range = "
        f"{state['date'].min().date()} "
        f"-> "
        f"{state['date'].max().date()}"
    )

    print(
        "\n[phase2] state head:"
    )

    print(
        state.head()
    )

    print(
        "\n[phase2] return/volatility summary:"
    )

    print(
        state[
            ["r", "sigma"]
        ].describe()
    )

    # ================================================================
    # ADF STATIONARITY TEST
    # ================================================================

    adf_stat, adf_p = (
        cfg.check_stationarity(
            state["r"]
        )
    )

    print(
        "\n[phase2] ADF stationarity "
        "test on r(t): "
        f"statistic={adf_stat:.4f}, "
        f"p={adf_p:.4g}"
    )

    if adf_p < 0.05:

        print(
            "[phase2] unit-root null rejected -- "
            "r(t) is consistent with the "
            "time-translation invariance "
            "required for fixed-coefficient "
            "estimation."
        )

    else:

        print(
            "[phase2] unit-root null NOT rejected -- "
            "reassess the return transform "
            "before proceeding to Phase 3."
        )

    # ================================================================
    # IMPORTANT SOURCE WARNING
    # ================================================================

    if source == "synthetic_demo":

        print(
            "\n[phase2] WARNING:"
        )

        print(
            "[phase2] This is SYNTHETIC validation data."
        )

        print(
            "[phase2] Do NOT interpret these "
            "statistics as empirical USD/INR evidence."
        )

    else:

        print(
            "\n[phase2] Real-data Phase 2 "
            "completed successfully."
        )

        print(
            "[phase2] Output written to:"
        )

        print(
            "    state_space.csv"
        )


if __name__ == "__main__":
    main()
