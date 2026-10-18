"""
PHASE 3 — OBSERVED EXOGENOUS DRIVES

This phase constructs the observed drive set for the GLE/VARX model
using only real market data already downloaded locally.

Inputs
------
state_space.csv
    Phase 2 state variables for USD/INR:
    date, price, r, sigma

market_drives.csv
    Real market levels and returns constructed from FRED/Yahoo data.

No synthetic fallback is used.
No zero-filling is used.
No historical news data are fabricated.

News is deliberately excluded because the available NewsAPI interface
cannot defensibly provide the required 2020-2024 historical coverage.

Outputs
-------
features.csv
    date, r, sigma, and real market-drive returns

cointegration_report.csv
    Engle-Granger diagnostics between USD/INR price level and each
    available market-drive level.
"""

import numpy as np
import pandas as pd
import phase1_config as cfg


MARKET_DRIVES = {
    "broad_usd": {
        "return": "broad_usd_r",
        "level": "broad_usd_level",
    },
    "brent_oil": {
        "return": "brent_oil_r",
        "level": "brent_oil_level",
    },
    "gold": {
        "return": "gold_r",
        "level": "gold_level",
    },
    "vix": {
        "return": "vix_r",
        "level": "vix_level",
    },
    "sp500": {
        "return": "sp500_r",
        "level": "sp500_level",
    },
    "nikkei": {
        "return": "nikkei_r",
        "level": "nikkei_level",
    },
    "us10y": {
        "return": "us10y_r",
        "level": "us10y_level",
    },
}


def load_state():
    """Load the Phase 2 USD/INR state-space data."""
    state = pd.read_csv("state_space.csv", parse_dates=["date"])

    required = {"date", "price", "r", "sigma"}
    missing = required - set(state.columns)

    if missing:
        raise ValueError(
            f"state_space.csv is missing required columns: {sorted(missing)}"
        )

    state = (
        state[["date", "price", "r", "sigma"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )

    return state


def load_market_drives():
    """Load the locally constructed real market-drive dataset."""
    market = pd.read_csv("market_drives.csv", parse_dates=["date"])

    if "date" not in market.columns:
        raise ValueError("market_drives.csv has no date column")

    market = (
        market
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )

    required = []
    for spec in MARKET_DRIVES.values():
        required.extend([spec["return"], spec["level"]])

    missing = [c for c in required if c not in market.columns]

    if missing:
        raise ValueError(
            "market_drives.csv is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    return market


def build_features(state, market):
    """
    Join Phase 2 state variables with real market-drive returns.

    We do not forward-fill returns or replace missing observations with
    synthetic/zero values. Rows with missing candidate-drive observations
    are removed so the regression receives only observed market data.
    """
    return_cols = [
        spec["return"]
        for spec in MARKET_DRIVES.values()
    ]

    state_cols = ["r", "sigma"]

    combined = state[state_cols].join(
        market[return_cols],
        how="inner",
    )

    before = len(combined)
    combined = combined.dropna()
    removed = before - len(combined)

    if combined.empty:
        raise ValueError(
            "No complete observations remain after joining state_space.csv "
            "with market_drives.csv."
        )

    print(f"[phase3] joined observations = {before}")
    print(f"[phase3] complete observations = {len(combined)}")
    print(f"[phase3] rows removed for missing real observations = {removed}")

    return combined


def run_cointegration_diagnostics(state, market):
    """
    Engle-Granger diagnostics between the USD/INR price level and each
    real market-drive level.

    This is a diagnostic only. Phase 4 models returns and does not impose
    an error-correction or cointegration relationship.

    Note:
    The Engle-Granger interpretation is formally most appropriate when
    both level series are I(1). Some market variables, especially VIX and
    interest-rate levels, need not satisfy that condition. Their results
    are therefore reported as diagnostics rather than treated as evidence
    of an economic long-run relationship.
    """
    currency_level = state["price"]

    results = []

    for name, spec in MARKET_DRIVES.items():
        level_col = spec["level"]

        aligned = pd.concat(
            [currency_level.rename("currency"), market[level_col].rename(name)],
            axis=1,
        ).dropna()

        if len(aligned) < 50:
            print(
                f"[phase3] cointegration '{name}' -> skipped "
                f"(only {len(aligned)} aligned observations)"
            )
            results.append(
                {
                    "neighbor": name,
                    "eg_statistic": np.nan,
                    "eg_pvalue": np.nan,
                    "n_observations": len(aligned),
                    "status": "insufficient_data",
                }
            )
            continue

        try:
            stat, pvalue = cfg.check_cointegration(
                aligned["currency"],
                aligned[name],
            )

            print(
                f"[phase3] cointegration USD/INR vs '{name}': "
                f"statistic={stat:.4f}, p={pvalue:.4g}, "
                f"n={len(aligned)}"
            )

            results.append(
                {
                    "neighbor": name,
                    "eg_statistic": stat,
                    "eg_pvalue": pvalue,
                    "n_observations": len(aligned),
                    "status": "computed",
                }
            )

        except Exception as exc:
            print(
                f"[phase3] cointegration '{name}' -> failed: {exc}"
            )

            results.append(
                {
                    "neighbor": name,
                    "eg_statistic": np.nan,
                    "eg_pvalue": np.nan,
                    "n_observations": len(aligned),
                    "status": f"failed: {exc}",
                }
            )

    report = pd.DataFrame(results)
    report.to_csv("cointegration_report.csv", index=False)

    print("[phase3] saved cointegration_report.csv")


def main():
    print("[phase3] real-data-only mode")
    print("[phase3] synthetic factors: DISABLED")
    print("[phase3] synthetic news: DISABLED")
    print("[phase3] zero-filled fallbacks: DISABLED")

    state = load_state()
    market = load_market_drives()

    print(
        f"[phase3] state range = "
        f"{state.index.min().date()} -> {state.index.max().date()}"
    )

    print(
        f"[phase3] market range = "
        f"{market.index.min().date()} -> {market.index.max().date()}"
    )

    features = build_features(state, market)

    full = features.reset_index()
    full.to_csv("features.csv", index=False)

    print(
        f"[phase3] final feature table: "
        f"{full.shape[0]} rows x {full.shape[1]} cols"
    )

    print("[phase3] feature columns:")
    for column in full.columns:
        print(f"  - {column}")

    run_cointegration_diagnostics(state, market)

    print("[phase3] Phase 3 completed successfully.")


if __name__ == "__main__":
    main()
