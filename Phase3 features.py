"""
PHASE 3 — OBSERVED DRIVES: NETWORK-NEIGHBOR AND NEWS FEATURE CONSTRUCTION

Role
----
The currency is an open system: its returns are influenced by other
instruments and by information flow that it does not itself determine.
These are the exogenous drives x_j(t) of the GLE. This phase assembles
the candidate drive set: for every candidate neighbor j, its return
r_j(t), by the same log-difference transform used for the currency in
Phase 2; and for every news topic m, a daily pair
(count_m(t), sentiment_m(t)).

Nothing is assumed here about which drives couple to the currency or
about the form of the coupling. The joint dynamics of the full system,
including its unobserved degrees of freedom, is not specified, and
neither equilibrium between the currency and its drives nor a Markovian
joint state is assumed. Coupling strength is estimated in Phase 4, which
selects from the complete candidate set { r_j(t) } union { news_m(t) }
supplied here.

News count and sentiment enter as exogenous scalar channels: external
driving with no state equation of their own in this formulation, and so
they enter the design matrix in the same way as a same-day neighbor
return.

Cointegration diagnostic
------------------------
This phase also tests each neighbor's price LEVEL for an Engle-Granger
cointegrating relationship with the currency's price level
(phase1_config.check_cointegration): whether some linear combination of
the two non-stationary series is itself stationary. This is a diagnostic
of the data only. The return-based model of Phase 4 neither assumes nor
imposes a cointegrating (error-correction) relation.

Output: features.csv (date, r, sigma, <name>_r ..., news_<topic>_sent,
        news_<topic>_count ...)
        cointegration_report.csv (neighbor, eg_statistic, eg_pvalue)
"""

import os
import numpy as np
import pandas as pd
import phase1_config as cfg


def fetch_real_neighbor_level(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=dates.min(), end=dates.max(), progress=False)
    if df.empty:
        raise ValueError("empty")
    return df["Close"].reindex(dates).rename(ticker)


def fetch_real_neighbor(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    level = fetch_real_neighbor_level(ticker, dates)
    return np.log(level / level.shift(1))


def load_synthetic_factors(dates: pd.DatetimeIndex):
    try:
        f = pd.read_csv("synthetic_factors.csv", parse_dates=["date"]).set_index("date")
        return f.reindex(dates).ffill().fillna(0.0)
    except FileNotFoundError:
        return None


def resolve_neighbor_list():
    if cfg.NEIGHBORS:
        return cfg.NEIGHBORS
    return [{"name": f"neighbor_{i+1}", "ticker": None}
            for i in range(cfg.NUM_SYNTHETIC_NEIGHBORS_DEFAULT)]


def build_neighbor_features(state: pd.DataFrame):
    """Returns (feature_df, levels). feature_df carries the return
    series used downstream by Phase 4's design matrix. levels carries
    the raw price-level series for every neighbor fetched from a real
    data source (not the synthetic fallback, which has no analogous
    price level), for use by the cointegration diagnostic below."""
    dates = pd.DatetimeIndex(state["date"])
    out = state.set_index("date")[["r", "sigma"]].copy()
    out.index = dates

    synthetic_factors = load_synthetic_factors(dates)
    levels = {}

    for entry in resolve_neighbor_list():
        name, ticker = entry["name"], entry.get("ticker")
        s, level, source = None, None, None
        if ticker:
            try:
                level = fetch_real_neighbor_level(ticker, dates)
                if level.isna().mean() > 0.3:
                    raise ValueError("too many NaNs")
                s = np.log(level / level.shift(1))
                source = "real data source"
            except Exception:
                s, level = None, None
        if s is None:
            if synthetic_factors is not None and name in synthetic_factors.columns:
                s, source = synthetic_factors[name], "synthetic (demo ground-truth factor)"
            else:
                s, source = pd.Series(np.zeros(len(dates)), index=dates), "unavailable -> zero-filled"
        out[f"{name}_r"] = s.reindex(dates).ffill().fillna(0.0)
        if level is not None:
            levels[name] = level.reindex(dates).ffill()
        print(f"[phase3] neighbor '{name}' -> {source}")

    return out, levels


def run_cointegration_diagnostics(state: pd.DataFrame, levels: dict):
    """Engle-Granger cointegration test (phase1_config.check_cointegration)
    between the currency's own price level and every neighbor for which
    a real price-level series was obtained. Not run against the
    synthetic fallback, whose data-generating process is specified
    directly on returns and has no corresponding notion of a
    cointegrating price-level relationship."""
    if not levels:
        print("[phase3] cointegration diagnostic skipped -- no real price-level "
              "series available (synthetic-data mode has no analogous quantity).")
        return
    currency_level = state.set_index("date")["price"]
    results = []
    for name, level in levels.items():
        stat, pvalue = cfg.check_cointegration(currency_level, level)
        results.append({"neighbor": name, "eg_statistic": stat, "eg_pvalue": pvalue})
        print(f"[phase3] cointegration (Engle-Granger), currency vs '{name}': "
              f"statistic={stat:.4f}, p={pvalue:.4g}")
    pd.DataFrame(results).to_csv("cointegration_report.csv", index=False)
    print("[phase3] saved cointegration_report.csv")


def resolve_news_categories():
    if cfg.NEWS_CATEGORIES:
        return cfg.NEWS_CATEGORIES
    return [f"topic_{i+1}" for i in range(cfg.NUM_SYNTHETIC_NEWS_DEFAULT)]


def fetch_real_news_batch(topic: str, dates: pd.DatetimeIndex, api_key: str) -> pd.DataFrame:
    """Issues a single paginated query per topic spanning the entire
    date range, then aggregates articles into per-date count and mean
    sentiment. See phase1_config.NEWS_API_LOOKBACK_WARNING regarding
    the news source's lookback window constraint."""
    import requests
    all_articles = []
    for page in range(1, 6):  # up to 500 articles/topic at pageSize=100
        params = {
            "q": topic,
            "from": dates.min().strftime("%Y-%m-%d"),
            "to": dates.max().strftime("%Y-%m-%d"),
            "apiKey": api_key,
            "language": "en",
            "pageSize": 100,
            "page": page,
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        arts = r.json().get("articles", [])
        if not arts:
            break
        all_articles.extend(arts)

    if not all_articles:
        raise ValueError("no articles returned")

    df = pd.DataFrame(all_articles)
    df["date"] = pd.to_datetime(df["publishedAt"]).dt.normalize()
    # plug a real sentiment model (VADER/TextBlob/etc.) on df["title"] here;
    # placeholder sentiment = 0.0 until wired up
    df["sent"] = 0.0
    daily = df.groupby("date").agg(count=("title", "size"), sent=("sent", "mean"))
    return daily.reindex(dates).fillna({"count": 0, "sent": 0.0})


def synthetic_news(topic: str, dates: pd.DatetimeIndex, seed_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.RANDOM_SEED + 100 + seed_offset)
    counts = rng.poisson(lam=3, size=len(dates))
    raw = rng.normal(0, 0.4, size=len(dates))
    sent = np.clip(pd.Series(raw).rolling(3, min_periods=1).mean().values, -1, 1)
    return pd.DataFrame({"count": counts, "sent": sent}, index=dates)


def build_news_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    api_key = os.environ.get(cfg.NEWS_API_KEY_ENV_VAR)
    if api_key:
        print(f"[phase3] NOTE: {cfg.NEWS_API_LOOKBACK_WARNING}")

    frames = []
    for i, topic in enumerate(resolve_news_categories()):
        df, source = None, None
        if api_key:
            try:
                df = fetch_real_news_batch(topic, dates, api_key)
                source = "real news source (batched)"
            except Exception:
                df = None
        if df is None:
            df = synthetic_news(topic, dates, seed_offset=i)
            source = "synthetic" if not api_key else "synthetic (fetch failed)"
        safe_topic = topic.replace(" ", "_")
        df.columns = [f"news_{safe_topic}_count", f"news_{safe_topic}_sent"]
        frames.append(df)
        print(f"[phase3] news topic '{topic}' -> {source}")
    return pd.concat(frames, axis=1)


def main():
    state = pd.read_csv("state_space.csv", parse_dates=["date"])
    feat, levels = build_neighbor_features(state)
    news = build_news_features(feat.index)
    full = feat.join(news).reset_index().rename(columns={"index": "date"})
    full.to_csv("features.csv", index=False)

    print(f"[phase3] final feature table: {full.shape[0]} rows x {full.shape[1]} cols")
    print(full.columns.tolist())

    run_cointegration_diagnostics(state, levels)


if __name__ == "__main__":
    main()
