"""
PHASE 3 — NETWORK-NEIGHBOR AND EXOGENOUS-FORCING FEATURE CONSTRUCTION

Physical and mathematical role
--------------------------------
In the network formulation, the currency pair is one node i of a graph
whose other nodes j are candidate coupled instruments, and whose
exogenous forcing is the flow of public information. The full,
unprojected equation of motion for node i takes the form

    ds_i/dt = F(s_i) + sum_j A_ij * G(s_i, s_j) + xi_i(t)

This phase materializes the observable side of the coupling sum: for
every candidate neighbor j, its return series r_j(t) is computed by
the identical log-difference transform used for the currency pair
itself in Phase 2, and for every news topic m, a daily scalar pair
(count_m(t), sentiment_m(t)) is computed as a proxy for the exogenous
forcing acting on node i through information channel m.

No claim is made here about which A_ij are nonzero: coupling strength
is not estimated until Phase 4. This phase supplies only the complete
candidate feature set { r_j(t) } union { news_m(t) } from which the
sparse regression in Phase 4 selects.

News count and sentiment are treated as parallel exogenous scalar
channels rather than as additional network nodes with their own
dynamical state, because in this formulation news acts as a forcing
term on the currency node, not as a self-propagating dynamical
variable; it therefore enters the design matrix identically to a
same-day neighbor return.

Output: features.csv (date, r, sigma, <name>_r ..., news_<topic>_sent,
        news_<topic>_count ...)
"""

import os
import numpy as np
import pandas as pd
import phase1_config as cfg


def fetch_real_neighbor(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=dates.min(), end=dates.max(), progress=False)
    if df.empty:
        raise ValueError("empty")
    s = np.log(df["Close"] / df["Close"].shift(1))
    return s.reindex(dates).rename(ticker)


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


def build_neighbor_features(state: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(state["date"])
    out = state.set_index("date")[["r", "sigma"]].copy()
    out.index = dates

    synthetic_factors = load_synthetic_factors(dates)

    for entry in resolve_neighbor_list():
        name, ticker = entry["name"], entry.get("ticker")
        s, source = None, None
        if ticker:
            try:
                s = fetch_real_neighbor(ticker, dates)
                if s.isna().mean() > 0.3:
                    raise ValueError("too many NaNs")
                source = "real data source"
            except Exception:
                s = None
        if s is None:
            if synthetic_factors is not None and name in synthetic_factors.columns:
                s, source = synthetic_factors[name], "synthetic (demo ground-truth factor)"
            else:
                s, source = pd.Series(np.zeros(len(dates)), index=dates), "unavailable -> zero-filled"
        out[f"{name}_r"] = s.reindex(dates).ffill().fillna(0.0)
        print(f"[phase3] neighbor '{name}' -> {source}")

    return out


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
    feat = build_neighbor_features(state)
    news = build_news_features(feat.index)
    full = feat.join(news).reset_index().rename(columns={"index": "date"})
    full.to_csv("features.csv", index=False)

    print(f"[phase3] final feature table: {full.shape[0]} rows x {full.shape[1]} cols")
    print(full.columns.tolist())


if __name__ == "__main__":
    main()