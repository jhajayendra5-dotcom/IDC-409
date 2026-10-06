""
PHASE 6 — MULTI-STEP FORWARD SIMULATION (MONTE CARLO)

Role
----
Phases 4 and 5 give a parameterized approximation of the reduced GLE
dynamics:

    r(t+1) = sum_k K_k*r(t-k+1) + sum_j Omega_j*x_j(t) + eta(t)

with eta(t) following the fitted GARCH-t process. This phase integrates
it forward from the last observed state over a horizon of H days.

The propagation is non-Markovian in r: each step depends on the p most
recent values of the path's own history through the memory kernel, so
the simulator carries a lag buffer of those values together with the
state of the conditional-variance recursion, not the current value
alone. No equilibrium is assumed: paths are driven by resampled
exogenous forcing and are not constrained to relax to any prescribed
distribution.

Monte Carlo simulation is required, not optional. r_sim(t+h) depends
recursively on r_sim(t+h-1), ..., r_sim(t+h-p), and the noise variance
at each step depends on the realized magnitude of the noise at the
previous step, so the distribution of r_sim(t+H) is obtainable only by
propagating an ensemble of sample paths through the full recursion; no
closed-form H-step-ahead marginal exists once H > 1.

Per Monte Carlo path i and step h = 1, ..., H:

  1. eta_sim^(i)(h) is drawn from the GARCH model's own simulated
     variance recursion, so successive noise draws within a path carry
     the fitted volatility-clustering structure rather than being
     sampled independently from a fixed distribution.

  2. Values for the drive terms x_j(t+h), which are unobserved at
     simulation time, are obtained by BLOCK bootstrap: for each path, a
     single random starting point is drawn from history, and the H
     consecutive days following it supply that path's entire sequence
     of exogenous values. Sampling a full historical day's row at once
     preserves cross-sectional co-movement across neighbors on that day
     (e.g. oil and gold moving together), and sampling one contiguous
     block per path, rather than one row per step, preserves the
     drives' own day-to-day autocorrelation across the simulated
     horizon. sigma(t+h) is excluded from this bootstrap: it is not an
     independent drive but a deterministic function of r's own recent
     history (Phase 2), so it is instead recomputed at every step from
     a rolling window of that path's own simulated returns, keeping it
     consistent with the path it is evaluated on.

  3. r_sim^(i)(t+h) = f(lag buffer, sigma from own history, sampled
     x_j) + eta_sim^(i)(h), where f is the linear map estimated in
     Phase 4. The lag buffer and the rolling-return history are then
     both updated with this newly simulated value, so subsequent steps
     feed back on the path's own simulated history exactly as the
     memory kernel prescribes.

The empirical distribution of {r_sim^(i)(t+h)} across the ensemble at
each horizon h approximates the predictive distribution of r(t+h) under
the estimated dynamics. Its mean is reported as the point forecast and
its 5th/95th percentiles as a forecast interval. The interval widens
with h, reflecting the compounding of memory-driven and
stochastic-volatility uncertainty over the horizon.

Output: simulation_forecast.csv (h, date, pred_mean, pred_p05, pred_p95)
"""

import pickle
import numpy as np
import pandas as pd
import phase1_config as cfg


def load_artifacts():
    with open("varx_model.pkl", "rb") as f:
        varx = pickle.load(f)
    with open("garch_model.pkl", "rb") as f:
        garch_res = pickle.load(f)
    df = pd.read_csv("features.csv", parse_dates=["date"]).set_index("date")
    return varx, garch_res, df


def simulate_forward(varx, garch_res, df, horizon: int, n_sims: int, seed: int):
    from phase5_noise_model import GARCH_SCALE

    pipeline, feature_names, p = varx["model"], varx["features"], varx["p"]
    has_sigma = "sigma" in feature_names
    exo_cols = [c for c in feature_names if not c.startswith("r_lag") and c != "sigma"]

    rng = np.random.default_rng(seed)

    # GARCH's own multi-step simulation forecast: shape (n_sims, horizon),
    # already accounts for the variance recursion, not flat noise
    fcast = garch_res.forecast(horizon=horizon, method="simulation",
                                simulations=n_sims, reindex=False)
    eta_sim = fcast.simulations.values[0] / GARCH_SCALE  # (n_sims, horizon)

    # block-bootstrap pool for unknown future drive (neighbor + news)
    # values: one contiguous historical window of length `horizon` per
    # simulation path, not independent per-day draws (see docstring).
    # sigma is deliberately excluded here; it is recomputed below.
    history_pool = df[exo_cols].values
    n_hist = len(history_pool)
    block_starts = rng.integers(0, n_hist - horizon, size=n_sims)          # (n_sims,)
    block_idx = block_starts[:, None] + np.arange(horizon)[None, :]        # (n_sims, horizon)
    sampled_exo_blocks = history_pool[block_idx]                          # (n_sims, horizon, num_exo)

    # initialize lag buffer identically across all paths: last p known r's,
    # ordered [r_lag1, r_lag2, ...] = [r(T), r(T-1), ...] matching
    # build_design_matrix's r_lag1 = r(t) convention exactly
    last_r = df["r"].values[-p:][::-1]
    lag_buffers = np.tile(last_r, (n_sims, 1))  # (n_sims, p)

    # separate rolling-history buffer for sigma, chronological order
    # (oldest first), long enough to cover VOLATILITY_WINDOW regardless
    # of how p compares to it
    hist_len = max(p, cfg.VOLATILITY_WINDOW)
    r_history = np.tile(df["r"].values[-hist_len:], (n_sims, 1))  # (n_sims, hist_len)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]

    records = []
    for h in range(horizon):
        sampled_exo = sampled_exo_blocks[:, h, :]  # (n_sims, num_exo_features)

        # assemble each simulated day's feature vector by column NAME:
        # a dict keyed by feature name is built from the three sources
        # below, then read out in the exact order feature_names
        # specifies, so correctness does not depend on any particular
        # column ordering in feature_names itself
        columns = {f"r_lag{k}": lag_buffers[:, k - 1] for k in range(1, p + 1)}
        if has_sigma:
            columns["sigma"] = r_history[:, -cfg.VOLATILITY_WINDOW:].std(axis=1, ddof=1)
        for idx, name in enumerate(exo_cols):
            columns[name] = sampled_exo[:, idx]
        X_step = np.stack([columns[name] for name in feature_names], axis=1)
        deterministic = pipeline.predict(X_step)  # (n_sims,)
        r_sim_h = deterministic + eta_sim[:, h]

        records.append({
            "h": h + 1,
            "date": future_dates[h],
            "pred_mean": r_sim_h.mean(),
            "pred_p05": np.percentile(r_sim_h, 5),
            "pred_p50": np.percentile(r_sim_h, 50),
            "pred_p95": np.percentile(r_sim_h, 95),
        })

        # shift lag buffer: newest simulated value becomes r_lag1
        lag_buffers = np.concatenate([r_sim_h.reshape(-1, 1), lag_buffers[:, :-1]], axis=1)
        # shift rolling-return history: newest simulated value appended at the end
        r_history = np.concatenate([r_history[:, 1:], r_sim_h.reshape(-1, 1)], axis=1)

    return pd.DataFrame(records)


def main():
    varx, garch_res, df = load_artifacts()
    forecast = simulate_forward(varx, garch_res, df, cfg.FORECAST_HORIZON, cfg.N_SIMS, cfg.RANDOM_SEED)

    forecast.to_csv("simulation_forecast.csv", index=False)
    print(f"[phase6] simulated {cfg.FORECAST_HORIZON} days forward, {cfg.N_SIMS} Monte Carlo paths")
    print(forecast.head(10))

    cum_mean = forecast["pred_mean"].sum()
    print(f"\n[phase6] cumulative expected log-return over {cfg.FORECAST_HORIZON} days: {cum_mean:.5f}")
    print(f"[phase6] band widens with horizon: day-1 p05/p95 = "
          f"[{forecast.iloc[0]['pred_p05']:.5f}, {forecast.iloc[0]['pred_p95']:.5f}]  |  "
          f"day-{cfg.FORECAST_HORIZON} p05/p95 = "
          f"[{forecast.iloc[-1]['pred_p05']:.5f}, {forecast.iloc[-1]['pred_p95']:.5f}]")
    print("[phase6] saved simulation_forecast.csv")


if __name__ == "__main__":
    main()
