"""
PHASE 6 — MULTI-STEP FORWARD SIMULATION (MONTE CARLO)

Mathematical role
-------------------
Phases 4 and 5 jointly specify a fully parameterized discretized GLE:

    r(t+1) = sum_k K_k*r(t-k) + sum_j Omega_j*x_j(t) + eta(t)

with eta(t) itself following a fitted GARCH(1,1) recursion. This phase
numerically integrates that equation forward from the last observed
state over a horizon of H days. This requires Monte Carlo simulation
rather than a closed-form propagation, for a precise reason: because
r_sim(t+h) depends recursively on r_sim(t+h-1),...,r_sim(t+h-p), and
because the noise variance h(t) is itself a function of the realized
magnitude of eta at the previous step, the distribution of r_sim(t+H)
is obtainable only by propagating an ensemble of sample paths through
the full nonlinear recursion; no closed-form expression for the
H-step-ahead marginal distribution exists once H>1.

Per Monte Carlo path i and step h=1,...,H:

  1. eta_sim^(i)(h) is drawn from the GARCH(1,1) model's own simulated
     variance recursion, so successive noise draws within a path
     inherit the correct volatility-clustering structure rather than
     being sampled independently from a fixed distribution.

  2. Values for the network-coupling terms x_j(t+h), which are
     genuinely unobserved at simulation time, are obtained by bootstrap
     resampling rows from the empirical joint distribution of
     historical neighbor/news values -- the minimal nonparametric
     representation of an unknown future exogenous input that remains
     consistent with its observed historical distribution, without
     imposing a parametric model on the neighbors' own dynamics.

  3. r_sim^(i)(t+h) = f(lag buffer, sampled x_j) + eta_sim^(i)(h),
     where f is the linear map estimated in Phase 4; the lag buffer is
     then updated with this newly simulated value, so subsequent steps
     feed back on the path's own simulated history exactly as
     prescribed by the memory kernel.

The empirical distribution of {r_sim^(i)(t+h)} across the ensemble at
each horizon h approximates the true predictive distribution of
r(t+h); its mean is reported as the point forecast, and its 5th/95th
percentiles as a forecast interval. The interval necessarily widens
with h, reflecting the genuine compounding of autoregressive and
stochastic-volatility uncertainty over the simulated horizon.

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
    exo_cols = [c for c in feature_names if not c.startswith("r_lag")]

    rng = np.random.default_rng(seed)

    # GARCH's own multi-step simulation forecast: shape (n_sims, horizon),
    # already accounts for the variance recursion, not flat noise
    fcast = garch_res.forecast(horizon=horizon, method="simulation",
                                simulations=n_sims, reindex=False)
    eta_sim = fcast.simulations.values[0] / GARCH_SCALE  # (n_sims, horizon)

    # bootstrap pool for unknown future exogenous (neighbor + news) values
    history_pool = df[exo_cols].values

    # initialize lag buffer identically across all paths: last p known r's
    last_r = df["r"].values[-p:][::-1]  # most recent first: r_lag1, r_lag2, ...
    lag_buffers = np.tile(last_r, (n_sims, 1))  # (n_sims, p)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]

    records = []
    for h in range(horizon):
        idx = rng.integers(0, len(history_pool), size=n_sims)
        sampled_exo = history_pool[idx]  # (n_sims, num_exo_features)

        X_step = np.concatenate([lag_buffers, sampled_exo], axis=1)
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