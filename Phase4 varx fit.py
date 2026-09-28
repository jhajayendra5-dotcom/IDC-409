"""
PHASE 4 — SPARSE ESTIMATION OF NETWORK COUPLING AND MEMORY KERNEL

Mathematical role
-------------------
This phase performs the numerical realization of the Mori-Zwanzig
projection. Given the discretized GLE

    r(t+1) = sum_{k=1}^{p} K_k*r(t-k) + sum_j Omega_j*x_j(t) + eta(t)

with x_j(t) ranging over every candidate neighbor return and news
channel produced in Phase 3, all coefficients {K_k, Omega_j} are
estimated jointly in a single regression, penalized by an elastic net:

    minimize_beta  (1/2n)||y - X*beta||_2^2
                    + alpha*[ l1_ratio*||beta||_1
                              + (1-l1_ratio)/2*||beta||_2^2 ]

on a standardized design matrix (each column rescaled to zero mean,
unit variance). Standardization is required because the L1 term
penalizes raw coefficient magnitude, which is not invariant to the
measurement units of x_j; absent it, sparse selection would partly
reflect feature scale rather than genuine explanatory contribution.
The penalty strength alpha and the L1/L2 mixing ratio l1_ratio are
chosen by time-series cross-validation, so the degree of
regularization is itself data-driven.

Physical interpretation of the output
----------------------------------------
The active set {j : Omega_j != 0} constitutes the estimated network
topology: a neighbor whose coefficient is shrunk exactly to zero is a
degree of freedom the data supports treating as fully projected out
(absorbed into eta(t)), while every surviving Omega_j is a directly
estimated instantaneous coupling strength A_ij. The nonzero {K_k} are
the discretized memory kernel -- the fingerprint left on r(t+1) by the
currency's own recently projected history. The regression residual
eta(t) = y(t) - X(t)*beta is the empirical noise term, characterized
dynamically in Phase 5.

Output:
  varx_model.pkl      (fitted pipeline + feature list + lag order p)
  residuals.csv        (date, eta)  -> feeds Phase 5
  network_edges.csv    (surviving neighbor columns, standardized weight)
"""

import pickle
import numpy as np
import pandas as pd
import phase1_config as cfg


def main():
    df = pd.read_csv("features.csv", parse_dates=["date"]).set_index("date")

    p = cfg.choose_lag_order(df["r"], cfg.MAX_LAG)
    print(f"[phase4] AIC-selected lag order p = {p}")

    X, y, feature_names = cfg.build_design_matrix(df, p)
    pipeline = cfg.make_scaled_elasticnet_pipeline()
    pipeline.fit(X.values, y.values)
    model = pipeline.named_steps["model"]

    # standardized coefficients -- directly comparable across features
    # of different natural scale/units, which raw coefficients are not
    coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)
    print(f"[phase4] alpha={model.alpha_:.2e}  l1_ratio={model.l1_ratio_}")
    nonzero = coefs[coefs.abs() > 1e-10]
    print("[phase4] nonzero standardized coefficients (network edges + memory kernel + news):")
    print(nonzero)

    kernel_terms = nonzero[[c for c in nonzero.index if c.startswith("r_lag")]]
    network_terms = nonzero[[c for c in nonzero.index
                              if c.endswith("_r") and not c.startswith("r_lag")]]
    news_terms = nonzero[[c for c in nonzero.index if c.startswith("news_")]]

    network_terms.rename("standardized_weight").to_frame().to_csv("network_edges.csv")
    print(f"[phase4] surviving network edges: {list(network_terms.index)}")
    print(f"[phase4] memory kernel (K) terms: {dict(kernel_terms.round(5))}")
    print(f"[phase4] surviving news terms: {list(news_terms.index)}")

    residuals = y.values - pipeline.predict(X.values)
    pd.DataFrame({"date": X.index, "eta": residuals}).to_csv("residuals.csv", index=False)

    with open("varx_model.pkl", "wb") as f:
        pickle.dump({"model": pipeline, "features": feature_names, "p": p}, f)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    print(f"[phase4] in-sample R^2 = {1 - ss_res / ss_tot:.4f}")
    print("[phase4] saved varx_model.pkl, residuals.csv, network_edges.csv")


if __name__ == "__main__":
    main()