"""
PHASE 4 — DATA-DRIVEN ESTIMATION OF THE MEMORY KERNEL AND DRIVE COUPLING

Role
----
This phase estimates the deterministic part of the reduced GLE dynamics
from data: the memory kernel {K_k} and the drive couplings {Omega_j} in

    r(t+1) = sum_{k=1}^{p} K_k*r(t-k+1) + sum_j Omega_j*x_j(t) + eta(t)

with x_j(t) ranging over every candidate neighbor return, news channel
and sigma(t) from Phase 3. All coefficients are estimated jointly by
elastic-net regression:

    minimize_beta  (1/2n)||y - X*beta||_2^2
                    + alpha*[ l1_ratio*||beta||_1
                              + (1-l1_ratio)/2*||beta||_2^2 ]

on a standardized design matrix (each column rescaled to zero mean and
unit variance). Standardization is required because the L1 term
penalizes raw coefficient magnitude, which depends on the measurement
units of x_j. The penalty strength alpha and the mixing ratio l1_ratio
are chosen by time-series cross-validation, so the regularization is
itself data-driven.

Status of the approximation
---------------------------
The Mori-Zwanzig identity defines K and eta exactly, but in terms of the
full microscopic dynamics, which are not available. The fit here is a
data-driven approximation: linear in the mean, with time-invariant
coefficients and finite memory depth p. These are properties of the
approximation, not claims about the system. p truncates a kernel whose
true support may be unbounded, and linearity and time-invariance are
tested in Phase 7.

No relation between K and the noise is imposed, and in particular no
fluctuation-dissipation relation. Consequently, from a single observed
trajectory, linear memory and any linear autocorrelation of the driving
noise enter together as linear temporal dependence and are not
separately identifiable. The fitted {K_k} therefore represent the total
linear temporal dependence of the projected dynamics, and eta(t) is the
innovation remaining after all linear dependence on the past and on the
drives is removed. Temporal dependence of the noise in second order is
characterized in Phase 5.

Interpretation of the output
----------------------------
The active set {j : Omega_j != 0} is the estimated set of drives coupled
to the currency: a drive shrunk exactly to zero is treated as absorbed
into eta(t), and every surviving Omega_j is an estimated coupling
strength. The nonzero {K_k} are the discretized memory kernel. The
residual eta(t) = y(t) - X(t)*beta is passed to Phase 5.

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
    state_terms = nonzero[[c for c in nonzero.index if c == "sigma"]]

    network_terms.rename("standardized_weight").to_frame().to_csv("network_edges.csv")
    print(f"[phase4] surviving network edges: {list(network_terms.index)}")
    print(f"[phase4] memory kernel (K) terms: {dict(kernel_terms.round(5))}")
    print(f"[phase4] surviving news terms: {list(news_terms.index)}")
    print(f"[phase4] surviving state-variable (sigma) term: {dict(state_terms.round(5))}")

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
