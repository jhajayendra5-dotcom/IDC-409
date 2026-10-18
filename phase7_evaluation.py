"""
PHASE 7 — WALK-FORWARD EVALUATION AND ADEQUACY DIAGNOSTICS

Role
----
The fitted GLE approximation makes three truncations: a finite memory
depth p, a finite-order noise recursion, and time-invariant
coefficients. None is presumed adequate, and none is a claim about the
system. This phase measures out-of-sample predictive accuracy and then
tests each truncation directly on the residuals.

Evaluation protocol
-------------------
Accuracy is assessed one step ahead under a walk-forward
(expanding-window) protocol: the model is refit every REFIT_EVERY days
using only data available up to that point, then evaluated on the
immediately following observation before the window advances. This
mirrors the information constraint present at deployment (no future data
enters a given fit) and gives a substantially more reliable estimate of
generalization than a single fixed train/test split, particularly for a
nonstationary financial time series.

Baselines
---------
Three comparisons isolate which structural component of the GLE
contributes predictive value:

  - naive persistence: r_hat(t+1) = r(t). The null model, with zero
    estimated parameters, against which any structural claim must be
    measured.
  - OLS without memory-kernel terms: the identical exogenous feature
    set as the candidate model with the own-lag columns removed --
    isolates whether the memory kernel contributes predictive value
    beyond instantaneous coupling to the drives alone.
  - Random forest on the identical feature set as the candidate model
    -- isolates whether nonlinearity in the coupling is warranted,
    relative to the candidate's linear specification.

Residual diagnostics
--------------------
Three named statistical tests examine the truncations, each with a
directly actionable interpretation:

  - The Ljung-Box test on the candidate model's residuals tests the
    null hypothesis of no residual autocorrelation up to a given lag.
    A significant result indicates the memory depth p (Phase 4) is too
    low: systematic own-history structure remains unexplained.
  - The ARCH-LM test on the same residuals tests the null hypothesis
    of no remaining autoregressive conditional heteroskedasticity. A
    significant result indicates the GARCH(1,1) specification
    (Phase 5) is insufficient to capture the residual's variance
    dynamics.
  - The CUSUM test, applied to the same residuals after dividing each
    by Phase 5's fitted conditional volatility sigma_eta(t), tests the
    null hypothesis of parameter stability: that a single fixed set of
    coefficients {K_k, Omega_j} is adequate across the entire evaluation
    window. Studentizing first removes the known volatility clustering
    from the series being tested, so a significant result reflects
    parameter drift rather than the heteroskedasticity Phase 5 already
    characterizes -- for a driven system whose external forcing changes
    over time, and for a managed-float currency subject to changes in
    central-bank intervention behavior, an expected possibility -- that
    the fixed-coefficient linear specification does not represent.

Passing all three tests is evidence, not proof, that the truncations
are adequate given the available data.

The candidate model's walk-forward refits reuse the (alpha, l1_ratio)
already selected by Phase 4's cross-validated fit (loaded from
varx_model.pkl) rather than re-running that selection at every refit:
coefficients are re-estimated on each expanding window, but the
regularization strength itself is treated as fixed over the evaluation
horizon, which is standard walk-forward practice and avoids repeating
an expensive hyperparameter search whose result is not expected to
change materially every REFIT_EVERY days.

Output: evaluation_report.txt
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import phase1_config as cfg


def walk_forward(X: pd.DataFrame, y: pd.Series, model_factory, refit_every: int, test_window: int):
    n = len(X)
    start = n - test_window
    preds, actuals, dates = [], [], []
    model = None
    for i in range(start, n):
        if model is None or (i - start) % refit_every == 0:
            model = model_factory()
            model.fit(X.iloc[:i].values, y.iloc[:i].values)
        preds.append(model.predict(X.iloc[i:i + 1].values)[0])
        actuals.append(y.iloc[i])
        dates.append(X.index[i])
    return pd.DataFrame({"date": dates, "pred": preds, "actual": actuals})


def report_regression(name: str, res: pd.DataFrame, lines: list):
    mae = mean_absolute_error(res["actual"], res["pred"])
    rmse = np.sqrt(mean_squared_error(res["actual"], res["pred"]))
    r2 = r2_score(res["actual"], res["pred"])
    lines.append(f"{name:32s} MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.4f}")


def report_classification(name: str, res: pd.DataFrame, lines: list):
    y_true = (res["actual"] > 0).astype(int)
    y_pred = (res["pred"] > 0).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    lines.append(f"{name:32s} F1={f1:.4f}  confusion_matrix={cm.tolist()}")


def main():
    df = pd.read_csv("features.csv", parse_dates=["date"]).set_index("date")

    # p is derived by the identical conditional ElasticNet CV procedure
    # Phase 4 applies, so the specification evaluated here matches the
    # specification fit in Phase 4 exactly
    p = cfg.choose_lag_order(df, cfg.MAX_LAG)
    print(f"[phase7] lag order p = {p} (matches Phase 4's conditional CV selection)")

    # Phase 4's cross-validated (alpha, l1_ratio) are reused rather than
    # re-selected at every walk-forward refit: the search itself
    # (~30 alphas x 4 l1_ratios x 5 CV folds) costs roughly two orders of
    # magnitude more than a single coefficient re-estimation, and the
    # regularization strength is not expected to change materially every
    # REFIT_EVERY days
    with open("varx_model.pkl", "rb") as f:
        varx = pickle.load(f)
    fitted_model = varx["model"].named_steps["model"]
    alpha, l1_ratio = fitted_model.alpha_, fitted_model.l1_ratio_
    print(f"[phase7] reusing Phase 4's tuned alpha={alpha:.2e}, l1_ratio={l1_ratio} "
          f"for all walk-forward refits")

    lines = [
        f"=== WALK-FORWARD EVALUATION (expanding window, refit every "
        f"{cfg.REFIT_EVERY} days, last {cfg.TEST_WINDOW} days tested) ===\n"
    ]

    X_full, y_full, _ = cfg.build_design_matrix(df, p)
    X_nomem, y_nomem = cfg.build_no_memory_matrix(df)

    # identical spec to Phase 4's candidate fit -- single source of truth
    varx_res = walk_forward(
        X_full, y_full, lambda: cfg.make_scaled_elasticnet_fixed(alpha, l1_ratio),
        cfg.REFIT_EVERY, cfg.TEST_WINDOW,
    )

    naive_res = varx_res.copy()
    naive_res["pred"] = df["r"].reindex(varx_res["date"]).values  # tomorrow = today's r

    ols_res = walk_forward(
        X_nomem, y_nomem, LinearRegression,
        cfg.REFIT_EVERY, cfg.TEST_WINDOW,
    )

    rf_res = walk_forward(
        X_full, y_full,
        lambda: RandomForestRegressor(n_estimators=200, max_depth=4, random_state=cfg.RANDOM_SEED),
        cfg.REFIT_EVERY, cfg.TEST_WINDOW,
    )

    for name, res in [
        ("naive persistence", naive_res),
        ("OLS, no memory kernel", ols_res),
        ("VARX-GLE (candidate)", varx_res),
        ("RandomForest (nonlinear check)", rf_res),
    ]:
        report_regression(name, res, lines)
        report_classification(name, res, lines)

    lines.append("\n=== DIAGNOSTIC TESTS (candidate model residuals) ===")
    resid = (varx_res["actual"] - varx_res["pred"]).values

    lb = acorr_ljungbox(resid, lags=[5, 10], return_df=True)
    lines.append(f"Ljung-Box (lag 5, 10) p-values:\n{lb['lb_pvalue'].to_string()}")
    lag_insufficient = (lb["lb_pvalue"] < 0.05).any()
    if lag_insufficient:
        lines.append(">> significant residual autocorrelation remains -> the selected linear conditional-mean specification does not fully explain temporal dependence; consider a richer conditional-mean model rather than increasing p solely to eliminate this diagnostic.")
    else:
        lines.append(">> no significant residual autocorrelation -> the selected conditional-mean specification appears adequate at the tested lags.")

    _, arch_p, _, _ = het_arch(resid, result_object=False)
    lines.append(f"\nARCH-LM test p-value: {arch_p:.4f}")
    garch_insufficient = arch_p < 0.05
    if garch_insufficient:
        lines.append(">> significant remaining heteroskedasticity -> GARCH(1,1) insufficient, try (1,2)/(2,1) in Phase 5.")
    else:
        lines.append(">> no significant remaining ARCH effect -> GARCH(1,1) noise model looks adequate.")

    # the CUSUM test's null assumes approximately homoskedastic input;
    # eta(t) is known (Phase 5) to have time-varying conditional
    # volatility, so residuals are studentized by that same
    # sigma_eta(t) before testing, isolating genuine parameter drift
    # from the already-characterized volatility clustering
    sigma_eta = pd.read_csv("sigma_eta.csv", parse_dates=["date"]).set_index("date")["sigma_eta"]
    sigma_aligned = sigma_eta.reindex(varx_res["date"]).ffill().bfill().values
    resid_studentized = resid / sigma_aligned
    cusum_stat, cusum_p = cfg.check_structural_break(resid_studentized)
    lines.append(f"\nCUSUM structural-break test: statistic={cusum_stat:.4f}, p={cusum_p:.4f}")
    regime_shift_detected = cusum_p < 0.05
    if regime_shift_detected:
        lines.append(">> parameter instability detected -> a single fixed-coefficient fit over "
                      "the full test window is misspecified; shorten REFIT_EVERY or consider a "
                      "regime-dependent (e.g. Markov-switching) extension of Phase 4.")
    else:
        lines.append(">> no significant parameter instability detected over the test window.")

    if lag_insufficient or garch_insufficient or regime_shift_detected:
        lines.append(
            "\n>> ACTION: adjust the flagged setting, re-run Phase 4 (and Phase 5 if GARCH order "
            "changed), then re-run this phase. This is the framework's iterative improvement "
            "loop -- statistical tests plus a re-fit, no gradient descent required."
        )

    report = "\n".join(lines)
    print(report)
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    print("\n[phase7] saved evaluation_report.txt")


if __name__ == "__main__":
    main()
