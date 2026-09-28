"""
PHASE 7 — WALK-FORWARD EVALUATION AND MODEL DIAGNOSTICS

Mathematical role
-------------------
This phase assesses one-step-ahead predictive accuracy under a
walk-forward (expanding-window) protocol: the model is refit every
REFIT_EVERY days using only data available up to that point, then
evaluated on the immediately following observation before the window
advances. This mirrors the information constraint present at
deployment time (no future data enters a given fit) and yields a
substantially more reliable estimate of generalization than a single
fixed train/test split, particularly for a nonstationary financial
time series.

Baselines
-----------
Three nested comparisons isolate which structural component of the
GLE contributes genuine predictive value:

  - naive persistence: r_hat(t+1) = r(t). The null model, with zero
    estimated parameters, against which any structural claim must be
    measured.
  - OLS without memory-kernel terms: the identical exogenous feature
    set as the candidate model with the own-lag columns removed --
    isolates whether the memory kernel contributes predictive value
    beyond instantaneous network coupling alone.
  - Random forest on the identical feature set as the candidate model
    -- isolates whether nonlinearity in the coupling function
    G(s_i, s_j) is warranted, relative to the candidate's linear
    specification.

Residual diagnostics
-----------------------
Two named statistical tests validate the two structural assumptions
made upstream, each with a directly actionable interpretation:

  - The Ljung-Box test on the candidate model's residuals tests the
    null hypothesis of no residual autocorrelation up to a given lag.
    A significant result indicates the memory-kernel order p (Phase 4)
    is too low: systematic own-history structure remains unexplained.
  - The ARCH-LM test on the same residuals tests the null hypothesis
    of no remaining autoregressive conditional heteroskedasticity. A
    significant result indicates the GARCH(1,1) specification
    (Phase 5) is insufficient to capture the residual's variance
    dynamics.

Passing both tests is evidence, not proof, that the fitted memory
kernel and noise model are adequate given the available data.

Output: evaluation_report.txt
"""

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
    p = cfg.MAX_LAG

    lines = [
        f"=== WALK-FORWARD EVALUATION (expanding window, refit every "
        f"{cfg.REFIT_EVERY} days, last {cfg.TEST_WINDOW} days tested) ===\n"
    ]

    X_full, y_full, _ = cfg.build_design_matrix(df, p)
    X_nomem, y_nomem = cfg.build_no_memory_matrix(df)

    # identical spec to Phase 4's candidate fit -- single source of truth
    varx_res = walk_forward(
        X_full, y_full, cfg.make_scaled_elasticnet_pipeline,
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
        lines.append(">> significant autocorrelation remains -> increase memory kernel lag order p (cfg.MAX_LAG).")
    else:
        lines.append(">> no significant residual autocorrelation -> memory kernel order p looks sufficient.")

    _, arch_p, _, _ = het_arch(resid)
    lines.append(f"\nARCH-LM test p-value: {arch_p:.4f}")
    garch_insufficient = arch_p < 0.05
    if garch_insufficient:
        lines.append(">> significant remaining heteroskedasticity -> GARCH(1,1) insufficient, try (1,2)/(2,1) in Phase 5.")
    else:
        lines.append(">> no significant remaining ARCH effect -> GARCH(1,1) noise model looks adequate.")

    if lag_insufficient or garch_insufficient:
        lines.append(
            "\n>> ACTION: adjust the flagged setting, re-run Phase 4 (and Phase 5 if GARCH order "
            "changed), then re-run this phase. This is the framework's iterative improvement "
            "loop -- two statistical tests plus a re-fit, no gradient descent required."
        )

    report = "\n".join(lines)
    print(report)
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    print("\n[phase7] saved evaluation_report.txt")


if __name__ == "__main__":
    main()