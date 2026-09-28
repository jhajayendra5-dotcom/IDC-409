"""
PHASE 5 — NOISE-TERM (eta) DYNAMICAL CHARACTERIZATION

Physical and mathematical role
--------------------------------
The GLE's noise term eta(t) is, formally, the projection of every
integrated-out (non-network) degree of freedom onto the observable
subspace. Empirically, financial return residuals exhibit volatility
clustering: eta(t) is not i.i.d., and its conditional variance is
itself autocorrelated. A constant-variance noise model is therefore
physically inadequate. This phase fits eta(t) with a GARCH(1,1)
process:

    eta(t) = sqrt(h(t)) * z(t),           z(t) ~ N(0,1)
    h(t)   = omega + alpha*eta(t-1)^2 + beta*h(t-1)

h(t) is the conditional variance; alpha and beta jointly determine the
persistence of a variance shock, with alpha+beta -> 1 indicating long
memory in volatility (strong clustering), and alpha+beta well below 1
indicating rapid mean-reversion of variance toward its unconditional
level omega/(1-alpha-beta).

Fluctuation-dissipation consistency check
---------------------------------------------
The classical fluctuation-dissipation theorem links the memory kernel
K(t) and the noise autocorrelation via

    <eta(t)*eta(t')> = k_B*T * K(|t-t'|)

i.e. under an equilibrium assumption, the two should decay at
commensurate rates. This phase compares the GARCH persistence
(alpha+beta), which governs the decay rate of eta's own
autocorrelation, against the magnitude of the memory-kernel
coefficients {K_k} estimated in Phase 4. Because currency markets are
not thermodynamic systems at equilibrium, exact agreement is not
expected; the comparison is reported purely as a diagnostic, and a
divergence between the two decay rates is itself an informative
statement about the degree of non-equilibrium behavior present in the
estimated system.

Output: sigma_eta.csv (date, sigma_eta)
        garch_model.pkl  (the fitted variance-recursion, used for
                           multi-step stochastic simulation in Phase 6)
"""

import pickle
import numpy as np
import pandas as pd
from arch import arch_model

GARCH_SCALE = 1000.0  # residuals are ~0.005-0.02; x1000 puts them in arch's
                       # recommended 1-1000 numerical range. Undone everywhere
                       # this constant is used (Phase 5 and Phase 6).


def fit_garch(residuals: pd.Series):
    am = arch_model(residuals.values * GARCH_SCALE, mean="Zero", vol="Garch",
                     p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    cond_vol = np.asarray(res.conditional_volatility) / GARCH_SCALE
    return res, cond_vol


def kernel_lag_coefficients(varx_pickle_path="varx_model.pkl"):
    with open(varx_pickle_path, "rb") as f:
        d = pickle.load(f)
    pipeline, features, p = d["model"], d["features"], d["p"]
    model = pipeline.named_steps["model"]
    return [model.coef_[features.index(f"r_lag{k}")] for k in range(1, p + 1)]


def main():
    resid = pd.read_csv("residuals.csv", parse_dates=["date"]).set_index("date")["eta"]

    res, cond_vol = fit_garch(resid)
    print(res.summary())

    alpha = res.params.get("alpha[1]", np.nan)
    beta = res.params.get("beta[1]", np.nan)
    persistence = alpha + beta
    print(f"\n[phase5] GARCH persistence (alpha+beta) = {persistence:.4f}")
    print("[phase5] (values close to 1 => long volatility memory / strong clustering)")

    lag_coefs = kernel_lag_coefficients()
    print(f"[phase5] memory-kernel own-lag (standardized) coefficient(s) K = {np.round(lag_coefs, 4)}")
    print(
        "[phase5] consistency check: compare decay implied by |K| to GARCH persistence.\n"
        "         Rough agreement supports a fluctuation-dissipation reading of the fit;\n"
        "         disagreement is a legitimate finding about non-equilibrium behavior,\n"
        "         not necessarily a bug -- report either outcome as-is."
    )

    pd.DataFrame({"date": resid.index, "sigma_eta": cond_vol}).to_csv("sigma_eta.csv", index=False)
    with open("garch_model.pkl", "wb") as f:
        pickle.dump(res, f)

    print("[phase5] saved sigma_eta.csv, garch_model.pkl")


if __name__ == "__main__":
    main()