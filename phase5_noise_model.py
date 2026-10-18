"""
PHASE 5 — NOISE-TERM CHARACTERIZATION AND IRREVERSIBILITY DIAGNOSTIC

Part A — the noise process
--------------------------
eta(t), the innovation left after Phase 4 removes the linear dependence
of r(t+1) on its own past and on the drives, is not assumed white in
second order, Gaussian, or tied to the memory kernel K by any
fluctuation-dissipation relation. It is modelled as a GARCH(1,1)
process with Student-t innovations:

    eta(t) = sqrt(h(t)) * z(t),           z(t) ~ standardized Student-t(nu)
    h(t)   = omega + alpha*eta(t-1)^2 + beta*h(t-1)

h(t) is the conditional variance. alpha+beta governs the persistence
of the noise's second-order dependence (volatility clustering);
alpha+beta close to 1 indicates extremely persistent conditional
variance and a near-IGARCH specification, but does not by itself prove
asymptotic long memory. The degrees of freedom nu set the tail weight
(nu -> 30 is near-Gaussian; nu of roughly 4-8 is pronounced excess
kurtosis). The finite-order recursion is an approximation to the
noise's temporal dependence, not a claim about its exact form.

Part B — irreversibility (entropy-production) diagnostic
--------------------------------------------------------
Away from equilibrium no fluctuation-dissipation relation constrains K
and eta, so none is tested. What is measured instead is how far the
estimated dynamics is from time-reversal symmetry, using path
statistics only. This is valid arbitrarily far from equilibrium and
requires no Markov property, because the object compared is the
probability measure of a whole window of L consecutive days.

For a window x = (x(0), ..., x(L-1)) of the joint series (the currency
return and the drives with nonzero coupling in Phase 4), let P[x] be
the forward path measure and P[Jx] the measure of the time-reversed
path, J reversing the order of the days. Define

    Sigma[x] = ln P[x] - ln P[Jx].

Then P(Sigma = s) / P(Sigma = -s) = e^s and <e^{-Sigma}> = 1. These are
identities of the path measure, holding for any stationary process; the
information about the system is in the size of

    <Sigma> = D_KL( P || P o J ) >= 0,

which is zero if and only if the process is statistically reversible.

Gaussian evaluation. With Gamma the (block-Toeplitz) covariance of the
stacked window, d channels and L days,

    <Sigma> = (1/2) [ tr( J Gamma^{-1} J Gamma ) - d*L ].

A stationary scalar Gaussian process is exactly reversible (its
covariance function is symmetric), so <Sigma> = 0 for the currency
alone. At second order, irreversibility arises only from directed
coupling, C_ij(tau) != C_ij(-tau) for i != j, which in the fitted model
is carried by the nonzero Omega edges.

Significance. Estimating Gamma from finite data gives <Sigma> > 0 even
for a reversible process, so the observed value is compared with a null
ensemble of reversible Gaussian surrogates: series of the same length
and cospectrum (real part of the smoothed cross-spectral matrix) as the
data, with zero quadrature spectrum (imaginary part), which is the
spectral signature of time-reversal symmetry. The p-value is the
fraction of surrogates whose <Sigma> is at least the observed value.

Scope. This measures second-order (Gaussian) irreversibility of the
observed channels: directed linear coupling. Asymmetry in the noise
process or in the tails, and contributions from unobserved degrees of
freedom, are not captured.

Output: sigma_eta.csv (date, sigma_eta)
        garch_model.pkl (fitted variance recursion, used for multi-step
                         stochastic simulation in Phase 6)
        irreversibility_report.csv (window, channels, observed <Sigma>,
                         null mean and 95th percentile, excess, p-value)
"""

import pickle
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.ndimage import uniform_filter1d
import phase1_config as cfg

GARCH_SCALE = 1000.0  # residuals are ~0.005-0.02; x1000 puts them in arch's
                       # recommended 1-1000 numerical range. Undone everywhere
                       # this constant is used (Phase 5 and Phase 6).


def fit_garch(residuals: pd.Series):
    am = arch_model(residuals.values * GARCH_SCALE, mean="Zero", vol="Garch",
                     p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off")
    cond_vol = np.asarray(res.conditional_volatility) / GARCH_SCALE
    return res, cond_vol


def block_toeplitz_covariance(X: np.ndarray, L: int) -> np.ndarray:
    """Biased sample covariance of the stacked window [x(0); ...; x(L-1)]
    of a stationary d-variate zero-mean series X (N x d). Block (a, b) is
    C(a-b), with C(tau) = E[x(t+tau) x(t)^T] and C(-tau) = C(tau)^T. The
    biased (1/N) estimator yields a positive semi-definite matrix."""
    N, d = X.shape
    C = [(X[tau:].T @ X[:N - tau]) / N for tau in range(L)]
    G = np.zeros((d * L, d * L))
    for a in range(L):
        for b in range(L):
            blk = C[a - b] if a >= b else C[b - a].T
            G[a * d:(a + 1) * d, b * d:(b + 1) * d] = blk
    return G


def path_kl_irreversibility(X: np.ndarray, L: int, ridge: float = 1e-6) -> float:
    """<Sigma> = (1/2)[tr(J Gamma^{-1} J Gamma) - d*L], the Gaussian
    path-level KL divergence between the forward and time-reversed
    measures of an L-day window. J reverses the order of the L blocks."""
    N, d = X.shape
    G = block_toeplitz_covariance(X, L) + ridge * np.eye(d * L)
    Ginv = np.linalg.inv(G)
    idx = np.concatenate([np.arange(a * d, (a + 1) * d) for a in range(L - 1, -1, -1)])
    JGinvJ = Ginv[np.ix_(idx, idx)]
    return 0.5 * (np.sum(JGinvJ * G) - d * L)


def reversible_surrogate_statistics(X: np.ndarray, L: int, n_surr: int,
                                     half_width: int, rng: np.random.Generator) -> np.ndarray:
    """<Sigma> evaluated on n_surr reversible Gaussian surrogates of X.
    Each surrogate is synthesized in the Fourier domain from a complex
    Gaussian vector at every frequency with covariance Re S(omega), the
    real part of the smoothed cross-spectral matrix of X. Its
    cross-covariance is therefore symmetric, C(tau) = C(-tau) = C(tau)^T,
    which is exact time-reversal symmetry, while the marginal and
    cross spectra (cospectrum) match the data."""
    N, d = X.shape
    Xf = np.fft.rfft(X, axis=0)
    K = Xf.shape[0]
    P = np.einsum("ki,kj->kij", Xf, Xf.conj()).real / N
    S = uniform_filter1d(P, size=2 * half_width + 1, axis=0, mode="reflect")
    w, V = np.linalg.eigh(S)
    A = V * np.sqrt(np.clip(w, 0.0, None))[:, None, :]   # A A^T = Re S
    stats = np.empty(n_surr)
    for s in range(n_surr):
        a = rng.standard_normal((K, d))
        b = rng.standard_normal((K, d))
        z = (np.einsum("kij,kj->ki", A, a) + 1j * np.einsum("kij,kj->ki", A, b)) / np.sqrt(2.0)
        z[0] = 0.0                                         # zero-mean surrogate
        if N % 2 == 0:
            z[-1] = np.einsum("ij,j->i", A[-1], a[-1])     # Nyquist bin is real
        Xs = np.fft.irfft(np.sqrt(N) * z, n=N, axis=0)
        Xs -= Xs.mean(axis=0)
        stats[s] = path_kl_irreversibility(Xs, L)
    return stats


def irreversibility_diagnostic(features: pd.DataFrame, coupled: list):
    channels = ["r"] + coupled
    if len(channels) < 2:
        print("[phase5] irreversibility diagnostic: no coupled drive survived Phase 4. "
              "A scalar stationary Gaussian process is exactly time-reversible, so "
              "second-order irreversibility cannot be detected from r(t) alone.")
        return None

    X = features[channels].values.astype(float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)

    L = cfg.IRREVERSIBILITY_WINDOW
    observed = path_kl_irreversibility(X, L)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    null = reversible_surrogate_statistics(X, L, cfg.N_REVERSIBLE_SURROGATES,
                                            cfg.SPECTRAL_SMOOTHING_HALF_WIDTH, rng)
    p_value = (1 + np.sum(null >= observed)) / (1 + len(null))

    report = {
        "window_L": L,
        "channels": ",".join(channels),
        "observed_sigma": observed,
        "null_mean": null.mean(),
        "null_p95": np.percentile(null, 95),
        "excess_over_null_mean": observed - null.mean(),
        "p_value": p_value,
    }
    return report


def main():
    resid = pd.read_csv("residuals.csv", parse_dates=["date"]).set_index("date")["eta"]

    # ── Part A: noise process ──
    res, cond_vol = fit_garch(resid)
    print(res.summary())

    alpha = res.params.get("alpha[1]", np.nan)
    beta = res.params.get("beta[1]", np.nan)
    print(f"\n[phase5] GARCH persistence (alpha+beta) = {alpha + beta:.4f}")
    print("[phase5] (values close to 1 => extremely persistent conditional "
          "variance / near-IGARCH behavior; this does not by itself prove "
          "asymptotic long memory)")

    nu = res.params.get("nu", np.nan)
    print(f"[phase5] Student-t degrees of freedom (nu) = {nu:.2f}")
    print("[phase5] (nu well below ~30 => tails heavier than Gaussian; nu below "
          "~6-8 is pronounced excess kurtosis)")

    pd.DataFrame({"date": resid.index, "sigma_eta": cond_vol}).to_csv("sigma_eta.csv", index=False)
    with open("garch_model.pkl", "wb") as f:
        pickle.dump(res, f)
    print("[phase5] saved sigma_eta.csv, garch_model.pkl")

    # ── Part B: irreversibility diagnostic ──
    features = pd.read_csv("features.csv", parse_dates=["date"]).set_index("date")
    coupled = pd.read_csv("network_edges.csv", index_col=0).index.tolist()
    report = irreversibility_diagnostic(features, coupled)
    if report is None:
        return

    print(f"\n[phase5] irreversibility diagnostic: window L={report['window_L']} days, "
          f"channels = {report['channels']}")
    print(f"[phase5] observed <Sigma>          = {report['observed_sigma']:.4f} nats/window")
    print(f"[phase5] reversible-null <Sigma>   : mean = {report['null_mean']:.4f}, "
          f"95th percentile = {report['null_p95']:.4f}")
    print(f"[phase5] excess over null mean     = {report['excess_over_null_mean']:.4f}  |  "
          f"p-value = {report['p_value']:.4f}")
    if report["p_value"] < 0.05:
        print("[phase5] time-reversal symmetry rejected: directed (non-reciprocal) coupling "
              "between the currency and its drives is present, i.e. the estimated dynamics "
              "is irreversible (out of equilibrium) at second order.")
    else:
        print("[phase5] time-reversal symmetry not rejected at this window and sample size; "
              "this does not establish equilibrium, only that no second-order asymmetry "
              "was resolved.")

    pd.DataFrame([report]).to_csv("irreversibility_report.csv", index=False)
    print("[phase5] saved irreversibility_report.csv")


if __name__ == "__main__":
    main()
