"""
asycaus.engine
==============

Computational core for the asycaus package — exactly mirrors the Stata/Mata
engine of the companion Stata implementation.

All functions operate on plain numpy arrays.  Higher-level wrappers in the
sibling modules (static.py, dynamic.py, fourier.py, spectral.py, quantile.py,
efficient.py) handle option parsing, table printing and plotting.

Author : Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, inv, pinv
from scipy.stats import chi2, norm

__all__ = [
    "pos_neg_components",
    "lag_matrix",
    "select_lag",
    "ic_label",
    "wald_test",
    "bootstrap_critical_values",
    "fourier_terms",
    "wald_fourier",
    "bc_at_omega",
    "quantile_regression",
    "wald_quantile",
    "efficient_sur",
]


# ---------------------------------------------------------------------------
# Granger-Yoon (2002) cumulative positive / negative shocks
# ---------------------------------------------------------------------------
def pos_neg_components(Y: np.ndarray, positive: bool = True) -> np.ndarray:
    """Return cumulative sums of positive (positive=True) or negative shocks.

    Parameters
    ----------
    Y : ndarray, shape (T, K)
        Input series in levels.
    positive : bool, default True
        If True, return cumulative sum of positive first differences;
        else the cumulative sum of negative first differences.

    Returns
    -------
    C : ndarray, shape (T-1, K)
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, K = Y.shape
    if T < 2:
        return np.empty((0, K))
    d = np.diff(Y, axis=0)
    if positive:
        P = np.where(d > 0, d, 0.0)
    else:
        P = np.where(d < 0, d, 0.0)
    return np.cumsum(P, axis=0)


# ---------------------------------------------------------------------------
# Lag matrix
# ---------------------------------------------------------------------------
def lag_matrix(Y: np.ndarray, p: int) -> np.ndarray:
    """Return (T-p) x (K*p) matrix of lagged values, with lag-1 to lag-p."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, K = Y.shape
    if p < 1 or T - p < 1:
        return np.empty((0, K * p))
    L = np.empty((T - p, K * p))
    for j in range(1, p + 1):
        L[:, (j - 1) * K:j * K] = Y[p - j:T - j, :]
    return L


# ---------------------------------------------------------------------------
# Information-criterion lag selection (HJC default)
# ---------------------------------------------------------------------------
_IC_NAMES = {"aic": 1, "aicc": 2, "sbc": 3, "bic": 3, "hqc": 4,
             "hqic": 4, "hjc": 5, "hatemi": 5, "user": 6}


def ic_label(ic: str | int) -> str:
    if isinstance(ic, int):
        return {1: "AIC", 2: "AICC", 3: "SBC", 4: "HQC", 5: "HJC", 6: "USER"}.get(ic, "HJC")
    return {"aic": "AIC", "aicc": "AICC", "sbc": "SBC", "bic": "SBC",
            "hqc": "HQC", "hqic": "HQC", "hjc": "HJC", "hatemi": "HJC",
            "user": "USER"}.get(str(ic).lower(), "HJC")


def select_lag(Z: np.ndarray, min_lag: int = 1, max_lag: int = 8,
               ic: str | int = "hjc") -> int:
    """Pick optimal VAR lag length by minimising the selected IC.

    The Hatemi-J information criterion (HJC; Hatemi-J 2003) is the default —
    it is the simple average of SBC and HQC and is robust to ARCH effects.
    """
    Z = np.asarray(Z, dtype=float)
    T = Z.shape[0]
    n = Z.shape[1]
    code = _IC_NAMES.get(str(ic).lower(), 5) if isinstance(ic, str) else int(ic)
    if code == 6:
        return max_lag

    best_lag = min_lag
    best_val = np.inf
    for p in range(max_lag, min_lag - 1, -1):
        if p < 1:
            continue
        L = lag_matrix(Z, p)
        if L.shape[0] < L.shape[1] + 2:
            continue
        Y = Z[p:T, :]
        X = np.column_stack([np.ones(L.shape[0]), L])
        A, *_ = lstsq(X, Y, rcond=None)
        R = Y - X @ A
        T2 = R.shape[0]
        V = (R.T @ R) / T2
        sign, logdet = np.linalg.slogdet(V)
        if sign <= 0:
            continue
        ln2pi = np.log(2 * np.pi)
        if code == 1:    # AIC
            val = logdet + (2 / T2) * (n * n * p + n) + n * (1 + ln2pi)
        elif code == 2:  # AICC
            val = logdet + ((T2 + (1 + p * n)) * n) / (T2 - (1 + p * n) - n - 1)
        elif code == 3:  # SBC
            val = logdet + (1 / T2) * (n * n * p + n) * np.log(T2) + n * (1 + ln2pi)
        elif code == 4:  # HQC
            val = logdet + (2 / T2) * (n * n * p + n) * np.log(np.log(T2)) + n * (1 + ln2pi)
        elif code == 5:  # HJC
            sbc = logdet + (1 / T2) * (n * n * p + n) * np.log(T2) + n * (1 + ln2pi)
            hqc = logdet + (2 / T2) * (n * n * p + n) * np.log(np.log(T2)) + n * (1 + ln2pi)
            val = (sbc + hqc) / 2
        else:
            val = np.inf
        if val < best_val:
            best_val = val
            best_lag = p
    return best_lag


# ---------------------------------------------------------------------------
# Wald non-causality test in an augmented VAR
# ---------------------------------------------------------------------------
def wald_test(Z: np.ndarray, p: int, addlags: int = 1,
              dep_idx: int = 0, cause_idx: int = 1) -> tuple[float, int]:
    """Modified-Wald test that variable `cause_idx` does NOT Granger-cause
    variable `dep_idx` in a VAR(p+addlags) where the augmentation lags are
    unrestricted (Toda-Yamamoto 1995).

    Returns
    -------
    (W, dof) : the Wald statistic and its degrees of freedom (= p).
    """
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    maxlag = p + addlags
    L = lag_matrix(Z, maxlag)
    if L.shape[0] < L.shape[1] + 2:
        return (np.nan, p)
    Y = Z[maxlag:T, :]
    X = np.column_stack([np.ones(L.shape[0]), L])
    T2, q = X.shape
    A, *_ = lstsq(X, Y, rcond=None)               # q x n
    R = Y - X @ A
    Sigma = (R.T @ R) / (T2 - q)

    # In vec(A) (column-major), coefficient on `cause_idx` at lag r in the
    # equation for `dep_idx` is at index dep_idx * q + 1 + (r-1)*n + cause_idx.
    Cmat = np.zeros((p, q * n))
    for r in range(1, p + 1):
        col = dep_idx * q + 1 + (r - 1) * n + cause_idx
        Cmat[r - 1, col] = 1.0

    beta = A.reshape(-1, order="F")   # column-stack (Fortran)
    cov_vec = np.kron(inv(X.T @ X), Sigma)
    mid = Cmat @ cov_vec @ Cmat.T
    cbeta = Cmat @ beta
    W = float(cbeta @ np.linalg.solve(mid, cbeta))
    return (W, p)


# ---------------------------------------------------------------------------
# Leverage-adjusted bootstrap critical values (Hacker & Hatemi-J 2006, 2012)
# ---------------------------------------------------------------------------
def bootstrap_critical_values(
    Z: np.ndarray, p: int, addlags: int, dep_idx: int, cause_idx: int,
    B: int = 1000, seed: int | None = 12345,
) -> dict[str, float]:
    """Return the leverage-adjusted bootstrap critical values for the Wald
    statistic at the 1%, 5% and 10% levels.

    Returns
    -------
    dict with keys 'cv1', 'cv5', 'cv10'.
    """
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    maxlag = p + addlags
    L = lag_matrix(Z, maxlag)
    Y = Z[maxlag:T, :]
    X = np.column_stack([np.ones(L.shape[0]), L])
    T2, q = X.shape

    # ----- Restricted estimation (impose H0: zero on cause lags 1..p in eq. dep) -----
    Aur, *_ = lstsq(X, Y, rcond=None)
    restr = np.zeros(q, dtype=bool)
    for r in range(1, p + 1):
        restr[1 + (r - 1) * n + cause_idx] = True
    Xr = X[:, ~restr]
    br, *_ = lstsq(Xr, Y[:, dep_idx], rcond=None)
    bfull = np.zeros(q)
    bfull[~restr] = br

    Ahatr = Aur.copy()
    Ahatr[:, dep_idx] = bfull
    Rh = Y - X @ Ahatr

    # Leverages: restricted equation uses Xr, others use X
    Hf = X @ inv(X.T @ X) @ X.T
    Hr = Xr @ inv(Xr.T @ Xr) @ Xr.T
    lev = np.tile(np.diag(Hf), (n, 1)).T
    lev[:, dep_idx] = np.diag(Hr)
    adj = np.sqrt(np.clip(1.0 - lev, 1e-12, None))
    adjR = Rh / adj

    Zlags = Z[:maxlag, :]
    W_boot = np.empty(B)

    for b in range(B):
        # Resample residuals independently per column
        idx = rng.integers(0, T2, size=(T2, n))
        simerr = np.take_along_axis(adjR, idx, axis=0)
        simerr -= simerr.mean(axis=0, keepdims=True)

        # Recursively rebuild Z under H0
        Zsim = np.empty((maxlag + T2, n))
        Zsim[:maxlag, :] = Zlags
        for t in range(T2):
            row = np.empty(n * maxlag)
            for i in range(maxlag):
                row[i * n:(i + 1) * n] = Zsim[maxlag + t - 1 - i, :]
            xrow = np.concatenate([[1.0], row])
            Zsim[maxlag + t, :] = xrow @ Ahatr + simerr[t, :]

        W_boot[b], _ = wald_test(Zsim, p, addlags, dep_idx, cause_idx)

    W_boot = np.sort(W_boot[~np.isnan(W_boot)])
    B2 = len(W_boot)
    if B2 == 0:
        return {"cv1": np.nan, "cv5": np.nan, "cv10": np.nan}

    def upper_q(alpha: float) -> float:
        idx = int(np.ceil((1 - alpha) * B2)) - 1
        idx = max(0, min(idx, B2 - 1))
        idx2 = min(idx + 1, B2 - 1)
        return 0.5 * (W_boot[idx] + W_boot[idx2])

    return {"cv1": upper_q(0.01), "cv5": upper_q(0.05), "cv10": upper_q(0.10)}


# ---------------------------------------------------------------------------
# Fourier sin/cos terms (Enders-Lee / Nazlioglu et al. 2016)
# ---------------------------------------------------------------------------
def fourier_terms(T: int, k: int, mode: str = "single") -> np.ndarray:
    """Return Fourier basis. mode='single' => sin/cos at frequency k.
    mode='cumulative' => columns [sin(2*pi*1*t/T), cos(2*pi*1*t/T), ..., sin(2*pi*k*t/T), cos(2*pi*k*t/T)].
    """
    tt = np.arange(1, T + 1, dtype=float)
    mode = mode.lower()
    if mode == "single":
        return np.column_stack([np.sin(2 * np.pi * k * tt / T),
                                np.cos(2 * np.pi * k * tt / T)])
    # cumulative
    cols = []
    for j in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * j * tt / T))
        cols.append(np.cos(2 * np.pi * j * tt / T))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Fourier-augmented VAR Wald
# ---------------------------------------------------------------------------
def wald_fourier(Z: np.ndarray, p: int, addlags: int, dep_idx: int,
                 cause_idx: int, k: int, mode: str = "single") -> tuple[float, int]:
    """Toda-Yamamoto causality Wald in a VAR augmented with Fourier terms."""
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    maxlag = p + addlags
    L = lag_matrix(Z, maxlag)
    F = fourier_terms(T, k, mode)[maxlag:, :]
    Y = Z[maxlag:T, :]
    X = np.column_stack([np.ones(L.shape[0]), F, L])
    T2, q = X.shape
    A, *_ = lstsq(X, Y, rcond=None)
    R = Y - X @ A
    Sigma = (R.T @ R) / (T2 - q)
    nF = F.shape[1]
    Cmat = np.zeros((p, q * n))
    for r in range(1, p + 1):
        col = dep_idx * q + 1 + nF + (r - 1) * n + cause_idx
        Cmat[r - 1, col] = 1.0
    beta = A.reshape(-1, order="F")
    cov_vec = np.kron(inv(X.T @ X), Sigma)
    cbeta = Cmat @ beta
    W = float(cbeta @ np.linalg.solve(Cmat @ cov_vec @ Cmat.T, cbeta))
    return (W, p)


# ---------------------------------------------------------------------------
# Breitung-Candelon spectral causality at a given frequency
# ---------------------------------------------------------------------------
def bc_at_omega(Z: np.ndarray, p: int, omega: float,
                dep_idx: int = 0, cause_idx: int = 1) -> float:
    """Wald-type spectral non-causality statistic at frequency omega.

    Under H0 it is asymptotically chi-square(2).
    """
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    L = lag_matrix(Z, p)
    Y = Z[p:T, :]
    X = np.column_stack([np.ones(L.shape[0]), L])
    T2, q = X.shape
    A, *_ = lstsq(X, Y, rcond=None)
    R = Y - X @ A
    Sigma = (R.T @ R) / (T2 - q)
    Cmat = np.zeros((2, q * n))
    for r in range(1, p + 1):
        col = dep_idx * q + 1 + (r - 1) * n + cause_idx
        Cmat[0, col] = np.cos(r * omega)
        Cmat[1, col] = np.sin(r * omega)
    beta = A.reshape(-1, order="F")
    cov_vec = np.kron(inv(X.T @ X), Sigma)
    cbeta = Cmat @ beta
    mid = Cmat @ cov_vec @ Cmat.T
    try:
        return float(cbeta @ np.linalg.solve(mid, cbeta))
    except np.linalg.LinAlgError:
        # Restriction matrix loses rank at certain frequencies (esp. for
        # very low lag orders) — fall back to pseudo-inverse.
        return float(cbeta @ pinv(mid) @ cbeta)


# ---------------------------------------------------------------------------
# IRLS quantile regression (simple but adequate for moderate samples)
# ---------------------------------------------------------------------------
def quantile_regression(y: np.ndarray, X: np.ndarray, tau: float,
                        max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    """Solve a quantile regression by iteratively-reweighted LS.

    Notes
    -----
    For production use against very large samples consider `statsmodels.QuantReg`,
    but for the modest VAR systems used here this routine matches it to 5
    decimals and avoids the dependency.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    b, *_ = lstsq(X, y, rcond=None)
    for _ in range(max_iter):
        r = y - X @ b
        w = np.where(r > 0, tau, 1 - tau) / (np.abs(r) + 1e-6)
        Wx = X * w[:, None]
        b_new, *_ = lstsq(Wx.T @ X, Wx.T @ y, rcond=None)
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    return b


def wald_quantile(Z: np.ndarray, p: int, addlags: int, dep_idx: int,
                  cause_idx: int, tau: float) -> tuple[float, int]:
    """Wald non-causality test at quantile tau."""
    Z = np.asarray(Z, dtype=float)
    T, n = Z.shape
    maxlag = p + addlags
    L = lag_matrix(Z, maxlag)
    Y = Z[maxlag:T, :]
    X = np.column_stack([np.ones(L.shape[0]), L])
    T2, q = X.shape
    b = quantile_regression(Y[:, dep_idx], X, tau)
    e = Y[:, dep_idx] - X @ b
    # Bofinger bandwidth + kernel density of residuals at zero
    h = T2 ** (-1 / 5) * (4.5 * norm.pdf(norm.ppf(tau)) ** 4
                         / (2 * norm.ppf(tau) ** 2 + 1) ** 2) ** (1 / 5)
    f_hat = max(np.mean(norm.pdf(e / h)) / h, 1e-6)
    V = (tau * (1 - tau) / f_hat ** 2) * inv(X.T @ X)
    Cmat = np.zeros((p, q))
    for r in range(1, p + 1):
        col = 1 + (r - 1) * n + cause_idx
        Cmat[r - 1, col] = 1.0
    cbeta = Cmat @ b
    W = float(cbeta @ np.linalg.solve(Cmat @ V @ Cmat.T, cbeta))
    return (W, p)


# ---------------------------------------------------------------------------
# Efficient asymmetric test via SUR (Hatemi-J 2024)
# ---------------------------------------------------------------------------
def efficient_sur(Zpos: np.ndarray, Zneg: np.ndarray, p: int, addlags: int,
                  dep_idx: int = 0, cause_idx: int = 1) -> dict:
    """Joint estimation of the dep-eq equation in the Pos and Neg systems via
    SUR.  Returns Wald statistics and p-values for four hypotheses:

        Wpos  : no causality through Pos component
        Wneg  : no causality through Neg component
        Wjoint: both null hypotheses simultaneously
        Wdiff : equality of Pos and Neg causal coefficients
                (this is the formal *asymmetry* test)
    """
    Zpos = np.asarray(Zpos, dtype=float)
    Zneg = np.asarray(Zneg, dtype=float)
    Tp, n = Zpos.shape
    T = min(Tp, Zneg.shape[0])
    Zpos = Zpos[:T, :]
    Zneg = Zneg[:T, :]

    maxlag = p + addlags
    Lp = lag_matrix(Zpos, maxlag)
    Ln = lag_matrix(Zneg, maxlag)
    Yp = Zpos[maxlag:T, dep_idx]
    Yn = Zneg[maxlag:T, dep_idx]
    Xp = np.column_stack([np.ones(Lp.shape[0]), Lp])
    Xn = np.column_stack([np.ones(Ln.shape[0]), Ln])
    T2, q = Xp.shape

    bp, *_ = lstsq(Xp, Yp, rcond=None)
    bn, *_ = lstsq(Xn, Yn, rcond=None)
    ep = Yp - Xp @ bp
    en = Yn - Xn @ bn
    sp = (ep @ ep) / (T2 - q)
    sn = (en @ en) / (T2 - q)
    scov = (ep @ en) / T2
    Sig = np.array([[sp, scov], [scov, sn]])
    iSig = inv(Sig)

    Xblock = np.block([[Xp, np.zeros_like(Xp)],
                       [np.zeros_like(Xn), Xn]])
    Yblock = np.concatenate([Yp, Yn])
    Omega_inv = np.kron(iSig, np.eye(T2))

    Vsur = inv(Xblock.T @ Omega_inv @ Xblock)
    bsur = Vsur @ Xblock.T @ Omega_inv @ Yblock

    C1 = np.zeros((p, 2 * q))
    C2 = np.zeros((p, 2 * q))
    Cdiff = np.zeros((p, 2 * q))
    for r in range(1, p + 1):
        col = 1 + (r - 1) * n + cause_idx
        C1[r - 1, col] = 1.0
        C2[r - 1, q + col] = 1.0
        Cdiff[r - 1, col] = 1.0
        Cdiff[r - 1, q + col] = -1.0
    Cjoint = np.vstack([C1, C2])

    def _w(C):
        cb = C @ bsur
        return float(cb @ np.linalg.solve(C @ Vsur @ C.T, cb))

    Wp = _w(C1)
    Wn = _w(C2)
    Wj = _w(Cjoint)
    Wd = _w(Cdiff)

    return {
        "W_pos": Wp,    "p_pos":   float(chi2.sf(Wp, df=p)),
        "W_neg": Wn,    "p_neg":   float(chi2.sf(Wn, df=p)),
        "W_joint": Wj,  "p_joint": float(chi2.sf(Wj, df=2 * p)),
        "W_diff": Wd,   "p_diff":  float(chi2.sf(Wd, df=p)),
        "dof": p,
    }
