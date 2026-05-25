"""
Basic smoke tests — run with `pytest tests/` from the package root.
"""

import numpy as np
import pytest
import asycaus


@pytest.fixture
def asym_dgp():
    """Bivariate DGP where x's POSITIVE shocks cause y."""
    rng = np.random.default_rng(30540)
    T = 250
    ex = rng.standard_normal(T)
    ey = rng.standard_normal(T)
    x = np.cumsum(ex)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.7 * max(ex[t - 1], 0) + ey[t]
    return y, x


def test_components_shape(asym_dgp):
    y, x = asym_dgp
    Y = np.column_stack([y, x])
    Cpos = asycaus.pos_neg_components(Y, positive=True)
    Cneg = asycaus.pos_neg_components(Y, positive=False)
    assert Cpos.shape == (len(y) - 1, 2)
    assert Cneg.shape == (len(y) - 1, 2)
    # Cumulative pos series is monotonically non-decreasing for column 0
    assert np.all(np.diff(Cpos[:, 0]) >= 0)


def test_static_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.static(y, x, shock="both", boot=80, seed=1,
                         show=False, plot=False)
    assert "Wald" in res.table.columns
    assert {"Positive", "Negative"}.issubset(set(res.table.index))


def test_fourier_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.fourier(y, x, kmax=2, form="single", show=False, plot=False)
    assert res.table.loc["Positive", "Wald"] >= 0


def test_spectral_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.spectral(y, x, nfreq=10, show=False, plot=False)
    assert (res.summary["n_reject_5pct"] >= 0).all()


def test_quantile_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.quantile(y, x, quantiles=(0.25, 0.5, 0.75),
                           show=False, plot=False)
    assert len(res.table) == 6   # 2 shocks x 3 quantiles


def test_efficient_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.efficient(y, x, max_lag=3, show=False, plot=False)
    for k in ("W_pos", "W_neg", "W_joint", "W_diff"):
        assert k in res.raw


def test_all_runs(asym_dgp):
    y, x = asym_dgp
    res = asycaus.all_tests(y, x, max_lag=3, boot=60,
                            skip_dynamic=True, skip_spectral=True,
                            show=False, plot=False)
    assert len(res.summary) > 0


def test_recovers_asymmetry(asym_dgp):
    """The asymmetric DGP should give stronger evidence for POS than NEG."""
    y, x = asym_dgp
    res = asycaus.fourier(y, x, kmax=2, form="single", show=False, plot=False)
    assert res.table.loc["Positive", "Wald"] > res.table.loc["Negative", "Wald"]
