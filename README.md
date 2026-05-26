# asycaus — Asymmetric Granger-Causality Suite for Python

[![PyPI version](https://img.shields.io/pypi/v/asycaus.svg?logo=pypi&logoColor=white&color=3776AB)](https://pypi.org/project/asycaus/)
[![Python versions](https://img.shields.io/pypi/pyversions/asycaus.svg?logo=python&logoColor=white)](https://pypi.org/project/asycaus/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-merwanroudane%2Fasycaus-181717?logo=github)](https://github.com/merwanroudane/asycaus)
[![Docs](https://img.shields.io/badge/docs-merwanroudane.github.io-1A8FE0?logo=readthedocs&logoColor=white)](https://merwanroudane.github.io/asycaus/)

**Links** &nbsp;·&nbsp; 📦 [PyPI](https://pypi.org/project/asycaus/) &nbsp;·&nbsp; 💻 [GitHub](https://github.com/merwanroudane/asycaus) &nbsp;·&nbsp; 📚 [Documentation site](https://merwanroudane.github.io/asycaus/) &nbsp;·&nbsp; 📊 Stata twin on SSC: `ssc install asycaus`

A complete Python implementation of asymmetric Granger-causality tests for
bivariate time series. The package is a faithful mirror of the companion
**Stata `asycaus`** library by the same author, with publication-quality
tables (powered by [`rich`](https://github.com/Textualize/rich)) and plots
(`matplotlib`).

> **Author:** Dr Merwan Roudane &nbsp;·&nbsp; <merwanroudane920@gmail.com>

---

## Methods implemented

| Function | Test | Reference |
|---|---|---|
| `asycaus.static` | Static asymmetric causality with leverage bootstrap | Hatemi-J (2012); Hacker & Hatemi-J (2006, 2012) |
| `asycaus.dynamic` | Rolling / recursive time-varying asymmetric causality | Hatemi-J (2021) |
| `asycaus.fourier` | Fourier-augmented asymmetric Toda-Yamamoto | Nazlioglu, Gormus & Soytas (2016); Pata (2020) |
| `asycaus.spectral` | Frequency-domain BC test on Pos / Neg components | Bahmani-Oskooee, Chang & Ranjbar (2016); Breitung & Candelon (2006) |
| `asycaus.quantile` | Quantile asymmetric causality (+ optional Fourier detrending) | Fang, Wang, Shieh & Chung (2026) |
| `asycaus.efficient` | SUR-based joint Pos / Neg / Joint / Pos=Neg test | Hatemi-J (2024) |
| `asycaus.all_tests` | Full battery with unified summary table | this package |
| `asycaus.pos_neg_components` | Utility: build cumulative shock components | Granger & Yoon (2002) |

The default lag-selection criterion across the library is the
**Hatemi-J information criterion** (HJC; Hatemi-J 2003), with AIC, AICC,
SBC, HQC also available.

---

## Installation

```bash
# Once published to PyPI:
pip install asycaus

# Directly from GitHub:
pip install git+https://github.com/merwanroudane/asycaus.git

# Or from a local clone:
git clone https://github.com/merwanroudane/asycaus.git
pip install -e asycaus
```

**Dependencies** (installed automatically):
`numpy`, `scipy`, `pandas`, `matplotlib`, `rich`.

Python **≥ 3.9** is required.

---

## Quick start

```python
import numpy as np
import asycaus

# An asymmetric DGP: positive shocks in x cause y, negative do not.
rng = np.random.default_rng(30540)
ex  = rng.standard_normal(300)
ey  = rng.standard_normal(300)
x   = np.cumsum(ex)
y   = np.zeros(300)
for t in range(1, 300):
    y[t] = 0.5 * y[t - 1] + 0.7 * max(ex[t - 1], 0) + ey[t]

# --- 1. Static Hatemi-J (2012) ----------------------------------------
asycaus.static(y, x, shock="both", boot=300)

# --- 2. Dynamic Hatemi-J (2021) rolling window ------------------------
asycaus.dynamic(y, x, shock="pos", mode="rolling", boot=150)

# --- 3. Fourier (Nazlioglu et al. 2016) -------------------------------
asycaus.fourier(y, x, kmax=3, form="single")

# --- 4. Spectral (Bahmani-Oskooee et al. 2016) ------------------------
asycaus.spectral(y, x, nfreq=50)

# --- 5. Quantile (Fang et al. 2026), with Fourier detrending ----------
asycaus.quantile(y, x, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                 fourier=True, kmax=2)

# --- 6. Efficient SUR (Hatemi-J 2024) ---------------------------------
asycaus.efficient(y, x, max_lag=4)

# --- Everything in one call -------------------------------------------
asycaus.all_tests(y, x, max_lag=4, boot=300, skip_dynamic=True)
```

Every test prints a coloured, boxed table (via `rich`) and renders a
publication-quality plot (set `plot=False` to suppress; `show=False` to
suppress the table; both return a `Result` dataclass).

---

## Detailed function reference

### `asycaus.static(y, x, *, shock='both', max_lag=8, ic='hjc', intorder=1, boot=1000, seed=12345, lnform=False, show=True, plot=True)`

**Hatemi-J (2012) static asymmetric causality with leverage-adjusted bootstrap.**

- `y`, `x`: 1-D arrays of equal length (length T). H₀: *x* does **not** Granger-cause *y*.
- `shock`: `'pos'` | `'neg'` | `'both'`. The cumulative components to test.
- `max_lag`: integer, max VAR lag for the IC search.
- `ic`: `'aic'` | `'aicc'` | `'sbc'` | `'hqc'` | `'hjc'` (default).
- `intorder`: integer ≥ 0, Toda-Yamamoto augmentation lags (max integration order).
- `boot`: bootstrap replications.
- `seed`: RNG seed (`None` for non-reproducible).
- `lnform`: take `np.log()` of inputs before the decomposition.
- `show` / `plot`: print the table / render the bar plot.

Returns `StaticResult`:
- `.table` — `pandas.DataFrame` indexed by shock, columns `Wald`, `lag`, `dof`, `asy_p`, `cv10`, `cv5`, `cv1`, `decision_5pct`.
- `.depvar`, `.causvar`, `.ic`, `.boot`, `.sample_size`, `.intorder`.

---

### `asycaus.dynamic(y, x, *, shock='pos', mode='rolling', window=None, max_lag=4, ic='hjc', intorder=1, boot=200, seed=12345, lnform=False, show=True, plot=True, progress=True)`

**Hatemi-J (2021) time-varying asymmetric causality.**

- `mode`: `'rolling'` (fixed window, slides by 1) or `'recursive'` (anchored at observation 1, expanding).
- `window`: window size *S*; defaults to the Phillips-Shi-Yu (2015) lower bound
  *S = ceil(T (0.01 + 1.8/√T))*.
- `progress`: print `subsample k/N` every 10 windows.

Returns `DynamicResult`:
- `.table` — DataFrame columns `sub_start`, `sub_end`, `lag`, `Wald`, `cv10`, `cv5`, `cv1`, `ratio_5pct`.
- `.mode`, `.window`, `.smin`, `.nsub`, `.shock`, `.ic`, `.boot`, `.intorder`.

---

### `asycaus.fourier(y, x, *, shock='both', kmax=5, form='single', max_lag=8, ic='hjc', intorder=1, lnform=False, show=True, plot=True)`

**Fourier-augmented asymmetric Toda-Yamamoto (Nazlioglu et al. 2016; Pata 2020).**

- `kmax`: maximum Fourier frequency. The function searches `k = 1..kmax` and selects the most informative *k*\*.
- `form`: `'single'` (sin/cos at frequency *k*) or `'cumulative'` (basis `k = 1..k_max`).

Returns `FourierResult` with `.table` indexed by shock, columns
`Wald`, `lag`, `k_opt`, `asy_p`, `sample`, `decision_5pct`.

---

### `asycaus.spectral(y, x, *, shock='both', nfreq=50, max_lag=8, ic='hjc', lnform=False, show=True, plot=True)`

**Asymmetric frequency-domain causality (Bahmani-Oskooee et al. 2016 — BC 2006 applied to Pos/Neg components).**

- `nfreq`: number of grid points in `ω ∈ (0, π]`.

Returns `SpectralResult` with:
- `.table` — full grid (`shock_id`, `shock`, `omega`, `Wald`, `cv10`, `cv5`, `cv1`, `lag`)
- `.summary` — per-shock count of rejections at 1%, 5%, 10%

---

### `asycaus.quantile(y, x, *, shock='both', quantiles=(0.1,0.25,0.5,0.75,0.9), max_lag=4, ic='hjc', intorder=1, fourier=False, kmax=3, lnform=False, show=True, plot=True)`

**Quantile asymmetric causality (Fang, Wang, Shieh & Chung 2026).**

- `quantiles`: iterable of quantile probabilities in (0, 1).
- `fourier=True`: project out a cumulative Fourier basis (sin/cos with *k = 1..kmax*) before estimating the quantile causality.

Returns `QuantileResult` with `.table` columns
`shock_id`, `shock`, `tau`, `Wald`, `lag`, `asy_p`, `decision_5pct`.

---

### `asycaus.efficient(y, x, *, max_lag=8, ic='hjc', intorder=1, lnform=False, show=True, plot=True)`

**Hatemi-J (2024) efficient asymmetric causality via SUR.**

Reports four hypotheses jointly:

1. No causality through **Pos** shocks
2. No causality through **Neg** shocks
3. **Joint** no causality
4. **Pos = Neg** causal coefficients — the formal asymmetry test

Returns `EfficientResult` with:
- `.table` indexed by hypothesis: `Wald`, `df`, `asy_p`, `decision_5pct`
- `.raw` dict with `W_pos`, `p_pos`, `W_neg`, `p_neg`, `W_joint`, `p_joint`, `W_diff`, `p_diff`, `dof`

---

### `asycaus.all_tests(y, x, *, max_lag=4, ic='hjc', intorder=1, boot=500, seed=12345, kmax=5, nfreq=50, quantiles=(0.1,0.25,0.5,0.75,0.9), window=None, lnform=False, skip_dynamic=False, skip_spectral=False, skip_quantile=False, show=True, plot=False)`

**Run the full battery and print a unified summary.**

Returns `AllResult` containing every individual result plus:
- `.summary` — DataFrame with rows of the form (Test, Shock, Statistic, p-value, Decision).
- `.plot()` — assembles a 6-panel dashboard.

`skip_dynamic`, `skip_spectral`, `skip_quantile` let you turn off the slowest tests.

---

### `asycaus.pos_neg_components(Y, positive=True)`

Utility: cumulative sum of positive (or negative) first differences of `Y` (T × K).
Returns a (T-1) × K matrix.

---

## Output styling

Tables use `rich` for colour output if available, falling back to a clean
ASCII boxed format otherwise. Plots use matplotlib with a consistent
publication-quality theme (navy / orange / green / red).

Each `Result` dataclass exposes `.print()` and `.plot()` so you can re-render
without re-running the test:

```python
r = asycaus.static(y, x, show=False, plot=False)   # silent
r.print()                                          # boxed table
fig = r.plot(save="static.png")                    # save the figure
```

---

## References

- Bahmani-Oskooee, M., Chang, T., & Ranjbar, O. (2016). Asymmetric causality using frequency-domain and time-frequency-domain (wavelet) approaches. *Economic Modelling*, 56, 66–78.
- Breitung, J., & Candelon, B. (2006). Testing for short- and long-run causality: a frequency-domain approach. *Journal of Econometrics*, 132, 363–378.
- Enders, W., & Lee, J. (2012). The flexible Fourier form and Dickey-Fuller type unit root tests. *Economics Letters*, 117(1), 196–199.
- Fang, H., Wang, C.-H., Shieh, J. C. P., & Chung, C.-P. (2026). The asymmetric Granger causality between banking-sector and stock-market development and economic growth in quantiles considering Fourier. *Applied Economics*, 58(20), 3822–3838.
- Granger, C. W. J., & Yoon, G. (2002). Hidden cointegration. *UCSD Discussion Paper* 2002-02.
- Hacker, R. S., & Hatemi-J, A. (2006). Tests for causality between integrated variables using asymptotic and bootstrap distributions: theory and application. *Applied Economics*, 38(13), 1489–1500.
- Hacker, R. S., & Hatemi-J, A. (2012). A bootstrap test for causality with endogenous lag length choice: theory and application in finance. *Journal of Economic Studies*, 39(2), 144–160.
- Hatemi-J, A. (2003). A new method to choose optimal lag order in stable and unstable VAR models. *Applied Economics Letters*, 10(3), 135–137.
- Hatemi-J, A. (2012). Asymmetric causality tests with an application. *Empirical Economics*, 43, 447–456.
- Hatemi-J, A. (2021). Dynamic Asymmetric Causality Tests with an Application. *arXiv* 2106.07612.
- Hatemi-J, A. (2024). Efficient Asymmetric Causality Tests. *arXiv* 2408.03137.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Nazlioglu, S., Gormus, N. A., & Soytas, U. (2016). Oil prices and real estate investment trusts (REITs): gradual-shift causality and volatility transmission analysis. *Energy Economics*, 60, 168–175.
- Pata, U. K. (2020). How is COVID-19 affecting environmental pollution in US cities? Evidence from asymmetric Fourier causality test. *Air Quality, Atmosphere & Health*, 13, 1149–1155.
- Phillips, P. C. B., Shi, S., & Yu, J. (2015). Testing for multiple bubbles: limit theory of real-time detectors. *International Economic Review*, 56(4), 1043–1078.
- Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66, 225–250.

---

## Citation

```
Roudane, M. (2026). asycaus: Asymmetric Granger-causality suite for Python.
Version 1.0.0.  https://github.com/merwanroudane/asycaus
```

---

## License

MIT — see [LICENSE](LICENSE).

## Companion package (Stata)

The Stata twin of this library, also by the same author, is available on SSC:

```stata
ssc install asycaus
help asycaus
```
