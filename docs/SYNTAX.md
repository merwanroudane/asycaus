# asycaus — Detailed Syntax Reference

Companion reference to [`README.md`](../README.md). Every public function,
every keyword argument, and every attribute of every returned object.

> **Convention.** Throughout this document, `y` is the *dependent* series and
> `x` is the *candidate cause*. All tests evaluate
> **H₀: x does NOT Granger-cause y**.

---

## Table of contents

1. [`asycaus.static`](#asycaus.static)
2. [`asycaus.dynamic`](#asycaus.dynamic)
3. [`asycaus.fourier`](#asycaus.fourier)
4. [`asycaus.spectral`](#asycaus.spectral)
5. [`asycaus.quantile`](#asycaus.quantile)
6. [`asycaus.efficient`](#asycaus.efficient)
7. [`asycaus.all_tests`](#asycaus.all_tests)
8. [`asycaus.pos_neg_components`](#asycaus.pos_neg_components)
9. [Result dataclasses](#result-dataclasses)
10. [Common arguments](#common-arguments)
11. [Information-criterion codes](#information-criterion-codes)

---

## Common arguments

These are accepted by most tests with consistent meaning:

| Argument | Type | Default | Description |
|---|---|---|---|
| `y`, `x` | array_like | — | One-dimensional series of equal length *T*. |
| `shock` | str | varies | `'pos'`, `'neg'`, or `'both'`. |
| `max_lag` | int | varies | Maximum VAR lag for the IC search. |
| `ic` | str | `'hjc'` | `'aic'` \| `'aicc'` \| `'sbc'` \| `'hqc'` \| `'hjc'`. |
| `intorder` | int | 1 | Toda-Yamamoto augmentation lags (max integration order). |
| `boot` | int | varies | Bootstrap replications (where applicable). |
| `seed` | int or None | 12345 | RNG seed; `None` for non-deterministic. |
| `lnform` | bool | False | Take `np.log()` of inputs before decomposition. |
| `show` | bool | True | Print the boxed table to stdout. |
| `plot` | bool | True | Render the publication-quality plot. |

Every test returns a dataclass; calling `.print()` or `.plot()` later
re-renders the table or graph without re-running the test.

---

## `asycaus.static`

```python
asycaus.static(
    y, x,
    *,
    shock:    str  = "both",
    max_lag:  int  = 8,
    ic:       str  = "hjc",
    intorder: int  = 1,
    boot:     int  = 1000,
    seed:     int | None = 12345,
    lnform:   bool = False,
    show:     bool = True,
    plot:     bool = True,
) -> StaticResult
```

**Hatemi-J (2012) static asymmetric Granger-causality test** with
leverage-adjusted bootstrap critical values (Hacker & Hatemi-J 2006, 2012).

**Returns `StaticResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table` | `pandas.DataFrame` | Indexed by `'Positive'` / `'Negative'`. Columns: `Wald`, `lag`, `dof`, `asy_p`, `cv10`, `cv5`, `cv1`, `decision_5pct`. |
| `.depvar`, `.causvar` | str | Names of the series. |
| `.ic`, `.boot`, `.sample_size`, `.intorder` | misc | Options used. |
| `.print(console=None)` | method | Reprints the boxed table. |
| `.plot(ax=None, save=None)` | method | Returns the matplotlib `Figure`. |

---

## `asycaus.dynamic`

```python
asycaus.dynamic(
    y, x,
    *,
    shock:    str  = "pos",
    mode:     str  = "rolling",
    window:   int | None = None,
    max_lag:  int  = 4,
    ic:       str  = "hjc",
    intorder: int  = 1,
    boot:     int  = 200,
    seed:     int | None = 12345,
    lnform:   bool = False,
    show:     bool = True,
    plot:     bool = True,
    progress: bool = True,
) -> DynamicResult
```

**Hatemi-J (2021) rolling / recursive time-varying asymmetric causality.**

- `mode` — `'rolling'` (fixed-width window of size *S* moved one step) or
  `'recursive'` (anchored at observation 1, expanding by one each step).
- `window` — window size *S*. If `None`, uses the Phillips-Shi-Yu (2015) lower
  bound &nbsp;*S = ⌈T (0.01 + 1.8 / √T)⌉*.
- `progress` — print `subsample k/N` every 10 windows.

**Returns `DynamicResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table` | DataFrame | Columns: `sub_start`, `sub_end`, `lag`, `Wald`, `cv10`, `cv5`, `cv1`, `ratio_5pct`. |
| `.mode`, `.window`, `.smin`, `.nsub`, `.shock`, `.ic`, `.boot`, `.intorder` | misc | Options used. |

---

## `asycaus.fourier`

```python
asycaus.fourier(
    y, x,
    *,
    shock:    str  = "both",
    kmax:     int  = 5,
    form:     str  = "single",
    max_lag:  int  = 8,
    ic:       str  = "hjc",
    intorder: int  = 1,
    lnform:   bool = False,
    show:     bool = True,
    plot:     bool = True,
) -> FourierResult
```

**Fourier-augmented asymmetric Toda-Yamamoto causality** (Nazlioglu, Gormus &
Soytas 2016; Pata 2020).

- `kmax` — maximum Fourier frequency. Searches `k = 1..kmax` and reports the
  most informative `k*`.
- `form` — `'single'` (sin/cos at frequency *k*) or `'cumulative'`
  (sin/cos basis for `k = 1..kmax`).

**Returns `FourierResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table` | DataFrame | Indexed by shock; columns `Wald`, `lag`, `k_opt`, `asy_p`, `sample`, `decision_5pct`. |
| `.form`, `.kmax`, `.ic`, `.intorder` | misc | Options used. |

---

## `asycaus.spectral`

```python
asycaus.spectral(
    y, x,
    *,
    shock:   str  = "both",
    nfreq:   int  = 50,
    max_lag: int  = 8,
    ic:      str  = "hjc",
    lnform:  bool = False,
    show:    bool = True,
    plot:    bool = True,
) -> SpectralResult
```

**Asymmetric frequency-domain causality** (Bahmani-Oskooee, Chang & Ranjbar
2016, applying Breitung & Candelon 2006 to the cumulative Pos/Neg components).

- `nfreq` — number of grid points in *ω ∈ (0, π]*.

At each frequency *ω* the BC Wald statistic is computed and compared with the
*χ²(2)* critical values.

**Returns `SpectralResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table`   | DataFrame | Full grid: `shock_id`, `shock`, `omega`, `Wald`, `cv10`, `cv5`, `cv1`, `lag`. |
| `.summary` | DataFrame | Per shock: `lag`, `nfreq`, `n_reject_1pct`, `n_reject_5pct`, `n_reject_10pct`, `pct_reject_5pct`. |
| `.nfreq`, `.ic` | misc | Options used. |

---

## `asycaus.quantile`

```python
asycaus.quantile(
    y, x,
    *,
    shock:     str  = "both",
    quantiles: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    max_lag:   int  = 4,
    ic:        str  = "hjc",
    intorder:  int  = 1,
    fourier:   bool = False,
    kmax:      int  = 3,
    lnform:    bool = False,
    show:      bool = True,
    plot:      bool = True,
) -> QuantileResult
```

**Quantile asymmetric causality** (Fang, Wang, Shieh & Chung 2026).

- `quantiles` — iterable of probabilities in (0, 1).
- `fourier=True` — first detrend the components against a cumulative Fourier
  basis (`k = 1..kmax`), then run the quantile test on the residuals (matches
  the Fang et al. specification).

**Returns `QuantileResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table` | DataFrame | Long-form: `shock_id`, `shock`, `tau`, `Wald`, `lag`, `asy_p`, `decision_5pct`. |
| `.ic`, `.intorder`, `.fourier_used`, `.kmax` | misc | Options used. |

---

## `asycaus.efficient`

```python
asycaus.efficient(
    y, x,
    *,
    max_lag:  int  = 8,
    ic:       str  = "hjc",
    intorder: int  = 1,
    lnform:   bool = False,
    show:     bool = True,
    plot:     bool = True,
) -> EfficientResult
```

**Hatemi-J (2024) efficient asymmetric causality** via SUR. Reports four
hypotheses jointly:

1. No causality through **Positive** shocks.
2. No causality through **Negative** shocks.
3. **Joint** no causality (1 AND 2).
4. **Pos = Neg** causal coefficients — the formal asymmetry test.

**Returns `EfficientResult`:**

| Attribute | Type | Description |
|---|---|---|
| `.table` | DataFrame | Indexed by hypothesis; columns `Wald`, `df`, `asy_p`, `decision_5pct`. |
| `.raw`   | dict      | `W_pos`, `p_pos`, `W_neg`, `p_neg`, `W_joint`, `p_joint`, `W_diff`, `p_diff`, `dof`. |
| `.ic`, `.intorder`, `.lag` | misc | Options used. |

---

## `asycaus.all_tests`

```python
asycaus.all_tests(
    y, x,
    *,
    max_lag:       int  = 4,
    ic:            str  = "hjc",
    intorder:      int  = 1,
    boot:          int  = 500,
    seed:          int | None = 12345,
    kmax:          int  = 5,
    nfreq:         int  = 50,
    quantiles:     Iterable[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    window:        int | None = None,
    lnform:        bool = False,
    skip_dynamic:  bool = False,
    skip_spectral: bool = False,
    skip_quantile: bool = False,
    show:          bool = True,
    plot:          bool = False,
) -> AllResult
```

Runs every test on the same `(y, x)` pair and prints a unified summary table
with one row per (test × shock). Returns `AllResult`:

| Attribute | Type | Description |
|---|---|---|
| `.static_`, `.fourier_`, `.efficient_`, `.spectral_`, `.quantile_`, `.dynamic_` | per-test Result objects | All individual results, retained for inspection. |
| `.summary` | DataFrame | Test, Shock, Statistic, p-value, Decision. |
| `.plot(save=None)` | method | Renders a 6-panel dashboard combining every test. |

Use the `skip_*` flags to omit the slowest tests for quick exploration.

---

## `asycaus.pos_neg_components`

```python
asycaus.pos_neg_components(Y: np.ndarray, positive: bool = True) -> np.ndarray
```

Utility from Granger & Yoon (2002). Returns the cumulative sum of positive
(`positive=True`) or negative first differences of `Y`.

`Y` may be `(T,)` or `(T, K)`. Returns an array of shape `(T-1, K)`.

---

## Result dataclasses

All result classes share the same interface:

- `.print()` — re-render the boxed table (also via `rich` if installed).
- `.plot(ax=None, save=None)` — re-render the plot. Returns the matplotlib
  `Figure`. Pass `save="figure.png"` to write to disk.
- `.table` — the main result `pandas.DataFrame`.

So you can run silently and inspect later:

```python
res = asycaus.static(y, x, show=False, plot=False)
res.table.to_csv("static_results.csv")     # pandas integration
res.print()                                # boxed table on demand
fig = res.plot(save="static.png")          # plot on demand, save to disk
```

---

## Information-criterion codes

`ic` accepts any of the following (case-insensitive):

| Token | Code | Criterion |
|---|---|---|
| `'aic'`  | 1 | Akaike |
| `'aicc'` | 2 | Corrected Akaike |
| `'sbc'`, `'bic'`  | 3 | Schwarz / Bayesian |
| `'hqc'`, `'hqic'` | 4 | Hannan-Quinn |
| `'hjc'`, `'hatemi'` (default) | 5 | Hatemi-J (2003) = (SBC + HQC) / 2 |
| `'user'` | 6 | Use `max_lag` directly (no search) |
