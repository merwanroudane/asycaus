"""
build_demo.py
=============

Programmatically generates the executed full-demo notebook:
    examples/full_demo.ipynb

Author : Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub : https://github.com/merwanroudane/asycaus
"""

from __future__ import annotations

import os, json
import nbformat as nbf


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells: list[dict] = []

# ------------------------------------------------------------------
cells.append(md(r"""# `asycaus` — Asymmetric Granger-Causality Suite

### Full demo on real-world data: crude-oil shocks → S&P 500

**Author:** Dr Merwan Roudane &nbsp;·&nbsp; <merwanroudane920@gmail.com>

**Links:**
- GitHub: <https://github.com/merwanroudane/asycaus>
- PyPI: <https://pypi.org/project/asycaus/>
- Stata twin (SSC): `ssc install asycaus`

---

This notebook runs every test in the `asycaus` library on a real dataset and
saves all tables and figures into `docs/tables/` and `docs/figures/` so they
can be reproduced and embedded in the GitHub Pages site.

The empirical setup follows **Hatemi-J (2012)**: do positive vs negative shocks
in crude-oil prices have *different* causal effects on the world's largest
stock market?
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 1. Setup"""))

cells.append(code(r"""import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import asycaus
print('asycaus version:', asycaus.__version__)
print('Author         :', asycaus.__author__, '<' + asycaus.__email__ + '>')
print('GitHub         :', asycaus.__url__)

# Where to drop tables and figures so docs/ embeds them automatically
ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) if os.path.basename(os.getcwd()) == 'examples' else os.getcwd()
FIG_DIR = os.path.join(ROOT, 'docs', 'figures')
TBL_DIR = os.path.join(ROOT, 'docs', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)
print('FIG_DIR:', FIG_DIR)
print('TBL_DIR:', TBL_DIR)
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 2. Real data — WTI crude oil and S&P 500

We download daily closing prices from Yahoo Finance (2015–2024) and convert to
monthly log levels, the standard frequency for causality studies in finance.
"""))

cells.append(code(r"""raw = yf.download(['CL=F', '^GSPC'], start='2015-01-01', end='2024-12-31',
                  progress=False, auto_adjust=True)['Close']
raw = raw.rename(columns={'CL=F': 'oil', '^GSPC': 'sp500'}).dropna()

# Monthly frequency, end-of-month closes -> log levels
monthly = raw.resample('M').last().apply(np.log).dropna()
monthly.columns = ['ln_oil', 'ln_sp500']
print(monthly.shape, 'monthly observations')
monthly.head()
"""))

cells.append(code(r"""fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(monthly.index, monthly['ln_oil'], color='#1f3a93', lw=1.8)
axes[0].set_title('log WTI crude oil (CL=F)', fontweight='bold')
axes[0].grid(alpha=0.3)
axes[1].plot(monthly.index, monthly['ln_sp500'], color='#27ae60', lw=1.8)
axes[1].set_title('log S&P 500 (^GSPC)', fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('Monthly data, log levels — 2015 to 2024',
             y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '00_data.png'), bbox_inches='tight', dpi=170)
plt.show()
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 3. Cumulative positive / negative shocks (Granger-Yoon 2002)

Each variable is split into the cumulative sum of its positive innovations and
the cumulative sum of its negative innovations. This is the heart of the
asymmetric causality framework: instead of testing whether `oil` causes
`sp500`, we test whether **positive oil shocks** cause **positive S&P shocks**
(or any combination).
"""))

cells.append(code(r"""y = monthly['ln_sp500'].to_numpy()   # dependent variable
x = monthly['ln_oil'].to_numpy()     # candidate cause
Y = np.column_stack([y, x])

C_pos = asycaus.pos_neg_components(Y, positive=True)
C_neg = asycaus.pos_neg_components(Y, positive=False)

comp = pd.DataFrame(
    np.hstack([C_pos, C_neg]),
    columns=['cum_pos_sp500', 'cum_pos_oil', 'cum_neg_sp500', 'cum_neg_oil'],
    index=monthly.index[1:],
)
comp.head()
"""))

cells.append(code(r"""fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(comp.index, comp['cum_pos_oil'],   color='#27ae60', label='cum POS oil',   lw=1.8)
axes[0].plot(comp.index, comp['cum_neg_oil'],   color='#c0392b', label='cum NEG oil',   lw=1.8)
axes[0].set_title('Cumulative positive vs negative oil shocks', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(comp.index, comp['cum_pos_sp500'], color='#27ae60', label='cum POS S&P',   lw=1.8)
axes[1].plot(comp.index, comp['cum_neg_sp500'], color='#c0392b', label='cum NEG S&P',   lw=1.8)
axes[1].set_title('Cumulative positive vs negative S&P shocks', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle('Granger-Yoon (2002) decomposition', y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_components.png'), bbox_inches='tight', dpi=170)
plt.show()
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 4. Static asymmetric causality — Hatemi-J (2012)

Modified Wald in a VAR(p+1) on the cumulative components, with bootstrap
critical values that use the **leverage-adjustment** of Hacker & Hatemi-J
(2006, 2012). The bootstrap is robust to non-normality and ARCH — both common
in oil and equity returns.
"""))

cells.append(code(r"""static = asycaus.static(y, x, shock='both', max_lag=6,
                        boot=500, seed=12345, plot=False)
static.table.to_csv(os.path.join(TBL_DIR, '01_static.csv'))
fig = static.plot(save=os.path.join(FIG_DIR, '02_static.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 5. Dynamic asymmetric causality — Hatemi-J (2021)

The static test gives a single Wald statistic for the whole sample, but the
oil-stock relationship is known to be unstable (Gulf War, 2014–16 oil glut,
COVID crash, 2022 invasion of Ukraine). We re-estimate the test on a rolling
window of size

$$S = \left\lceil T\bigl(0.01 + 1.8/\sqrt T\bigr)\right\rceil$$

(Phillips, Shi & Yu 2015) for *positive* shocks, then for *negative* shocks.
The resulting Wald time-series shows precisely **when** asymmetric causation
turns on and off.
"""))

cells.append(code(r"""dyn_pos = asycaus.dynamic(y, x, shock='pos', mode='rolling',
                          max_lag=3, boot=120, seed=12345,
                          plot=False, progress=False)
dyn_pos.table.to_csv(os.path.join(TBL_DIR, '02_dynamic_pos.csv'))
fig = dyn_pos.plot(save=os.path.join(FIG_DIR, '03_dynamic_pos.png'))
"""))

cells.append(code(r"""dyn_neg = asycaus.dynamic(y, x, shock='neg', mode='rolling',
                          max_lag=3, boot=120, seed=12345,
                          plot=False, progress=False)
dyn_neg.table.to_csv(os.path.join(TBL_DIR, '03_dynamic_neg.csv'))
fig = dyn_neg.plot(save=os.path.join(FIG_DIR, '04_dynamic_neg.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 6. Fourier-augmented asymmetric TY — Nazlioglu et al. (2016) / Pata (2020)

Smooth structural breaks of unknown number, timing and form are absorbed by
sine/cosine terms inserted into the VAR. The frequency `k*` is chosen
automatically over `k ∈ {1,…,k_max}`.
"""))

cells.append(code(r"""fr = asycaus.fourier(y, x, shock='both', kmax=4, form='single',
                     max_lag=6, plot=False)
fr.table.to_csv(os.path.join(TBL_DIR, '04_fourier.csv'))
fig = fr.plot(save=os.path.join(FIG_DIR, '05_fourier.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 7. Frequency-domain asymmetric causality — Bahmani-Oskooee, Chang & Ranjbar (2016)

Breitung-Candelon (2006) spectral causality applied separately to the cumulative
positive and negative components. For each angular frequency $\omega \in (0, \pi]$
we report a $\chi^2(2)$ Wald statistic. The plot shows the *Wald-vs-frequency*
curve with 1%, 5%, 10% critical lines — the canonical Bahmani-Oskooee
presentation.
"""))

cells.append(code(r"""sp = asycaus.spectral(y, x, shock='both', nfreq=50, max_lag=6, plot=False)
sp.summary.to_csv(os.path.join(TBL_DIR, '05_spectral_summary.csv'))
sp.table.to_csv(os.path.join(TBL_DIR, '05_spectral_full.csv'), index=False)
fig = sp.plot(save=os.path.join(FIG_DIR, '06_spectral.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 8. Quantile asymmetric causality — Fang, Wang, Shieh & Chung (2026)

Tests whether oil causes the S&P 500 **at different quantiles** of the
S&P 500's distribution — i.e., is the causal effect bigger in bull markets,
in bear markets, or near the median? Optionally, smooth structural breaks
are first removed via a cumulative Fourier basis (`fourier=True`).
"""))

cells.append(code(r"""q = asycaus.quantile(y, x, shock='both',
                     quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                     max_lag=4, fourier=True, kmax=2, plot=False)
q.table.to_csv(os.path.join(TBL_DIR, '06_quantile.csv'), index=False)
fig = q.plot(save=os.path.join(FIG_DIR, '07_quantile.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 9. Efficient asymmetric tests — Hatemi-J (2024)

Joint SUR estimation of the positive and negative systems. Four hypotheses
are produced in one shot:

| # | Null |
|---|---|
| 1 | No causality via **positive** shocks |
| 2 | No causality via **negative** shocks |
| 3 | **Joint** no causality (1 AND 2) |
| 4 | **Positive = Negative** causal coefficients ← formal asymmetry test |
"""))

cells.append(code(r"""eff = asycaus.efficient(y, x, max_lag=6, plot=False)
eff.table.to_csv(os.path.join(TBL_DIR, '07_efficient.csv'))
fig = eff.plot(save=os.path.join(FIG_DIR, '08_efficient.png'))
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 10. Full battery + unified summary

A single `asycaus.all_tests(...)` call runs everything above and prints one
unified row per (test × shock).
"""))

cells.append(code(r"""battery = asycaus.all_tests(y, x, max_lag=4, boot=200,
                            skip_dynamic=True, skip_spectral=False,
                            plot=False)
battery.summary.to_csv(os.path.join(TBL_DIR, '08_battery_summary.csv'), index=False)
battery.summary
"""))

# ------------------------------------------------------------------
cells.append(md(r"""## 11. Conclusion

The `asycaus` library produces a **clean, end-to-end** asymmetric-causality
analysis on real financial data with one consistent API, professional tables
and publication-quality plots.

Every artifact above is saved under `docs/figures/` and `docs/tables/` for
embedding in the GitHub Pages site.

---

**Citation**

> Roudane, M. (2026). *asycaus: Asymmetric Granger-causality suite for Python*.
> Version 1.0.0. https://github.com/merwanroudane/asycaus

**Companion Stata package on SSC:** `ssc install asycaus`
"""))

nb["cells"] = cells

out_path = os.path.join(os.path.dirname(__file__), "full_demo.ipynb")
nbf.write(nb, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
