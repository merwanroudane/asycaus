"""
example_simulation.py
=====================

A controlled asymmetric DGP that asycaus should recover cleanly.

Setup
-----
y_t = 0.5 * y_{t-1} + 0.7 * max(e_{x,t-1}, 0) + e_{y,t}

So x's positive shocks Granger-cause y, but x's negative shocks do not.
Every asycaus test should find significant causality on positive shocks
and (much) weaker evidence on negative shocks.

Run with:  python -m asycaus_py.examples.example_simulation
       or:  python examples/example_simulation.py
"""

import numpy as np
import asycaus


def main():
    rng = np.random.default_rng(30540)
    T = 300
    ex = rng.standard_normal(T)
    ey = rng.standard_normal(T)
    x = np.cumsum(ex)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.7 * max(ex[t - 1], 0) + ey[t]

    print("\n>>> 1. Static (Hatemi-J 2012)")
    asycaus.static(y, x, shock="both", boot=300, plot=False)

    print("\n>>> 2. Fourier (Nazlioglu et al. 2016)")
    asycaus.fourier(y, x, kmax=3, form="single", plot=False)

    print("\n>>> 3. Efficient SUR (Hatemi-J 2024)")
    asycaus.efficient(y, x, max_lag=4, plot=False)

    print("\n>>> 4. Spectral (Bahmani-Oskooee et al. 2016) -- nfreq=20")
    asycaus.spectral(y, x, nfreq=20, plot=False)

    print("\n>>> 5. Quantile (Fang et al. 2026)")
    asycaus.quantile(y, x, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                     plot=False)

    print("\n>>> 6. Dynamic (Hatemi-J 2021) rolling window")
    asycaus.dynamic(y, x, shock="pos", mode="rolling",
                    boot=80, max_lag=2, plot=False, progress=True)

    print("\n>>> 7. Unified battery")
    asycaus.all_tests(y, x, max_lag=3, boot=200,
                      skip_dynamic=True, skip_spectral=True, plot=False)


if __name__ == "__main__":
    main()
