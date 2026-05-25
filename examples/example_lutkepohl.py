"""
example_lutkepohl.py
====================

Use Stata's Lutkepohl macro data (`dln_inv`, `dln_inc`) loaded from a CSV
exported from Stata to demonstrate the asycaus API on a real dataset.

To produce the CSV from Stata, run:

    sysuse lutkepohl2, clear
    export delimited qtr dln_inv dln_inc using "lutkepohl.csv", replace
"""

import pandas as pd
import asycaus

CSV = "lutkepohl.csv"   # adjust path as needed


def main():
    df = pd.read_csv(CSV).dropna(subset=["dln_inv", "dln_inc"])

    y = df["dln_inv"].to_numpy()
    x = df["dln_inc"].to_numpy()
    print(f"Loaded {len(y)} observations.")

    print("\n=== Static asymmetric (Hatemi-J 2012) ===")
    asycaus.static(y, x, shock="both", maxlag := 4, boot=400, plot=True)

    print("\n=== Dynamic asymmetric (Hatemi-J 2021) ===")
    asycaus.dynamic(y, x, shock="pos", mode="rolling",
                    window=40, max_lag=4, boot=150, plot=True)

    print("\n=== Fourier asymmetric TY (Nazlioglu et al. 2016) ===")
    asycaus.fourier(y, x, kmax=3, form="single", plot=True)

    print("\n=== Spectral asymmetric (Bahmani-Oskooee et al. 2016) ===")
    asycaus.spectral(y, x, nfreq=50, plot=True)

    print("\n=== Quantile asymmetric (Fang et al. 2026) ===")
    asycaus.quantile(y, x, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                     fourier=True, kmax=2, plot=True)

    print("\n=== Efficient SUR (Hatemi-J 2024) ===")
    asycaus.efficient(y, x, max_lag=4, plot=True)

    print("\n=== Full battery + unified summary ===")
    res = asycaus.all_tests(y, x, max_lag=4, boot=200,
                            skip_dynamic=True, plot=True)
    res.summary.to_csv("asycaus_summary.csv", index=False)
    print("Saved asycaus_summary.csv")


if __name__ == "__main__":
    main()
